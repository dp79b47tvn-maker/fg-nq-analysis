"""
抓取 CNN 恐懼貪婪指數(股市版)歷史資料，合併官方即時 + 第三方重建，並標註來源。

官方 CNN API (production.dataviz.cnn.io) 實測結果，分兩階段發現（2026-07-22）：

  第一階段（只測HTTP狀態碼）：不管帶多早的 start_date，實際能回傳200的最早資料點固定
  在2020-07-15，早於2020-07-14一律500。當時據此把官方/第三方切點設在2020-07-15。

  第二階段（畫時間軸圖時肉眼發現異常，回頭檢查實際數值）：2020-07-15~2021-01-21這段
  「官方」資料裡，有122天的分數是完全打死的50.0（明顯是後端某個計算窗口還沒暖機完成的
  佔位值，不是真正逐日計算出來的分數），中間還夾雜幾天離譜的近0異常值（例如2020-09-03
  的2.6分），推測是backfill過程的暫時性錯誤。也就是說，光看HTTP 200不代表資料是真的——
  這是本專案第一版分析的一個真實錯誤，已經修正：OFFICIAL_FLOOR改為2021-01-22之後第一個
  乾淨的月初2021-02-01（往後檢查到今天為止沒有再出現整段打平50的情況），巧合的是這跟
  使用者原信最早假設的2021-02-01幾乎一模一樣——原本以為「實測比較早」，其實是實測方法
  本身不夠嚴謹，使用者原本的假設反而更準。2020-07-15~2021-01-31這段改回用第三方重建
  資料，不使用官方API在這段回傳的假資料。

第三方重建資料來源：https://github.com/whit3rabbit/fear-greed-data (fear-greed.csv)
  - 涵蓋 2011-01-03 至今，且看起來持續更新中（本身也可能是每天爬CNN網頁重建）。
  - 只取其 2021-01-31（含）以前的部分，取代為第三方重建段；2021-02-01 之後一律
    以官方API資料為準，第三方資料在重疊段只用來做「重建準確度」驗證，不混入正式序列。
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

OFFICIAL_FLOOR = pd.Timestamp("2021-02-01")  # 修正後：官方API資料真正可信的下限（見上方說明）
CNN_API_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{start_date}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.cnn.com/markets/fear-greed",
}
THIRDPARTY_URL = "https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/main/fear-greed.csv"


def fetch_official(start_date="2020-07-14"):
    """呼叫CNN官方API。start_date早於2020-07-14會500，這裡固定用實測到的最早可用日期。"""
    resp = requests.get(CNN_API_URL.format(start_date=start_date), headers=HEADERS, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    raw = (DATA_DIR / "cnn_official_raw.json")
    raw.write_text(json.dumps(payload))
    hist = payload["fear_and_greed_historical"]["data"]
    df = pd.DataFrame(hist)
    df["Date"] = pd.to_datetime(df["x"], unit="ms").dt.normalize()
    df = df.rename(columns={"y": "score", "rating": "rating"})[["Date", "score", "rating"]]
    df["source"] = "official"
    df = df.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    return df


def fetch_thirdparty():
    resp = requests.get(THIRDPARTY_URL, timeout=30)
    resp.raise_for_status()
    raw_path = DATA_DIR / "thirdparty_raw.csv"
    raw_path.write_bytes(resp.content)
    df = pd.read_csv(raw_path, parse_dates=["Date"])
    df = df.rename(columns={"Fear Greed": "score", "Rating": "rating"})
    df["rating"] = df["rating"].str.lower()
    df["source"] = "reconstructed"
    df = df.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    return df[["Date", "score", "rating", "source"]]


def detect_placeholder_runs(official_df, min_run=3):
    """防呆檢查：官方API裡有沒有連續3天以上打死同一個分數的可疑佔位值段落。
    2020-07-15~2021-01-21那次就是這樣被抓到的（122天卡在50.0）。每次重新抓資料
    都跑一次，避免同樣的問題在OFFICIAL_FLOOR之後的『乾淨』區間又悄悄發生而沒被發現。"""
    s = official_df.sort_values("Date").reset_index(drop=True)
    same_as_prev = s["score"].diff().eq(0)
    run_id = (~same_as_prev).cumsum()
    runs = []
    for _, g in s.groupby(run_id):
        if len(g) >= min_run:
            runs.append({
                "start": str(g["Date"].min().date()), "end": str(g["Date"].max().date()),
                "n_days": len(g), "score": float(g["score"].iloc[0]),
            })
    return runs


def validate_reconstruction(official_df, thirdparty_df):
    """用重疊期(OFFICIAL_FLOOR至今，只取已驗證乾淨的官方資料段落)比對第三方重建 vs
    官方資料的準確度，量化可信度落差。"""
    official_df = official_df[official_df["Date"] >= OFFICIAL_FLOOR]
    merged = pd.merge(
        official_df[["Date", "score", "rating"]],
        thirdparty_df[["Date", "score", "rating"]],
        on="Date", suffixes=("_official", "_reconstructed"), how="inner",
    )
    from scipy import stats
    rho, pval = stats.spearmanr(merged["score_official"], merged["score_reconstructed"])
    mae = (merged["score_official"] - merged["score_reconstructed"]).abs().mean()
    rmse = ((merged["score_official"] - merged["score_reconstructed"]) ** 2).mean() ** 0.5
    rating_match = (merged["rating_official"] == merged["rating_reconstructed"]).mean()
    return {
        "n_overlap_days": len(merged),
        "overlap_start": str(merged["Date"].min().date()),
        "overlap_end": str(merged["Date"].max().date()),
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "mae": float(mae),
        "rmse": float(rmse),
        "rating_exact_match_rate": float(rating_match),
    }


def build_merged_series():
    official = fetch_official()
    thirdparty = fetch_thirdparty()

    placeholder_runs = detect_placeholder_runs(official)
    placeholder_runs_before_floor = [r for r in placeholder_runs if r["start"] < str(OFFICIAL_FLOOR.date())]
    placeholder_runs_after_floor = [r for r in placeholder_runs if r["start"] >= str(OFFICIAL_FLOOR.date())]
    if placeholder_runs_after_floor:
        print("⚠️  警告：OFFICIAL_FLOOR之後仍偵測到疑似佔位值段落，需要人工檢查：")
        for r in placeholder_runs_after_floor:
            print("   ", r)

    validation = validate_reconstruction(official, thirdparty)

    recon_segment = thirdparty[thirdparty["Date"] < OFFICIAL_FLOOR].copy()
    official_segment = official[official["Date"] >= OFFICIAL_FLOOR].copy()

    merged = pd.concat([recon_segment, official_segment], ignore_index=True)
    merged = merged.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)

    merged.to_csv(DATA_DIR / "fg_merged.csv", index=False)
    with open(DATA_DIR / "reconstruction_validation.json", "w") as f:
        json.dump({
            **validation,
            "official_floor": str(OFFICIAL_FLOOR.date()),
            "placeholder_runs_excluded": placeholder_runs_before_floor,
            "placeholder_runs_after_floor_check": placeholder_runs_after_floor,
        }, f, indent=2)

    print(f"合併完成：{len(merged)} 筆，{merged['Date'].min().date()} ~ {merged['Date'].max().date()}")
    print(f"  官方即時：{(merged['source']=='official').sum()} 筆（{OFFICIAL_FLOOR.date()} 起）")
    print(f"  第三方重建：{(merged['source']=='reconstructed').sum()} 筆")
    print(f"  官方API在{OFFICIAL_FLOOR.date()}之前有{len(placeholder_runs_before_floor)}段疑似佔位值"
          f"（共{sum(r['n_days'] for r in placeholder_runs_before_floor)}天）已排除、改用第三方重建：")
    for r in placeholder_runs_before_floor:
        print("   ", r)
    print(f"重建準確度驗證（重疊期 {validation['overlap_start']} ~ {validation['overlap_end']}，"
          f"n={validation['n_overlap_days']}）：")
    print(f"  Spearman rho={validation['spearman_rho']:.4f}, MAE={validation['mae']:.2f}, "
          f"RMSE={validation['rmse']:.2f}, rating完全一致率={validation['rating_exact_match_rate']:.1%}")
    return merged, validation


if __name__ == "__main__":
    build_merged_series()
