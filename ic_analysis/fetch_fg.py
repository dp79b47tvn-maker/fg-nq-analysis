"""
抓取 CNN 恐懼貪婪指數(股市版)歷史資料，合併官方即時 + 第三方重建，並標註來源。

官方 CNN API (production.dataviz.cnn.io) 實測結果（2026-07-22）：
  - 不管帶多早的 start_date，實際能回傳的最早資料點固定在 2020-07-15。
  - start_date 早於 2020-07-14 一律回傳 HTTP 500。
  - 這代表 CNN 後端本身的每日歷史資料庫下限就是 2020-07-15，不是「近N年」的滾動窗口。
  - 因此本專案「官方即時」與「第三方重建」的切點定在 2020-07-15，而不是使用者原信
    假設的 2021-02-01（實測結果比假設更早，官方可信資料範圍更長）。

第三方重建資料來源：https://github.com/whit3rabbit/fear-greed-data (fear-greed.csv)
  - 涵蓋 2011-01-03 至今，且看起來持續更新中（本身也可能是每天爬CNN網頁重建）。
  - 只取其 2020-07-14（含）以前的部分，取代為第三方重建段；2020-07-15 之後一律
    以官方API資料為準，第三方資料在重疊段只用來做「重建準確度」驗證，不混入正式序列。
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

OFFICIAL_FLOOR = pd.Timestamp("2020-07-15")  # 實測得出的官方API資料下限
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


def validate_reconstruction(official_df, thirdparty_df):
    """用重疊期(2020-07-15至今)比對第三方重建 vs 官方資料的準確度，量化可信度落差。"""
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

    validation = validate_reconstruction(official, thirdparty)

    recon_segment = thirdparty[thirdparty["Date"] < OFFICIAL_FLOOR].copy()
    official_segment = official[official["Date"] >= OFFICIAL_FLOOR].copy()

    merged = pd.concat([recon_segment, official_segment], ignore_index=True)
    merged = merged.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)

    merged.to_csv(DATA_DIR / "fg_merged.csv", index=False)
    with open(DATA_DIR / "reconstruction_validation.json", "w") as f:
        json.dump(validation, f, indent=2)

    print(f"合併完成：{len(merged)} 筆，{merged['Date'].min().date()} ~ {merged['Date'].max().date()}")
    print(f"  官方即時：{(merged['source']=='official').sum()} 筆（{OFFICIAL_FLOOR.date()} 起）")
    print(f"  第三方重建：{(merged['source']=='reconstructed').sum()} 筆")
    print(f"重建準確度驗證（重疊期 {validation['overlap_start']} ~ {validation['overlap_end']}，"
          f"n={validation['n_overlap_days']}）：")
    print(f"  Spearman rho={validation['spearman_rho']:.4f}, MAE={validation['mae']:.2f}, "
          f"RMSE={validation['rmse']:.2f}, rating完全一致率={validation['rating_exact_match_rate']:.1%}")
    return merged, validation


if __name__ == "__main__":
    build_merged_series()
