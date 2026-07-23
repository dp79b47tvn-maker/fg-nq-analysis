"""
CNN 恐懼貪婪指數(股市版) 對 NQ=F / SP500 的因子驗證分析

方法論沿用 bond_data_pipeline/factor_validation_analysis.py 的做法：
  1. IC分析：CNN指數分數 vs 未來20個交易日報酬，Spearman等級相關係數，
     一律用「不重疊取樣」(每隔20個交易日才取一筆樣本)，避免報酬期間重疊互相
     高度相關、把統計顯著性灌水。
  2. 分位數分桶：qcut切20等分，看未來20日平均報酬是否隨分數單調變化。
     這裡分桶分析改用「逐日重疊」資料(不做不重疊取樣)——這點跟IC分析的方法論
     不同，是刻意的取捨：20組要拆兩段子時期後還有夠的樣本可看，不重疊取樣在
     子時期會讓每組平均樣本數掉到個位數(尤其官方API期間，約1500個交易日/20組
     只剩約75筆再打散——仍可以撐、但更早的內部測試顯示不重疊+20組+拆子期在
     更短的期間會直接跌破自訂的n=10門檻)。分桶是描述性統計、不是拿來算顯著性
     p值，重疊窗口造成的自相關對「平均數」的偏誤遠比對「IC的p值」小，這是能
     接受的權衡，但明確寫在報告方法論頁，不要讓讀者誤以為分桶跟IC用同一套取樣。
  3. 樣本切分：官方API實測資料下限是2020-07-15(見fetch_fg.py內的說明，不是
     使用者原信假設的2021-02-01)，因此兩段子時期以此為界：
       - 第三方重建期間：2011-01-03 ~ 2020-07-14
       - 官方API期間：    2020-07-15 ~ 今
"""
import base64
import io
import json
from pathlib import Path

import glob as _glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

_cjk_font_names = []
for _p in (_glob.glob("/usr/share/fonts/**/*CJK*", recursive=True)
           + _glob.glob("/usr/share/fonts/**/*NotoSansCJK*", recursive=True)):
    try:
        _fm.fontManager.addfont(_p)
        _cjk_font_names.append(_fm.FontProperties(fname=_p).get_name())
    except Exception:
        pass
matplotlib.rcParams["font.sans-serif"] = (
    ["PingFang TC", "Heiti TC", "Arial Unicode MS"]
    + list(dict.fromkeys(_cjk_font_names))
    + ["Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"]
)
matplotlib.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

HORIZON = 20
N_BUCKETS = 20
LOW_N_WARN = 10
OFFICIAL_FLOOR = pd.Timestamp("2020-07-15")

TARGETS = {"NQ": "NQ=F (那斯達克100期貨)", "SPX": "^GSPC (標普500指數)"}


def load_data():
    fg = pd.read_csv(DATA_DIR / "fg_merged.csv", parse_dates=["Date"]).set_index("Date")
    prices = pd.read_csv(DATA_DIR / "prices.csv", parse_dates=["Date"]).set_index("Date")
    df = fg.join(prices, how="inner").sort_index()
    return df


def forward_return(price, horizon=HORIZON):
    return (price.shift(-horizon) / price - 1) * 100


def split_periods(df):
    p1 = df.loc[df.index < OFFICIAL_FLOOR].copy()
    p2 = df.loc[df.index >= OFFICIAL_FLOOR].copy()
    return {"full": df, "period1_reconstructed": p1, "period2_official": p2}


# ---------------------------------------------------------------- 1. IC分析(不重疊取樣)
def non_overlapping_ic(df, target_col, horizon=HORIZON):
    sub = df[["score"]].copy()
    sub["fwd"] = forward_return(df[target_col], horizon)
    sub = sub.dropna()
    if len(sub) < LOW_N_WARN:
        return {"rho": None, "pval": None, "n": len(sub)}
    sampled = sub.iloc[::horizon]
    if len(sampled) < 8:
        return {"rho": None, "pval": None, "n": len(sampled)}
    rho, pval = stats.spearmanr(sampled["score"], sampled["fwd"])
    return {
        "rho": float(rho) if pd.notna(rho) else None,
        "pval": float(pval) if pd.notna(pval) else None,
        "n": len(sampled),
    }


def unconditional_stats(df, target_col, horizon=HORIZON):
    """不管CNN分數是多少,這段期間『隨便挑一天』往後看horizon個交易日的報酬長怎樣。
    用來當分桶圖的『地心引力』基準線——大盤長期上漲期，這個數字天生就是正的，
    分桶柱狀圖疊在這個基準之上，柱子沒有跨過0不代表訊號無效，見報告說明。"""
    fwd = forward_return(df[target_col], horizon).dropna()
    if len(fwd) == 0:
        return None
    return {
        "mean": float(fwd.mean()),
        "median": float(fwd.median()),
        "pct_negative": float((fwd < 0).mean() * 100),
        "n": int(len(fwd)),
    }


# ---------------------------------------------------------------- 2. 分位數分桶(逐日重疊)
def bucket_analysis(df, target_col, horizon=HORIZON, buckets=N_BUCKETS):
    sub = df[["score"]].copy()
    sub["fwd"] = forward_return(df[target_col], horizon)
    sub = sub.dropna()
    if len(sub) < buckets * 3:
        return None
    try:
        sub["bucket"] = pd.qcut(sub["score"], buckets, labels=False, duplicates="drop")
    except ValueError:
        return None
    grp = sub.groupby("bucket").agg(
        mean_score=("score", "mean"),
        mean_fwd_ret=("fwd", "mean"),
        median_fwd_ret=("fwd", "median"),
        n=("fwd", "count"),
    ).reset_index()
    grp["label"] = [f"{i+1}" for i in range(len(grp))]
    grp["low_confidence"] = grp["n"] < LOW_N_WARN
    uncond = unconditional_stats(df, target_col, horizon)
    if uncond is not None:
        grp["excess_fwd_ret"] = grp["mean_fwd_ret"] - uncond["mean"]
    mono_rho, _ = stats.spearmanr(grp["bucket"], grp["mean_fwd_ret"]) if len(grp) >= 3 else (None, None)
    return {"table": grp, "monotonicity": float(mono_rho) if pd.notna(mono_rho) else None,
            "unconditional": uncond}


FEAR_HEX = "#3E5C76"   # 分桶1端：最恐懼
GREED_HEX = "#B8863B"  # 分桶20端：最貪婪
WARN_HEX = "#8E3B3B"   # 低樣本數警示（語意色，跟恐懼/貪婪主色分開）


def _lerp_hex(c1, c2, t):
    c1 = tuple(int(c1[i:i+2], 16) for i in (1, 3, 5))
    c2 = tuple(int(c2[i:i+2], 16) for i in (1, 3, 5))
    mixed = tuple(round(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
    return f"#{mixed[0]:02x}{mixed[1]:02x}{mixed[2]:02x}"


def bucket_chart_base64(bucket_result, title):
    if bucket_result is None:
        return None
    grp = bucket_result["table"]
    n_bars = max(len(grp) - 1, 1)
    fig, ax = plt.subplots(figsize=(9.5, 3.4), dpi=140)
    colors = [_lerp_hex(FEAR_HEX, GREED_HEX, i / n_bars) for i in range(len(grp))]
    bars = ax.bar(grp["label"], grp["mean_fwd_ret"], color=colors, width=0.72,
                   edgecolor=[WARN_HEX if lc else "none" for lc in grp["low_confidence"]],
                   linewidth=[2.2 if lc else 0 for lc in grp["low_confidence"]],
                   hatch=["///" if lc else None for lc in grp["low_confidence"]])
    ax.axhline(0, color="#9a9488", linewidth=0.8)
    ax.set_ylabel(f"未來{HORIZON}日平均報酬 (%)", fontsize=9)
    ax.set_xlabel("CNN指數分數分桶（1=最恐懼　→　20=最貪婪）", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.tick_params(axis="y", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for bar, v, n in zip(bars, grp["mean_fwd_ret"], grp["n"]):
        ax.annotate(f"{v:+.2f}\nn={n}", (bar.get_x() + bar.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4 if v >= 0 else -22),
                    ha="center", fontsize=6.5, linespacing=1.3)
    fig.tight_layout()
    return fig_to_base64(fig)


def excess_bucket_chart_base64(bucket_result, title):
    """跟bucket_chart_base64同一份資料，但Y軸改成『扣掉這段期間無條件平均20日報酬』後的
    超額報酬——0線代表『跟隨便挑一天沒兩樣』，負的才代表這組真的比同期平均還差。"""
    if bucket_result is None or bucket_result.get("unconditional") is None:
        return None
    grp = bucket_result["table"]
    uncond = bucket_result["unconditional"]
    n_bars = max(len(grp) - 1, 1)
    fig, ax = plt.subplots(figsize=(9.5, 3.4), dpi=140)
    colors = [_lerp_hex(FEAR_HEX, GREED_HEX, i / n_bars) for i in range(len(grp))]
    bars = ax.bar(grp["label"], grp["excess_fwd_ret"], color=colors, width=0.72,
                   edgecolor=[WARN_HEX if lc else "none" for lc in grp["low_confidence"]],
                   linewidth=[2.2 if lc else 0 for lc in grp["low_confidence"]],
                   hatch=["///" if lc else None for lc in grp["low_confidence"]])
    ax.axhline(0, color="#3a3a36", linewidth=1.1)
    ax.axhline(0, color="#9a9488", linewidth=0.6, linestyle=(0, (1, 1)))
    ax.set_ylabel("超額報酬 (%)\n（減去同期無條件平均）", fontsize=8.5)
    ax.set_xlabel("CNN指數分數分桶（1=最恐懼　→　20=最貪婪）", fontsize=9)
    ax.set_title(f"{title}\n同期無條件平均{HORIZON}日報酬基準線＝{uncond['mean']:+.2f}%（0線＝這條基準）",
                 fontsize=9)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.tick_params(axis="y", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for bar, v, n in zip(bars, grp["excess_fwd_ret"], grp["n"]):
        ax.annotate(f"{v:+.2f}\nn={n}", (bar.get_x() + bar.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4 if v >= 0 else -22),
                    ha="center", fontsize=6.5, linespacing=1.3)
    fig.tight_layout()
    return fig_to_base64(fig)


def price_trend_chart_base64(df, periods):
    """NQ=F、SP500 全樣本價格走勢（指數化為期初=100，對數座標），用來直接說明分桶圖
    疊在多大的『大盤長期上漲』基準之上——不是抽象的一句話，讀者可以直接看到那條曲線。"""
    fig, ax = plt.subplots(figsize=(9.5, 3.6), dpi=140)
    nq_idx = df["NQ"] / df["NQ"].iloc[0] * 100
    spx_idx = df["SPX"] / df["SPX"].iloc[0] * 100
    ax.plot(df.index, nq_idx, color="#23262B", linewidth=1.3, label="NQ=F（那斯達克100期貨）")
    ax.plot(df.index, spx_idx, color="#8A8577", linewidth=1.1,
            linestyle="--", label="^GSPC（標普500指數）")
    ax.set_yscale("log")
    ax.set_ylabel("指數化價格（期初=100，對數座標）", fontsize=9)
    for pname in ["period1_reconstructed", "period2_official"]:
        boundary = pd.Timestamp(periods[pname]["date_range"][0])
        if pname == "period2_official":
            ax.axvline(boundary, color="#9a9488", linewidth=0.8, linestyle=":")
            ax.annotate("官方API期間起點\n2020-07-15", (boundary, ax.get_ylim()[1]),
                        fontsize=7, color="#6C7268", ha="left", va="top",
                        xytext=(4, -4), textcoords="offset points")
    nq_valid, spx_valid = nq_idx.dropna(), spx_idx.dropna()
    nq_total = nq_valid.iloc[-1] - 100
    spx_total = spx_valid.iloc[-1] - 100
    nq_years = (nq_valid.index[-1] - nq_valid.index[0]).days / 365.25
    spx_years = (spx_valid.index[-1] - spx_valid.index[0]).days / 365.25
    nq_cagr = ((nq_valid.iloc[-1] / 100) ** (1 / nq_years) - 1) * 100
    spx_cagr = ((spx_valid.iloc[-1] / 100) ** (1 / spx_years) - 1) * 100
    years = nq_years
    ax.set_title(
        f"NQ=F 全期間累積 {nq_total:+.0f}%（年化 {nq_cagr:+.1f}%）　｜　"
        f"SP500 全期間累積 {spx_total:+.0f}%（年化 {spx_cagr:+.1f}%）　（{years:.1f}年）",
        fontsize=9.5,
    )
    ax.tick_params(axis="x", labelsize=7.5)
    ax.tick_params(axis="y", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig_to_base64(fig)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_all():
    df = load_data()
    periods = split_periods(df)

    results = {"periods": {}}
    for pname, pdf in periods.items():
        results["periods"][pname] = {
            "date_range": [str(pdf.index.min().date()), str(pdf.index.max().date())],
            "n_rows": len(pdf),
            "ic": {},
            "bucket": {},
        }
        for tcol in TARGETS:
            ic = non_overlapping_ic(pdf, tcol)
            results["periods"][pname]["ic"][tcol] = ic

            bucket = bucket_analysis(pdf, tcol)
            chart = bucket_chart_base64(
                bucket, f"{pname} | CNN指數分桶 vs 未來{HORIZON}日{TARGETS[tcol]}報酬"
            )
            excess_chart = excess_bucket_chart_base64(
                bucket, f"{pname} | 超額報酬版（vs 同期無條件平均）"
            )
            if bucket is not None:
                bucket["table"].to_csv(OUT_DIR / f"bucket_{pname}_{tcol}.csv", index=False)
                results["periods"][pname]["bucket"][tcol] = {
                    "monotonicity": bucket["monotonicity"],
                    "n_buckets": len(bucket["table"]),
                    "n_low_confidence": int(bucket["table"]["low_confidence"].sum()),
                    "chart_base64": chart,
                    "excess_chart_base64": excess_chart,
                    "unconditional": bucket["unconditional"],
                }
            else:
                results["periods"][pname]["bucket"][tcol] = None

    results["price_trend_chart_base64"] = price_trend_chart_base64(df, results["periods"])

    with open(DATA_DIR / "reconstruction_validation.json") as f:
        results["reconstruction_validation"] = json.load(f)

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps(
        {k: v["ic"] for k, v in results["periods"].items()}, indent=2
    ))
    return results


if __name__ == "__main__":
    run_all()
