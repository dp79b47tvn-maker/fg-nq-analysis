"""
CNN 恐懼貪婪指數(股市版) 對 ^NDX / SP500 的因子驗證分析

方法論沿用 bond_data_pipeline/factor_validation_analysis.py 的做法：
  1. IC分析（2026-07-23改版）：CNN指數分數 vs 未來20個交易日報酬，Spearman等級相關係數。
     rho(點估計)用「重疊取樣」(全部每日配對，使用者要求)；但顯著性p值用「不重疊取樣」
     (每隔20天取一筆)，因為重疊樣本自相關會把顯著性灌水。詳見 compute_ic() docstring。
  2. 分位數分桶：qcut切20等分，看未來20日平均報酬是否隨分數單調變化。
     這裡分桶分析改用「逐日重疊」資料(不做不重疊取樣)——這點跟IC分析的方法論
     不同，是刻意的取捨：20組要拆兩段子時期後還有夠的樣本可看，不重疊取樣在
     子時期會讓每組平均樣本數掉到個位數(尤其官方API期間，約1500個交易日/20組
     只剩約75筆再打散——仍可以撐、但更早的內部測試顯示不重疊+20組+拆子期在
     更短的期間會直接跌破自訂的n=10門檻)。分桶是描述性統計、不是拿來算顯著性
     p值，重疊窗口造成的自相關對「平均數」的偏誤遠比對「IC的p值」小，這是能
     接受的權衡，但明確寫在報告方法論頁，不要讓讀者誤以為分桶跟IC用同一套取樣。
  3. 樣本切分：官方API真正可信的資料下限是2021-02-01(見fetch_fg.py內的說明——
     2020-07-15~2021-01-21這段官方API雖然HTTP 200，但回傳的分數有122天是打死的
     50.0佔位值，不是真的逐日資料，已改用第三方重建資料取代)，因此兩段子時期以此為界：
       - 第三方重建期間：2011-01-03 ~ 2021-01-31
       - 官方API期間：    2021-02-01 ~ 今
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
import numpy as np
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
OFFICIAL_FLOOR = pd.Timestamp("2021-02-01")  # 見fetch_fg.py：修正後的官方API資料真正可信下限

TARGETS = {"NDX": "^NDX (那斯達克100指數)", "SPX": "^GSPC (標普500指數)"}


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


# ---------------------------------------------------------------- 1. IC分析(重疊取樣rho + 不重疊顯著性)
def compute_ic(df, target_col, horizon=HORIZON):
    """2026-07-23起改版：rho(點估計)改用『重疊取樣』——用全部每日的(分數, 未來報酬)配對算，
    用到全部資料、點估計更穩定(這是使用者要求的改動)。但p值/顯著性不採用重疊版本，因為重疊樣本
    相鄰點的報酬窗口大量重疊、有強烈自相關，會把顯著性灌水成虛假的『顯著』。因此另外用『不重疊
    取樣』(每隔horizon天取一筆)算一個乾淨的p值，當作可信的顯著性依據：
      - rho / n            → 重疊取樣(headline)
      - pval / n_nonoverlap → 不重疊取樣(顯著性用這個，pval欄位就是可信的那個)
      - pval_overlap        → 重疊取樣的p值，只保留給報告當『看，這個被灌水了』的對照，不作為結論依據
    """
    sub = df[["score"]].copy()
    sub["fwd"] = forward_return(df[target_col], horizon)
    sub = sub.dropna()
    empty = {"rho": None, "pval": None, "pval_overlap": None, "n": len(sub),
             "n_nonoverlap": 0, "rho_nonoverlap": None}
    if len(sub) < LOW_N_WARN:
        return empty
    rho_ov, p_ov = stats.spearmanr(sub["score"], sub["fwd"])
    sampled = sub.iloc[::horizon]
    if len(sampled) >= 8:
        rho_no, p_no = stats.spearmanr(sampled["score"], sampled["fwd"])
    else:
        rho_no, p_no = (float("nan"), float("nan"))
    return {
        "rho": float(rho_ov) if pd.notna(rho_ov) else None,
        "pval_overlap": float(p_ov) if pd.notna(p_ov) else None,
        "pval": float(p_no) if pd.notna(p_no) else None,
        "n": int(len(sub)),
        "n_nonoverlap": int(len(sampled)),
        "rho_nonoverlap": float(rho_no) if pd.notna(rho_no) else None,
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


# ---------------------------------------------------------------- 3. 中間分數的規律性(排除頭尾兩組)
MIDDLE_EXCLUDE_EACH_END = 4  # 前後各排除幾組(不是只排除頭尾各1組)，剩中間段=20-2*4=12組


def middle_zone_ic(df, target_col, horizon=HORIZON, buckets=N_BUCKETS,
                    exclude_each_end=MIDDLE_EXCLUDE_EACH_END):
    """把最恐懼端、最貪婪端各exclude_each_end組剔除，只看中間那些組『分數高低』跟
    『未來報酬』還有沒有單調關係——回答『拿掉頭尾兩端的極端讀數之後，中間是不是就
    沒有方向性了』這個問題。跟主IC表同步(2026-07-23)：rho用重疊取樣(全部每日中間段
    配對)算、顯著性用不重疊取樣的p值。"""
    sub = df[["score"]].copy()
    sub["fwd"] = forward_return(df[target_col], horizon)
    sub = sub.dropna()
    if len(sub) < buckets * 3:
        return None
    try:
        sub["bucket"] = pd.qcut(sub["score"], buckets, labels=False, duplicates="drop")
    except ValueError:
        return None
    n_actual = sub["bucket"].nunique()
    in_middle = (sub["bucket"] >= exclude_each_end) & (sub["bucket"] <= n_actual - 1 - exclude_each_end)
    middle_all = sub[in_middle]                 # 重疊：全部每日中間段配對
    middle_no = sub.iloc[::horizon]
    middle_no = middle_no[(middle_no["bucket"] >= exclude_each_end)
                          & (middle_no["bucket"] <= n_actual - 1 - exclude_each_end)]
    if len(middle_all) < 8:
        return {"rho": None, "pval": None, "n": len(middle_all)}
    rho_ov, p_ov = stats.spearmanr(middle_all["score"], middle_all["fwd"])
    if len(middle_no) >= 8:
        _, p_no = stats.spearmanr(middle_no["score"], middle_no["fwd"])
    else:
        p_no = float("nan")
    return {
        "rho": float(rho_ov) if pd.notna(rho_ov) else None,
        "pval": float(p_no) if pd.notna(p_no) else None,          # 顯著性：不重疊
        "pval_overlap": float(p_ov) if pd.notna(p_ov) else None,  # 重疊p(灌水，僅對照)
        "n": len(middle_all),
        "n_nonoverlap": len(middle_no),
        "score_range": [float(middle_all["score"].min()), float(middle_all["score"].max())],
    }


def middle_bucket_chart_base64(bucket_result, title, exclude_each_end=MIDDLE_EXCLUDE_EACH_END):
    """重用主分桶表，把最恐懼端、最貪婪端各exclude_each_end組拿掉，放大看中間段。"""
    if bucket_result is None:
        return None
    grp = bucket_result["table"]
    if len(grp) < exclude_each_end * 2 + 2:
        return None
    mid = grp.iloc[exclude_each_end:len(grp) - exclude_each_end].copy()
    n_bars = max(len(grp) - 1, 1)
    fig, ax = plt.subplots(figsize=(9.5, 3.2), dpi=140)
    colors = [_lerp_hex(FEAR_HEX, GREED_HEX, (i + exclude_each_end) / n_bars) for i in range(len(mid))]
    bars = ax.bar(mid["label"], mid["excess_fwd_ret"], color=colors, width=0.72)
    ax.axhline(0, color="#3a3a36", linewidth=1.1)
    ax.set_ylabel("超額報酬 (%)\n（減去同期無條件平均）", fontsize=8.5)
    ax.set_xlabel(f"CNN指數分數分桶（已排除最恐懼端與最貪婪端各{exclude_each_end}組）", fontsize=9)
    ax.set_title(title, fontsize=9.5)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.tick_params(axis="y", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for bar, v, n in zip(bars, mid["excess_fwd_ret"], mid["n"]):
        ax.annotate(f"{v:+.2f}", (bar.get_x() + bar.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 3 if v >= 0 else -12),
                    ha="center", fontsize=6.5)
    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------- 3b. 不同持有期的訊號強度掃描
HORIZON_SCAN_SET = [3, 5, 10, 15, 20, 30, 40, 60, 90]


def horizon_scan(df, target_col, horizons=HORIZON_SCAN_SET, buckets=N_BUCKETS):
    """固定用全樣本，把IC分析／分桶分析在一整組不同持有期(3~90個交易日)上各跑一次，
    直接回答『這個訊號到底在多短/多長的持有期內有參考價值』，而不是只看單一個20日窗口。
    每個horizon都用該horizon各自的『不重疊取樣』(採樣間距=horizon)，方法論跟主IC分析
    完全一致，只是horizon本身當成變數掃過去。"""
    rows = []
    for h in horizons:
        ic = compute_ic(df, target_col, horizon=h)
        bucket = bucket_analysis(df, target_col, horizon=h, buckets=buckets)
        fear_excess = greed_excess = fear_n = greed_n = None
        if bucket is not None:
            grp = bucket["table"]
            fear_excess = float(grp.iloc[0]["excess_fwd_ret"])
            greed_excess = float(grp.iloc[-1]["excess_fwd_ret"])
            fear_n = int(grp.iloc[0]["n"])
            greed_n = int(grp.iloc[-1]["n"])
        rows.append({
            "horizon": h, "ic": ic,
            "fear_excess": fear_excess, "greed_excess": greed_excess,
            "fear_n": fear_n, "greed_n": greed_n,
        })
    return rows


def horizon_scan_chart_base64(scan_ndx, scan_spx, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 5.6), dpi=140, sharex=True,
                                     gridspec_kw={"hspace": 0.12})
    horizons = [r["horizon"] for r in scan_ndx]

    for scan, color, label in [(scan_ndx, "#23262B", "^NDX"), (scan_spx, "#8A8577", "^GSPC")]:
        rhos = [r["ic"]["rho"] if r["ic"] and r["ic"]["rho"] is not None else float("nan") for r in scan]
        sig = [r["ic"] is not None and r["ic"]["pval"] is not None and r["ic"]["pval"] < 0.05 for r in scan]
        ax1.plot(horizons, rhos, color=color, linewidth=1.3, marker="o", markersize=4, label=label)
        for x, y, s in zip(horizons, rhos, sig):
            if s:
                ax1.scatter([x], [y], s=70, facecolors="none", edgecolors=color, linewidths=1.6, zorder=5)
    ax1.axhline(0, color="#9a9488", linewidth=0.8)
    ax1.set_ylabel("IC（Spearman rho）", fontsize=9)
    ax1.set_title(title, fontsize=10)
    ax1.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    ax1.tick_params(axis="y", labelsize=8)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)

    for scan, ls, label in [(scan_ndx, "-", "^NDX 第1組(最恐懼)超額報酬"), (scan_spx, "--", "^GSPC 第1組(最恐懼)超額報酬")]:
        vals = [r["fear_excess"] if r["fear_excess"] is not None else float("nan") for r in scan]
        ax2.plot(horizons, vals, color=FEAR_HEX, linewidth=1.3, linestyle=ls, marker="o", markersize=4, label=label)
    for scan, ls, label in [(scan_ndx, "-", "^NDX 最後一組(最貪婪)超額報酬"), (scan_spx, "--", "^GSPC 最後一組(最貪婪)超額報酬")]:
        vals = [r["greed_excess"] if r["greed_excess"] is not None else float("nan") for r in scan]
        ax2.plot(horizons, vals, color=GREED_HEX, linewidth=1.3, linestyle=ls, marker="o", markersize=4, label=label)
    ax2.axhline(0, color="#9a9488", linewidth=0.8)
    ax2.set_ylabel("超額報酬 (%)\n（vs 同期無條件平均）", fontsize=8.5)
    ax2.set_xlabel("持有期（交易日）", fontsize=9)
    ax2.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    ax2.tick_params(axis="x", labelsize=8)
    ax2.tick_params(axis="y", labelsize=8)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------- 4. 策略回測
def build_strategy_returns(df, target_col, mode):
    """訊號用第t日收盤分數決定，套用在第t+1日的報酬（shift(1)，不偷看未來）。
    long_only_tilt：分數0分=100%多單，分數50分以上=空手(不放空)，線性內插。
    long_short：跟bond_data_pipeline的position_size()公式一樣，分數100分=100%空單。
    buy_hold：對照組，全程100%多單。"""
    sub = df[["score", target_col]].copy().dropna()
    sub["ret"] = sub[target_col].pct_change()
    if mode == "buy_hold":
        pos = pd.Series(1.0, index=sub.index)
    elif mode == "long_only_tilt":
        pos = ((50 - sub["score"]) / 50).clip(0, 1)
    elif mode == "long_short":
        pos = ((50 - sub["score"]) / 50).clip(-1, 1)
    else:
        raise ValueError(mode)
    sub["pos"] = pos
    sub["strat_ret"] = sub["pos"].shift(1) * sub["ret"]
    return sub["strat_ret"].dropna(), sub["pos"]


def strategy_metrics(strat_ret, avg_pos=None, periods_per_year=252):
    strat_ret = strat_ret.dropna()
    if len(strat_ret) < 60:
        return None
    cum = (1 + strat_ret).cumprod()
    years = len(strat_ret) / periods_per_year
    cagr = cum.iloc[-1] ** (1 / years) - 1
    vol = strat_ret.std() * (periods_per_year ** 0.5)
    sharpe = (strat_ret.mean() * periods_per_year) / vol if vol > 0 else None
    downside = strat_ret[strat_ret < 0]
    downside_vol = downside.std() * (periods_per_year ** 0.5) if len(downside) > 5 else None
    sortino = (strat_ret.mean() * periods_per_year) / downside_vol if downside_vol else None
    running_max = cum.cummax()
    max_dd = (cum / running_max - 1).min()
    return {
        "total_return": float(cum.iloc[-1] - 1) * 100,
        "cagr": float(cagr) * 100,
        "vol": float(vol) * 100,
        "sharpe": float(sharpe) if sharpe is not None else None,
        "sortino": float(sortino) if sortino is not None else None,
        "max_dd": float(max_dd) * 100,
        "win_rate": float((strat_ret > 0).mean()) * 100,
        "avg_abs_position": float(avg_pos.abs().mean()) if avg_pos is not None else None,
        "n": len(strat_ret),
        "years": float(years),
    }


def backtest_equity_chart_base64(curves, title):
    fig, ax = plt.subplots(figsize=(9.5, 3.6), dpi=140)
    style_map = {
        "buy_hold": {"color": "#8A8577", "linestyle": "--", "label": "買進持有(對照組)"},
        "long_only_tilt": {"color": FEAR_HEX, "linestyle": "-", "label": "恐懼多單/貪婪空手(不放空)"},
        "long_short": {"color": GREED_HEX, "linestyle": "-", "label": "恐懼多單/貪婪放空(對稱)"},
    }
    for mode, cum in curves.items():
        s = style_map.get(mode, {})
        ax.plot(cum.index, cum.values, linewidth=1.3, **s)
    ax.set_yscale("log")
    ax.set_ylabel("累積淨值（期初=100，對數座標）", fontsize=9)
    ax.set_title(title, fontsize=9.5)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.tick_params(axis="y", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig_to_base64(fig)


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


# ---------------------------------------------------------------- 2b. 第20分位往下拆細
def top_bucket_drilldown(df, target_col, horizon=HORIZON, buckets=N_BUCKETS, sub_groups=5):
    """把第20分位(最貪婪那~5%)單獨抓出來，往下拆細看內部有沒有趨勢。同時做兩種切法：
      (a) fixed：在最貪婪組的分數範圍內切成 sub_groups 個等寬分數區間(例如78-82,82-86...)，
          可讀性高、可以直接講『分數90以上 vs 78-82』，但每組樣本數不均。
      (b) qcut：把最貪婪組的樣本再平均切成 sub_groups 等分，每組樣本數均勻、統計上較穩，
          但每組的分數範圍標籤不漂亮。
    兩種都看未來horizon日的『超額報酬』(減去全樣本無條件平均，跟其它超額報酬圖一致)。
    分桶跟主分桶圖一樣用逐日重疊資料。"""
    sub = df[["score"]].copy()
    sub["fwd"] = forward_return(df[target_col], horizon)
    sub = sub.dropna()
    if len(sub) < buckets * 3:
        return None
    try:
        sub["bucket"] = pd.qcut(sub["score"], buckets, labels=False, duplicates="drop")
    except ValueError:
        return None
    top = sub[sub["bucket"] == sub["bucket"].max()].copy()
    if len(top) < sub_groups * 3:
        return None
    uncond = unconditional_stats(df, target_col, horizon)
    base = uncond["mean"] if uncond else 0.0

    lo, hi = float(top["score"].min()), float(top["score"].max())
    edges = [lo + (hi - lo) * i / sub_groups for i in range(sub_groups + 1)]
    edges[-1] = hi + 1e-6
    top["fixed"] = pd.cut(top["score"], bins=edges, labels=False, include_lowest=True)
    fixed_rows = []
    for g in range(sub_groups):
        grp = top[top["fixed"] == g]
        if len(grp) == 0:
            fixed_rows.append({"label": f"{edges[g]:.0f}-{edges[g+1]:.0f}", "excess": None,
                               "mean_fwd": None, "n": 0})
        else:
            fixed_rows.append({
                "label": f"{grp['score'].min():.0f}-{grp['score'].max():.0f}",
                "excess": float(grp["fwd"].mean() - base),
                "mean_fwd": float(grp["fwd"].mean()), "n": int(len(grp)),
            })

    try:
        top["q"] = pd.qcut(top["score"], sub_groups, labels=False, duplicates="drop")
    except ValueError:
        top["q"] = 0
    qcut_rows = []
    for g in sorted(top["q"].unique()):
        grp = top[top["q"] == g]
        qcut_rows.append({
            "label": f"{grp['score'].min():.1f}-{grp['score'].max():.1f}",
            "excess": float(grp["fwd"].mean() - base),
            "mean_fwd": float(grp["fwd"].mean()), "n": int(len(grp)),
        })
    return {"fixed": fixed_rows, "qcut": qcut_rows, "base": float(base),
            "top_score_range": [lo, hi], "top_n": int(len(top))}


def top_bucket_drilldown_chart_base64(dd_map, title):
    """單張圖：只畫 SP500 等數量分位(qcut)。樣本數<10的子組用警示色標。"""
    dd = dd_map.get("SPX")
    if dd is None:
        return None
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.2), dpi=140)
    rows = dd["qcut"]
    labels = [r["label"] for r in rows]
    vals = [r["excess"] if r["excess"] is not None else 0.0 for r in rows]
    ns = [r["n"] for r in rows]
    colors = [GREED_HEX if n >= LOW_N_WARN else WARN_HEX for n in ns]
    ax.bar(range(len(rows)), vals, color=colors, width=0.7)
    ax.axhline(0, color="#3a3a36", linewidth=1.0)
    vmax = max(vals + [0]); vmin = min(vals + [0])
    pad = (vmax - vmin) * 0.28 or 0.5
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_title("^GSPC｜等數量分位(qcut)", fontsize=9, pad=8)
    ax.set_ylabel("超額報酬 (%)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.annotate(f"{v:+.2f}\nn={n}", (i, v), textcoords="offset points",
                    xytext=(0, 3 if v >= 0 else -15), ha="center", fontsize=6.2, linespacing=1.2)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig_to_base64(fig)


# ---------------------------------------------------------------- 3c. 報酬趨勢熱力圖（分數十分位 × 持有期1~60）
HEATMAP_MAX_HORIZON = 60
HEATMAP_DECILES = 10


def horizon_return_heatmap(df, target_col, max_h=HEATMAP_MAX_HORIZON, deciles=HEATMAP_DECILES):
    """回傳 raw / excess 兩個矩陣（列=分數十分位 D1恐懼→D{deciles}貪婪，欄=持有期1~max_h天）。
    raw[d, h-1]    = 分數第d十分位、持有h天的平均未來報酬（重疊取樣：用全部每日配對，
                     每個持有期各自算一次，這是使用者要求的重疊取樣，橫軸要每日顆粒度也非用它不可）。
    excess[d, h-1] = raw[d, h-1] - 該h天『所有樣本不分分數』的無條件平均報酬（扣掉大盤地心引力，
                     0=跟隨便挑一天沒兩樣，才看得出分數的影響）。
    分組(qcut十分位)固定用分數本身切一次、跨所有持有期共用，讓每一列在不同持有期之間可比較。"""
    dec = pd.qcut(df["score"], deciles, labels=False, duplicates="drop")
    n_dec = int(dec.max()) + 1 if dec.notna().any() else 0
    raw = np.full((n_dec, max_h), np.nan)
    excess = np.full((n_dec, max_h), np.nan)
    for h in range(1, max_h + 1):
        fwd = forward_return(df[target_col], h)
        uncond = fwd.dropna().mean()
        tmp = pd.DataFrame({"dec": dec, "fwd": fwd}).dropna()
        g = tmp.groupby("dec")["fwd"].mean()
        for d in g.index:
            raw[int(d), h - 1] = g[d]
            excess[int(d), h - 1] = g[d] - uncond
    tmp0 = pd.DataFrame({"dec": dec, "score": df["score"]}).dropna()
    ranges, ns = [], []
    for d in range(n_dec):
        s = tmp0[tmp0["dec"] == d]["score"]
        ranges.append([float(s.min()), float(s.max())])
        ns.append(int(len(s)))
    return {"raw": raw.tolist(), "excess": excess.tolist(),
            "score_ranges": ranges, "decile_n": ns, "max_h": max_h}


def heatmap_chart_base64(hm_map, metric_key, title, cbar_label):
    """一張圖、上下兩個panel（^NDX / SP500）。coolwarm色階：紅=報酬高、藍=報酬低
    （刻意不用紅綠，因為台股紅=漲綠=跌跟美國相反，紅綠會誤導；紅=高剛好對齊台灣紅=賺）。
    色階對稱、以0為中心，上下限用98百分位穩健值，避免單一極端格洗白整張圖。"""
    targets = [t for t in hm_map if hm_map[t] is not None]
    if not targets:
        return None
    mats = {t: np.array(hm_map[t][metric_key], dtype=float) for t in targets}
    allvals = np.concatenate([m[~np.isnan(m)].ravel() for m in mats.values()])
    vlim = float(np.nanpercentile(np.abs(allvals), 98)) or 1.0

    fig, axes = plt.subplots(len(targets), 1, figsize=(9.8, 2.7 * len(targets) + 0.6), dpi=140,
                             constrained_layout=True)
    if len(targets) == 1:
        axes = [axes]
    im = None
    for ax, t in zip(axes, targets):
        m = mats[t]
        im = ax.imshow(m, aspect="auto", cmap="coolwarm", vmin=-vlim, vmax=vlim, origin="lower")
        ranges = hm_map[t]["score_ranges"]
        ax.set_yticks(range(m.shape[0]))
        ax.set_yticklabels([f"D{d+1}（{ranges[d][0]:.0f}-{ranges[d][1]:.0f}分）" for d in range(m.shape[0])],
                           fontsize=6.5)
        xt = [0, 9, 19, 29, 39, 49, 59]
        ax.set_xticks([x for x in xt if x < m.shape[1]])
        ax.set_xticklabels([str(x + 1) for x in xt if x < m.shape[1]], fontsize=7.5)
        ax.set_xlabel("持有期（交易日）", fontsize=8.5)
        ax.set_ylabel("CNN分數十分位\nD1恐懼→D10貪婪", fontsize=8)
        ax.set_title(TARGETS[t].split(" ")[0], fontsize=9)
    fig.suptitle(title, fontsize=10.5)
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.015)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    return fig_to_base64(fig)


def price_trend_chart_base64(df, periods):
    """^NDX、SP500 全樣本價格走勢（指數化為期初=100，對數座標），用來直接說明分桶圖
    疊在多大的『大盤長期上漲』基準之上——不是抽象的一句話，讀者可以直接看到那條曲線。"""
    fig, ax = plt.subplots(figsize=(9.5, 3.6), dpi=140)
    ndx_idx = df["NDX"] / df["NDX"].iloc[0] * 100
    spx_idx = df["SPX"] / df["SPX"].iloc[0] * 100
    ax.plot(df.index, ndx_idx, color="#23262B", linewidth=1.3, label="^NDX（那斯達克100指數）")
    ax.plot(df.index, spx_idx, color="#8A8577", linewidth=1.1,
            linestyle="--", label="^GSPC（標普500指數）")
    ax.set_yscale("log")
    ax.set_ylabel("指數化價格（期初=100，對數座標）", fontsize=9)
    for pname in ["period1_reconstructed", "period2_official"]:
        boundary = pd.Timestamp(periods[pname]["date_range"][0])
        if pname == "period2_official":
            ax.axvline(boundary, color="#9a9488", linewidth=0.8, linestyle=":")
            ax.annotate("官方API期間起點\n2021-02-01", (boundary, ax.get_ylim()[1]),
                        fontsize=7, color="#6C7268", ha="left", va="top",
                        xytext=(4, -4), textcoords="offset points")
    ndx_valid, spx_valid = ndx_idx.dropna(), spx_idx.dropna()
    ndx_total = ndx_valid.iloc[-1] - 100
    spx_total = spx_valid.iloc[-1] - 100
    ndx_years = (ndx_valid.index[-1] - ndx_valid.index[0]).days / 365.25
    spx_years = (spx_valid.index[-1] - spx_valid.index[0]).days / 365.25
    ndx_cagr = ((ndx_valid.iloc[-1] / 100) ** (1 / ndx_years) - 1) * 100
    spx_cagr = ((spx_valid.iloc[-1] / 100) ** (1 / spx_years) - 1) * 100
    years = ndx_years
    ax.set_title(
        f"^NDX 全期間累積 {ndx_total:+.0f}%（年化 {ndx_cagr:+.1f}%）　｜　"
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


EXTREME_FEAR_TH = 25
EXTREME_GREED_TH = 75


def timeline_comparison_chart_base64(df):
    """上下兩層對齊圖：上層是CNN指數本身的時間軸(標出25/75門檻)，下層是^NDX/SP500指數化
    價格走勢，並在股價線上用綠點標出CNN指數<25(最恐懼)、紅點標出>75(最貪婪)的那些日子，
    方便直接用肉眼比對『極端讀數出現時，股價實際在做什麼』。"""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.5, 6.6), dpi=140, sharex=True,
        gridspec_kw={"height_ratios": [1, 1.7], "hspace": 0.06},
    )

    is_official = df["source"] == "official"
    ax1.plot(df.index[~is_official], df["score"][~is_official], color=FEAR_HEX,
              linewidth=0.7, alpha=0.5, label="第三方重建")
    ax1.plot(df.index[is_official], df["score"][is_official], color=FEAR_HEX,
              linewidth=0.9, label="官方API")
    ax1.axhspan(0, EXTREME_FEAR_TH, color="#2E7D4F", alpha=0.08, linewidth=0)
    ax1.axhspan(EXTREME_GREED_TH, 100, color="#B23B3B", alpha=0.08, linewidth=0)
    ax1.axhline(EXTREME_FEAR_TH, color="#2E7D4F", linewidth=0.8, linestyle="--")
    ax1.axhline(EXTREME_GREED_TH, color="#B23B3B", linewidth=0.8, linestyle="--")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("CNN指數分數", fontsize=9)
    ax1.text(df.index[10], EXTREME_FEAR_TH - 3, f"{EXTREME_FEAR_TH}＝最恐懼門檻",
              fontsize=7, color="#2E7D4F", va="top")
    ax1.text(df.index[10], EXTREME_GREED_TH + 3, f"{EXTREME_GREED_TH}＝最貪婪門檻",
              fontsize=7, color="#B23B3B", va="bottom")
    ax1.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=2)
    ax1.tick_params(axis="y", labelsize=8)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)

    ndx_idx = df["NDX"] / df["NDX"].iloc[0] * 100
    spx_idx = df["SPX"] / df["SPX"].iloc[0] * 100
    ax2.plot(df.index, ndx_idx, color="#23262B", linewidth=1.0, label="^NDX（那斯達克100指數）")
    ax2.plot(df.index, spx_idx, color="#8A8577", linewidth=0.9, linestyle="--",
              label="^GSPC（標普500指數）")
    ax2.set_yscale("log")

    fear_mask = df["score"] < EXTREME_FEAR_TH
    greed_mask = df["score"] > EXTREME_GREED_TH
    n_fear, n_greed = int(fear_mask.sum()), int(greed_mask.sum())
    ax2.scatter(df.index[fear_mask], ndx_idx[fear_mask], color="#2E7D4F", s=9, zorder=5,
                label=f"分數<{EXTREME_FEAR_TH}最恐懼（{n_fear}天）")
    ax2.scatter(df.index[fear_mask], spx_idx[fear_mask], color="#2E7D4F", s=7, zorder=5)
    ax2.scatter(df.index[greed_mask], ndx_idx[greed_mask], color="#B23B3B", s=9, zorder=5,
                label=f"分數>{EXTREME_GREED_TH}最貪婪（{n_greed}天）")
    ax2.scatter(df.index[greed_mask], spx_idx[greed_mask], color="#B23B3B", s=7, zorder=5)
    ax2.set_ylabel("指數化價格（期初=100，對數座標）", fontsize=9)
    ax2.tick_params(axis="x", labelsize=8)
    ax2.tick_params(axis="y", labelsize=8)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    ax2.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=2)

    fig.tight_layout()
    return fig_to_base64(fig), {"n_fear": n_fear, "n_greed": n_greed}


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
        results["periods"][pname]["middle_ic"] = {}
        for tcol in TARGETS:
            ic = compute_ic(pdf, tcol)
            results["periods"][pname]["ic"][tcol] = ic
            results["periods"][pname]["middle_ic"][tcol] = middle_zone_ic(pdf, tcol)

            bucket = bucket_analysis(pdf, tcol)
            chart = bucket_chart_base64(
                bucket, f"{pname} | CNN指數分桶 vs 未來{HORIZON}日{TARGETS[tcol]}報酬"
            )
            excess_chart = excess_bucket_chart_base64(
                bucket, f"{pname} | 超額報酬版（vs 同期無條件平均）"
            )
            mid_chart = middle_bucket_chart_base64(
                bucket, f"{pname} | 排除頭尾兩極端後的中間段（超額報酬版）"
            )
            if bucket is not None:
                bucket["table"].to_csv(OUT_DIR / f"bucket_{pname}_{tcol}.csv", index=False)
                results["periods"][pname]["bucket"][tcol] = {
                    "monotonicity": bucket["monotonicity"],
                    "n_buckets": len(bucket["table"]),
                    "n_low_confidence": int(bucket["table"]["low_confidence"].sum()),
                    "chart_base64": chart,
                    "excess_chart_base64": excess_chart,
                    "middle_chart_base64": mid_chart,
                    "unconditional": bucket["unconditional"],
                }
            else:
                results["periods"][pname]["bucket"][tcol] = None

    results["price_trend_chart_base64"] = price_trend_chart_base64(df, results["periods"])

    # ---------------------------------------------------- 第20分位往下拆細（全樣本）
    dd_map = {t: top_bucket_drilldown(df, t) for t in TARGETS}
    results["top_bucket_drilldown"] = {
        "data": dd_map,
        "chart_base64": top_bucket_drilldown_chart_base64(
            dd_map, "第20分位（最貪婪~5%）往下拆細：未來20日超額報酬"),
    }

    timeline_chart, timeline_stats = timeline_comparison_chart_base64(df)
    results["timeline_chart_base64"] = timeline_chart
    results["timeline_stats"] = timeline_stats

    # ---------------------------------------------------- 報酬趨勢熱力圖（十分位 × 持有期1~60，重疊取樣）
    hm_map = {t: horizon_return_heatmap(df, t) for t in TARGETS}
    results["heatmap"] = {
        "raw_chart_base64": heatmap_chart_base64(
            hm_map, "raw", "報酬趨勢熱力圖：原始平均報酬（分數十分位 × 持有期1~60天）",
            "未來N日平均報酬 (%)"),
        "excess_chart_base64": heatmap_chart_base64(
            hm_map, "excess", "報酬趨勢熱力圖：超額報酬（減去同持有期無條件平均）",
            "超額報酬 (%)"),
        "max_h": HEATMAP_MAX_HORIZON,
    }

    # ---------------------------------------------------- 不同持有期的訊號強度掃描（全樣本）
    scan_ndx = horizon_scan(df, "NDX")
    scan_spx = horizon_scan(df, "SPX")
    results["horizon_scan"] = {
        "NDX": scan_ndx, "SPX": scan_spx,
        "chart_base64": horizon_scan_chart_base64(scan_ndx, scan_spx, "不同持有期的IC與極端分組超額報酬（全樣本）"),
    }

    # ---------------------------------------------------- 策略回測（全樣本，daily rebalance）
    results["backtest"] = {}
    for tcol in TARGETS:
        curves = {}
        metrics = {}
        for mode in ["buy_hold", "long_only_tilt", "long_short"]:
            strat_ret, pos = build_strategy_returns(df, tcol, mode)
            m = strategy_metrics(strat_ret, avg_pos=pos)
            metrics[mode] = m
            if m is not None:
                curves[mode] = (1 + strat_ret).cumprod() * 100
        chart = backtest_equity_chart_base64(curves, f"策略回測（全樣本）| {TARGETS[tcol]}")
        results["backtest"][tcol] = {"metrics": metrics, "chart_base64": chart}

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
