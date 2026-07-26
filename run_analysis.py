"""
CNN Fear & Greed vs SPX / NQ(NASDAQ-100) 對比分析
==================================================
延續先前 SPX 版方法論（forward-only、避免套套邏輯），加入 NQ 對比：
  1. 三個時間軸對齊（F&G ∩ SPX ∩ NQ 共同交易日）
  2. 極度恐懼(<25)/極度貪婪(>75) 標記在 SPX 與 NQ 走勢圖上
  3. streak 連續天數 vs 事件結束後 N 日守住率（forward-only）
  4. Hold-rate vs bootstrap 隨機基準，SPX / NQ 並列
輸出：./output/ 下的 PNG 圖表與 CSV 統計表
"""

import os
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA = "/private/tmp/claude-501/-Users-wujohn-Claude/5fcf72db-6d83-41b1-82fc-4a812a0ab679/scratchpad/files"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

# ---- dataviz palette (light mode) ----
C = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
         grid="#e1e0d9", axis="#c3c2b7",
         spx="#2a78d6", nq="#eb6834", fear="#d03b3b", greed="#0ca30c")
plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
    "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink2"],
    "xtick.color": C["muted"], "ytick.color": C["muted"],
    "grid.color": C["grid"], "grid.linewidth": 0.8,
    "text.color": C["ink"], "font.family": "sans-serif",
    "font.sans-serif": ["PingFang TC", "Helvetica Neue", "Arial"],
    "axes.unicode_minus": False,
})

FWD_WINDOWS = (3, 5, 10, 15, 20, 30, 40, 60)
STREAK_WINDOW = 15
N_BOOT = 2000

# ---------------------------------------------------------------
# 讀取與對齊
# ---------------------------------------------------------------
def load_fg():
    old = pd.read_csv(f"{DATA}/fg_old.csv")
    old.columns = [c.strip() for c in old.columns]
    old = old[["Date", "Fear Greed"]].rename(columns={"Fear Greed": "FG"})
    new = pd.read_csv(f"{DATA}/fg_combined.csv")
    new.columns = [c.strip() for c in new.columns]
    new = new[["Date", "Fear Greed"]].rename(columns={"Fear Greed": "FG"})
    fg = pd.concat([old, new])
    fg["Date"] = pd.to_datetime(fg["Date"])
    fg = fg.groupby("Date", as_index=False)["FG"].mean().sort_values("Date")
    return fg

fg = load_fg()
spx = pd.read_csv(f"{DATA}/spx_clean.csv", parse_dates=["Date"])
spx = spx[spx["Date"] >= "2020-05-14"].reset_index(drop=True)  # 去除前段零星月頻點
nq = pd.read_csv(f"{DATA}/nq_daily.csv", parse_dates=["Date"]).dropna()

df_all = fg.merge(spx, on="Date").merge(nq, on="Date").sort_values("Date").reset_index(drop=True)
df_nq_ext = fg.merge(nq, on="Date").sort_values("Date").reset_index(drop=True)  # NQ 延伸樣本 2016-07 起
print(f"對齊樣本 (F&G ∩ SPX ∩ NQ): {df_all['Date'].min():%Y-%m-%d} ~ {df_all['Date'].max():%Y-%m-%d}, n={len(df_all)}")
print(f"NQ 延伸樣本 (F&G ∩ NQ):    {df_nq_ext['Date'].min():%Y-%m-%d} ~ {df_nq_ext['Date'].max():%Y-%m-%d}, n={len(df_nq_ext)}")

# ---------------------------------------------------------------
# 工具：streak / 守住判定（forward-only，從 streak 結束日的收盤價起算）
# 資料缺口（F&G 斷檔 >30 天）處理：前瞻窗口跨越缺口的事件與抽樣位置一律排除
# ---------------------------------------------------------------
def gap_boundaries(dates):
    dd = dates.diff().dt.days.values
    return set(np.where(dd > 30)[0])  # index b: b-1 與 b 之間有缺口

def window_valid(e, w, n, gaps):
    if e + w >= n:
        return False
    return not any(e < b <= e + w for b in gaps)

def find_streaks(mask):
    streaks, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            streaks.append((i, j - 1, j - i))  # (start, end, length)
            i = j
        else:
            i += 1
    return streaks

def held_at(pv, e, w, kind):
    """streak 結束日 e 的收盤為錨，之後 w 個交易日：恐懼=最低不破底, 貪婪=最高不過頂"""
    fwd = pv[e + 1:e + 1 + w]
    edge = pv[e]
    return bool(fwd.min() >= edge) if kind == "fear" else bool(fwd.max() <= edge)

def analyze(df, price_col, label):
    pv = df[price_col].values
    fgv = df["FG"].values
    dates = df["Date"]
    n = len(df)
    gaps = gap_boundaries(dates)
    res = {"label": label, "n": n}

    for kind, mask in [("fear", fgv < 25), ("greed", fgv > 75)]:
        streaks = find_streaks(mask)
        res[f"{kind}_days"] = int(mask.sum())
        res[f"{kind}_events"] = len(streaks)

        # --- streak 長度 vs 守住率 (window=STREAK_WINDOW) ---
        rows = []
        for s, e, L in streaks:
            if window_valid(e, STREAK_WINDOW, n, gaps):
                rows.append((L, held_at(pv, e, STREAK_WINDOW, kind)))
        res[f"{kind}_streak_df"] = pd.DataFrame(rows, columns=["length", "held"])

        # --- hold-rate 掃描 + bootstrap 隨機基準 ---
        rng = np.random.default_rng(42)
        actual, rand_mean = [], []
        ends = [e for _, e, _ in streaks]
        for w in FWD_WINDOWS:
            ev = [held_at(pv, e, w, kind) for e in ends if window_valid(e, w, n, gaps)]
            actual.append(np.mean(ev) * 100 if ev else np.nan)
            # 向量化：所有合法位置的前瞻 min/max
            sw = sliding_window_view(pv[1:], w)          # sw[i] = pv[i+1 : i+1+w]
            valid = np.array([window_valid(i, w, n, gaps) for i in range(len(sw))])
            pos = np.where(valid)[0]
            hold_all = (sw.min(axis=1) >= pv[:len(sw)]) if kind == "fear" else (sw.max(axis=1) <= pv[:len(sw)])
            k = max(len(ev), 1)
            samp = rng.choice(pos, size=(N_BOOT, k), replace=True)
            rand_mean.append(hold_all[samp].mean() * 100)
        res[f"{kind}_actual"] = actual
        res[f"{kind}_random"] = rand_mean
    return res

R_spx = analyze(df_all, "SPX", "SPX (對齊樣本)")
R_nq = analyze(df_all, "NQ", "NQ (對齊樣本)")
R_nqx = analyze(df_nq_ext, "NQ", "NQ (延伸 2016-07 起)")

# ---------------------------------------------------------------
# 統計表輸出
# ---------------------------------------------------------------
def holdrate_table(res_list):
    rows = []
    for res in res_list:
        for kind, zh in [("fear", "極度恐懼"), ("greed", "極度貪婪")]:
            for w, a, r in zip(FWD_WINDOWS, res[f"{kind}_actual"], res[f"{kind}_random"]):
                rows.append(dict(樣本=res["label"], 訊號=zh, 窗口=w, 實際守住率=round(a, 1),
                                 隨機基準=round(r, 1), 倍數=round(a / r, 2) if r else np.nan,
                                 事件數=res[f"{kind}_events"]))
    return pd.DataFrame(rows)

tbl_hold = holdrate_table([R_spx, R_nq, R_nqx])
tbl_hold.to_csv(f"{OUT}/holdrate_spx_vs_nq.csv", index=False)

def streak_table(res_list):
    bins = [(1, 1, "1天"), (2, 3, "2-3天"), (4, 7, "4-7天"), (8, 999, "8天以上")]
    rows = []
    for res in res_list:
        for kind, zh in [("fear", "極度恐懼"), ("greed", "極度貪婪")]:
            d = res[f"{kind}_streak_df"]
            for lo, hi, name in bins:
                sub = d[(d["length"] >= lo) & (d["length"] <= hi)]
                if len(sub):
                    rows.append(dict(樣本=res["label"], 訊號=zh, 連續天數=name,
                                     事件數=len(sub), 守住率=round(sub["held"].mean() * 100, 0)))
    return pd.DataFrame(rows)

tbl_streak = streak_table([R_spx, R_nq, R_nqx])
tbl_streak.to_csv(f"{OUT}/streak_length_holdrate.csv", index=False)

print("\n===== Hold-rate vs 隨機基準 =====")
print(tbl_hold.to_string(index=False))
print(f"\n===== 連續天數 vs 守住率 (窗口={STREAK_WINDOW}日) =====")
print(tbl_streak.to_string(index=False))

# ---------------------------------------------------------------
# 圖 1：SPX 與 NQ 走勢 + 極端 F&G 標記（共用時間軸）
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.subplots_adjust(hspace=0.12, left=0.07, right=0.97, top=0.90, bottom=0.07)

gap_lo = pd.Timestamp("2020-09-18")
gap_hi = pd.Timestamp("2021-02-01")

for ax, dfp, col, cc, name in [(axes[0], spx, "SPX", C["spx"], "S&P 500 (SPX)"),
                               (axes[1], nq, "NQ", C["nq"], "NASDAQ-100 (NDX)")]:
    m = fg.merge(dfp, on="Date")
    ax.plot(dfp["Date"], dfp[col], color=cc, lw=1.4, zorder=2)
    f = m[m["FG"] < 25]
    g = m[m["FG"] > 75]
    ax.scatter(f["Date"], f[col], s=16, color=C["fear"], marker="v", zorder=3, lw=0)
    ax.scatter(g["Date"], g[col], s=16, color=C["greed"], marker="^", zorder=3, lw=0)
    ax.axvspan(gap_lo, gap_hi, color=C["grid"], alpha=0.5, zorder=1)
    ax.set_ylabel(name, fontsize=10)
    ax.grid(True, axis="y", zorder=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.margins(x=0.01)

axes[1].set_xlim(pd.Timestamp("2016-06-01"), nq["Date"].max() + pd.Timedelta(days=30))
handles = [Line2D([], [], marker="v", color=C["fear"], ls="", ms=7, label="極度恐懼 F&G<25"),
           Line2D([], [], marker="^", color=C["greed"], ls="", ms=7, label="極度貪婪 F&G>75"),
           Line2D([], [], marker="s", color=C["grid"], ls="", ms=9, label="F&G 資料缺口 2020-10~2021-01")]
fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, ncol=3, bbox_to_anchor=(0.97, 0.97))
fig.suptitle("SPX 與 NASDAQ-100：CNN Fear & Greed 極端值標記（SPX 日線自 2020-05 起）",
             fontsize=12, x=0.07, ha="left", color=C["ink"])
fig.savefig(f"{OUT}/price_with_fg_signals.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------
# 圖 2：Hold-rate vs 隨機基準（恐懼 / 貪婪 兩面板，SPX vs NQ）
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.13, wspace=0.18)
for ax, kind, zh in [(axes[0], "fear", "極度恐懼（結束後不破底）"),
                     (axes[1], "greed", "極度貪婪（結束後不過頂）")]:
    x = list(FWD_WINDOWS)
    ax.plot(x, R_spx[f"{kind}_actual"], color=C["spx"], lw=2, marker="o", ms=5, label="SPX 實際")
    ax.plot(x, R_nq[f"{kind}_actual"], color=C["nq"], lw=2, marker="o", ms=5, label="NQ 實際")
    ax.plot(x, R_spx[f"{kind}_random"], color=C["spx"], lw=1.2, ls="--", alpha=0.45, label="SPX 隨機基準")
    ax.plot(x, R_nq[f"{kind}_random"], color=C["nq"], lw=1.2, ls="--", alpha=0.45, label="NQ 隨機基準")
    ax.set_title(zh, fontsize=11, color=C["ink"], loc="left")
    ax.set_xlabel("事件結束後前瞻窗口（交易日）", fontsize=9)
    ax.set_ylabel("守住率 %", fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", zorder=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
nev = f"事件數：恐懼 SPX n={R_spx['fear_events']} / NQ n={R_nq['fear_events']}；貪婪 SPX n={R_spx['greed_events']} / NQ n={R_nq['greed_events']}（對齊樣本 2020-05~2026-07）"
fig.suptitle("Forward-only Hold-rate vs Bootstrap 隨機基準 — SPX / NQ 並列\n" + nev,
             fontsize=11, x=0.06, ha="left", color=C["ink"])
fig.savefig(f"{OUT}/holdrate_spx_vs_nq.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------
# 圖 3：連續天數 vs 守住率（恐懼訊號，SPX / NQ 對齊 + NQ 延伸）
# ---------------------------------------------------------------
bins_order = ["1天", "2-3天", "4-7天", "8天以上"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
fig.subplots_adjust(left=0.06, right=0.98, top=0.80, bottom=0.13, wspace=0.18)
for ax, zh in [(axes[0], "極度恐懼"), (axes[1], "極度貪婪")]:
    sub = tbl_streak[tbl_streak["訊號"] == zh]
    width = 0.27
    xs = np.arange(len(bins_order))
    for off, (lab, cc, alpha) in zip([-width, 0, width],
                                     [("SPX (對齊樣本)", C["spx"], 1.0),
                                      ("NQ (對齊樣本)", C["nq"], 1.0),
                                      ("NQ (延伸 2016-07 起)", C["nq"], 0.45)]):
        s = sub[sub["樣本"] == lab].set_index("連續天數")
        vals = [s["守住率"].get(b, np.nan) for b in bins_order]
        ns = [s["事件數"].get(b, 0) for b in bins_order]
        bars = ax.bar(xs + off, vals, width * 0.92, color=cc, alpha=alpha, label=lab, zorder=2)
        for b, nn in zip(bars, ns):
            if not np.isnan(b.get_height()):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"n={nn}",
                        ha="center", fontsize=7, color=C["muted"])
    ax.set_xticks(xs, bins_order)
    ax.set_title(f"{zh}：連續天數 vs 守住率（窗口 {STREAK_WINDOW} 日）", fontsize=11, color=C["ink"], loc="left")
    ax.set_ylabel("守住率 %", fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", zorder=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("連續天數（streak 長度）是否影響事件後守住率 — forward-only 方法",
             fontsize=11, x=0.06, ha="left", color=C["ink"])
fig.savefig(f"{OUT}/streak_length_holdrate.png", dpi=150)
plt.close(fig)

# 對齊後的合併資料也存一份
df_all.to_csv(f"{OUT}/aligned_fg_spx_nq.csv", index=False)
df_nq_ext.to_csv(f"{OUT}/nq_extended_fg.csv", index=False)
print(f"\n輸出完成 → {OUT}")
