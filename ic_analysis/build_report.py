"""組出獨立的HTML因子驗證報告：CNN恐懼貪婪指數(股市版) vs ^NDX / SP500。"""
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent / "output"

PERIOD_LABELS = {
    "full": "全樣本",
    "period1_reconstructed": "第三方重建期間",
    "period2_official": "官方API期間",
}
TARGET_LABELS = {"NDX": "^NDX（那斯達克100指數）", "SPX": "^GSPC（標普500指數）"}


def fmt_rho(cell):
    """rho用重疊取樣(headline)、n=重疊樣本數；顯著性星號與p值一律用『不重疊』的可信版本
    （cell['pval']=不重疊p）。重疊p會被自相關灌水，不拿來判定顯著性。"""
    if cell is None or cell.get("rho") is None:
        return f"<span class='dim'>樣本不足(n={cell.get('n', 0) if cell else 0})</span>"
    rho, n = cell["rho"], cell["n"]
    p_sig = cell.get("pval")
    sig = ""
    if p_sig is not None:
        sig = "**" if p_sig < 0.01 else ("*" if p_sig < 0.05 else "")
    p_txt = f"{p_sig:.3f}" if p_sig is not None else "—"
    cls = "fear-tilt" if rho < 0 else "greed-tilt"
    return (f"<span class='{cls}'>{rho:+.3f}{sig}</span>"
            f"<br><span class='sub'>n={n:,}（重疊）<br>顯著性p={p_txt}（不重疊）</span>")


def fmt_mono(bucket):
    if bucket is None or bucket.get("monotonicity") is None:
        return "<span class='dim'>—</span>"
    v = bucket["monotonicity"]
    cls = "fear-tilt" if v < 0 else "greed-tilt"
    warn = f"<span class='warn-chip'>{bucket['n_low_confidence']}組樣本過少</span>" if bucket["n_low_confidence"] else ""
    return f"<span class='{cls}'>{v:+.3f}</span> {warn}"


def build():
    with open(OUT_DIR / "results.json") as f:
        r = json.load(f)
    val = r["reconstruction_validation"]

    periods = r["periods"]
    full, p1, p2 = periods["full"], periods["period1_reconstructed"], periods["period2_official"]

    ic_rows = ""
    for pname in ["full", "period1_reconstructed", "period2_official"]:
        pd_ = periods[pname]
        ic_rows += f"""
        <tr>
          <td>{PERIOD_LABELS[pname]}<br><span class='sub'>{pd_['date_range'][0]} ~ {pd_['date_range'][1]}</span></td>
          <td>{fmt_rho(pd_['ic']['NDX'])}</td>
          <td>{fmt_rho(pd_['ic']['SPX'])}</td>
          <td>{fmt_mono(pd_['bucket']['NDX'])}</td>
          <td>{fmt_mono(pd_['bucket']['SPX'])}</td>
        </tr>"""

    def chart_img(pname, tcol, key="chart_base64"):
        b = periods[pname]["bucket"][tcol]
        if b is None or b.get(key) is None:
            return "<p class='dim'>（無法產生此圖）</p>"
        return f"<img src='data:image/png;base64,{b[key]}' style='width:100%;max-width:760px;'/>"

    full_charts = f"""
      <div class="chart-pair">
        <div>{chart_img('full', 'NDX')}</div>
        <div>{chart_img('full', 'SPX')}</div>
      </div>"""

    full_excess_charts = f"""
      <div class="chart-pair">
        <div>{chart_img('full', 'NDX', 'excess_chart_base64')}</div>
        <div>{chart_img('full', 'SPX', 'excess_chart_base64')}</div>
      </div>"""

    price_trend_img = (
        f"<img src='data:image/png;base64,{r['price_trend_chart_base64']}' "
        f"style='width:100%;max-width:820px;'/>"
        if r.get("price_trend_chart_base64") else "<p class='dim'>（無法產生走勢圖）</p>"
    )

    dd = r.get("top_bucket_drilldown", {})
    drilldown_img = (
        f"<img src='data:image/png;base64,{dd['chart_base64']}' style='width:100%;max-width:860px;'/>"
        if dd.get("chart_base64") else "<p class='dim'>（無法產生拆細圖）</p>"
    )
    dd_ndx = (dd.get("data") or {}).get("NDX") or {}
    dd_spx = (dd.get("data") or {}).get("SPX") or {}

    uncond_nq = full["bucket"]["NDX"]["unconditional"]
    uncond_spx = full["bucket"]["SPX"]["unconditional"]

    period_compare_charts = ""
    for pname in ["period1_reconstructed", "period2_official"]:
        u_nq = periods[pname]["bucket"]["NDX"]["unconditional"]
        u_spx = periods[pname]["bucket"]["SPX"]["unconditional"]
        period_compare_charts += f"""
      <h4>{PERIOD_LABELS[pname]}（{periods[pname]['date_range'][0]} ~ {periods[pname]['date_range'][1]}，n={periods[pname]['n_rows']}筆；
      同期無條件平均20日報酬：^NDX {u_nq['mean']:+.2f}%、SP500 {u_spx['mean']:+.2f}%）</h4>
      <div class="chart-pair">
        <div>{chart_img(pname, 'NDX')}</div>
        <div>{chart_img(pname, 'SPX')}</div>
      </div>
      <p class="sub">超額報酬版（扣掉上面這條同期基準線）：</p>
      <div class="chart-pair">
        <div>{chart_img(pname, 'NDX', 'excess_chart_base64')}</div>
        <div>{chart_img(pname, 'SPX', 'excess_chart_base64')}</div>
      </div>"""

    # 方向一致性判斷
    def sign(cell):
        if cell is None or cell.get("rho") is None:
            return None
        return cell["rho"] < 0

    nq_signs = [sign(periods[p]["ic"]["NDX"]) for p in ["period1_reconstructed", "period2_official"]]
    spx_signs = [sign(periods[p]["ic"]["SPX"]) for p in ["period1_reconstructed", "period2_official"]]
    nq_consistent = len(set(nq_signs)) == 1 and None not in nq_signs
    spx_consistent = len(set(spx_signs)) == 1 and None not in spx_signs

    stability_note_nq = "一致" if nq_consistent else "不一致"
    stability_note_spx = "一致" if spx_consistent else "不一致"

    # ---- 中間分數規律性 ----
    middle_rows = ""
    for pname in ["full", "period1_reconstructed", "period2_official"]:
        m = periods[pname]["middle_ic"]
        rng = m["NDX"].get("score_range") if m.get("NDX") else None
        rng_txt = f"{rng[0]:.0f}~{rng[1]:.0f}分" if rng else "—"
        middle_rows += f"""
        <tr>
          <td>{PERIOD_LABELS[pname]}<br><span class='sub'>分數範圍 {rng_txt}</span></td>
          <td>{fmt_rho(m['NDX'])}</td>
          <td>{fmt_rho(m['SPX'])}</td>
        </tr>"""

    middle_charts_full = f"""
      <div class="chart-pair">
        <div>{chart_img('full', 'NDX', 'middle_chart_base64')}</div>
        <div>{chart_img('full', 'SPX', 'middle_chart_base64')}</div>
      </div>"""

    # ---- 不同持有期的訊號強度掃描 ----
    hs = r.get("horizon_scan", {})
    horizon_scan_img = (
        f"<img src='data:image/png;base64,{hs['chart_base64']}' style='width:100%;max-width:820px;'/>"
        if hs.get("chart_base64") else "<p class='dim'>（無法產生持有期掃描圖）</p>"
    )

    # ---- 報酬趨勢熱力圖 ----
    hmp = r.get("heatmap", {})
    raw_heatmap_img = (
        f"<img src='data:image/png;base64,{hmp['raw_chart_base64']}' style='width:100%;max-width:900px;'/>"
        if hmp.get("raw_chart_base64") else "<p class='dim'>（無法產生原始報酬熱力圖）</p>"
    )
    excess_heatmap_img = (
        f"<img src='data:image/png;base64,{hmp['excess_chart_base64']}' style='width:100%;max-width:900px;'/>"
        if hmp.get("excess_chart_base64") else "<p class='dim'>（無法產生超額報酬熱力圖）</p>"
    )

    def horizon_scan_rows(tcol):
        rows = ""
        for row in hs.get(tcol, []):
            ic = row["ic"]
            sig = ic is not None and ic.get("pval") is not None and ic["pval"] < 0.05
            rho_txt = f"{ic['rho']:+.3f}" if ic and ic.get("rho") is not None else "—"
            pval_txt = f"{ic['pval']:.3f}" if ic and ic.get("pval") is not None else "—"
            rho_cls = "fear-tilt" if (ic and ic.get("rho") or 0) < 0 else "greed-tilt"
            fear_txt = f"{row['fear_excess']:+.2f}%" if row["fear_excess"] is not None else "—"
            greed_txt = f"{row['greed_excess']:+.2f}%" if row["greed_excess"] is not None else "—"
            rows += f"""
            <tr>
              <td>{row['horizon']}</td>
              <td class="{rho_cls}">{rho_txt}{' *' if sig else ''}<br><span class="sub">p={pval_txt}, n={ic['n'] if ic else '—'}</span></td>
              <td class="fear-tilt">{fear_txt}</td>
              <td class="greed-tilt">{greed_txt}</td>
            </tr>"""
        return rows

    # ---- 策略回測 ----
    def fmt_metric(m):
        if m is None:
            return "<td class='dim'>—</td>" * 7
        sharpe = f"{m['sharpe']:.2f}" if m['sharpe'] is not None else "—"
        sortino = f"{m['sortino']:.2f}" if m['sortino'] is not None else "—"
        avg_pos = f"{m['avg_abs_position']*100:.0f}%" if m['avg_abs_position'] is not None else "—"
        return (
            f"<td>{m['total_return']:+.0f}%</td><td>{m['cagr']:+.1f}%</td>"
            f"<td>{m['vol']:.1f}%</td><td>{sharpe}</td><td>{sortino}</td>"
            f"<td class='fear-tilt' style='color:var(--warn)'>{m['max_dd']:.1f}%</td><td>{avg_pos}</td>"
        )

    strategy_labels = {
        "buy_hold": "買進持有（對照組）",
        "long_only_tilt": "恐懼多單／貪婪空手（不放空）",
        "long_short": "恐懼多單／貪婪放空（對稱，跟美債專案position_size公式相同）",
    }

    def backtest_table(tcol):
        rows = ""
        for mode in ["buy_hold", "long_only_tilt", "long_short"]:
            m = r["backtest"][tcol]["metrics"].get(mode)
            rows += f"<tr><td>{strategy_labels[mode]}</td>{fmt_metric(m)}</tr>"
        return f"""
        <div class="table-wrap"><table>
          <tr><th>策略</th><th>累積報酬</th><th>年化CAGR</th><th>年化波動</th><th>Sharpe</th>
              <th>Sortino</th><th>最大回撤</th><th>平均持倉水位</th></tr>
          {rows}
        </table></div>"""

    def backtest_chart(tcol):
        c = r["backtest"][tcol].get("chart_base64")
        if not c:
            return "<p class='dim'>（無法產生回測圖）</p>"
        return f"<img src='data:image/png;base64,{c}' style='width:100%;max-width:820px;'/>"

    timeline_img = (
        f"<img src='data:image/png;base64,{r['timeline_chart_base64']}' "
        f"style='width:100%;max-width:820px;'/>"
        if r.get("timeline_chart_base64") else "<p class='dim'>（無法產生時間軸對照圖）</p>"
    )
    ts = r.get("timeline_stats", {})

    html = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"/>
<title>CNN恐懼貪婪指數(股市版) 對 ^NDX / SP500 預測力驗證報告</title>
<style>
  :root {{
    --paper: #EEF0EA;
    --ink: #20242B;
    --card: #FBFCF9;
    --border: #D6D9CF;
    --fear: #3E5C76;
    --greed: #A9782E;
    --warn: #8E3B3B;
    --dim: #6C7268;
    --font-display: "Noto Serif TC", "Songti TC", "PingFang TC", serif;
    --font-body: "PingFang TC", "Noto Sans TC", -apple-system, "Helvetica Neue", sans-serif;
    --font-num: ui-monospace, "SF Mono", Menlo, "PingFang TC", monospace;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --paper:#14171C; --ink:#E7E9E2; --card:#1B1F26; --border:#2C313A;
      --fear:#7FA3C4; --greed:#D9A75C; --warn:#C97575; --dim:#9AA0AC;
    }}
  }}
  :root[data-theme="dark"] {{
    --paper:#14171C; --ink:#E7E9E2; --card:#1B1F26; --border:#2C313A;
    --fear:#7FA3C4; --greed:#D9A75C; --warn:#C97575; --dim:#9AA0AC;
  }}
  :root[data-theme="light"] {{
    --paper: #EEF0EA; --ink: #20242B; --card: #FBFCF9; --border: #D6D9CF;
    --fear: #3E5C76; --greed: #A9782E; --warn: #8E3B3B; --dim: #6C7268;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--paper); color: var(--ink); font-family: var(--font-body);
    line-height: 1.75; max-width: 920px; margin: 0 auto; padding: 40px 20px 90px;
  }}
  h1, h2, h3, h4 {{ font-family: var(--font-display); text-wrap: balance; font-weight: 600; }}
  h1 {{ font-size: 1.7em; letter-spacing: 0.02em; margin-bottom: 0.3em; }}
  h2 {{ font-size: 1.22em; margin-top: 2.6em; padding-bottom: 8px; border-bottom: 1px solid var(--border); }}
  h3 {{ font-size: 1.02em; margin-top: 1.7em; }}
  h4 {{ font-size: 0.95em; color: var(--dim); margin-top: 1.4em; font-family: var(--font-body); }}
  p {{ max-width: 66ch; }}
  .lede {{ font-size: 0.95em; color: var(--dim); max-width: 70ch; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 4px; padding: 20px 24px; margin: 16px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.92em; font-variant-numeric: tabular-nums; }}
  .table-wrap {{ overflow-x: auto; }}
  th, td {{ border: 1px solid var(--border); padding: 9px 11px; text-align: center; vertical-align: middle; }}
  th {{ font-family: var(--font-display); font-weight: 600; background: color-mix(in srgb, var(--fear) 8%, transparent); }}
  td {{ font-family: var(--font-num); }}
  td:first-child {{ font-family: var(--font-body); }}
  .sub {{ font-size: 0.82em; color: var(--dim); font-family: var(--font-body); }}
  .dim {{ color: var(--dim); }}
  .fear-tilt {{ color: var(--fear); font-weight: 700; }}
  .greed-tilt {{ color: var(--greed); font-weight: 700; }}
  .quality-good {{ color: var(--ink); font-weight: 700; }}
  .warn-chip {{
    display: inline-block; font-family: var(--font-body); font-size: 0.78em; color: var(--warn);
    border: 1px solid color-mix(in srgb, var(--warn) 50%, transparent); border-radius: 3px; padding: 1px 6px;
  }}
  .chart-pair {{ display: flex; gap: 16px; flex-wrap: wrap; justify-content: center; }}
  .chart-pair > div {{ flex: 1 1 360px; text-align: center; }}
  .chart-pair img {{ border-radius: 4px; border: 1px solid var(--border); }}
  .callout {{
    border-left: 3px solid var(--fear); background: color-mix(in srgb, var(--fear) 7%, var(--card));
    padding: 14px 18px; border-radius: 0 4px 4px 0; margin: 16px 0; max-width: none;
  }}
  .callout.warn {{ border-left-color: var(--warn); background: color-mix(in srgb, var(--warn) 9%, var(--card)); }}
  code {{
    font-family: var(--font-num); background: color-mix(in srgb, var(--ink) 8%, transparent);
    padding: 1px 6px; border-radius: 3px; font-size: 0.88em;
  }}
  ul {{ padding-left: 1.3em; }}
  li {{ margin: 0.4em 0; }}
  .gauge-row {{ display: flex; align-items: center; gap: 14px; margin: 18px 0 24px; }}
  .gauge-bar {{
    flex: 1; height: 10px; border-radius: 5px;
    background: linear-gradient(90deg, var(--fear), color-mix(in srgb, var(--fear) 40%, var(--greed)), var(--greed));
  }}
  .gauge-label {{ font-family: var(--font-display); font-size: 0.82em; color: var(--dim); white-space: nowrap; }}
  footer {{ margin-top: 4.5em; padding-top: 1.2em; border-top: 1px solid var(--border); color: var(--dim); font-size: 0.85em; }}
</style>
</head>
<body>

<h1>CNN恐懼貪婪指數（股市版）對 ^NDX 與 SP500 未來報酬的預測力驗證</h1>
<p class="lede">
  獨立分析專案，與美債恐懼貪婪儀表板專案（<code>bond_data_pipeline</code>）分開存放於
  <code>fg_nq_analysis/ic_analysis/</code>。方法論沿用美債專案「因子驗證分析報告」的IC與分位數分桶架構，
  套用在CNN官方股市版指數上。
</p>
<div class="gauge-row">
  <span class="gauge-label">0 極度恐懼</span>
  <div class="gauge-bar"></div>
  <span class="gauge-label">100 極度貪婪</span>
</div>

<h3>指數時間軸總覽：CNN分數 vs ^NDX／SP500走勢</h3>
<p>上層是CNN指數本身逐日的分數（淺色＝第三方重建、深色＝官方API），標出25分（最恐懼門檻）、
75分（最貪婪門檻）；下層是^NDX、SP500指數化後的價格走勢，在對應日期用<span class="fear-tilt"
style="color:#2E7D4F">綠點標出分數&lt;25</span>、<span class="fear-tilt" style="color:#B23B3B">紅點
標出分數&gt;75</span>，方便直接用肉眼比對「極端讀數出現時，股價實際在做什麼」——
全樣本裡分數低於25的有{ts.get('n_fear','—')}天、高於75的有{ts.get('n_greed','—')}天。</p>
<div style="text-align:center;">{timeline_img}</div>
<div class="callout warn">
  <b>這張圖在製作過程中，直接肉眼發現了一個資料品質問題：</b>上層CNN指數線在2020年下半段有一截
  異常地打平成一條直線。往下追查後發現，CNN官方API在2020-07-15~2021-01-21這段回傳的分數，
  有122天被打死在50.0（後端某個計算窗口顯然還沒暖機完成的佔位值），不是真的逐日計算結果。
  這個問題已經修正——詳見下方「兩段子時期對照」與「資料來源與可信度」，修正後全篇報告的官方/
  第三方切點從原本推測的2020-07-15改成2021-02-01，所有IC、分桶、回測數字都已經用修正後的資料
  重新計算。這是這次做圖表意外抓到的一個真實錯誤，不是假設性的風險提示。
</div>

<h2>一、摘要：全樣本結果</h2>
<div class="card">
  <div class="table-wrap"><table>
    <tr><th>樣本</th><th>IC vs ^NDX（20日、不重疊取樣）</th><th>IC vs SP500（20日、不重疊取樣）</th>
        <th>分桶單調性 vs ^NDX</th><th>分桶單調性 vs SP500</th></tr>
    <tr>
      <td>{PERIOD_LABELS['full']}<br><span class='sub'>{full['date_range'][0]} ~ {full['date_range'][1]}，{full['n_rows']}筆</span></td>
      <td>{fmt_rho(full['ic']['NDX'])}</td>
      <td>{fmt_rho(full['ic']['SPX'])}</td>
      <td>{fmt_mono(full['bucket']['NDX'])}</td>
      <td>{fmt_mono(full['bucket']['SPX'])}</td>
    </tr>
  </table></div>
  <p class="sub">* p&lt;0.05　** p&lt;0.01。IC為負，代表分數低（恐懼）對應未來報酬較高，方向符合「恐懼買入」的傳統解讀；IC為正則相反。
  <b>rho（相關係數）用重疊取樣算</b>（全樣本每日的分數vs未來20日報酬配對，n≈3,866，點估計用到全部資料）；
  <b>顯著性（星號與p值）用不重疊取樣算</b>（每隔20天取一筆、n≈194），理由見下方紅框。</p>
</div>

<div class="callout warn">
  <b>為什麼rho用重疊、顯著性卻堅持用不重疊？這次的資料剛好是最好的示範。</b>
  你要求把IC取樣改成重疊——rho（相關係數的點估計）用重疊取樣完全沒問題，用到全部每日資料反而更穩定。
  但<b>p值（顯著性）不能用重疊版本</b>：重疊取樣下，今天跟明天的「未來20日報酬」有19天是重疊看同一段行情，
  相鄰樣本高度自相關，會把樣本數灌水（n從194灌到3,866），讓p值假性地變超小。具體看全樣本：
  <ul>
    <li>^NDX：重疊p算出來是 <b>{full['ic']['NDX']['pval_overlap']:.4f}</b>（看起來「顯著」！）——但這是假的；
    不重疊的可信p是 <b>{full['ic']['NDX']['pval']:.3f}</b>（根本不顯著）。</li>
    <li>SP500：重疊p <b>{full['ic']['SPX']['pval_overlap']:.4f}</b>（看起來「極顯著」！）——一樣是假的；
    不重疊可信p是 <b>{full['ic']['SPX']['pval']:.3f}</b>（不顯著）。</li>
  </ul>
  兩個rho點估計本身（重疊vs不重疊）其實很接近（^NDX {full['ic']['NDX']['rho']:+.3f} vs {full['ic']['NDX']['rho_nonoverlap']:+.3f}、
  SP500 {full['ic']['SPX']['rho']:+.3f} vs {full['ic']['SPX']['rho_nonoverlap']:+.3f}），會爆炸的只有p值。
  所以本報告一律：<b>rho看重疊、顯著性看不重疊</b>，這樣既給你重疊版的IC，又不會謊報「顯著」。
</div>

<h3>全樣本20分位分桶（依CNN指數分數切20等分，看未來20日平均報酬）</h3>
{full_charts}
<p class="sub">分桶分析用逐日重疊資料計算（不做不重疊取樣），跟上面IC表用的「不重疊取樣」方法不同，理由見下方「方法論」段落。</p>

<h3>為什麼最右邊（最貪婪）那組沒有變成負的？</h3>
<div class="callout">
  直覺上「貪婪時未來報酬應該轉差」，甚至轉負——但上面的圖裡，就連第20組（最貪婪）^NDX與SP500的
  未來20日平均報酬<b>仍然是正的</b>。原因是整條分桶曲線<b>疊在「大盤長期上漲」這個基準之上</b>，
  不是以0為中心畫出來的。
</div>
<p>把CNN指數分數完全丟開，這15.5年裡「隨便挑一天」往後看20個交易日，平均報酬長這樣：</p>
<div class="table-wrap"><table>
  <tr><th></th><th>無條件平均20日報酬</th><th>中位數</th><th>報酬為負的比例</th></tr>
  <tr><td>^NDX</td><td class="greed-tilt">{uncond_nq['mean']:+.2f}%</td><td>{uncond_nq['median']:+.2f}%</td><td>{uncond_nq['pct_negative']:.1f}%</td></tr>
  <tr><td>SP500</td><td class="greed-tilt">{uncond_spx['mean']:+.2f}%</td><td>{uncond_spx['median']:+.2f}%</td><td>{uncond_spx['pct_negative']:.1f}%</td></tr>
</table></div>
<p>這是2011~2026這段罕見長多頭的結果——下面是同一段期間^NDX、SP500的實際走勢（對數座標）：</p>
<div style="text-align:center;">{price_trend_img}</div>
<p class="sub">^NDX這15.5年累積漲了約1186%（年化約18.0%），SP500累積約490%（年化約12.2%）。
在這種背景下，隨便挑一個交易日往後看20天，平均本來就已經是正的（^NDX約+1.46%、SP500約+1.01%），
而且六成以上的窗口都是正報酬。分桶柱狀圖每一根都疊在這個「地心引力」水位上，
IC與分桶單調性算的是「分數越高、柱子相對而言有沒有變矮」這個<b>相對關係</b>，不是「柱子有沒有跨過0」——
就算訊號完全成立，貪婪那組理論上也只需要比恐懼那組矮，不一定要跌破0。</p>

<h4>扣掉這條「地心引力」之後的乾淨版本</h4>
<p>把每一組的平均報酬，減掉上面那條同期無條件平均基準線，0就代表「跟隨便挑一天沒兩樣」，
負的才代表這組真的比同期平均還差：</p>
{full_excess_charts}
<p class="sub">扣掉基準線之後，中段到偏貪婪的分組（約第6~19組）大多轉為負的超額報酬，符合「恐懼相對較好、
貪婪相對較差」的方向。但值得注意的是：<b>第20組（最極端貪婪）在超額報酬版本裡依然反彈回正值</b>，
是全場數一數二高的一組——不是單調遞減到最低。這代表市場在最極端的貪婪階段，短期內經常
<b>動能延續、繼續噴出</b>，不是立刻反轉向下；分桶分析捕捉到的比較像是「中段偏貪婪時動能開始減弱」，
而不是「越貪婪就越應該賣出」這種單純的線性關係。這也是上面分桶單調性只有-0.3左右（方向對但不強）
的原因之一。</p>

<h3>把第20分位（最貪婪那~5%）單獨抓出來往下拆細，內部有沒有趨勢？</h3>
<p>既然第20組整組反彈成正的違反直覺，就把這一組（分數約 {dd_ndx.get('top_score_range',[0,0])[0]:.0f}~{dd_ndx.get('top_score_range',[0,0])[1]:.0f} 分，
n={dd_ndx.get('top_n','—')}）單獨拉出來、再往下切細看內部。用兩種切法並列對照：<b>固定等寬分數區間</b>
（例如81-84、85-87…，可讀性高但每組樣本數不均）跟<b>等數量分位qcut</b>（每組樣本數一樣多、統計上較穩，
但分數標籤不漂亮）。兩種都看未來20日超額報酬，樣本數少於10的子組用紅色標示（可信度低）：</p>
<div style="text-align:center;">{drilldown_img}</div>
<div class="callout">
  <b>拆細之後看到一個「倒U」形，原本整組的反彈其實藏著內部結構：</b>
  <ul>
    <li><b>從「偏貪婪」往「很貪婪」走，超額報酬是先往上爬的</b>（動能延續）：^NDX大約在分數88~91這一段
    衝到最高（等寬區間88-90組來到+1.99%、qcut最高那組約+1.75%），這段是撐起「第20組整組反彈」的主力。</li>
    <li><b>但真正最極端的貪婪（分數約92以上）就開始往下掉，不再是最高：</b>^NDX最極端那個qcut子組
    （{dd_ndx.get('qcut',[{}])[-1].get('label','—')} 分）超額報酬掉到 {dd_ndx.get('qcut',[{}])[-1].get('excess',0):+.2f}%，
    SP500更明顯、直接翻負（最極端子組 {dd_spx.get('qcut',[{}])[-1].get('excess',0):+.2f}%，固定區間94-97那格也是負的）。</li>
    <li>換句話說：<b>「第20組整組反彈成正」這件事，主要是分數85~91這種「貪婪但還沒到極端」的區間貢獻的，
    不是真正最極端那一小撮</b>。真正最極端貪婪（92分以上）其實已經開始出現「退燒」的跡象，方向跟直覺一致，
    只是這一小撮的樣本非常薄（等寬區間最高那格n只有個位數），統計上很不可靠，不能當成確定的結論。</li>
  </ul>
  這也解釋了為什麼前面用20組看的時候，最右邊那根會反彈——因為第20組把「貪婪但不極端」跟「極端貪婪」
  混在同一組了，一拆細就看得出來這兩者的行為其實不一樣。
</div>

<h3>把最恐懼、最貪婪兩端各拿掉4組，中間段還有沒有方向性？</h3>
<p>整條分桶曲線的「方向」，有多少其實是被頭尾兩端的極端讀數撐起來的？<b>頭尾各拿掉4組</b>
（20組裡拿掉8組，只留中間12組，大約對應30~68分這個更嚴格定義的「中性」區間），重新算一次
CNN分數跟未來報酬的Spearman相關（rho用重疊取樣、顯著性用不重疊，方法跟主IC表一致）：</p>
<div class="card">
  <div class="table-wrap"><table>
    <tr><th>樣本</th><th>中間段IC vs ^NDX</th><th>中間段IC vs SP500</th></tr>
    {middle_rows}
  </table></div>
  <p class="sub">對照：拿掉頭尾前的全樣本IC是 ^NDX {fmt_rho(full['ic']['NDX'])}　SP500 {fmt_rho(full['ic']['SPX'])}（見上方摘要表）。</p>
</div>
<p>把中間12組的超額報酬攤開來看：</p>
{middle_charts_full}
<div class="callout">
  <b>改用重疊取樣（更多樣本、點估計更穩）之後，中間段的答案變得更乾脆：幾乎是0，沒有例外。</b>
  全樣本^NDX中間段IC只有{full['middle_ic']['NDX']['rho']:+.3f}（p={full['middle_ic']['NDX']['pval']:.3f}，
  不顯著），SP500 {full['middle_ic']['SPX']['rho']:+.3f}（p={full['middle_ic']['SPX']['pval']:.3f}，也不顯著）；
  兩段子時期同樣都貼著0：第三方重建期間^NDX {p1['middle_ic']['NDX']['rho']:+.3f}、官方API期間^NDX
  {p2['middle_ic']['NDX']['rho']:+.3f}。（附帶一提：早先用「不重疊、樣本較小」的取樣方式算過一次，
  曾經算出一個勉強顯著的正值——現在用重疊取樣、樣本數從一百多筆增加到兩千多筆之後，那個訊號就消失了，
  這正是小樣本下的雜訊會被誤讀成「訊號」的活教材。）中間段長條圖也看不出清楚的階梯狀。
  <b>結論沒變、而且更確定了：這個指數的方向性幾乎全部集中在頭尾的極端讀數（尤其是最恐懼那端），
  中間的「中性」分數區間對未來20日報酬沒有可靠的區辨力</b>。這對操作上的含意是：這個指數比較適合當
  「極端值警示器」（分數落到最低或最高的一小段區間時多留意），而不是拿來對整條0~100分做連續的
  「分數越低倉位越重」這種線性訊號——下一節的策略回測會直接驗證這個想法。
</div>

<h2>二、策略回測：把訊號變成一套可交易的策略，實際表現如何？</h2>
<div class="callout">
  上面的IC、分桶分析都是「事後統計」，回答的是「分數高低跟未來報酬有沒有關聯」；這裡改成回答一個更
  貼近操作的問題：<b>如果真的照這個訊號建倉位，長期下來會不會贏過乾脆买進持有？</b>
  三套策略都是用第t日收盤的CNN分數，決定第t+1日的持倉比例（不偷看未來），每天依分數重新調整：
  <ul>
    <li><b>買進持有（對照組）</b>：全程100%多單，不管分數。</li>
    <li><b>恐懼多單／貪婪空手（不放空）</b>：分數0分＝100%多單、分數50分以上＝空手(0%)，中間線性內插——
    直接操作化你的假設「恐懼買、貪婪不放空只是相對少賺」。</li>
    <li><b>恐懼多單／貪婪放空（對稱）</b>：分數100分＝100%空單，跟美債恐懼貪婪儀表板專案
    <code>position_size()</code> 用的公式完全一樣（<code>(50-分數)/50</code>），拿來對照「如果真的在
    最貪婪放空，會發生什麼事」。</li>
  </ul>
</div>

<h3>^NDX 回測（全樣本，2011-01 ~ 2026-07）</h3>
{backtest_table('NDX')}
<div style="text-align:center;margin-top:10px;">{backtest_chart('NDX')}</div>

<h3>SP500 回測（全樣本，2011-01 ~ 2026-07）</h3>
{backtest_table('SPX')}
<div style="text-align:center;margin-top:10px;">{backtest_chart('SPX')}</div>

<div class="callout warn">
  <b>結果跟直覺可能相反：兩套訊號策略都遠遠輸給乾脆買進持有。</b>
  ^NDX買進持有15.5年累積 {r['backtest']['NDX']['metrics']['buy_hold']['total_return']:+.0f}%
  （年化{r['backtest']['NDX']['metrics']['buy_hold']['cagr']:+.1f}%、Sharpe
  {r['backtest']['NDX']['metrics']['buy_hold']['sharpe']:.2f}）；「恐懼多單／貪婪空手」只累積
  {r['backtest']['NDX']['metrics']['long_only_tilt']['total_return']:+.0f}%
  （年化{r['backtest']['NDX']['metrics']['long_only_tilt']['cagr']:+.1f}%）；「恐懼多單／貪婪放空」
  幾乎打平，累積只有{r['backtest']['NDX']['metrics']['long_short']['total_return']:+.1f}%
  （年化{r['backtest']['NDX']['metrics']['long_short']['cagr']:+.2f}%，15年幾乎沒賺錢）。SP500也是同樣的型態。
  <br><br>
  原因直接跟第一節「地心引力」的發現連在一起：這15.5年是罕見的長多頭，「恐懼多單／貪婪空手」策略平均
  只有約19%的時間持有多單（其餘時間空手在等分數變低），錯過了大部分的漲幅；「恐懼多單／貪婪放空」
  更慘，貪婪時的空單持續在跟這段長期上漲的趨勢對作，把買進持有原本該賺到的報酬幾乎全部吃掉。
  「恐懼多單／貪婪空手」的最大回撤（^NDX {r['backtest']['NDX']['metrics']['long_only_tilt']['max_dd']:.1f}%）
  確實比買進持有（{r['backtest']['NDX']['metrics']['buy_hold']['max_dd']:.1f}%）小很多，波動也低很多，
  如果目標是「降低回撤、犧牲一部分報酬換穩定」，這個策略在風險控制上是有作用的；但如果目標是
  「打敗大盤」，即使風險調整後看Sharpe值，買進持有（{r['backtest']['NDX']['metrics']['buy_hold']['sharpe']:.2f}）
  仍然明顯贏過恐懼多單／貪婪空手（{r['backtest']['NDX']['metrics']['long_only_tilt']['sharpe']:.2f}）。
  <br><br>
  這跟上一節「不宜放空」的結論方向一致，但這裡的回測把它量化得更清楚：不只是「放空會少賺一點」，
  而是<b>放空在這段長多頭裡幾乎抵銷了訊號原本該有的優勢，讓15年的報酬幾乎歸零</b>。
  這個結果高度依賴這15.5年剛好是大多頭的樣本背景，如果换成一段長期盤整或空頭主導的期間，
  三套策略的相對表現可能完全不同——這是用歷史回測評估任何時機策略時都有的限制，不是這個指數獨有的問題。
</div>

<h2>三、不同持有期的訊號強度：短線到底多短算數？</h2>
<div class="callout">
  前面全部分析都固定用20個交易日（約一個月）當持有期。這裡把持有期從3天掃到90天
  （3、5、10、15、20、30、40、60、90個交易日），同一套IC與分桶方法論在每個持有期各跑一次，
  直接回答「這個訊號在多短、多長的持有期內才看得出東西」，而不是只看單一個20日窗口的結果。
</div>
<div style="text-align:center;">{horizon_scan_img}</div>

<h3>^NDX 逐持有期數字</h3>
<div class="table-wrap"><table>
  <tr><th>持有期（交易日）</th><th>全樣本IC（0~100分整體）</th>
      <th>第1組（最恐懼）超額報酬</th><th>最後一組（最貪婪）超額報酬</th></tr>
  {horizon_scan_rows('NDX')}
</table></div>
<p class="sub">* p&lt;0.05。超額報酬＝該分組平均未來報酬減去同期無條件平均（見第一節說明），分桶用逐日重疊資料。</p>

<div class="callout warn">
  <b>答案跟直覺可能相反：這個訊號不是短線訊號，反而更像中長線訊號。</b>
  三個觀察：
  <ul>
    <li><b>整體IC在任何持有期都沒有達到統計顯著</b>（3天到90天，p值全部大於0.1，大部分甚至大於0.3）——
    不是「短線比較有效、長線比較沒用」，而是不管持有期長短，CNN分數對「整條0~100分」的區辨力
    始終偏弱，這點在3天、20天、90天都一樣。</li>
    <li><b>最恐懼那組的超額報酬，是隨著持有期拉長而愈來愈明顯，不是愈來愈短愈準：</b>
    3天只有+0.13%，5天甚至轉負（-0.00%），10天+0.45%，20天+1.92%，一路到90天累積到+5.40%。
    如果「恐懼買入」這個效應是真的，看起來比較像是「市場從恐慌中花數週到數月慢慢修復」的中期現象，
    不是「恐慌隔天就反彈」的短線現象——3~5天的窗口幾乎看不到任何優勢。</li>
    <li><b>最貪婪那組的超額報酬，要拉長到60天以上才轉負</b>（40天以前都還是正的，60天-0.88%、
    90天-1.79%）。意思是：如果貪婪之後真的有「退燒」，這個退燒也是要數個月的時間才會顯現，
    不是進場後幾天內就會發生的事。</li>
  </ul>
  SP500的型態完全一樣（最恐懼超額報酬從3天的+0.02%一路爬升到90天的+3.09%；最貪婪超額報酬在
  40天轉負、90天來到-3.63%），不是^NDX單獨的巧合。
</div>

<h3>報酬趨勢熱力圖：所有分數水位 × 每一天到第60天，一次看完</h3>
<p>上面的線圖只畫了最恐懼、最貪婪兩條。這張熱力圖把<b>分數切成10等分（直軸，D1最恐懼→D10最貪婪）</b>、
<b>持有期從第1天排到第60天（橫軸）</b>，每一格的顏色是那個「分數水位 × 持有天數」的平均報酬。挑任一條分數列
從左掃到右，就能看那個水位的報酬隨持有天數怎麼變化。取樣用重疊（每日），因為橫軸要有每日顆粒度、非用重疊不可；
熱力圖只畫平均報酬、不涉及顯著性，重疊完全沒問題。<b>色階刻意用藍↔紅、不用紅綠</b>（台股紅=漲綠=跌跟美股相反，
紅綠會誤導）：<span style="color:#b1592f;font-weight:700">紅＝報酬較高</span>、<span style="color:#3E5C76;font-weight:700">藍＝報酬較低</span>。</p>

<h4>① 原始平均報酬（會被「大盤地心引力」主導）</h4>
<div style="text-align:center;">{raw_heatmap_img}</div>
<p class="sub">整張圖幾乎都是紅的、而且越往右（持有越久）越紅——這就是前面講過的「地心引力」：這15.5年是長多頭，
不管分數多少、持有越久平均報酬越高。光看這張，看不太出分數的影響，因為所有列都被大盤上漲墊高了。</p>

<h4>② 超額報酬（扣掉同持有期的無條件平均，才看得出分數的影響）</h4>
<div style="text-align:center;">{excess_heatmap_img}</div>
<div class="callout">
  <b>扣掉地心引力之後，分數的影響一目了然，而且跟前面所有分析完全對得起來：</b>
  <ul>
    <li><b>最底下兩列（D1、D2，最恐懼，分數約0~29）整條紅、而且越往右越紅</b>——恐懼時買、持有越久（往30~60天）
    超額報酬越明顯。這跟線圖「最恐懼超額報酬隨持有期爬升」是同一件事，只是這裡看得到它是一整片、不是單一個數字。</li>
    <li><b>中間（D5、D6，分數約44~57的「中性」）整條最藍</b>——這是超額報酬最差的一區，印證了前面「中間分數
    沒有可靠正向訊號、甚至偏弱」的結論。</li>
    <li><b>最上面兩列（D9、D10，最貪婪）短持有期還偏中性，但拉長到40天以上就轉藍（負超額）</b>——貪婪的「退燒」
    確實存在，但要數個月才顯現，短線看不到。這也跟前面「最貪婪要60天以上才轉負」一致。</li>
  </ul>
  一句話：<b>紅在左下、藍在中間跟右上</b>——恐懼＋中長線持有是唯一持續偏紅（偏強）的角落，中性最弱，貪婪要很久才退燒。
</div>

<h3>那「不同分數，短線該怎麼操作」這個問題，誠實的答案是什麼？</h3>
<div class="card">
  <table>
    <tr><th>CNN分數</th><th>幾天內的短線（3~10個交易日）</th><th>幾週到幾個月（20~90個交易日）</th></tr>
    <tr>
      <td><b>&lt;25 最恐懼</b></td>
      <td class="dim">沒有可靠的短線優勢（3~5天的超額報酬接近0，甚至可能為負）</td>
      <td class="fear-tilt">歷史上優勢會逐漸浮現，20天以上有一致、跨標的的正超額報酬，但仍非統計顯著、且第三方重建期間資料可信度較低，要打折扣</td>
    </tr>
    <tr>
      <td><b>25~75 中性</b></td>
      <td class="dim">沒有可靠規律（見上一節：連拿掉頭尾各4組後方向都不穩定）</td>
      <td class="dim">同樣沒有可靠規律，不建議把中性分數本身當成操作依據</td>
    </tr>
    <tr>
      <td><b>&gt;75 最貪婪</b></td>
      <td class="dim">沒有證據支持放空或大幅減碼（3~40天超額報酬多為正）</td>
      <td class="greed-tilt">歷史上60天以上開始出現負超額報酬，但樣本已經偏薄（見上表n值），不宜當成精準的出場時間點</td>
    </tr>
  </table>
  <p class="sub">
    整體來說：這個指數比較適合當「情緒溫度計」搭配數週到數月的視野去解讀，不太適合當成幾天內
    進出的短線交易訊號——短線窗口（3~10天）不管在哪個分數區間，超額報酬都小、方向也不穩定。
    如果操作目標真的是短線（幾天內進出），這份分析沒有找到支持這個指數有用的證據；如果目標是
    在極端恐懼時提高警覺、考慮分批布局，並且有耐心持有數週到數月，歷史數據方向上支持這個做法，
    但強度仍然不到能拿來精準計時進出場的程度——上一節的策略回測已經證明，即使方向正確，
    真的拿來決定持倉去留，長期報酬還是大幅跑輸單純買進持有。
  </p>
</div>

<h2>四、兩段子時期對照</h2>
<div class="callout warn">
  <b>切點是2021-02-01，這個日期背後有一段修正過程，值得寫清楚：</b>一開始只測試CNN官方API的
  HTTP狀態碼，發現不管帶多早的start_date，2020-07-14以前一律回傳500、2020-07-15開始回傳200，
  於是把切點設在2020-07-15。後來畫「CNN指數時間軸 vs 股價走勢」對照圖時，肉眼發現2020年那段
  官方資料長時間打平在一條直線上，回頭檢查數值才發現：2020-07-15~2021-01-21這段「官方」資料裡，
  有<b>122天的分數被打死在50.0</b>（後端某個計算窗口顯然還沒暖機完成，回傳的是佔位值，不是真的
  逐日計算結果），中間還夾雜幾天離譜的近0異常值。HTTP 200不代表資料是真的——這是分析過程中一個
  真實的方法論錯誤，現在已修正：<b>2021-01-22之後才是連續、真正逐日計算的官方資料</b>，切點抓
  2021-02-01（留幾天緩衝），2020-07-15~2021-01-31這段改回用第三方重建資料頂替。巧的是，這個
  修正後的日期，跟使用者原信最早提出的假設「2021-02-01」幾乎一模一樣——一開始以為「實測結果比
  假設更早」，其實是實測方法本身不夠嚴謹（只測了HTTP狀態，沒檢查數值），原本的假設反而更準。
</div>
<div class="card">
  <div class="table-wrap"><table>
    <tr><th>樣本</th><th>IC vs ^NDX</th><th>IC vs SP500</th><th>分桶單調性 vs ^NDX</th><th>分桶單調性 vs SP500</th></tr>
    {ic_rows}
  </table></div>
</div>

<h3>兩段子時期分桶圖</h3>
{period_compare_charts}

<div class="callout {'warn' if not (nq_consistent and spx_consistent) else ''}">
  <b>穩定性判讀：</b>^NDX的IC方向在兩段子時期{stability_note_nq}；SP500的IC方向在兩段子時期{stability_note_spx}。
  <br>
  請注意：如果兩段結果差異很大，這個差異<b>可能來自資料品質的落差</b>（第三方重建資料本身可能不夠精確，見下方
  「重建準確度驗證」的量化數字），<b>不必然代表市場行為真的不同</b>——不要只看到某段IC比較強就直接下結論說
  「那段時期訊號比較有效」，兩段樣本數也不對等（第三方重建期間約{p1['n_rows']}個交易日、官方API期間約{p2['n_rows']}個交易日），
  差異本身就可能只是統計雜訊。
</div>

<h2>五、資料來源與可信度</h2>
<div class="card">
  <h3>CNN指數（股市版）</h3>
  <div class="table-wrap"><table>
    <tr><th>期間</th><th>來源</th><th>可信度</th></tr>
    <tr><td>2011-01-03 ~ 2021-01-31</td><td>第三方重建（GitHub: <code>whit3rabbit/fear-greed-data</code>）</td><td>中——見下方說明</td></tr>
    <tr><td>2021-02-01 ~ 今</td><td>CNN官方API（<code>production.dataviz.cnn.io</code>）</td><td>高——官方即時資料</td></tr>
  </table></div>
  <p>合併後的資料表（<code>data/fg_merged.csv</code>）在每一列都標註 <code>source</code> 欄位
  （<code>official</code> / <code>reconstructed</code>），兩種可信度的資料在分析全程都可以被獨立篩選、不會被混在一起處理。
  官方API裡2021-02-01之前那122天佔位值（見上方callout）已經被抓出來剔除，不使用。</p>

  <h3>重建準確度「驗證」——以及為什麼這個驗證其實驗不出2011~2021那段的準確度</h3>
  <p>第三方重建資料本身持續更新到今天，理論上可以拿2021-02-01之後「官方API」與「第三方重建」
  同時存在的重疊期，比較兩者準確度：</p>
  <div class="table-wrap"><table>
    <tr><th>重疊期</th><th>樣本數</th><th>Spearman相關</th><th>平均絕對誤差 MAE</th><th>均方根誤差 RMSE</th><th>情緒標籤(fear/greed等)完全一致率</th></tr>
    <tr>
      <td>{val['overlap_start']} ~ {val['overlap_end']}</td>
      <td>{val['n_overlap_days']}</td>
      <td class="quality-good">{val['spearman_rho']:.4f}</td>
      <td>{val['mae']:.4f}分</td>
      <td>{val['rmse']:.4f}分</td>
      <td>{val['rating_exact_match_rate']:.1%}</td>
    </tr>
  </table></div>
  <p class="sub"><b>結果幾乎是完美吻合（ρ≈1.0000、MAE≈0）——但這不是好消息，是這個驗證方法本身
  失效的訊號。</b>完美吻合最合理的解釋是：這個GitHub第三方資料源，至少在近期，本身就是每天直接
  照抄／爬取CNN官方那個API，不是獨立用其他方法重建的。換句話說，2021-02-01之後的「重疊比對」，
  比的其實是同一個資料源跟它自己，<b>不能拿來證明2011~2021這段真正靠獨立方法重建的資料有多準確</b>——
  那段用的多半是別的方法（例如從舊版網頁存檔或不同端點回推），跟近期這種「直接抄官方」的做法不是
  同一回事，準確度沒辦法用這個重疊期去反推。誠實的說法是：<b>2011~2021這段第三方重建資料的實際
  準確度，這份報告沒有找到獨立的驗證方式</b>，只能提醒讀者這段的可信度低於官方API期間，程度未知，
  解讀時要留更大的安全邊際，不宜過度解讀那段的細節數字。</p>
  <p class="sub">另外，第三方原始資料在2020-06-06 ~ 2020-07-08（約33天）之間有一段缺漏（該來源本身缺這段），
  已如實反映在合併後的資料表裡（缺漏期間沒有列），沒有用插值或前值填補去掩蓋這個缺口。</p>

  <h3>^NDX / SP500 價格資料</h3>
  <p>皆取自 Yahoo Finance：^NDX（那斯達克100現貨指數）、^GSPC（標普500現貨指數），
  時間範圍對齊到CNN指數資料最早可取得的日期（2011-01-03）至今。<b>兩邊都是現貨指數</b>——
  這是2026-07-23的更新：這一側原本用的是 NQ=F（那斯達克100期貨），但期貨有轉倉／展期的價差雜訊，
  跟SP500用現貨指數不是完全對等的比較基礎；改成 ^NDX 現貨指數之後，「^NDX vs SP500」這組對照
  就是兩個同性質的現貨指數在比，資料層面更乾淨，沒有期貨/現貨混用的問題。代價是 ^NDX 是指數、
  不是可直接交易的商品，但本報告的策略回測本來就是示意性質（SP500 那側也一樣是不可直接交易的指數），
  這個取捨不影響任何結論。</p>
</div>

<h2>六、^NDX 與 SP500 兩組結果，為什麼不能當作對等的兩次驗證</h2>
<div class="callout">
  CNN恐懼貪婪指數的七項分項成分（動能、強度、廣度、避險需求、垃圾債需求、選擇權Put/Call、波動度VIX）
  裡，動能、強度、廣度這幾項本身就是直接拿標普500或紐約證交所相關資料算出來的。這代表：
  <ul>
    <li><b>「CNN指數對SP500未來報酬的預測力」</b>測試，某種程度上是指數在對自己的資料來源做預測——
    分數本身部分是由SP500近期走勢決定的，兩者天生就會有一定程度的統計關聯，這個關聯不完全能算作
    「指數真的預測到了什麼」，也可能只是「指數的計算方式本來就跟SP500的近況綁在一起」。</li>
    <li><b>「CNN指數對^NDX未來報酬的預測力」</b>測試，因為^NDX完全不是CNN指數計算時用到的任何一項輸入
    資料，是一個更乾淨、更能檢驗這個訊號能不能類推到「指數本身沒看過」的其他市場的測試。如果^NDX這邊
    的IC強度、方向跟SP500差不多，那才是訊號有跨市場類推能力的比較強證據；如果^NDX明顯比SP500弱，
    比較合理的解讀是SP500那組數字有一部分只是「同源效應」，不是真正的預測力。</li>
  </ul>
  下面用本次算出來的全樣本數字具體對照：
</div>
<div class="card">
  <div class="table-wrap"><table>
    <tr><th></th><th>^NDX（乾淨的跨市場測試）</th><th>SP500（部分同源，解讀要打折扣）</th></tr>
    <tr><td>全樣本IC（20日、不重疊）</td><td>{fmt_rho(full['ic']['NDX'])}</td><td>{fmt_rho(full['ic']['SPX'])}</td></tr>
    <tr><td>全樣本分桶單調性</td><td>{fmt_mono(full['bucket']['NDX'])}</td><td>{fmt_mono(full['bucket']['SPX'])}</td></tr>
  </table></div>
</div>

<h2>七、方法論</h2>
<div class="card">
  <ul>
    <li><b>IC分析（2026-07-23改版：重疊rho + 不重疊顯著性）：</b>CNN指數分數 vs 未來20個交易日報酬，
    Spearman等級相關係數。<b>rho（點估計）用重疊取樣</b>（全部每日配對），用到全部資料、點估計更穩定；
    <b>但顯著性（星號與p值）用不重疊取樣</b>（每隔20交易日取一筆），因為重疊樣本相鄰報酬窗口大量重疊、
    自相關強，會把樣本數與顯著性灌水（重疊p會假性地變超小，見摘要頁紅框的實例）。所以表裡rho是重疊、
    p值是不重疊——兩者分工，既給重疊IC又不謊報顯著性。</li>
    <li><b>分位數分桶：</b>用pandas <code>qcut</code> 把CNN指數分數切20等分，看每組未來20日平均報酬。
    這裡改用「逐日重疊」資料（沒有做不重疊取樣）——這跟IC分析的取樣方法不同，是刻意的取捨：
    20組分桶如果也套不重疊取樣，拆成兩段子時期後每組平均樣本數會掉到個位數，20組裡大半會被自訂的
    n&lt;10門檻標成低可信度、圖表會看不出訊息。分桶分析算的是「描述性平均數」，不是拿來做統計顯著性
    檢定的p值，重疊窗口造成的自相關對平均數的偏誤，遠比對IC的p值有效性影響小，因此接受這個折衷，
    但明確寫在這裡，不要讓讀者誤以為兩種分析用同一套取樣邏輯。</li>
    <li><b>訊號時間點：</b>用第t日收盤時的CNN指數分數，對照第t日收盤到第t+20個交易日收盤的價格報酬，
    不使用未來資料。</li>
    <li><b>子時期切點：</b>2021-02-01，對齊CNN官方API真正可信的資料下限（見上方「資料來源與可信度」
    裡2020-07-15~2021-01-21佔位值段落的說明）。</li>
    <li><b>樣本數標示：</b>分桶圖表裡每一根柱子都標n值；n&lt;10的分組會用橘色標示、代表可信度較低
    （本次分桶因為改用逐日重疊資料，實際上每組樣本數都遠高於10，橘色警示在這次的圖表裡應該不會出現）。</li>
    <li><b>無條件基準線／超額報酬：</b>不管CNN分數是多少，同一段期間「隨便挑一天」往後看N日的
    平均報酬，拿來當分桶圖的地心引力基準；每組平均報酬減掉這條基準，就是超額報酬版本，0代表
    「跟隨便選一天沒兩樣」。</li>
    <li><b>中間段IC：</b>用跟主IC分析完全相同的不重疊取樣，取完樣之後才把第1組（最恐懼）、
    最後一組（最貪婪）的樣本點剔除，避免在篩選後才取樣、重新引入報酬窗口重疊的問題。</li>
    <li><b>策略回測：</b>用第t日收盤的CNN分數，決定第t+1日的持倉比例（<code>shift(1)</code>，
    不偷看未來），逐日計算報酬並複利累積；三套持倉公式分別是買進持有（恆為100%多單）、
    恐懼多單／貪婪空手（<code>clip((50-分數)/50, 0, 1)</code>）、恐懼多單／貪婪放空
    （<code>clip((50-分數)/50, -1, 1)</code>，跟美債專案<code>position_size()</code>公式相同）。
    績效指標（CAGR、年化波動、Sharpe、Sortino、最大回撤）皆為自行用日報酬序列計算，
    無風險利率簡化為0；只跑全樣本（2011-01~2026-07），沒有拆兩段子時期，因為策略回測本來就需要
    夠長的期間才有意義，拆開後樣本太短、複利效果會被扭曲，參考價值有限。</li>
  </ul>
</div>

<h2>八、結論</h2>
<div class="card">
  <p>
    全樣本下，CNN恐懼貪婪指數（股市版）對^NDX與SP500未來20個交易日報酬的IC分別為
    {full['ic']['NDX']['rho']:+.3f}（p={full['ic']['NDX']['pval']:.3f}）與
    {full['ic']['SPX']['rho']:+.3f}（p={full['ic']['SPX']['pval']:.3f}），
    方向都是負的，符合「恐懼時買入、未來報酬較高」的傳統解讀方向，但強度偏弱，
    且都<b>{'未達' if full['ic']['NDX']['pval']>=0.05 and full['ic']['SPX']['pval']>=0.05 else '部分達到'}</b>
    常見的p&lt;0.05統計顯著門檻。
  </p>
  <p>
    拆開兩段子時期看，^NDX的方向{stability_note_nq}、SP500的方向{stability_note_spx}，官方API期間
    （2021-02至今）修正資料品質問題之後，IC強度（^NDX {p2['ic']['NDX']['rho']:+.3f}、SP500
    {p2['ic']['SPX']['rho']:+.3f}）其實跟第三方重建期間相當接近，比修正前的版本（當時被佔位值
    污染、^NDX的IC被拉到接近0）更穩定、更可信。不過官方期樣本數仍只有約{p2['n_rows']}個交易日、
    顯著性用的不重疊取樣只剩約{p2['ic']['NDX']['n_nonoverlap']}筆，統計檢定力本來就偏低，加上這段期間本身經歷
    2022升息、2023-2025AI狂熱等風格迥異的市場階段，IC結果的穩健性還是需要保守看待，不宜只憑
    這一段的數字就對訊號下強烈結論。
  </p>
  <p>
    ^NDX與SP500兩組結果之間，因為SP500本身是CNN指數部分計算輸入的來源，SP500那組IC不能被當成
    獨立於指數本身的「乾淨」驗證；^NDX那組因為完全不是指數的計算輸入，是這次驗證裡比較有參考價值的
    跨市場類推證據。整體而言，這個指數看起來更接近「同期市場情緒的溫度計」，而不是一個能提前20個
    交易日、有統計顯著把握去predict報酬方向的獨立訊號。
  </p>
  <p>
    <b>訊號的方向性幾乎完全集中在頭尾的極端讀數，中間的「中性」區間沒有規律。</b>把最恐懼、
    最貪婪兩端各拿掉4組、只留中間12組（約30~68分）重算IC，全樣本^NDX是
    {full['middle_ic']['NDX']['rho']:+.3f}（p={full['middle_ic']['NDX']['pval']:.3f}，不顯著），
    SP500是{full['middle_ic']['SPX']['rho']:+.3f}（p={full['middle_ic']['SPX']['pval']:.3f}，也不顯著），
    兩段子時期同樣都貼著0。意思是：CNN分數落在中性區間時，對未來20日報酬沒有清楚、可靠、能重複驗證的
    區辨力；真正有訊息量的只有最恐懼那一小段（最貪婪那段則因為動能延續效應，也不像單純的反向訊號）。
  </p>
  <p>
    <b>把訊號變成策略之後，長期報酬遠遠落後買進持有。</b>全樣本回測顯示，「恐懼多單／貪婪空手」
    策略15.5年只累積{r['backtest']['NDX']['metrics']['long_only_tilt']['total_return']:+.0f}%（^NDX），
    「恐懼多單／貪婪放空」更只有{r['backtest']['NDX']['metrics']['long_short']['total_return']:+.1f}%，
    兩者都遠遠不敵買進持有的{r['backtest']['NDX']['metrics']['buy_hold']['total_return']:+.0f}%——
    根本原因就是第一節講的「地心引力」：這15.5年是罕見長多頭，任何讓你長時間空手或做空的策略，
    都會持續錯過這段漲幅。這印證了「貪婪時不宜放空」的直覺，但也進一步顯示：<b>就連「貪婪時只是
    空手觀望、不做多不做空」，付出的機會成本可能已經比想像中大得多。</b>如果只看風險（最大回撤、
    波動度），恐懼多單／貪婪空手確實比買進持有溫和很多，這是它唯一站得住腳的優勢，但這是用犧牲
    大部分長期報酬換來的，不是「風險降低、報酬照樣好」的免費午餐。
  </p>
  <p>
    整體結論：這個指數的「恐懼」端有一致、值得留意的逆向訊號，「貪婪」端不宜當成放空理由（甚至
    不宜當成大幅減碼的理由），中間九成的分數區間本身規律性很弱，比較適合當成極端值的警示器，而不是
    拿來對整條0~100分做連續、線性的持倉調整訊號。以上所有結論都建立在2011~2026這段特定的歷史樣本
    （尤其是長期大多頭的背景）之上，換一段市場風格不同的樣本，結果可能不同。
  </p>
</div>

<footer>
  資料來源：CNN官方API（production.dataviz.cnn.io）、GitHub whit3rabbit/fear-greed-data、Yahoo Finance（^NDX、^GSPC）。
  分析與報告程式碼位於 <code>fg_nq_analysis/ic_analysis/</code>（<code>fetch_fg.py</code> /
  <code>fetch_prices.py</code> / <code>analysis.py</code> / <code>build_report.py</code>），與美債恐懼貪婪
  儀表板專案（<code>bond_data_pipeline/</code>）完全分開存放，不共用任何資料或程式碼。
</footer>

</body>
</html>
"""
    out_path = OUT_DIR / "report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"報告已產出：{out_path}")
    return out_path


if __name__ == "__main__":
    build()
