"""組出獨立的HTML因子驗證報告：CNN恐懼貪婪指數(股市版) vs NQ=F / SP500。"""
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent / "output"

PERIOD_LABELS = {
    "full": "全樣本",
    "period1_reconstructed": "第三方重建期間",
    "period2_official": "官方API期間",
}
TARGET_LABELS = {"NQ": "NQ=F（那斯達克100期貨）", "SPX": "^GSPC（標普500指數）"}


def fmt_rho(cell):
    if cell is None or cell.get("rho") is None:
        return f"<span class='dim'>樣本不足(n={cell.get('n', 0) if cell else 0})</span>"
    rho, pval, n = cell["rho"], cell["pval"], cell["n"]
    sig = "**" if pval < 0.01 else ("*" if pval < 0.05 else "")
    cls = "fear-tilt" if rho < 0 else "greed-tilt"
    return f"<span class='{cls}'>{rho:+.3f}{sig}</span><br><span class='sub'>p={pval:.3f}, n={n}</span>"


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
          <td>{fmt_rho(pd_['ic']['NQ'])}</td>
          <td>{fmt_rho(pd_['ic']['SPX'])}</td>
          <td>{fmt_mono(pd_['bucket']['NQ'])}</td>
          <td>{fmt_mono(pd_['bucket']['SPX'])}</td>
        </tr>"""

    def chart_img(pname, tcol, key="chart_base64"):
        b = periods[pname]["bucket"][tcol]
        if b is None or b.get(key) is None:
            return "<p class='dim'>（無法產生此圖）</p>"
        return f"<img src='data:image/png;base64,{b[key]}' style='width:100%;max-width:760px;'/>"

    full_charts = f"""
      <div class="chart-pair">
        <div>{chart_img('full', 'NQ')}</div>
        <div>{chart_img('full', 'SPX')}</div>
      </div>"""

    full_excess_charts = f"""
      <div class="chart-pair">
        <div>{chart_img('full', 'NQ', 'excess_chart_base64')}</div>
        <div>{chart_img('full', 'SPX', 'excess_chart_base64')}</div>
      </div>"""

    price_trend_img = (
        f"<img src='data:image/png;base64,{r['price_trend_chart_base64']}' "
        f"style='width:100%;max-width:820px;'/>"
        if r.get("price_trend_chart_base64") else "<p class='dim'>（無法產生走勢圖）</p>"
    )

    uncond_nq = full["bucket"]["NQ"]["unconditional"]
    uncond_spx = full["bucket"]["SPX"]["unconditional"]

    period_compare_charts = ""
    for pname in ["period1_reconstructed", "period2_official"]:
        u_nq = periods[pname]["bucket"]["NQ"]["unconditional"]
        u_spx = periods[pname]["bucket"]["SPX"]["unconditional"]
        period_compare_charts += f"""
      <h4>{PERIOD_LABELS[pname]}（{periods[pname]['date_range'][0]} ~ {periods[pname]['date_range'][1]}，n={periods[pname]['n_rows']}筆；
      同期無條件平均20日報酬：NQ {u_nq['mean']:+.2f}%、SP500 {u_spx['mean']:+.2f}%）</h4>
      <div class="chart-pair">
        <div>{chart_img(pname, 'NQ')}</div>
        <div>{chart_img(pname, 'SPX')}</div>
      </div>
      <p class="sub">超額報酬版（扣掉上面這條同期基準線）：</p>
      <div class="chart-pair">
        <div>{chart_img(pname, 'NQ', 'excess_chart_base64')}</div>
        <div>{chart_img(pname, 'SPX', 'excess_chart_base64')}</div>
      </div>"""

    # 方向一致性判斷
    def sign(cell):
        if cell is None or cell.get("rho") is None:
            return None
        return cell["rho"] < 0

    nq_signs = [sign(periods[p]["ic"]["NQ"]) for p in ["period1_reconstructed", "period2_official"]]
    spx_signs = [sign(periods[p]["ic"]["SPX"]) for p in ["period1_reconstructed", "period2_official"]]
    nq_consistent = len(set(nq_signs)) == 1 and None not in nq_signs
    spx_consistent = len(set(spx_signs)) == 1 and None not in spx_signs

    stability_note_nq = "方向一致" if nq_consistent else "方向不一致"
    stability_note_spx = "方向一致" if spx_consistent else "方向不一致"

    # ---- 中間分數規律性 ----
    middle_rows = ""
    for pname in ["full", "period1_reconstructed", "period2_official"]:
        m = periods[pname]["middle_ic"]
        rng = m["NQ"].get("score_range") if m.get("NQ") else None
        rng_txt = f"{rng[0]:.0f}~{rng[1]:.0f}分" if rng else "—"
        middle_rows += f"""
        <tr>
          <td>{PERIOD_LABELS[pname]}<br><span class='sub'>分數範圍 {rng_txt}</span></td>
          <td>{fmt_rho(m['NQ'])}</td>
          <td>{fmt_rho(m['SPX'])}</td>
        </tr>"""

    middle_charts_full = f"""
      <div class="chart-pair">
        <div>{chart_img('full', 'NQ', 'middle_chart_base64')}</div>
        <div>{chart_img('full', 'SPX', 'middle_chart_base64')}</div>
      </div>"""

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

    html = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"/>
<title>CNN恐懼貪婪指數(股市版) 對 NQ / SP500 預測力驗證報告</title>
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

<h1>CNN恐懼貪婪指數（股市版）對 NQ 與 SP500 未來報酬的預測力驗證</h1>
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

<h2>一、摘要：全樣本結果</h2>
<div class="card">
  <div class="table-wrap"><table>
    <tr><th>樣本</th><th>IC vs NQ=F（20日、不重疊取樣）</th><th>IC vs SP500（20日、不重疊取樣）</th>
        <th>分桶單調性 vs NQ</th><th>分桶單調性 vs SP500</th></tr>
    <tr>
      <td>{PERIOD_LABELS['full']}<br><span class='sub'>{full['date_range'][0]} ~ {full['date_range'][1]}，{full['n_rows']}筆</span></td>
      <td>{fmt_rho(full['ic']['NQ'])}</td>
      <td>{fmt_rho(full['ic']['SPX'])}</td>
      <td>{fmt_mono(full['bucket']['NQ'])}</td>
      <td>{fmt_mono(full['bucket']['SPX'])}</td>
    </tr>
  </table></div>
  <p class="sub">* p&lt;0.05　** p&lt;0.01（Spearman檢定）。IC為負，代表分數低（恐懼）對應未來報酬較高，方向符合「恐懼買入」的傳統解讀；IC為正則相反。</p>
</div>

<h3>全樣本20分位分桶（依CNN指數分數切20等分，看未來20日平均報酬）</h3>
{full_charts}
<p class="sub">分桶分析用逐日重疊資料計算（不做不重疊取樣），跟上面IC表用的「不重疊取樣」方法不同，理由見下方「方法論」段落。</p>

<h3>為什麼最右邊（最貪婪）那組沒有變成負的？</h3>
<div class="callout">
  直覺上「貪婪時未來報酬應該轉差」，甚至轉負——但上面的圖裡，就連第20組（最貪婪）NQ與SP500的
  未來20日平均報酬<b>仍然是正的</b>。原因是整條分桶曲線<b>疊在「大盤長期上漲」這個基準之上</b>，
  不是以0為中心畫出來的。
</div>
<p>把CNN指數分數完全丟開，這15.5年裡「隨便挑一天」往後看20個交易日，平均報酬長這樣：</p>
<div class="table-wrap"><table>
  <tr><th></th><th>無條件平均20日報酬</th><th>中位數</th><th>報酬為負的比例</th></tr>
  <tr><td>NQ=F</td><td class="greed-tilt">{uncond_nq['mean']:+.2f}%</td><td>{uncond_nq['median']:+.2f}%</td><td>{uncond_nq['pct_negative']:.1f}%</td></tr>
  <tr><td>SP500</td><td class="greed-tilt">{uncond_spx['mean']:+.2f}%</td><td>{uncond_spx['median']:+.2f}%</td><td>{uncond_spx['pct_negative']:.1f}%</td></tr>
</table></div>
<p>這是2011~2026這段罕見長多頭的結果——下面是同一段期間NQ=F、SP500的實際走勢（對數座標）：</p>
<div style="text-align:center;">{price_trend_img}</div>
<p class="sub">NQ=F這15.5年累積漲了約1193%（年化約17.9%），SP500累積約490%（年化約12.1%）。
在這種背景下，隨便挑一個交易日往後看20天，平均本來就已經是正的（NQ約+1.47%、SP500約+1.01%），
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

<h3>把最恐懼、最貪婪兩組拿掉，中間段還有沒有方向性？</h3>
<p>整條分桶曲線的「方向」，有多少其實是被頭尾兩組(第1組最恐懼、第20組最貪婪)撐起來的？
把這兩組樣本剔除，只留中間段(約12~85分之間，依樣本區間略有不同)，重新算一次CNN分數
跟未來報酬的Spearman相關(取樣方法跟主IC表完全一樣，一樣先做不重疊取樣、再篩掉頭尾)：</p>
<div class="card">
  <div class="table-wrap"><table>
    <tr><th>樣本</th><th>中間段IC vs NQ=F</th><th>中間段IC vs SP500</th></tr>
    {middle_rows}
  </table></div>
  <p class="sub">對照：拿掉頭尾前的全樣本IC是 NQ {fmt_rho(full['ic']['NQ'])}　SP500 {fmt_rho(full['ic']['SPX'])}（見上方摘要表）。</p>
</div>
<p>全樣本中間段的IC是 NQ {full['middle_ic']['NQ']['rho']:+.3f}（p={full['middle_ic']['NQ']['pval']:.3f}）、
SP500 {full['middle_ic']['SPX']['rho']:+.3f}（p={full['middle_ic']['SPX']['pval']:.3f}）——
比拿掉頭尾前的全樣本IC還要更弱，官方API期間NQ的中間段IC甚至幾乎是0
（{p2['middle_ic']['NQ']['rho']:+.3f}，p={p2['middle_ic']['NQ']['pval']:.3f}）。
把中間18組的超額報酬攤開來看也看不出穩定的階梯狀：</p>
{middle_charts_full}
<p class="sub">中間段長條圖高高低低沒有清楚的遞減趨勢(例如全樣本NQ第3組是+1.14%全場數一數二高，
緊接著第6組就轉負-0.51%)。合理的解讀是：<b>這個訊號的「方向性」幾乎完全集中在頭尾兩個極端
(尤其是最恐懼那組)，中間九成的分數區間本身沒有清楚、可靠的規律</b>——分數從35分走到65分，
不太能說「未來報酬會怎麼系統性變化」，比較像雜訊。這對操作上的含意是：這個指數比較適合當
「極端值警示器」（分數落到最低或最高的一小段區間時多留意），而不是拿來對整條0~100分做連續的
「分數越低倉位越重」這種線性訊號——下一節的策略回測會直接驗證這個想法。</p>

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

<h3>NQ=F 回測（全樣本，2011-01 ~ 2026-07）</h3>
{backtest_table('NQ')}
<div style="text-align:center;margin-top:10px;">{backtest_chart('NQ')}</div>

<h3>SP500 回測（全樣本，2011-01 ~ 2026-07）</h3>
{backtest_table('SPX')}
<div style="text-align:center;margin-top:10px;">{backtest_chart('SPX')}</div>

<div class="callout warn">
  <b>結果跟直覺可能相反：兩套訊號策略都遠遠輸給乾脆買進持有。</b>
  NQ買進持有15.5年累積 {r['backtest']['NQ']['metrics']['buy_hold']['total_return']:+.0f}%
  （年化{r['backtest']['NQ']['metrics']['buy_hold']['cagr']:+.1f}%、Sharpe
  {r['backtest']['NQ']['metrics']['buy_hold']['sharpe']:.2f}）；「恐懼多單／貪婪空手」只累積
  {r['backtest']['NQ']['metrics']['long_only_tilt']['total_return']:+.0f}%
  （年化{r['backtest']['NQ']['metrics']['long_only_tilt']['cagr']:+.1f}%）；「恐懼多單／貪婪放空」
  幾乎打平，累積只有{r['backtest']['NQ']['metrics']['long_short']['total_return']:+.1f}%
  （年化{r['backtest']['NQ']['metrics']['long_short']['cagr']:+.2f}%，15年幾乎沒賺錢）。SP500也是同樣的型態。
  <br><br>
  原因直接跟第一節「地心引力」的發現連在一起：這15.5年是罕見的長多頭，「恐懼多單／貪婪空手」策略平均
  只有約19%的時間持有多單（其餘時間空手在等分數變低），錯過了大部分的漲幅；「恐懼多單／貪婪放空」
  更慘，貪婪時的空單持續在跟這段長期上漲的趨勢對作，把買進持有原本該賺到的報酬幾乎全部吃掉。
  「恐懼多單／貪婪空手」的最大回撤（NQ {r['backtest']['NQ']['metrics']['long_only_tilt']['max_dd']:.1f}%）
  確實比買進持有（{r['backtest']['NQ']['metrics']['buy_hold']['max_dd']:.1f}%）小很多，波動也低很多，
  如果目標是「降低回撤、犧牲一部分報酬換穩定」，這個策略在風險控制上是有作用的；但如果目標是
  「打敗大盤」，即使風險調整後看Sharpe值，買進持有（{r['backtest']['NQ']['metrics']['buy_hold']['sharpe']:.2f}）
  仍然明顯贏過恐懼多單／貪婪空手（{r['backtest']['NQ']['metrics']['long_only_tilt']['sharpe']:.2f}）。
  <br><br>
  這跟上一節「不宜放空」的結論方向一致，但這裡的回測把它量化得更清楚：不只是「放空會少賺一點」，
  而是<b>放空在這段長多頭裡幾乎抵銷了訊號原本該有的優勢，讓15年的報酬幾乎歸零</b>。
  這個結果高度依賴這15.5年剛好是大多頭的樣本背景，如果换成一段長期盤整或空頭主導的期間，
  三套策略的相對表現可能完全不同——這是用歷史回測評估任何時機策略時都有的限制，不是這個指數獨有的問題。
</div>

<h2>三、兩段子時期對照</h2>
<div class="callout">
  子時期切點定在 <b>2020-07-15</b>，這是CNN官方API實測得到的資料下限（不是原先假設的2021-02-01；
  API對早於2020-07-14的start_date一律回傳500錯誤，2020-07-15是後端資料庫本身的下限，不是「近一年」
  這種滾動窗口限制）。因此「第三方重建期間」與「官方API期間」的切點，直接對齊資料可信度真正改變的那一天。
</div>
<div class="card">
  <div class="table-wrap"><table>
    <tr><th>樣本</th><th>IC vs NQ=F</th><th>IC vs SP500</th><th>分桶單調性 vs NQ</th><th>分桶單調性 vs SP500</th></tr>
    {ic_rows}
  </table></div>
</div>

<h3>兩段子時期分桶圖</h3>
{period_compare_charts}

<div class="callout {'warn' if not (nq_consistent and spx_consistent) else ''}">
  <b>穩定性判讀：</b>NQ的IC方向在兩段子時期{stability_note_nq}；SP500的IC方向在兩段子時期{stability_note_spx}。
  <br>
  請注意：如果兩段結果差異很大，這個差異<b>可能來自資料品質的落差</b>（第三方重建資料本身可能不夠精確，見下方
  「重建準確度驗證」的量化數字），<b>不必然代表市場行為真的不同</b>——不要只看到某段IC比較強就直接下結論說
  「那段時期訊號比較有效」，兩段樣本數也不對等（第三方重建期間約{p1['n_rows']}個交易日、官方API期間約{p2['n_rows']}個交易日），
  差異本身就可能只是統計雜訊。
</div>

<h2>四、資料來源與可信度</h2>
<div class="card">
  <h3>CNN指數（股市版）</h3>
  <div class="table-wrap"><table>
    <tr><th>期間</th><th>來源</th><th>可信度</th></tr>
    <tr><td>2011-01-03 ~ 2020-07-14</td><td>第三方重建（GitHub: <code>whit3rabbit/fear-greed-data</code>）</td><td>中——見下方量化驗證</td></tr>
    <tr><td>2020-07-15 ~ 今</td><td>CNN官方API（<code>production.dataviz.cnn.io</code>）</td><td>高——官方即時資料</td></tr>
  </table></div>
  <p>合併後的資料表（<code>data/fg_merged.csv</code>）在每一列都標註 <code>source</code> 欄位
  （<code>official</code> / <code>reconstructed</code>），兩種可信度的資料在分析全程都可以被獨立篩選、不會被混在一起處理。</p>

  <h3>重建準確度驗證（不只是文字警語，用重疊期實際比對）</h3>
  <p>第三方重建資料在近年其實持續更新到今天，因此可以拿 2020-07-15 之後「官方API」與「第三方重建」
  同時存在的重疊期，直接比較兩者準確度：</p>
  <div class="table-wrap"><table>
    <tr><th>重疊期</th><th>樣本數</th><th>Spearman相關</th><th>平均絕對誤差 MAE</th><th>均方根誤差 RMSE</th><th>情緒標籤(fear/greed等)完全一致率</th></tr>
    <tr>
      <td>{val['overlap_start']} ~ {val['overlap_end']}</td>
      <td>{val['n_overlap_days']}</td>
      <td class="quality-good">{val['spearman_rho']:.3f}</td>
      <td>{val['mae']:.2f}分</td>
      <td>{val['rmse']:.2f}分</td>
      <td>{val['rating_exact_match_rate']:.1%}</td>
    </tr>
  </table></div>
  <p class="sub">第三方重建資料跟官方資料在等級相關上高度一致（ρ≈0.95），平均誤差約1.5分（滿分100分），
  但RMSE比MAE明顯大，代表存在少數誤差較大的離群日；情緒標籤（fear/greed等5級）完全一致率約92.5%，
  表示約每13天會有一天標籤等級對不上（例如官方判「fear」但重建資料判「neutral」）。整體而言重建資料
  可用但不是逐日精確複製官方數字，這是2011~2020-07這段第三方重建期間結果解讀時要放在心上的落差。</p>
  <p class="sub">另外，第三方原始資料在2020-06-06 ~ 2020-07-08（約33天）之間有一段缺漏（該來源本身缺這段），
  已如實反映在合併後的資料表裡（缺漏期間沒有列），沒有用插值或前值填補去掩蓋這個缺口。</p>

  <h3>NQ / SP500 價格資料</h3>
  <p>皆取自 Yahoo Finance：NQ=F（那斯達克100期貨連續合約）、^GSPC（標普500現貨指數），
  時間範圍對齊到CNN指數資料最早可取得的日期（2011-01-03）至今。NQ用期貨而非 ^NDX 現貨指數是使用者
  的明確選擇，好處是報酬更貼近實際可交易商品，代價是期貨轉倉可能帶來的價差雜訊，讓NQ這一側的報酬
  序列跟SP500的「現貨指數」在資料性質上不是完全對等的比較基礎——解讀時建議把這一點也算進去。</p>
</div>

<h2>五、NQ 與 SP500 兩組結果，為什麼不能當作對等的兩次驗證</h2>
<div class="callout">
  CNN恐懼貪婪指數的七項分項成分（動能、強度、廣度、避險需求、垃圾債需求、選擇權Put/Call、波動度VIX）
  裡，動能、強度、廣度這幾項本身就是直接拿標普500或紐約證交所相關資料算出來的。這代表：
  <ul>
    <li><b>「CNN指數對SP500未來報酬的預測力」</b>測試，某種程度上是指數在對自己的資料來源做預測——
    分數本身部分是由SP500近期走勢決定的，兩者天生就會有一定程度的統計關聯，這個關聯不完全能算作
    「指數真的預測到了什麼」，也可能只是「指數的計算方式本來就跟SP500的近況綁在一起」。</li>
    <li><b>「CNN指數對NQ未來報酬的預測力」</b>測試，因為NQ完全不是CNN指數計算時用到的任何一項輸入
    資料，是一個更乾淨、更能檢驗這個訊號能不能類推到「指數本身沒看過」的其他市場的測試。如果NQ這邊
    的IC強度、方向跟SP500差不多，那才是訊號有跨市場類推能力的比較強證據；如果NQ明顯比SP500弱，
    比較合理的解讀是SP500那組數字有一部分只是「同源效應」，不是真正的預測力。</li>
  </ul>
  下面用本次算出來的全樣本數字具體對照：
</div>
<div class="card">
  <div class="table-wrap"><table>
    <tr><th></th><th>NQ=F（乾淨的跨市場測試）</th><th>SP500（部分同源，解讀要打折扣）</th></tr>
    <tr><td>全樣本IC（20日、不重疊）</td><td>{fmt_rho(full['ic']['NQ'])}</td><td>{fmt_rho(full['ic']['SPX'])}</td></tr>
    <tr><td>全樣本分桶單調性</td><td>{fmt_mono(full['bucket']['NQ'])}</td><td>{fmt_mono(full['bucket']['SPX'])}</td></tr>
  </table></div>
</div>

<h2>六、方法論</h2>
<div class="card">
  <ul>
    <li><b>IC分析：</b>CNN指數分數 vs 未來20個交易日報酬，Spearman等級相關係數，一律用「不重疊取樣」
    （每隔20個交易日才取一筆樣本），避免相鄰樣本因報酬窗口重疊而互相高度相關、把統計顯著性灌水。</li>
    <li><b>分位數分桶：</b>用pandas <code>qcut</code> 把CNN指數分數切20等分，看每組未來20日平均報酬。
    這裡改用「逐日重疊」資料（沒有做不重疊取樣）——這跟IC分析的取樣方法不同，是刻意的取捨：
    20組分桶如果也套不重疊取樣，拆成兩段子時期後每組平均樣本數會掉到個位數，20組裡大半會被自訂的
    n&lt;10門檻標成低可信度、圖表會看不出訊息。分桶分析算的是「描述性平均數」，不是拿來做統計顯著性
    檢定的p值，重疊窗口造成的自相關對平均數的偏誤，遠比對IC的p值有效性影響小，因此接受這個折衷，
    但明確寫在這裡，不要讓讀者誤以為兩種分析用同一套取樣邏輯。</li>
    <li><b>訊號時間點：</b>用第t日收盤時的CNN指數分數，對照第t日收盤到第t+20個交易日收盤的價格報酬，
    不使用未來資料。</li>
    <li><b>子時期切點：</b>2020-07-15，對齊CNN官方API實測資料下限（見上方「資料來源與可信度」）。</li>
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

<h2>七、結論</h2>
<div class="card">
  <p>
    全樣本下，CNN恐懼貪婪指數（股市版）對NQ與SP500未來20個交易日報酬的IC分別為
    {full['ic']['NQ']['rho']:+.3f}（p={full['ic']['NQ']['pval']:.3f}）與
    {full['ic']['SPX']['rho']:+.3f}（p={full['ic']['SPX']['pval']:.3f}），
    方向都是負的，符合「恐懼時買入、未來報酬較高」的傳統解讀方向，但強度偏弱，
    且都<b>{'未達' if full['ic']['NQ']['pval']>=0.05 and full['ic']['SPX']['pval']>=0.05 else '部分達到'}</b>
    常見的p&lt;0.05統計顯著門檻。
  </p>
  <p>
    拆開兩段子時期看，NQ的方向{stability_note_nq}、SP500的方向{stability_note_spx}，
    但官方API期間（2020-07至今）的樣本數只有約{p2['n_rows']}個交易日、不重疊取樣後IC只剩約
    {p2['ic']['NQ']['n']}筆，統計檢定力本來就偏低，加上這段期間本身經歷2020疫情復甦、2022升息、
    2023-2025AI狂熱等多個風格迥異的市場階段，IC結果的穩健性需要更保守看待，不宜只憑這一段的數字
    就對訊號下強烈結論。
  </p>
  <p>
    NQ與SP500兩組結果之間，因為SP500本身是CNN指數部分計算輸入的來源，SP500那組IC不能被當成
    獨立於指數本身的「乾淨」驗證；NQ那組因為完全不是指數的計算輸入，是這次驗證裡比較有參考價值的
    跨市場類推證據。整體而言，這個指數看起來更接近「同期市場情緒的溫度計」，而不是一個能提前20個
    交易日、有統計顯著把握去predict報酬方向的獨立訊號。
  </p>
  <p>
    <b>訊號的方向性幾乎完全集中在頭尾兩個極端。</b>把最恐懼、最貪婪兩組拿掉，中間段的IC
    （NQ {full['middle_ic']['NQ']['rho']:+.3f}、SP500 {full['middle_ic']['SPX']['rho']:+.3f}）
    比全樣本IC更弱，官方API期間的中間段IC甚至接近0。意思是：CNN分數從中性偏恐懼走到中性偏貪婪這段
    （分數大約落在15~80分之間），對未來20日報酬幾乎沒有穩定、可靠的區辨力；真正有訊息量的只有
    最恐懼那一小段（最貪婪那段則因為動能延續效應、不像單純的反向訊號）。
  </p>
  <p>
    <b>把訊號變成策略之後，長期報酬遠遠落後買進持有。</b>全樣本回測顯示，「恐懼多單／貪婪空手」
    策略15.5年只累積{r['backtest']['NQ']['metrics']['long_only_tilt']['total_return']:+.0f}%（NQ），
    「恐懼多單／貪婪放空」更只有{r['backtest']['NQ']['metrics']['long_short']['total_return']:+.1f}%，
    兩者都遠遠不敵買進持有的{r['backtest']['NQ']['metrics']['buy_hold']['total_return']:+.0f}%——
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
  資料來源：CNN官方API（production.dataviz.cnn.io）、GitHub whit3rabbit/fear-greed-data、Yahoo Finance（NQ=F、^GSPC）。
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
