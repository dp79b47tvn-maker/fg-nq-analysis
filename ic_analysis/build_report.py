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

    def chart_img(pname, tcol):
        b = periods[pname]["bucket"][tcol]
        if b is None or b.get("chart_base64") is None:
            return "<p class='dim'>（無法產生此分桶圖）</p>"
        return f"<img src='data:image/png;base64,{b['chart_base64']}' style='width:100%;max-width:760px;'/>"

    full_charts = f"""
      <div class="chart-pair">
        <div>{chart_img('full', 'NQ')}</div>
        <div>{chart_img('full', 'SPX')}</div>
      </div>"""

    period_compare_charts = ""
    for pname in ["period1_reconstructed", "period2_official"]:
        period_compare_charts += f"""
      <h4>{PERIOD_LABELS[pname]}（{periods[pname]['date_range'][0]} ~ {periods[pname]['date_range'][1]}，n={periods[pname]['n_rows']}筆）</h4>
      <div class="chart-pair">
        <div>{chart_img(pname, 'NQ')}</div>
        <div>{chart_img(pname, 'SPX')}</div>
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

<h2>二、兩段子時期對照</h2>
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

<h2>三、資料來源與可信度</h2>
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

<h2>四、NQ 與 SP500 兩組結果，為什麼不能當作對等的兩次驗證</h2>
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

<h2>五、方法論</h2>
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
  </ul>
</div>

<h2>六、結論</h2>
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
