# fg_nq_analysis — 給接手的 AI Agent 看的專案說明

這份文件是為了讓任何 AI coding agent（不限 Claude Code——Cursor、Codex CLI、Windsurf、Copilot
等都適用）在沒有先前對話記憶的情況下，能快速看懂這個專案在做什麼、怎麼重跑、有哪些已知的坑。
如果你是被叫來接手這個專案的 agent，請先讀完這份文件再開始改東西。

## 這個 repo 裡有兩個東西，只有一個是目前在維護的

- **`ic_analysis/`** — 目前唯一在維護的專案。CNN 恐懼貪婪指數（股市版）對 ^NDX、SP500 的
  因子驗證分析：IC 相關性、分位數分桶、策略回測、報酬趨勢熱力圖。**所有後續開發都在這裡做。**
- **`run_analysis.py`（repo 根目錄）+ 根目錄的 `output/`** — 舊的、方法論完全不同的分析
  （forward-only streak 分析），2026-07-15 做的，跟 `ic_analysis/` 不共用任何程式碼或資料。
  **除非使用者明確要求，不要動這個檔案**，它是獨立保留的舊產出，不是這個專案的一部分。

以下說明全部是關於 `ic_analysis/`。

## 專案在做什麼

驗證 CNN 恐懼貪婪指數（股市版，0~100分，0=極度恐懼、100=極度貪婪）對那斯達克100（^NDX）
與標普500（^GSPC）未來報酬有沒有預測力。輸出是一份獨立的 HTML 報告
（`ic_analysis/output/report.html`）。

🌐 上線網址（GitHub Actions 每個交易日自動重跑+部署，見下方「發布流程」）：
https://dp79b47tvn-maker.github.io/fg-nq-analysis/

📊 另有一份較早手動發布的 Claude Artifact 版本（不會自動同步，可能落後）：
https://claude.ai/code/artifact/d69aa4a5-f2c4-4f5b-9c76-4900e15fff4b

跟同一個使用者的另一個專案 `bond_data_pipeline/`（美債恐懼貪婪儀表板）方法論同源
（沿用它的 `factor_validation_analysis.py` 架構），但兩個專案完全獨立、不共用資料或程式碼。

## 執行順序（一定要照這個順序跑）

```bash
cd ic_analysis
python3 fetch_fg.py       # 抓CNN指數(官方API+第三方重建)，合併輸出 data/fg_merged.csv
python3 fetch_prices.py   # 抓^NDX、^GSPC價格，輸出 data/prices.csv
python3 analysis.py       # 核心分析(IC/分桶/回測/熱力圖等)，輸出 output/results.json + 各種PNG(base64存在json裡)
python3 build_report.py   # 讀 results.json，組出 output/report.html
```

`data/` 和 `output/` 大部分都在 `.gitignore` 裡（每次重跑會重新產生，不追蹤歷史），**唯一
例外是 `output/report.html`**——這個檔案刻意不被忽略、會進版本控制，因為 GitHub Actions
的部署流程需要它被 commit 回 repo 才能追蹤歷史／供 Pages 服務讀取（見下方「發布流程」）。
改完程式碼後，
四支腳本**不一定要全部重跑**——如果只改了 `build_report.py` 的排版/文字，只要 `results.json`
還在，直接重跑 `build_report.py` 就好；如果改了 `analysis.py` 裡的計算邏輯，要重跑
`analysis.py`（會用到已經抓好的 `data/*.csv`，不需要重新 fetch）；只有資料本身可能過期時
（例如很久沒跑、想要最新的股價/CNN分數）才需要重跑 `fetch_fg.py` / `fetch_prices.py`。

### 需要的套件
`pandas`, `numpy`, `scipy`, `matplotlib`, `yfinance`, `requests`。沒有 `requirements.txt`
（目前用系統 Python 直接跑，沒有 venv）——如果你的環境沒有這些套件，先 `pip install` 對應套件。

## 已知的坑（不要重蹈覆轍）

1. **CNN 官方 API 的「HTTP 200」不代表資料是真的。**
   `production.dataviz.cnn.io/index/fearandgreed/graphdata/{start_date}` 這個端點，
   2020-07-15~2021-01-21 這段回傳的分數有 122 天是打死的 `50.0`（後端某個計算窗口還沒暖機
   完成的佔位值），中間還夾雜幾天離譜的近 0 異常值。`fetch_fg.py` 裡的 `OFFICIAL_FLOOR`
   已經設成實測驗證過的 **2021-02-01**（不是單純測 HTTP 狀態碼測出來的 2020-07-15）。
   `detect_placeholder_runs()` 會在每次重抓資料時自動檢查有沒有連續打平的可疑段落——
   如果哪天官方 API 又出現這種問題、`OFFICIAL_FLOOR` 之後的乾淨區間也中標，這個函式會印警告，
   不要略過那個警告。
2. **CNN F&G 資料最早只到 2011-01-03**，指數本身大約 2012 年才推出，2010 年以前的資料
   在任何地方（含第三方重建）都不存在，不要假設可以往前拉。
3. **NQ 這一側用的是 `^NDX`（現貨指數），不是 `NQ=F`（期貨）**——這是 2026-07-23 從期貨改過來的，
   原因是避免期貨轉倉/展期價差雜訊跟 SP500 的現貨指數混用。程式裡的欄位／變數名稱統一是
   `NDX`，不要看到 "NQ" 就以為裝的是 NQ=F 期貨資料。
4. **IC 分析用「重疊 rho + 不重疊顯著性」的混合取樣**（2026-07-23 定案）：
   - `rho`（相關係數點估計）用**重疊取樣**算（全部每日的分數/未來報酬配對），用到全部資料、
     點估計更穩。
   - **p 值/顯著性星號**一律用**不重疊取樣**算（每隔 horizon 天才取一筆樣本），因為重疊取樣
     下相鄰樣本的報酬窗口大量重疊、有強自相關，會把顯著性嚴重灌水（實測：全樣本 ^NDX 重疊
     p=0.0018 看起來「顯著」，不重疊 p=0.622 其實不顯著）。
   - `analysis.py` 裡的 `compute_ic()` 同時回傳兩者：`rho`/`n`（重疊）、`pval`/`n_nonoverlap`
     （不重疊，這個才是可信的顯著性）、`pval_overlap`（重疊 p，只放著對照用，**不能拿來下結論**）。
   - **改任何 IC 相關的程式碼時，千萬不要把 `pval` 換成重疊版本**，那會讓報告的顯著性判斷
     全部變成錯的。
5. **分桶分析（`bucket_analysis`）刻意用逐日重疊資料**，跟 IC 分析的取樣邏輯不同——這是有意的
   設計，不是疏漏。分桶算的是描述性平均數，不是拿來做顯著性檢定，重疊造成的偏誤影響小，
   但拆兩段子時期後如果也用不重疊取樣，每組樣本數會掉到太少。詳細理由寫在 `analysis.py`
   檔頭注解跟 `build_report.py` 的「方法論」章節。
6. **熱力圖顏色故意不用紅綠**，用藍↔紅（`coolwarm`）：台股「紅漲綠跌」跟美股「綠漲紅跌」
   相反，紅綠配色對台灣讀者是誤導的（這個 repo 服務的使用者是台灣人）。維持這個色階選擇，
   不要為了「看起來更像財經圖表」就換回紅綠。
7. **`bucket_analysis` 的第 20 組（最貪婪端）不是單調遞減到最低**——內部有「倒 U」形，
   `top_bucket_drilldown()` 專門拆這個。改動分桶/貪婪端的敘述時，記得這個發現，不要走回
   「越貪婪越差」這種過度簡化的單調敘述。

## 資料來源

- CNN 官方 API：`production.dataviz.cnn.io/index/fearandgreed/graphdata/{start_date}`
  （2021-02-01 起可信；帶 header 偽裝瀏覽器，否則會被擋回 418）
- 第三方重建（2011-01-03~2021-01-31 用這段）：GitHub `whit3rabbit/fear-greed-data`
  （`fear-greed.csv`）——2021-02 之後跟官方幾乎完全一致（Spearman ρ≈1.0），這**不能**當成
  對 2011~2021 那段重建準確度的驗證，那段的真實準確度沒有獨立驗證方式，解讀要更保守。
- 價格：Yahoo Finance（`yfinance`），`^NDX`、`^GSPC`。

## 發布流程（2026-07-26起：GitHub Actions 自動化，比照 bond_data_pipeline 的模式）

`.github/workflows/update-and-deploy.yml` 會在三種情況觸發：(1) 每個交易日美股收盤後
（cron，22:00 UTC）(2) push 到 `main` (3) 手動 `workflow_dispatch`。流程是：

1. 安裝中文字型（matplotlib 圖表要用）、裝 `ic_analysis/requirements.txt` 的套件。
2. 依序跑 `fetch_fg.py` → `fetch_prices.py` → `analysis.py` → `build_report.py`。
3. 跑 `ic_analysis/scripts/verify_report.py` 驗證（`<div>`/`<table>` 標籤配對、沒有
   f-string 殘留痕跡、嵌入圖表都是合法PNG、必要章節都在）——**沒過就不會往下部署**。
4. 把重新產生的 `output/report.html` commit 回 repo（`.gitignore` 對它開了例外，見上方）。
5. 複製一份成 `output/index.html`，把整個 `output/` 資料夾部署到 GitHub Pages。

**本機開發流程**：改完 `build_report.py` 或 `analysis.py` 後，照「執行順序」重新產生
`output/report.html`，然後手動跑一次 `python3 ic_analysis/scripts/verify_report.py`
確認過關，再考慮要不要 push（push到main會觸發上面的自動部署）。Commit 前一定要先問過
使用者要不要 commit（見 `~/.claude/CLAUDE.md` 的全域 git 規則，如果你是在讀那份文件的
環境下工作）。

**如果 Pages 網址打不開**：GitHub Pages 需要在 repo 設定裡手動啟用過一次
（Settings → Pages → Source 選 "GitHub Actions"），這是一次性的手動步驟，git push
沒辦法自動做這件事——如果部署流程跑到 `deploy` job 失敗，先檢查這個有沒有設定。

## 使用者背景（有助於判斷語氣跟優先順序）

使用者用繁體中文溝通，是這個分析的主導者、會對統計方法論細節追問到底（多次要求把某個
取捨講清楚、要求「先討論再執行」）。回應風格上，這個使用者重視：誠實揭露方法論限制
（不要美化結果）、發現問題主動講清楚不要含糊帶過、每個決定點列選項讓使用者拍板而不是自己
悶著頭選。報告的語氣也延續這個原則——寧可誠實地說「這裡看不出規律」，也不要為了讓報告
好看而過度解讀雜訊。
