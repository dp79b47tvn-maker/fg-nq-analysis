# CNN 恐懼貪婪指數（股市版）對 ^NDX / SP500 預測力驗證

驗證 CNN 恐懼貪婪指數（Fear & Greed Index，股市版，0~100分）對那斯達克100指數（^NDX）與
標普500指數（^GSPC）未來報酬有沒有預測力：IC 相關性分析、分位數分桶、不同持有期的訊號強度、
策略回測、報酬趨勢熱力圖。

📊 **報告連結**：https://claude.ai/code/artifact/d69aa4a5-f2c4-4f5b-9c76-4900e15fff4b

跟同一位使用者的另一個專案 [`bond-fear-greed-dashboard`](https://github.com/dp79b47tvn-maker/bond-fear-greed-dashboard)
（美債版恐懼貪婪指數）方法論同源，但兩者完全獨立，不共用資料或程式碼。

## 主要發現（摘要，完整推導與資料見報告）

- 全樣本 IC（20 個交易日）方向符合「恐懼買入」的傳統解讀，但強度弱、未達統計顯著。
- 訊號的方向性幾乎全部集中在最恐懼那端；中性區間（分數約 25~75）沒有可靠規律。
- 最恐懼那組的優勢是隨持有期拉長才愈明顯（3 天幾乎看不到，90 天累積到 +5% 左右），比較像
  中長線現象，不是短線訊號。
- 把訊號直接變成交易策略（做多恐懼／放空貪婪），長期報酬遠遠跑輸單純買進持有——這 15 年
  剛好是罕見長多頭，任何讓你空手或做空的策略都會持續錯過漲幅。

## 專案結構

```
ic_analysis/
├── fetch_fg.py        # 抓CNN恐懼貪婪指數(官方API + 第三方重建)，輸出 data/fg_merged.csv
├── fetch_prices.py     # 抓 ^NDX / ^GSPC 價格，輸出 data/prices.csv
├── analysis.py         # 核心分析：IC、分位數分桶、策略回測、熱力圖，輸出 output/results.json
├── build_report.py     # 組出最終的 output/report.html
└── output/, data/       # 執行產物，不進版本控制(.gitignore)，每次重跑會重新產生

run_analysis.py         # 舊的、方法論不同的分析(forward-only streak)，跟上面獨立、不維護
```

## 重現報告

```bash
cd ic_analysis
pip install pandas numpy scipy matplotlib yfinance requests
python3 fetch_fg.py
python3 fetch_prices.py
python3 analysis.py
python3 build_report.py
# 產出 output/report.html
```

目前是**手動**重跑、手動發布到 Claude Artifact，還沒有像美債專案那樣接 GitHub Actions
自動排程更新。

## 給 AI Agent 的說明

如果你是被叫來接手這個專案的 AI coding agent（不限 Claude Code），完整的執行細節、環境
需求、還有 7 個已知的資料/方法論陷阱，請讀 [`AGENTS.md`](./AGENTS.md)。
