"""抓取 ^NDX(那斯達克100指數) 與 ^GSPC(標普500指數) 歷史日線，對齊到CNN指數可取得的最早日期。

2026-07-23：NQ 這一側從原本的 NQ=F(那斯達克100期貨) 改成 ^NDX(那斯達克100現貨指數)。
原因：兩邊都用現貨指數之後，「NQ vs SP500」這組對照就沒有「一個期貨、一個現貨」的資料性質
不對稱問題（期貨的轉倉／展期價差雜訊），是更乾淨的比較。欄位識別碼也一併從 NQ 正名為 NDX，
不留「欄位叫NQ、裝的卻是^NDX」這種會誤導人的地雷。
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent / "data"
START = "2011-01-01"


def fetch(ticker, name):
    df = yf.download(ticker, start=START, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"].rename(name)
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close


def main():
    ndx = fetch("^NDX", "NDX")
    spx = fetch("^GSPC", "SPX")
    print("^NDX:", ndx.index.min().date(), "~", ndx.index.max().date(), f"({len(ndx)}筆)")
    print("^GSPC:", spx.index.min().date(), "~", spx.index.max().date(), f"({len(spx)}筆)")

    combined = pd.concat([ndx, spx], axis=1, sort=True)
    combined.index.name = "Date"
    combined.to_csv(DATA_DIR / "prices.csv")
    return combined


if __name__ == "__main__":
    main()
