"""抓取 NQ=F(那斯達克100期貨) 與 ^GSPC(標普500指數) 歷史日線，對齊到CNN指數可取得的最早日期。"""
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
    nq = fetch("NQ=F", "NQ")
    spx = fetch("^GSPC", "SPX")
    print("NQ=F:", nq.index.min().date(), "~", nq.index.max().date(), f"({len(nq)}筆)")
    print("^GSPC:", spx.index.min().date(), "~", spx.index.max().date(), f"({len(spx)}筆)")

    combined = pd.concat([nq, spx], axis=1, sort=True)
    combined.index.name = "Date"
    combined.to_csv(DATA_DIR / "prices.csv")
    return combined


if __name__ == "__main__":
    main()
