from pathlib import Path

import pandas as pd
import yfinance as yf

from configs.config import (
    TICKERS,
    START_DATE,
    END_DATE,
    RAW_DATA_PATH
)

def download_stock_data(tickers, start_date, end_date):
    """
    Download historical data from Yahoo Finance
    """

    all_data = []

    for ticker in tickers:
        print(f"Downloading {ticker}...")

        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False, multi_level_index=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            print(f"NO, Data for {ticker}")
            continue

        df = df.reset_index()

        df["Ticker"] = ticker
        expected_columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Ticker"
        ]
        missing = set(expected_columns) - set(df.columns)

        if missing:
            raise ValueError(f"{ticker}: Missing columns {missing}")

        all_data.append(df)

    final_df = pd.concat(
        all_data,
        ignore_index=True
    )

    return final_df

def save_data(df):
    Path(RAW_DATA_PATH).mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (Path(RAW_DATA_PATH)/"combined_stock_data.csv")

    df.to_csv(
        output_path,
        index=False
    )

    print(f"Saved to {output_path}")

def main():
    df = download_stock_data(TICKERS, START_DATE, END_DATE)
    print(df.head())
    print(df.shape)
    save_data(df)

if __name__ == "__main__":
    main()