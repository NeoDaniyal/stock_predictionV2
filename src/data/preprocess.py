from pathlib import Path

import pandas as pd
import numpy as np

from configs.config import (RAW_DATA_PATH, PROCESSED_DATA_PATH)

def load_data():

    file_path = (
        Path(RAW_DATA_PATH)/ "combined_stock_data.csv"
    )
    
    df = pd.read_csv(file_path, parse_dates=["Date"])
    return df

def clean_data(df):
    print("~"*60)
    print(f"Original Shape: {df.shape}")
    print("~"*60)

    df = df.drop_duplicates()

    df = df.sort_values(["Ticker", "Date"])

    df = df.reset_index(drop=True)

    print(f"After cleaning: {df.shape}")
    return df

def add_returns(df):

    df["Daily_Returns"] = (df.groupby("Ticker")["Close"].pct_change())

    return df

def add_log_returns(df):

    df["Log_Returns"] = np.log(df["Close"]/df.groupby("Ticker")["Close"].shift(1))

    return df

def add_volatility(df):

    df["Volatility_30"] = (df.groupby("Ticker")["Daily_Returns"].rolling(window=30).std().reset_index(level=0, drop=True))
    return df

def  report_missing_values(df):
    missing = (df.isnull().sum().sort_values(ascending=False))
    print("="*60)
    print("\nMissing values\n")
    print("="*60)

    print(missing[missing>0])

def save_data(df):
    Path(PROCESSED_DATA_PATH).mkdir(parents=True, exist_ok=True)
    output_path = (Path(PROCESSED_DATA_PATH)/"processed_stock_data.csv")
    df.to_csv(output_path, index=False)

    print(f"\nSaved to {output_path}")

def main():

    df = load_data()

    df = clean_data(df)

    df = add_returns(df)

    df = add_log_returns(df)

    df = add_volatility(df)

    report_missing_values(df)

    save_data(df)

    print("\nFinal shape:")

    print(df.shape)

    print(df.head())

if __name__ == "__main__":
    main()