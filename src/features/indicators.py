from pathlib import Path

import pandas as pd
import pandas_ta_classic as ta

from configs.config import (PROCESSED_DATA_PATH, FEATURE_DATA_PATH)

def load_data():

    file_path = (Path(PROCESSED_DATA_PATH)/"processed_stock_data.csv")
    df = pd.read_csv(file_path, parse_dates=["Date"])
    return df
def add_indicators(group):
    group = group.copy()

    group["SMA_10"] = ta.sma(group["Close"], length=10)
    group["SMA_30"] = ta.sma(group["Close"], length=30)

    group["EMA_10"] = ta.ema(group["Close"], length=10)
    group["EMA_30"] = ta.ema(group["Close"], length=30)

    group["RSI_14"] = ta.rsi(group["Close"], length=14)
    
    macd = ta.macd(group["Close"])
    group["MACD"] = macd["MACD_12_26_9"]
    group["MACD_SIGNAL"] = macd["MACDs_12_26_9"]

    group["ATR_14"] = ta.atr(group["High"], group["Low"], group["Close"], length=14)

    bbands = ta.bbands(group["Close"])
    group["BB_UPPER"] = bbands["BBU_5_2.0"]
    group["BB_LOWER"] = bbands["BBL_5_2.0"]
    return group

def build_indicators(df):
    results = []
    for ticker, group in df.groupby("Ticker"):
        group = add_indicators(group)
        group["Ticker"] = ticker
        results.append(group)
    
    df = pd.concat(results, ignore_index=True)
    return df

def save_df(df):
    Path(FEATURE_DATA_PATH).mkdir(parents=True, exist_ok=True)
    output_path = (Path(FEATURE_DATA_PATH)/"stock_features.csv")

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

def main():
    df = load_data()
    print(df.shape)

    df = build_indicators(df)
    print(df.shape)

    print(df.columns.tolist())
    save_df(df)

if __name__ == "__main__":
    main()
    