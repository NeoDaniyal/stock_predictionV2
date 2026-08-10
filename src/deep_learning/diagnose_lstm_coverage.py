import os
from pathlib import Path
import pandas as pd

if os.path.exists("/content/drive"):
    PROJECT_ROOT = Path("/content/drive/MyDrive/ML_Projects/stock_predictionV2")
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT /"reports"

def inspect_oof_coverage():
    xbg_path = REPORTS_DIR /"xgb_oof_predictions.csv"
    lstm_path = REPORTS_DIR /"lstm_oof_predictions.csv"

    xgb_df = pd.read_csv(xbg_path, parse_dates=["Date"])
    lstm_df = pd.read_csv(lstm_path, parse_dates=["Date"])

    if "Year" not in xgb_df.columns:
        xgb_df["Year"] = xgb_df["Date"].dt.year
    if "Year" not in lstm_df.columns:
        lstm_df["Year"] = lstm_df["Date"].dt.year
    print("\n=================== YEAR-BY-YEAR ROW COUNT COMPARISON ===================")
    xgb_by_yr = xgb_df.groupby("Year")["Ticker"].nunique().to_frame("XGB_Ticker")
    xgb_by_yr["XGB_Rows"] = xgb_df.groupby("Year")["Date"].count()

    lstm_by_yr = lstm_df.groupby("Year")["Ticker"].nunique().to_frame("LSTM_Ticker")
    lstm_by_yr["LSTM_Rows"] = lstm_df.groupby("Year")["Date"].count()

    comp = xgb_by_yr.join(lstm_by_yr, how="outer").fillna(0).astype(int)

    print("\n=================== MISSING TICKERS IN LSTM ===================")
    all_xgb_tickers = set(xgb_df["Ticker"].unique())
    all_lstm_tickers = set(lstm_df["Ticker"].unique())
    missing_tickers = all_xgb_tickers - all_lstm_tickers
    print(f"Total XGB Ticker: {len(all_xgb_tickers)}")
    print(f"Total LSTM Ticker: {len(all_lstm_tickers)}")
    print(f"Missing Tickers in LSTM OOF ({len(missing_tickers)}): {sorted(list(missing_tickers))}")

if __name__ == "__main__":
    inspect_oof_coverage()