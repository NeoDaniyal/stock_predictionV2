from pathlib import Path
import numpy as np
import pandas as pd
from configs.config import FEATURE_DATA_PATH


def load_data():
    path = Path(FEATURE_DATA_PATH) / "stock_features.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    lags = [1, 3, 7, 14]
    for lag in lags:
        df[f"Close_lag_{lag}"] = df.groupby("Ticker")["Close"].shift(lag)
        df[f"Volume_lag_{lag}"] = df.groupby("Ticker")["Volume"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    windows = [7, 14, 30]
    for window in windows:
        df[f"Rolling_Mean_{window}"] = (
            df.groupby("Ticker")["Close"]
            .transform(lambda x: x.rolling(window).mean())
        )
        df[f"Rolling_STD_{window}"] = (
            df.groupby("Ticker")["Close"]
            .transform(lambda x: x.rolling(window).std())
        )
    return df


def add_cross_sectional_features(
    df: pd.DataFrame, date_col: str = "Date"
) -> pd.DataFrame:
    """Module 1: Cross-Sectional Features."""
    df = df.copy()
    rank_target_cols = [
        "Daily_Returns",
        "Volume",
        "RSI_14",
        "Volatility_30",
        "Log_Returns",
    ]
    cols_to_rank = [col for col in rank_target_cols if col in df.columns]

    for col in cols_to_rank:
        df[f"{col}_CS_Rank"] = df.groupby(date_col)[col].rank(
            pct=True, method="average"
        )
    return df


def add_rolling_percentiles(
    df: pd.DataFrame, window: int = 252
) -> pd.DataFrame:
    """Module 2: Min-Max Normalized Rolling Percentiles."""
    df = df.copy()
    target_cols = ["RSI_14", "ATR_14", "Volume", "Volatility_30"]
    cols_to_pct = [c for c in target_cols if c in df.columns]

    for col in cols_to_pct:
        roll_min = df.groupby("Ticker")[col].transform(
            lambda x: x.rolling(window, min_periods=30).min()
        )
        roll_max = df.groupby("Ticker")[col].transform(
            lambda x: x.rolling(window, min_periods=30).max()
        )
        df[f"{col}_Pct_252"] = (df[col] - roll_min) / (
            roll_max - roll_min + 1e-8
        )
    return df


def add_rolling_zscores(
    df: pd.DataFrame, windows: list = [20, 60]
) -> pd.DataFrame:
    """Module 3: Rolling Z-Score features."""
    df = df.copy()
    target_cols = [
        "RSI_14",
        "ATR_14",
        "Volume",
        "Daily_Returns",
        "Volatility_30",
    ]
    cols_to_z = [c for c in target_cols if c in df.columns]

    for window in windows:
        for col in cols_to_z:
            mean = df.groupby("Ticker")[col].transform(
                lambda x: x.rolling(window, min_periods=10).mean()
            )
            std = df.groupby("Ticker")[col].transform(
                lambda x: x.rolling(window, min_periods=10).std()
            )
            df[f"{col}_Z_{window}"] = (df[col] - mean) / (std + 1e-8)

    return df


def add_trend_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """Module 4: Distance from Moving Averages and Cross Ratios."""
    df = df.copy()

    if "SMA_10" in df.columns:
        df["Distance_SMA_10"] = (df["Close"] / df["SMA_10"]) - 1
    if "SMA_30" in df.columns:
        df["Distance_SMA_30"] = (df["Close"] / df["SMA_30"]) - 1

    if "SMA_10" in df.columns and "SMA_30" in df.columns:
        df["SMA_Cross_Ratio"] = (df["SMA_10"] / df["SMA_30"]) - 1

    if "EMA_10" in df.columns and "EMA_30" in df.columns:
        df["EMA_Cross_Ratio"] = (df["EMA_10"] / df["EMA_30"]) - 1

    return df


def add_momentum_acceleration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Module 5: RSI Velocity and MACD Histogram Acceleration."""
    df = df.copy()

    if "RSI_14" in df.columns:
        df["RSI_Change_3"] = df.groupby("Ticker")["RSI_14"].diff(3)
        df["RSI_Change_7"] = df.groupby("Ticker")["RSI_14"].diff(7)

    if "MACD" in df.columns and "MACD_SIGNAL" in df.columns:
        df["MACD_Hist"] = df["MACD"] - df["MACD_SIGNAL"]
        df["MACD_Hist_Slope_3"] = df.groupby("Ticker")["MACD_Hist"].diff(3)

    return df


def add_relative_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """Module 6: Asset performance relative to market cross-sectional mean."""
    df = df.copy()

    mkt_return = df.groupby("Date")["Daily_Returns"].transform("mean")
    df["Relative_Return_1D"] = df["Daily_Returns"] - mkt_return

    stock_5d = df.groupby("Ticker")["Daily_Returns"].transform(
        lambda x: x.rolling(5).sum()
    )
    mkt_5d = df.groupby("Date")["Daily_Returns"].mean().rolling(5).sum()
    mkt_5d_mapped = df["Date"].map(mkt_5d)
    df["Relative_Return_5D"] = stock_5d - mkt_5d_mapped

    return df


def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Module 7: Broad Market Volatility & Trend Indicators."""
    df = df.copy()

    daily_mkt = df.groupby("Date")["Daily_Returns"].mean().reset_index()
    daily_mkt["Market_Vol_20"] = (
        daily_mkt["Daily_Returns"].rolling(20, min_periods=5).std()
    )

    mkt_sma_20 = daily_mkt["Daily_Returns"].rolling(20, min_periods=5).mean()
    mkt_sma_50 = daily_mkt["Daily_Returns"].rolling(50, min_periods=10).mean()
    daily_mkt["Market_Trend_Ratio"] = (mkt_sma_20 - mkt_sma_50) / (
        mkt_sma_50.abs() + 1e-8
    )

    df = df.merge(
        daily_mkt[["Date", "Market_Vol_20", "Market_Trend_Ratio"]],
        on="Date",
        how="left",
    )
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Module 8: Explicit Feature Interactions."""
    df = df.copy()

    if "Volatility_30" in df.columns and "RSI_14" in df.columns:
        df["Vol_x_RSI"] = df["Volatility_30"] * df["RSI_14"]

    if "Volatility_30" in df.columns and "Daily_Returns" in df.columns:
        df["Vol_x_Return"] = df["Volatility_30"] * df["Daily_Returns"]

    if "Volume_Z_20" in df.columns and "Daily_Returns_Z_20" in df.columns:
        df["Volume_x_Return"] = df["Volume_Z_20"] * df["Daily_Returns_Z_20"]

    return df


def add_rolling_statistics_features(
    df: pd.DataFrame, window: int = 30
) -> pd.DataFrame:
    """Module 9: Rolling Skew, Kurtosis, and Range Expansion Ratio."""
    df = df.copy()

    if "Daily_Returns" in df.columns:
        df["Rolling_Skew_30"] = df.groupby("Ticker")["Daily_Returns"].transform(
            lambda x: x.rolling(window, min_periods=15).skew()
        )
        df["Rolling_Kurt_30"] = df.groupby("Ticker")["Daily_Returns"].transform(
            lambda x: x.rolling(window, min_periods=15).kurt()
        )

    roll_max = df.groupby("Ticker")["Close"].transform(
        lambda x: x.rolling(window, min_periods=10).max()
    )
    roll_min = df.groupby("Ticker")["Close"].transform(
        lambda x: x.rolling(window, min_periods=10).min()
    )
    df["Rolling_Max_Min_Ratio_30"] = (roll_max / (roll_min + 1e-8)) - 1

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    horizon = 5
    df["Future_Return"] = (
        df.groupby("Ticker")["Close"].shift(-horizon) / df["Close"] - 1
    )
    df["Target_Threshold"] = 0.5 * df["Volatility_30"] * np.sqrt(horizon)

    conditions = [
        df["Future_Return"] > df["Target_Threshold"],
        df["Future_Return"] < -df["Target_Threshold"],
    ]
    choices = [2, 0]  # 2: BUY, 0: SELL

    df["Target"] = np.select(conditions, choices, default=1)  # 1: HOLD
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Building lag features...")
    df = add_lag_features(df)

    print("Building rolling features...")
    df = add_rolling_features(df)

    print("Building Module 1: Cross-Sectional features...")
    df = add_cross_sectional_features(df, date_col="Date")

    print("Building Module 2: Rolling Percentile features...")
    df = add_rolling_percentiles(df, window=252)

    print("Building Module 3: Rolling Z-Score features...")
    df = add_rolling_zscores(df, windows=[20, 60])

    print("Building Module 4: Trend Strength features...")
    df = add_trend_strength_features(df)

    print("Building Module 5: Momentum Acceleration features...")
    df = add_momentum_acceleration_features(df)

    print("Building Module 6: Relative Strength features...")
    df = add_relative_strength_features(df)

    print("Building Module 7: Market Regime features...")
    df = add_market_regime_features(df)

    print("Building Module 8: Interaction features...")
    df = add_interaction_features(df)

    print("Building Module 9: Rolling Statistics features...")
    df = add_rolling_statistics_features(df, window=30)

    print("Creating targets...")
    df = create_target(df)

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print("\nBefore dropping NaN:")
    print(df.shape)
    df = df.dropna().reset_index(drop=True)
    print("After dropping NaN:")
    print(df.shape)
    return df


def save_data(df: pd.DataFrame):
    output_path = Path(FEATURE_DATA_PATH) / "final_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


def main():
    df = load_data()
    print("Original shape:", df.shape)

    df = build_features(df)
    df = clean_dataset(df)

    save_data(df)
    print("\nFinal shape:", df.shape)


if __name__ == "__main__":
    main()