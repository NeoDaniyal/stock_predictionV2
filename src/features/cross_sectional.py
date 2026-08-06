import numpy as np
import pandas as pd

def add_cross_sectional_features(df:pd.DataFrame, data_col: str ="Date")-> pd.DataFrame:
    """Calcuates daily cross-sectional ranks and percentile ranks for key metrics across all stocks.
    
    For each unique trading date, stocks are ranked relative to each other.
    Returns normalized percentile ranks scaled between 0.0 and 1.0"""
    df = df.copy()

    rank_target_cols = ["Daily_Returns", "Volume", "RSI_14", "Volatility_30", "Log_Returns"]
    cols_to_rank = [col for col in rank_target_cols if col in df.columns]

    for col in cols_to_rank:
        df[f"{col}_CS_Rank"] = df.groupby(data_col)[col].rank(pct=True, method="average")
        
    return df

if __name__ == "__main__":
    sample_data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "Ticker": ["AAPL", "MSFT", "TSLA", "AAPL", "MSFT", "TSLA"],
            "Daily_Returns": [0.021, 0.007, 0.054, -0.010, 0.015, -0.025],
            "Volume": [1000, 2000, 1500, 1100, 1900, 1600],
            "RSI_14": [55.0, 62.0, 70.0, 52.0, 65.0, 48.0],
            "Volatility_30": [0.012, 0.010, 0.025, 0.013, 0.009, 0.027],
        }
    )

    processed_df = add_cross_sectional_features(sample_data)
    print("\nSample Cross-Sectional Features Output:")
    print(
        processed_df[
            [
                "Date",
                "Ticker",
                "Daily_Returns",
                "Daily_Returns_CS_Rank",
                "RSI_14_CS_Rank",
            ]
        ]
    )