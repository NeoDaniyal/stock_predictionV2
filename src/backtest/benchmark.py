from pathlib import Path
import pandas as pd

from configs.config import FEATURE_DATA_PATH


def load_data():
    path = Path(FEATURE_DATA_PATH) / "final_dataset.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def calculate_daily_returns(df):
    df = df.copy()
    # Fill NaN returns on the first day per ticker with 0.0
    df["Daily_Return"] = (
        df.groupby("Ticker")["Close"].pct_change().fillna(0.0)
    )
    return df


def build_benchmark(df):
    # Average return across all tickers on each date
    benchmark = df.groupby("Date")["Daily_Return"].mean().reset_index()
    return benchmark


def simulate_benchmark(benchmark, initial_capital=10000.0):
    capital = initial_capital
    portfolio = []

    for row in benchmark.itertuples():
        capital *= 1 + row.Daily_Return
        portfolio.append({"Date": row.Date, "Portfolio": capital})

    # Return statement moved outside the loop
    return pd.DataFrame(portfolio)


def save_results(df):
    Path("reports").mkdir(exist_ok=True)
    output = Path("reports") / "benchmark_equity.csv"
    df.to_csv(output, index=False)
    print(f"Saved benchmark equity to: {output}")


def main():
    df = load_data()
    df = calculate_daily_returns(df)

    benchmark = build_benchmark(df)
    equity = simulate_benchmark(benchmark)

    save_results(equity)

    print("\nFirst 5 benchmark records:")
    print(equity.head())

    print("\nLast 5 benchmark records:")
    print(equity.tail())


if __name__ == "__main__":
    main()