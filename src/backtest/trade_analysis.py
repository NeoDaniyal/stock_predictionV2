from pathlib import Path
import pandas as pd


def load_trades() -> pd.DataFrame:
    path = Path("reports") / "trade_log.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Trade log file not found at {path}. Run portfolio.py first."
        )
    return pd.read_csv(path, parse_dates=["Entry_Date", "Exit_Date"])


def analyze_by_ticker(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyzes trade count, total profit, average return, and win rate per ticker."""
    summary = (
        trades.groupby("Ticker")
        .agg(
            Trade_Count=("Profit", "count"),
            Total_Profit=("Profit", "sum"),
            Mean_Profit=("Profit", "mean"),
            Mean_Return=("Return", "mean"),
            Win_Rate=("Return", lambda x: (x > 0).mean()),
        )
        .sort_values("Total_Profit", ascending=False)
    )
    return summary


def analyze_by_signal(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyzes trade count, total profit, average return, and win rate by trade signal (1: BUY, -1: SELL)."""
    signal_map = {1: "BUY", -1: "SELL"}
    trades_copy = trades.copy()
    trades_copy["Signal_Name"] = trades_copy["Signal"].map(signal_map)

    summary = (
        trades_copy.groupby("Signal_Name")
        .agg(
            Trade_Count=("Profit", "count"),
            Total_Profit=("Profit", "sum"),
            Mean_Profit=("Profit", "mean"),
            Mean_Return=("Return", "mean"),
            Win_Rate=("Return", lambda x: (x > 0).mean()),
        )
        .sort_values("Total_Profit", ascending=False)
    )
    return summary


def analyze_by_confidence(trades: pd.DataFrame) -> pd.DataFrame:
    """Groups trades into confidence bins and evaluates performance per bin."""
    trades_copy = trades.copy()
    trades_copy["Confidence_Bin"] = pd.cut(
        trades_copy["Confidence"],
        bins=[0.69, 0.70, 0.80, 0.90, 1.00],
        labels=["<0.70", "0.70-0.79", "0.80-0.89", "0.90-1.00"],
    )

    summary = (
        trades_copy.groupby("Confidence_Bin", observed=False)
        .agg(
            Trade_Count=("Profit", "count"),
            Total_Profit=("Profit", "sum"),
            Mean_Profit=("Profit", "mean"),
            Mean_Return=("Return", "mean"),
            Win_Rate=("Return", lambda x: (x > 0).mean()),
        )
    )
    return summary


def save_reports(
    ticker_summary: pd.DataFrame,
    signal_summary: pd.DataFrame,
    confidence_summary: pd.DataFrame,
):
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    ticker_summary.to_csv(reports_dir / "trade_analysis_by_ticker.csv")
    signal_summary.to_csv(reports_dir / "trade_analysis_by_signal.csv")
    confidence_summary.to_csv(reports_dir / "trade_analysis_by_confidence.csv")

    print(f"\nTrade analysis reports saved to: {reports_dir}/")


def main():
    print("Loading trade log...")
    trades = load_trades()

    print(f"Total trades loaded: {len(trades)}")

    ticker_summary = analyze_by_ticker(trades)
    signal_summary = analyze_by_signal(trades)
    confidence_summary = analyze_by_confidence(trades)

    print("\n" + "=" * 60)
    print("1. PERFORMANCE BY TICKER")
    print("=" * 60)
    print(ticker_summary.to_string())

    print("\n" + "=" * 60)
    print("2. PERFORMANCE BY SIGNAL (BUY vs SELL)")
    print("=" * 60)
    print(signal_summary.to_string())

    print("\n" + "=" * 60)
    print("3. PERFORMANCE BY CONFIDENCE LEVEL")
    print("=" * 60)
    print(confidence_summary.to_string())

    save_reports(ticker_summary, signal_summary, confidence_summary)


if __name__ == "__main__":
    main()