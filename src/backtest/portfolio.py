from pathlib import Path
import numpy as np
import pandas as pd

# Configs
INITIAL_CAPITAL = 10_000.0
TRANSACTION_COST = 0.001  # 0.10% per transaction
HOLD_DAYS = 5


def load_signals() -> pd.DataFrame:
    path = Path("reports/strategy_signals.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Strategy file not found at {path}. Run strategy.py first."
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def prepare_execution_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Computes next day Open price per ticker to eliminate look-ahead bias."""
    df = df.copy()
    df["Next_Open"] = df.groupby("Ticker")["Open"].shift(-1)
    return df


def simulate_portfolio(
    df: pd.DataFrame, initial_cash: float = INITIAL_CAPITAL, hold_days: int = HOLD_DAYS
):
    df = prepare_execution_prices(df)
    trade_log = []
    equity_records = []
    cash = initial_cash

    for ticker, group in df.groupby("Ticker"):
        group = group.reset_index(drop=True)

        for i in range(len(group)):
            row = group.iloc[i]

            # Track daily equity entry
            equity_records.append({"Date": row["Date"], "Portfolio": cash})

            # Skip non-actionable signals
            if row["Signal"] == 0:
                continue

            # Ensure valid entry/exit indices and valid prices
            exit_idx = i + hold_days
            if (
                pd.isna(row["Next_Open"])
                or exit_idx >= len(group)
                or pd.isna(group.iloc[exit_idx]["Next_Open"])
            ):
                continue

            entry_date = group.iloc[i + 1]["Date"]
            entry_price = row["Next_Open"]
            exit_date = group.iloc[exit_idx]["Date"]
            exit_price = group.iloc[exit_idx]["Next_Open"]
            position_size = row["Position_Size"]

            # Calculate trade return based on direction (1: BUY, -1: SELL)
            if row["Signal"] == 1:
                gross_return = (exit_price - entry_price) / entry_price
            elif row["Signal"] == -1:
                gross_return = (entry_price - exit_price) / entry_price
            else:
                gross_return = 0.0

            # Apply entry & exit transaction costs
            net_return = gross_return - (2 * TRANSACTION_COST)

            trade_amount = cash * position_size
            profit = trade_amount * net_return
            cash += profit

            trade_log.append(
                {
                    "Ticker": ticker,
                    "Entry_Date": entry_date,
                    "Exit_Date": exit_date,
                    "Signal": row["Signal"],
                    "Confidence": row["Confidence"],
                    "Entry_Price": entry_price,
                    "Exit_Price": exit_price,
                    "Position_Size": position_size,
                    "Return": net_return,
                    "Profit": profit,
                    "Portfolio_Value": cash,
                }
            )

    trades_df = pd.DataFrame(trade_log)

    equity_df = pd.DataFrame(equity_records)
    if not equity_df.empty:
        equity_df = (
            equity_df.groupby("Date")["Portfolio"]
            .mean()
            .reset_index()
            .sort_values("Date")
        )

    return trades_df, equity_df

def calculate_metrics(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    initial_cash: float = 10000.0,
    risk_free_rate: float = 0.02,
) -> dict:
    """Calculates comprehensive performance and risk metrics.

    Note on parameters:
    `trades_df` is expected first, and `equity_df` second.
    """
    portfolio = equity_df["Portfolio"].values
    dates = pd.to_datetime(equity_df["Date"])

    # Total Return & Annualized Return (CAGR)
    total_return = (portfolio[-1] - initial_cash) / initial_cash
    num_days = (dates.iloc[-1] - dates.iloc[0]).days
    years = max(num_days / 365.25, 1e-6)
    cagr = (portfolio[-1] / initial_cash) ** (1 / years) - 1

    # Daily Return Statistics
    daily_returns = np.diff(portfolio) / portfolio[:-1]
    daily_returns = daily_returns[~np.isnan(daily_returns)]

    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_returns = daily_returns - daily_rf

    ann_volatility = (
        daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0.0
    )

    sharpe_ratio = (
        (excess_returns.mean() / daily_returns.std()) * np.sqrt(252)
        if daily_returns.std() > 0
        else 0.0
    )

    # Maximum Drawdown
    running_max = np.maximum.accumulate(portfolio)
    drawdown = (portfolio - running_max) / running_max
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0

    # Trade Log Specifics
    total_trades = len(trades_df) if trades_df is not None else 0
    win_rate = 0.0
    profit_factor = 0.0
    avg_trade_return = 0.0
    avg_win = 0.0
    avg_loss = 0.0

    if trades_df is not None and not trades_df.empty and "Profit" in trades_df.columns:
        wins = trades_df[trades_df["Profit"] > 0]["Profit"]
        losses = trades_df[trades_df["Profit"] < 0]["Profit"]

        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else (gross_profit if gross_profit > 0 else 0.0)
        )

        if "Return" in trades_df.columns:
            avg_trade_return = trades_df["Return"].mean()
            avg_win = trades_df[trades_df["Return"] > 0]["Return"].mean()
            avg_loss = trades_df[trades_df["Return"] < 0]["Return"].mean()

    return {
        "Initial Capital": initial_cash,
        "Final Capital": portfolio[-1],
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": ann_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": max_drawdown,
        "Total Trades": total_trades,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Average Trade Return": avg_trade_return,
        "Average Winning Trade": avg_win,
        "Average Losing Trade": avg_loss,
    }

def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Loading strategy signals...")
    signals_df = load_signals()

    print("Simulating 5-day hold portfolio strategy...")
    trades_df, equity_df = simulate_portfolio(signals_df)

    if trades_df.empty:
        print("\nNo trades were executed based on the current strategy parameters.")
        return

    # Save output reports
    trades_path = reports_dir / "trade_log.csv"
    trades_df.to_csv(trades_path, index=False)
    print(f"Trade log saved to: {trades_path}")

    equity_path = reports_dir / "equity_curve.csv"
    equity_df.to_csv(equity_path, index=False)
    print(f"Equity curve saved to: {equity_path}")

    metrics = calculate_metrics(trades_df, equity_df)
    metrics_df = pd.DataFrame([metrics])
    
    metrics_path = reports_dir / "portfolio_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Portfolio metrics saved to: {metrics_path}")

    print("\n" + "=" * 60)
    print("PORTFOLIO PERFORMANCE SUMMARY")
    print("=" * 60)
    for metric, value in metrics.items():
        if isinstance(value, float):
            if any(k in metric for k in ["Return", "CAGR", "Volatility", "Drawdown", "Rate", "Trade"]):
                print(f"{metric:<25}: {value:.2%}")
            else:
                print(f"{metric:<25}: {value:.4f}")
        else:
            print(f"{metric:<25}: {value}")


if __name__ == "__main__":
    main()