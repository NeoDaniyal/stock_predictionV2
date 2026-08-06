import json
from pathlib import Path
import numpy as np
import pandas as pd


def calculate_metrics(
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame = None,
    initial_cash: float = 10000.0,
    risk_free_rate: float = 0.02,
) -> dict:
    """Computes comprehensive quantitative performance and risk metrics.

    Args:
        equity_curve: DataFrame containing 'Date' and 'Portfolio' columns.
        trade_log: DataFrame containing individual trade logs (optional).
        initial_cash: Initial portfolio capital.
        risk_free_rate: Annualized risk-free rate (default 2.0%).

    Returns:
        dict: Performance and risk metrics dictionary.
    """
    portfolio = equity_curve["Portfolio"].values
    dates = pd.to_datetime(equity_curve["Date"])

    # 1. Basic Return Metrics
    total_return = (portfolio[-1] - initial_cash) / initial_cash

    # Calculate exact trading days and annualized CAGR
    num_days = (dates.iloc[-1] - dates.iloc[0]).days
    years = max(num_days / 365.25, 1e-6)
    cagr = (portfolio[-1] / initial_cash) ** (1 / years) - 1

    # 2. Daily Returns Analysis
    daily_returns = np.diff(portfolio) / portfolio[:-1]
    daily_returns = daily_returns[~np.isnan(daily_returns)]

    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_returns = daily_returns - daily_rf

    # Annualized Volatility
    ann_volatility = (
        daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0.0
    )

    # Sharpe Ratio
    sharpe_ratio = (
        (excess_returns.mean() / daily_returns.std()) * np.sqrt(252)
        if daily_returns.std() > 0
        else 0.0
    )

    # Sortino Ratio (Downside Volatility Risk)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
    sortino_ratio = (
        (excess_returns.mean() / downside_std) * np.sqrt(252)
        if downside_std > 0
        else 0.0
    )

    # 3. Maximum Drawdown & Drawdown Duration
    running_max = np.maximum.accumulate(portfolio)
    drawdown = (portfolio - running_max) / running_max
    max_drawdown = drawdown.min()

    # 4. Trade Log Statistics (if available)
    win_rate = 0.0
    profit_factor = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    total_trades = 0

    if trade_log is not None and not trade_log.empty and "Profit" in trade_log.columns:
        total_trades = len(trade_log)
        wins = trade_log[trade_log["Profit"] > 0]["Profit"]
        losses = trade_log[trade_log["Profit"] < 0]["Profit"]

        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else (gross_profit if gross_profit > 0 else 0.0)
        )
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0

    return {
        "Initial Capital": initial_cash,
        "Final Value": portfolio[-1],
        "Total Return": total_return,
        "CAGR (Annual Return)": cagr,
        "Annualized Volatility": ann_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown": max_drawdown,
        "Total Trades": total_trades,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Average Win ($)": avg_win,
        "Average Loss ($)": avg_loss,
    }


def main():
    reports_dir = Path("reports")
    trades_path = reports_dir / "risk_managed_trade_log.csv"

    if not trades_path.exists():
        print(
            f"Trade log not found at {trades_path}. Run ranking strategy backtest first."
        )
        return

    print(f"Loading backtest results from {trades_path}...")
    trades_df = pd.read_csv(trades_path)

    # Reconstruct portfolio equity curve from trade log profits
    if "Portfolio_Value" in trades_df.columns:
        portfolio_values = trades_df["Portfolio_Value"].values
        dates = trades_df["Exit_Date"].values
    else:
        # Fallback build equity curve
        initial_cash = 10000.0
        trades_df["Cumulative_Profit"] = trades_df["Profit"].cumsum()
        portfolio_values = initial_cash + trades_df["Cumulative_Profit"].values
        dates = trades_df["Exit_Date"].values

    equity_df = pd.DataFrame({"Date": dates, "Portfolio": portfolio_values})

    metrics = calculate_metrics(
        equity_curve=equity_df, trade_log=trades_df, initial_cash=10000.0
    )

    print("\n" + "=" * 60)
    print("QUANTITATIVE STRATEGY PERFORMANCE METRICS")
    print("=" * 60)

    for metric, val in metrics.items():
        if isinstance(val, float):
            if any(
                k in metric
                for k in [
                    "Return",
                    "CAGR",
                    "Volatility",
                    "Drawdown",
                    "Rate",
                ]
            ):
                print(f"{metric:<25}: {val:.2%}")
            elif "Ratio" in metric or "Factor" in metric:
                print(f"{metric:<25}: {val:.4f}")
            else:
                print(f"{metric:<25}: ${val:,.2f}")
        else:
            print(f"{metric:<25}: {val}")

    # Save metrics JSON for tracking
    output_path = reports_dir / "performance_summary.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nPerformance summary saved to: {output_path}")


if __name__ == "__main__":
    main()