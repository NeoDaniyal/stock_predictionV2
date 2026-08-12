import json
from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================================
# GLOBAL BACKTEST CONFIGURATION
# =====================================================================
INITIAL_CAPITAL = 10000.0
HOLD_DAYS = 5
TRANSACTION_COST = 0.0010  # 0.10% commission per side
SLIPPAGE = 0.0005          # 0.05% execution slippage
MAX_POSITION_SIZE = 0.20   # Cap per position at 20% of total portfolio equity
MIN_CASH_BUFFER = 0.05     # Maintain 5% uninvested cash buffer


# =====================================================================
# 1. BENCHMARK ENGINE: True Equal-Weighted Buy & Hold
# =====================================================================
def calculate_equal_weight_buy_and_hold(
    df: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL
) -> pd.DataFrame:
    """Calculates true Equal-Weighted Buy & Hold portfolio without sequential 
    compounding or daily rebalancing distortion."""
    df = df.copy().sort_values(["Date", "Ticker"])
    tickers = df["Ticker"].unique()
    n_tickers = len(tickers)
    capital_per_ticker = initial_capital / n_tickers

    first_date = df["Date"].min()
    initial_prices = (
        df[df["Date"] == first_date].set_index("Ticker")["Close"].to_dict()
    )

    daily_portfolio = []
    for date, day_df in df.groupby("Date"):
        portfolio_val = 0.0
        for _, row in day_df.iterrows():
            ticker = row["Ticker"]
            p0 = initial_prices.get(ticker, row["Close"])
            portfolio_val += capital_per_ticker * (row["Close"] / p0)

        daily_portfolio.append({"Date": date, "Portfolio": portfolio_val})

    return pd.DataFrame(daily_portfolio)


# =====================================================================
# 2. PROBABILITY MONOTONICITY DIAGNOSTIC
# =====================================================================
def analyze_probability_monotonicity(
    df: pd.DataFrame, hold_days: int = HOLD_DAYS
) -> pd.DataFrame:
    """Evaluates whether higher predicted P_BUY corresponds to higher future returns."""
    df = df.copy().sort_values(["Ticker", "Date"])
    df["Future_Return"] = (
        df.groupby("Ticker")["Close"].shift(-hold_days) - df["Close"]
    ) / df["Close"]

    bins = [0.0, 0.35, 0.45, 0.55, 0.65, 0.75, 1.00]
    labels = ["<0.35", "0.35-0.44", "0.45-0.54", "0.55-0.64", "0.65-0.74", ">=0.75"]
    df["Prob_Bin"] = pd.cut(df["P_BUY"], bins=bins, labels=labels)

    summary = (
        df.groupby("Prob_Bin", observed=False)
        .agg(
            Samples=("Future_Return", "count"),
            Mean_Realized_Return=("Future_Return", "mean"),
            Win_Rate=("Future_Return", lambda x: (x > 0).mean()),
        )
        .reset_index()
    )
    return summary


# =====================================================================
# 3. CHRONOLOGICAL MULTI-TICKER BACKTEST ENGINE
# =====================================================================
def simulate_chronological_strategy(
    df: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
    initial_cash: float = INITIAL_CAPITAL,
    hold_days: int = HOLD_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulates portfolio execution chronologically across all tickers.
    
    Includes mark-to-market valuation, total equity position sizing, 
    slippage, commissions, and HOLD zone enforcement.
    """
    df = df.copy().sort_values(["Date", "Ticker"]).reset_index(drop=True)
    df["Next_Open"] = df.groupby("Ticker")["Open"].shift(-1)

    dates = sorted(df["Date"].unique())
    cash = initial_cash
    active_positions = []
    trade_log = []
    equity_records = []

    for date in dates:
        day_df = df[df["Date"] == date].set_index("Ticker")

        # A. Process Exits for positions maturing TODAY
        remaining_positions = []
        for pos in active_positions:
            ticker = pos["ticker"]
            if date >= pos["exit_date"]:
                # Exit at Next_Open of current bar with slippage
                if ticker in day_df.index and not pd.isna(day_df.loc[ticker, "Next_Open"]):
                    exit_price = day_df.loc[ticker, "Next_Open"] * (1 - SLIPPAGE)
                else:
                    exit_price = pos["current_price"] * (1 - SLIPPAGE)

                gross_ret = (
                    (exit_price - pos["entry_price"]) / pos["entry_price"]
                    if pos["direction"] == 1
                    else (pos["entry_price"] - exit_price) / pos["entry_price"]
                )
                net_ret = gross_ret - (2 * TRANSACTION_COST)
                profit = pos["allocated_capital"] * net_ret

                cash += pos["allocated_capital"] + profit

                trade_log.append({
                    "Ticker": ticker,
                    "Entry_Date": pos["entry_date"],
                    "Exit_Date": date,
                    "Direction": pos["direction"],
                    "Entry_Price": pos["entry_price"],
                    "Exit_Price": exit_price,
                    "Net_Return": net_ret,
                    "Profit": profit,
                })
            else:
                # Mark to market current price update
                if ticker in day_df.index:
                    pos["current_price"] = day_df.loc[ticker, "Close"]
                remaining_positions.append(pos)

        active_positions = remaining_positions

        # B. Mark-to-Market Portfolio Equity Calculation
        unrealized_equity = 0.0
        for pos in active_positions:
            unrealized_pnl = (
                (pos["current_price"] - pos["entry_price"]) / pos["entry_price"]
                if pos["direction"] == 1
                else (pos["entry_price"] - pos["current_price"]) / pos["entry_price"]
            )
            unrealized_equity += pos["allocated_capital"] * (1 + unrealized_pnl)

        total_portfolio_equity = cash + unrealized_equity
        equity_records.append({"Date": date, "Portfolio": total_portfolio_equity})

        # C. Process Entries with HOLD Zone Enforcement
        currently_held = {p["ticker"] for p in active_positions}

        for ticker, row in day_df.iterrows():
            if ticker in currently_held or pd.isna(row["Next_Open"]):
                continue

            p_buy = row.get("P_BUY", 0.0)
            p_sell = row.get("P_SELL", 0.0)

            # Enforce HOLD Zone
            signal = 0
            if p_buy >= buy_threshold:
                signal = 1
            elif p_sell >= sell_threshold:
                signal = -1

            if signal == 0:
                continue  # Model is not confident; stay in HOLD

            ticker_df = df[df["Ticker"] == ticker].reset_index(drop=True)
            try:
                curr_idx = ticker_df[ticker_df["Date"] == date].index[0]
            except IndexError:
                continue

            exit_idx = curr_idx + hold_days
            if exit_idx >= len(ticker_df) or pd.isna(ticker_df.iloc[exit_idx]["Next_Open"]):
                continue

            entry_price = row["Next_Open"] * (1 + SLIPPAGE)
            exit_date = ticker_df.iloc[exit_idx]["Date"]

            # Position Sizing based on TOTAL PORTFOLIO EQUITY (not leftover cash)
            target_alloc = total_portfolio_equity * MAX_POSITION_SIZE
            allocated_capital = min(target_alloc, cash * (1 - MIN_CASH_BUFFER))

            if allocated_capital < 50.0:  # Minimum cash deployment floor
                continue

            cash -= allocated_capital
            active_positions.append({
                "ticker": ticker,
                "direction": signal,
                "entry_date": date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "current_price": row["Close"],
                "allocated_capital": allocated_capital,
            })

    return pd.DataFrame(trade_log), pd.DataFrame(equity_records)


# =====================================================================
# 4. QUANTITATIVE METRICS EVALUATOR
# =====================================================================
def compute_performance_metrics(
    equity_df: pd.DataFrame, trades_df: pd.DataFrame
) -> dict:
    if equity_df.empty or len(equity_df) < 2:
        return {}

    portfolio = equity_df["Portfolio"].values
    dates = pd.to_datetime(equity_df["Date"])

    total_return = (portfolio[-1] - portfolio[0]) / portfolio[0]
    num_days = max((dates.iloc[-1] - dates.iloc[0]).days, 1)
    years = num_days / 365.25
    cagr = (portfolio[-1] / portfolio[0]) ** (1 / max(years, 1e-6)) - 1

    daily_returns = np.diff(portfolio) / portfolio[:-1]
    ann_vol = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0.0
    sharpe = (
        (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        if daily_returns.std() > 0
        else 0.0
    )

    peak = np.maximum.accumulate(portfolio)
    drawdown = (portfolio - peak) / peak
    max_dd = drawdown.min()

    total_trades = len(trades_df)
    win_rate = (trades_df["Net_Return"] > 0).mean() if total_trades > 0 else 0.0

    return {
        "Final Capital": portfolio[-1],
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Total Trades": total_trades,
        "Win Rate": win_rate,
    }


# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================
def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    oof_path = reports_dir / "oof_predictions.csv"

    if not oof_path.exists():
        print(f"Error: Predictions file not found at {oof_path}.")
        return

    print("Loading out-of-fold predictions...")
    oof_df = pd.read_csv(oof_path, parse_dates=["Date"])

    # 1. Evaluate Probability Monotonicity
    print("\n" + "=" * 60)
    print("PROBABILITY VS REALIZED RETURN MONOTONICITY AUDIT")
    print("=" * 60)
    mono_df = analyze_probability_monotonicity(oof_df)
    print(mono_df.to_string(index=False))

    # 2. Compute Benchmark (Equal-Weighted Buy & Hold)
    print("\n" + "=" * 60)
    print("EQUAL-WEIGHTED BUY & HOLD BENCHMARK")
    print("=" * 60)
    bh_df = calculate_equal_weight_buy_and_hold(oof_df)
    bh_metrics = compute_performance_metrics(bh_df, pd.DataFrame())
    for k, v in bh_metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.2%}" if "Return" in k or "CAGR" in k or "Drawdown" in k else f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")

    # 3. Threshold Sweep Execution
    print("\n" + "=" * 60)
    print("THRESHOLD SWEEP & OPTIMIZATION (HOLD ZONE ENFORCED)")
    print("=" * 60)

    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    sweep_results = []

    for t in thresholds:
        trades_df, equity_df = simulate_chronological_strategy(
            oof_df, buy_threshold=t, sell_threshold=t
        )
        metrics = compute_performance_metrics(equity_df, trades_df)
        metrics["Threshold"] = t
        sweep_results.append(metrics)

        print(
            f"Threshold: {t:.2f} | Trades: {metrics.get('Total Trades', 0):>4} | "
            f"Return: {metrics.get('Total Return', 0.0):>7.2%} | Sharpe: {metrics.get('Sharpe Ratio', 0.0):>6.4f} | "
            f"Max DD: {metrics.get('Max Drawdown', 0.0):>7.2%} | Win Rate: {metrics.get('Win Rate', 0.0):>6.2%}"
        )

    # 4. Export Final Reports
    sweep_df = pd.DataFrame(sweep_results)
    summary_path = reports_dir / "final_backtest_summary.csv"
    sweep_df.to_csv(summary_path, index=False)
    print(f"\nFinal backtest summary saved to: {summary_path}")


if __name__ == "__main__":
    main()