from pathlib import Path
import pandas as pd

from src.backtest.portfolio import calculate_metrics
from src.backtest.risk_manager import RiskConfig, RiskManager

# Strategy & Backtest Parameters
TOP_N = 3
HOLD_DAYS = 5
INITIAL_CAPITAL = 10_000.0
TRANSACTION_COST = 0.0010  # 0.10% commission
SLIPPAGE = 0.0005          # 0.05% execution slippage
MIN_P_BUY = 0.50


def load_oof_predictions() -> pd.DataFrame:
    path = Path("reports") / "oof_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"OOF predictions file not found at {path}. Run train_xgboost.py first."
        )
    return pd.read_csv(path, parse_dates=["Date"])


def simulate_risk_managed_ranking(
    df: pd.DataFrame,
    risk_manager: RiskManager,
    top_n: int = TOP_N,
    hold_days: int = HOLD_DAYS,
    initial_cash: float = INITIAL_CAPITAL,
    min_prob: float = MIN_P_BUY,
):
    """
    Simulates top-N ranking strategy with risk-managed position sizing
    and execution at Next_Open with transaction costs and slippage.
    """
    df = df.copy().sort_values(["Date", "Ticker"]).reset_index(drop=True)
    df["Next_Open"] = df.groupby("Ticker")["Open"].shift(-1)

    dates = sorted(df["Date"].unique())
    trade_log = []

    cash = initial_cash
    active_positions = []  # List of dicts tracking open positions
    daily_equity = []

    for date in dates:
        day_df = df[df["Date"] == date].copy()

        # 1. Process Exits for positions maturing TODAY
        remaining_positions = []
        for pos in active_positions:
            if date >= pos["exit_date"]:
                # Realize trade proceeds and profit
                cash += pos["allocated_capital"] + pos["profit"]
            else:
                remaining_positions.append(pos)
        active_positions = remaining_positions

        # 2. Identify new trade candidates
        currently_held_tickers = {p["ticker"] for p in active_positions}
        candidates = day_df[
            (~day_df["Ticker"].isin(currently_held_tickers))
            & (day_df["P_BUY"] >= min_prob)
        ].copy()

        top_candidates = candidates.sort_values("P_BUY", ascending=False).head(top_n)

        # 3. Process Entries
        if not top_candidates.empty and cash > 0:
            weights = risk_manager.calculate_weights(
                candidates=top_candidates, prices_df=df
            )

            # Available capital base for this rebalance round
            capital_base = cash

            for _, row in top_candidates.iterrows():
                ticker = row["Ticker"]
                weight = weights.get(ticker, 0.0)

                if weight <= 0.0:
                    continue

                ticker_df = df[df["Ticker"] == ticker].reset_index(drop=True)
                try:
                    curr_idx = ticker_df[ticker_df["Date"] == date].index[0]
                except IndexError:
                    continue

                exit_idx = curr_idx + hold_days

                if (
                    pd.isna(row["Next_Open"])
                    or exit_idx >= len(ticker_df)
                    or pd.isna(ticker_df.iloc[exit_idx]["Next_Open"])
                ):
                    continue

                entry_date = ticker_df.iloc[curr_idx + 1]["Date"]
                entry_price = row["Next_Open"] * (1 + SLIPPAGE)
                exit_date = ticker_df.iloc[exit_idx]["Date"]
                exit_price = ticker_df.iloc[exit_idx]["Next_Open"] * (1 - SLIPPAGE)

                # Returns & Friction
                gross_return = (exit_price - entry_price) / entry_price
                net_return = gross_return - (2 * TRANSACTION_COST)

                # Size position relative to available cash
                trade_amount = capital_base * weight

                # Ensure we don't deploy more cash than available
                if trade_amount > cash:
                    trade_amount = cash

                if trade_amount <= 0:
                    continue

                profit = trade_amount * net_return

                # Deduct allocated capital immediately upon entry
                cash -= trade_amount

                position_info = {
                    "ticker": ticker,
                    "allocated_capital": trade_amount,
                    "profit": profit,
                    "exit_date": exit_date,
                }
                active_positions.append(position_info)

                trade_log.append(
                    {
                        "Ticker": ticker,
                        "Signal_Date": date,
                        "Entry_Date": entry_date,
                        "Exit_Date": exit_date,
                        "P_BUY": row["P_BUY"],
                        "Weight": weight,
                        "Entry_Price": entry_price,
                        "Exit_Price": exit_price,
                        "Return": net_return,
                        "Profit": profit,
                        "Trade_Amount": trade_amount,
                    }
                )

        # 4. Record Daily Total Portfolio Equity (Cash + Capital tied up in active positions)
        invested_capital = sum(p["allocated_capital"] for p in active_positions)
        total_equity = cash + invested_capital
        daily_equity.append({"Date": date, "Portfolio": total_equity})

    trades_df = pd.DataFrame(trade_log)
    equity_df = pd.DataFrame(daily_equity)

    return trades_df, equity_df


def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Loading predictions for Risk-Managed Ranking Strategy...")
    oof_df = load_oof_predictions()

    # Configure Risk Manager: Confidence Sizing with 35% max per position
    risk_cfg = RiskConfig(
        sizing_method="confidence",
        max_position_size=0.35,
        cash_buffer=0.05
    )
    risk_manager = RiskManager(config=risk_cfg)

    print(
        f"Simulating Strategy (Method: '{risk_cfg.sizing_method}', "
        f"Top-{TOP_N}, Friction: Commission {TRANSACTION_COST:.2%} + Slippage {SLIPPAGE:.2%})..."
    )

    trades_df, equity_df = simulate_risk_managed_ranking(
        oof_df, risk_manager=risk_manager
    )

    if trades_df.empty:
        print("No trades executed.")
        return

    # Save output logs
    trades_path = reports_dir / "risk_managed_trade_log.csv"
    trades_df.to_csv(trades_path, index=False)
    print(f"Trade log saved to: {trades_path}")

    metrics = calculate_metrics(trades_df, equity_df)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(reports_dir / "risk_managed_metrics.csv", index=False)

    print("\n" + "=" * 60)
    print("RISK-MANAGED RANKING PORTFOLIO PERFORMANCE SUMMARY")
    print("=" * 60)
    for metric, value in metrics.items():
        if isinstance(value, float):
            if any(
                k in metric
                for k in [
                    "Return",
                    "CAGR",
                    "Volatility",
                    "Drawdown",
                    "Rate",
                    "Trade",
                ]
            ):
                print(f"{metric:<25}: {value:.2%}")
            else:
                print(f"{metric:<25}: {value:.4f}")
        else:
            print(f"{metric:<25}: {value}")


if __name__ == "__main__":
    main()