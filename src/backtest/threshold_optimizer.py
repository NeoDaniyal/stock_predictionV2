from pathlib import Path
import pandas as pd

from src.backtest.portfolio import calculate_metrics, simulate_portfolio
from src.backtest.strategy import (
    StrategyConfig,
    generate_signals,
    load_oof_predictions,
)


def run_threshold_optimization():
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    results = []

    print("Loading out-of-fold predictions...")
    oof_df = load_oof_predictions()

    print("\nStarting threshold optimization sweep with diagnostic checks...\n")

    for threshold in thresholds:
        # Generate signals for the given threshold
        config = StrategyConfig(buy_threshold=threshold)
        signals_df = generate_signals(oof_df, config=config)

        buy_signal_count = (signals_df["Signal"] == 1).sum()

        # Run portfolio simulation
        trades_df, equity_df = simulate_portfolio(signals_df)
        executed_trades_count = len(trades_df) if not trades_df.empty else 0

        # Print diagnostic logging
        print(
            f"Threshold: {threshold:.2f} | BUY signals generated: {buy_signal_count:>4} | Executed trades: {executed_trades_count:>4}"
        )

        if trades_df.empty or equity_df.empty:
            results.append(
                {
                    "Threshold": threshold,
                    "BUY_Signals": buy_signal_count,
                    "Executed_Trades": 0,
                    "Return": 0.0,
                    "CAGR": 0.0,
                    "Sharpe": 0.0,
                    "Max_DD": 0.0,
                    "Win_Rate": 0.0,
                    "Profit_Factor": 0.0,
                }
            )
            continue

        # Compute performance metrics
        metrics = calculate_metrics(trades_df, equity_df)

        results.append(
            {
                "Threshold": threshold,
                "BUY_Signals": buy_signal_count,
                "Executed_Trades": metrics.get("Total Trades", 0),
                "Return": metrics.get("Total Return", 0.0),
                "CAGR": metrics.get("CAGR", 0.0),
                "Sharpe": metrics.get("Sharpe Ratio", 0.0),
                "Max_DD": metrics.get("Maximum Drawdown", 0.0),
                "Win_Rate": metrics.get("Win Rate", 0.0),
                "Profit_Factor": metrics.get("Profit Factor", 0.0),
            }
        )

    # Save to reports/threshold_results.csv
    results_df = pd.DataFrame(results)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "threshold_results.csv"

    results_df.to_csv(output_path, index=False)
    print(f"\nResults successfully saved to: {output_path}")


if __name__ == "__main__":
    run_threshold_optimization()