from dataclasses import dataclass
from pathlib import Path
import pandas as pd

# Configuration constants
BUY_THRESHOLD = 0.80


@dataclass
class StrategyConfig:
    buy_threshold: float = BUY_THRESHOLD
    confidence_position_sizing: bool = True


def calculate_position_size(confidence: float) -> float:
    if confidence >= 0.90:
        return 1.00
    if confidence >= 0.80:
        return 0.75
    if confidence >= 0.70:
        return 0.50
    return 0.00


def load_oof_predictions() -> pd.DataFrame:
    path = Path("reports") / "oof_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"OOF predictions file not found at {path}. Run train_xgboost.py first."
        )
    return pd.read_csv(path, parse_dates=["Date"])


def generate_signals(
    df: pd.DataFrame, config: StrategyConfig = StrategyConfig()
) -> pd.DataFrame:
    df = df.copy()

    # Calculate max confidence across all class probabilities
    df["Confidence"] = df[["P_BUY", "P_HOLD", "P_SELL"]].max(axis=1)

    # Long-Only Signal Logic: Only BUY if predicted class is 2 (BUY) AND P_BUY >= threshold
    buy_mask = (df["Prediction"] == 2) & (df["P_BUY"] >= config.buy_threshold)

    df["Signal"] = 0
    df.loc[buy_mask, "Signal"] = 1  # BUY signal only

    # Position sizing
    if config.confidence_position_sizing:
        df["Position_Size"] = df["Confidence"].apply(calculate_position_size)
    else:
        df["Position_Size"] = 1.0

    return df


def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating Long-Only Signals (BUY Threshold >= {BUY_THRESHOLD:.2f})...")
    oof_df = load_oof_predictions()

    signals_df = generate_signals(oof_df)

    # Verification summary
    signal_counts = signals_df["Signal"].value_counts()
    print("\nSignal Distribution:")
    print(f"  HOLD (0) : {signal_counts.get(0, 0)}")
    print(f"  BUY  (1) : {signal_counts.get(1, 0)}")
    print(f"  SELL (-1): {signal_counts.get(-1, 0)} (Disabled)")

    output_path = reports_dir / "strategy_signals.csv"
    signals_df.to_csv(output_path, index=False)
    print(f"\nStrategy signals saved to: {output_path}")


if __name__ == "__main__":
    main()