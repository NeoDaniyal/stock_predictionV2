from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
LSTM_OOF_PATH = REPORTS_DIR / "lstm_oof_predictions.csv"


def standardize_and_evaluate_lstm_oof():
    if not LSTM_OOF_PATH.exists():
        raise FileNotFoundError(
            f"OOF predictions file not found at {LSTM_OOF_PATH}. "
            "Please run src/deep_learning/train_lstm.py first."
        )

    df = pd.read_csv(LSTM_OOF_PATH, parse_dates=["Date"])

    # Standardize probabilities and predictions naming
    if "Prob_0" in df.columns:
        df = df.rename(
            columns={
                "Prob_0": "P_SELL",
                "Prob_1": "P_HOLD",
                "Prob_2": "P_BUY",
                "Pred_Class": "Prediction",
            }
        )

    # Ensure required columns are present
    required_cols = [
        "Date",
        "Ticker",
        "Target",
        "P_SELL",
        "P_HOLD",
        "P_BUY",
        "Prediction",
        "Year",
    ]
    df = df[required_cols]

    # Save standardized version back to reports/
    df.to_csv(LSTM_OOF_PATH, index=False)
    print(f"Standardized LSTM OOF predictions saved to: {LSTM_OOF_PATH}")

    # Calculate per-year and overall performance
    years = sorted(df["Year"].unique())
    metrics_list = []

    print(
        "\n===================== LSTM OOF EVALUATION BY YEAR ====================="
    )
    for yr in years:
        sub = df[df["Year"] == yr]
        acc = accuracy_score(sub["Target"], sub["Prediction"])
        bal_acc = balanced_accuracy_score(sub["Target"], sub["Prediction"])
        f1_macro = f1_score(sub["Target"], sub["Prediction"], average="macro")
        ll = log_loss(
            sub["Target"], sub[["P_SELL", "P_HOLD", "P_BUY"]].values
        )

        metrics_list.append(
            {
                "Year": yr,
                "Accuracy": acc,
                "Balanced_Accuracy": bal_acc,
                "Macro_F1": f1_macro,
                "Log_Loss": ll,
            }
        )

    metrics_df = pd.DataFrame(metrics_list)
    print(metrics_df.to_string(index=False))

    # Overall Summary
    overall_acc = accuracy_score(df["Target"], df["Prediction"])
    overall_bal_acc = balanced_accuracy_score(df["Target"], df["Prediction"])
    overall_f1_macro = f1_score(df["Target"], df["Prediction"], average="macro")
    overall_ll = log_loss(
        df["Target"], df[["P_SELL", "P_HOLD", "P_BUY"]].values
    )

    print(
        "\n===================== OVERALL LSTM OOF SUMMARY ====================="
    )
    print(f"Accuracy:          {overall_acc:.4f}")
    print(f"Balanced Accuracy: {overall_bal_acc:.4f}")
    print(f"Macro F1:          {overall_f1_macro:.4f}")
    print(f"Log Loss:          {overall_ll:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            df["Target"],
            df["Prediction"],
            target_names=["SELL", "HOLD", "BUY"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    standardize_and_evaluate_lstm_oof()