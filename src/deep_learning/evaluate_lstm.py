import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, f1_score, log_loss)

if os.path.exists("/content/drive"):
    PROJECT_ROOT = Path("/content/drive/MyDrive/ML_Projects/stock_predictionV2")
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = PROJECT_ROOT/"reports"
LSTM_OOF_PATH = REPORT_DIR/ "lstm_oof_prediction.csv"

def standardize_and_evaluation_lstm_oof():
    if not LSTM_OOF_PATH.exists():
        raise FileNotFoundError(f"OOF prediction file not found at {LSTM_OOF_PATH}."
                                "Please run the src/deep_learning/train_lstm.py first."
                                )
    df = pd.read_csv(LSTM_OOF_PATH, parse_dates=["Date"])

    if "prob_0" in df.columns:
        df = df.rename(
            columns={
                "Prob_0":"P_SELL",
                "Prob_1":"P_HOLD",
                "Prob_2":"P_BUY",
                "Pred_Class": "Prediction"
            }
        )
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

        df.to_csv(LSTM_OOF_PATH, index=False)
        print(f"Standardized LSTM OOF predictions saved to: {LSTM_OOF_PATH}")

        years = sorted(df["Year"].unique())
        metrics_list = []

        print("\n===================== LSTM OOF EVALUATION BY YEAR =====================")
        for yr in years:
            sub = df[df["Year"] == yr]
            acc = accuracy_score(sub["Target"], sub["Prediction"])
            bal_acc = balanced_accuracy_score(sub["Target"], sub["Prediction"])
            f1_macro = f1_score(sub["Target"], sub["Prediction"], average="macro")
            ll = log_loss(sub["Target"], sub[["P_SELL", "P_HOLD", "P_BUY"]].values)

            metrics_list.append({
                "Year": yr,
                "Accuracy": acc,
                "Balanced_Accuracy": bal_acc,
                "Macro_F1": f1_macro,
                "Log_Loss": ll
            })
        metrics_df = pd.DataFrame(metrics_list)
        print(metrics_df.to_string(index=False))

        overall_acc = accuracy_score(df["Target"], df["Prediction"])
        overall_bal_acc = balanced_accuracy_score(df["Target"], df["Prediction"])
        overall_f1_macro = f1_score(df["Target"], df["Prediction"], average="macro")
        overall_ll = log_loss(sub["Target"], sub[["P_SELL", "P_HOLD", "P_BUY"]].values)

        print("\n===================== OVERALL LSTM OOF SUMMARY =====================")
        print(f"Accuracy: {overall_acc:.4f}")
        print(f"Balance Accuracy: {overall_bal_acc:.4f}")
        print(f"Macro F1: {overall_f1_macro:.4f}")
        print(f"Log Loss: {overall_ll:.4f}")
        print("\nClassification Report:")
        print(
            classification_report(df["Target"], df["Prediction"], target_names=["SELL", "HOLD", "BUY"], zero_division=0)
        )

if __name__ == "__main__":
    standardize_and_evaluation_lstm_oof()
        