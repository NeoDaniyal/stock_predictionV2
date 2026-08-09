import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

if os.path.exists("/content/drive"):
    PROJECT_ROOT = Path("/content/drive/MyDrive/ML_Projects/stock_predictionV2")
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = PROJECT_ROOT /"deep_learning"


def load_oof_predictions():
    """Loads all model out-of-fold prediction CSV files."""
    models = ["rf", "xgboost", "lgbm", "catboost", "lstm"]
    oof_data = {}

    for m in models:
        path = REPORTS_DIR / f"{m}_oof_predictions.csv"
        if not path.exists():
            print(f"Warning: {path} does not exist. Skipping {m}.")
            continue
        df = pd.read_csv(path, parse_dates=["Date"])
        # Ensure consistent column naming
        if "Prob_0" in df.columns:
            df = df.rename(
                columns={
                    "Prob_0": "P_SELL",
                    "Prob_1": "P_HOLD",
                    "Prob_2": "P_BUY",
                    "Pred_Class": "Prediction",
                }
            )
        oof_data[m] = df

    return oof_data


def align_predictions(oof_data):
    """Aligns OOF predictions from all models on (Date, Ticker)."""
    base_m = "xgboost"
    if base_m not in oof_data:
        base_m = list(oof_data.keys())[0]

    base_df = oof_data[base_m][["Date", "Ticker", "Target", "Year"]].copy()

    for m, df in oof_data.items():
        probs_df = df[["Date", "Ticker", "P_SELL", "P_HOLD", "P_BUY"]].copy()
        probs_df.columns = [
            "Date",
            "Ticker",
            f"{m}_P_SELL",
            f"{m}_P_HOLD",
            f"{m}_P_BUY",
        ]
        base_df = pd.merge(
            base_df, probs_df, on=["Date", "Ticker"], how="inner"
        )

    return base_df


def compute_metrics(y_true, y_pred, y_probs):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    ll = log_loss(y_true, y_probs)

    # Class-level breakdown
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)

    return {
        "Accuracy": acc,
        "Balanced_Accuracy": bal_acc,
        "Macro_F1": macro_f1,
        "Log_Loss": ll,
        "SELL_Precision": prec[0],
        "SELL_Recall": rec[0],
        "HOLD_Precision": prec[1],
        "HOLD_Recall": rec[1],
        "BUY_Precision": prec[2],
        "BUY_Recall": rec[2],
    }


def compare_ensembles():
    oof_data = load_oof_predictions()
    if len(oof_data) < 2:
        print("Not enough OOF prediction files found to run ensemble evaluation.")
        return

    merged_df = align_predictions(oof_data)
    print(f"Aligned {len(merged_df)} common out-of-fold samples for ensemble evaluation.")

    tree_models = [m for m in ["rf", "xgboost", "lgbm", "catboost"] if m in oof_data]
    all_models = [m for m in ["rf", "xgboost", "lgbm", "catboost", "lstm"] if m in oof_data]

    # --- 1. Compute 4-Model Tree Ensemble Probabilities ---
    p_sell_4 = merged_df[[f"{m}_P_SELL" for m in tree_models]].mean(axis=1)
    p_hold_4 = merged_df[[f"{m}_P_HOLD" for m in tree_models]].mean(axis=1)
    p_buy_4 = merged_df[[f"{m}_P_BUY" for m in tree_models]].mean(axis=1)

    probs_4 = np.column_stack([p_sell_4, p_hold_4, p_buy_4])
    preds_4 = np.argmax(probs_4, axis=1)

    # --- 2. Compute 5-Model Heterogeneous Ensemble Probabilities ---
    p_sell_5 = merged_df[[f"{m}_P_SELL" for m in all_models]].mean(axis=1)
    p_hold_5 = merged_df[[f"{m}_P_HOLD" for m in all_models]].mean(axis=1)
    p_buy_5 = merged_df[[f"{m}_P_BUY" for m in all_models]].mean(axis=1)

    probs_5 = np.column_stack([p_sell_5, p_hold_5, p_buy_5])
    preds_5 = np.argmax(probs_5, axis=1)

    y_true = merged_df["Target"].values

    metrics_4 = compute_metrics(y_true, preds_4, probs_4)
    metrics_5 = compute_metrics(y_true, preds_5, probs_5)

    comp_df = pd.DataFrame([metrics_4, metrics_5], index=["4-Model Ensemble (Trees)", "5-Model Ensemble (Trees + LSTM)"])

    print("\n=================== OVERALL ENSEMBLE COMPARISON ===================")
    print(comp_df[["Accuracy", "Balanced_Accuracy", "Macro_F1", "Log_Loss"]].to_string())

    print("\n=================== CLASS-LEVEL PERFORMANCE COMPARISON ===================")
    print(comp_df[["BUY_Precision", "BUY_Recall", "SELL_Precision", "SELL_Recall", "HOLD_Recall"]].to_string())

    # Yearly Breakdown Comparison
    years = sorted(merged_df["Year"].unique())
    yearly_records = []

    for yr in years:
        idx = merged_df["Year"] == yr
        y_yr = y_true[idx]

        p4_yr = probs_4[idx]
        pred4_yr = preds_4[idx]
        m4_yr = compute_metrics(y_yr, pred4_yr, p4_yr)

        p5_yr = probs_5[idx]
        pred5_yr = preds_5[idx]
        m5_yr = compute_metrics(y_yr, pred5_yr, p5_yr)

        yearly_records.append({
            "Year": yr,
            "4M_MacroF1": m4_yr["Macro_F1"],
            "5M_MacroF1": m5_yr["Macro_F1"],
            "4M_BalAcc": m4_yr["Balanced_Accuracy"],
            "5M_BalAcc": m5_yr["Balanced_Accuracy"],
            "4M_BUY_Recall": m4_yr["BUY_Recall"],
            "5M_BUY_Recall": m5_yr["BUY_Recall"],
            "4M_SELL_Recall": m4_yr["SELL_Recall"],
            "5M_SELL_Recall": m5_yr["SELL_Recall"],
        })

    yearly_df = pd.DataFrame(yearly_records)
    print("\n=================== YEAR-BY-YEAR ENSEMBLE BREAKDOWN ===================")
    print(yearly_df.to_string(index=False))

    # Save summary report
    comp_df.to_csv(REPORTS_DIR / "ensemble_comparison_summary.csv")
    yearly_df.to_csv(REPORTS_DIR / "ensemble_yearly_breakdown.csv", index=False)
    print(f"\nEnsemble comparison report saved to: {REPORTS_DIR / 'ensemble_comparison_summary.csv'}")


if __name__ == "__main__":
    compare_ensembles()