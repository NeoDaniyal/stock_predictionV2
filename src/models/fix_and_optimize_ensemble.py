import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

if os.path.exists("/content/drive"):
    PROJECT_ROOT = Path("/content/drive/MyDrive/ML_Projects/stock_predictionV2")
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT /"reports"


def audit_and_normalize_probabilities(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Checks and normalizes probability columns to sum to 1.0 strictly."""
    prob_cols = ["P_SELL", "P_HOLD", "P_BUY"]
    probs = df[prob_cols].values

    # Check row sums
    row_sums = probs.sum(axis=1)
    is_valid = np.allclose(row_sums, 1.0, atol=1e-5)

    if not is_valid:
        print(f"  ⚠️ [{model_name}] Probabilities do NOT sum to 1.0 (min={row_sums.min():.4f}, max={row_sums.max():.4f}). Normalizing...")
        # Prevent division by zero
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        probs = probs / row_sums[:, np.newaxis]
        df[prob_cols] = probs
    else:
        print(f"  ✅ [{model_name}] Probabilities correctly sum to 1.0.")

    return df


def load_and_audit_all_oofs():
    """Loads all model OOF files, normalizes probabilities, and prints row count breakdown."""
    models = ["rf", "xgboost", "lgbm", "catboost", "lstm"]
    oof_dict = {}

    print("\n================ Step 1: Auditing Probability Integrity ================")
    for m in models:
        path = REPORTS_DIR / f"{m}_oof_predictions.csv"
        if not path.exists():
            print(f"  ❌ Missing file: {path}")
            continue

        df = pd.read_csv(path, parse_dates=["Date"])

        # Standardize column names if legacy names exist
        if "Prob_0" in df.columns:
            df = df.rename(
                columns={
                    "Prob_0": "P_SELL",
                    "Prob_1": "P_HOLD",
                    "Prob_2": "P_BUY",
                    "Pred_Class": "Prediction",
                }
            )

        df = audit_and_normalize_probabilities(df, m)
        if "Year" not in df.columns and "Date" in df.columns:
            df["Year"] = df["Date"].dt.year

        oof_dict[m] = df
        print(f"     -> {m.upper()} OOF Row Count: {len(df):,}")

    return oof_dict


def analyze_sample_mismatch(oof_dict):
    """Investigates why common sample count collapses when combining models."""
    print("\n================ Step 2: Investigating Sample Count Mismatch ================")
    tree_m = "xgboost"
    lstm_m = "lstm"

    if tree_m not in oof_dict or lstm_m not in oof_dict:
        print("  Required models for alignment check not found.")
        return

    xgb_df = oof_dict[tree_m]
    lstm_df = oof_dict[lstm_m]

    xgb_keys = set(zip(xgb_df["Date"].dt.strftime("%Y-%m-%d"), xgb_df["Ticker"]))
    lstm_keys = set(zip(lstm_df["Date"].dt.strftime("%Y-%m-%d"), lstm_df["Ticker"]))

    common = xgb_keys.intersection(lstm_keys)
    only_xgb = xgb_keys - lstm_keys
    only_lstm = lstm_keys - xgb_keys

    print(f"  Tree OOF Total Samples:   {len(xgb_keys):,}")
    print(f"  LSTM OOF Total Samples:   {len(lstm_keys):,}")
    print(f"  Exact Common Intersection: {len(common):,}")
    print(f"  Tree-only Keys:           {len(only_xgb):,}")
    print(f"  LSTM-only Keys:           {len(only_lstm):,}")

    if len(common) < len(xgb_keys) * 0.5:
        print("\n  🔎 Diagnostic Reason for Mismatch:")
        print("     LSTM sequence builder skips the first T=30 trading days PER TICKER.")
        print("     Also check if ticker sets or test years differ in data/deep_learning/ sequence generation.")


def align_datasets(oof_dict):
    """Inner joins all models on Date and Ticker."""
    base_m = "xgboost"
    base_df = oof_dict[base_m][["Date", "Ticker", "Target", "Year"]].copy()

    for m, df in oof_dict.items():
        sub = df[["Date", "Ticker", "P_SELL", "P_HOLD", "P_BUY"]].copy()
        sub.columns = ["Date", "Ticker", f"{m}_P_SELL", f"{m}_P_HOLD", f"{m}_P_BUY"]
        base_df = pd.merge(base_df, sub, on=["Date", "Ticker"], how="inner")

    return base_df


def evaluate_blend(df, weights_dict):
    """Computes metrics for a given weighted combination of models."""
    sell_probs = np.zeros(len(df))
    hold_probs = np.zeros(len(df))
    buy_probs = np.zeros(len(df))

    total_w = sum(weights_dict.values())

    for m, w in weights_dict.items():
        norm_w = w / total_w
        sell_probs += df[f"{m}_P_SELL"].values * norm_w
        hold_probs += df[f"{m}_P_HOLD"].values * norm_w
        buy_probs += df[f"{m}_P_BUY"].values * norm_w

    probs = np.column_stack([sell_probs, hold_probs, buy_probs])
    preds = np.argmax(probs, axis=1)
    y_true = df["Target"].values

    acc = accuracy_score(y_true, preds)
    bal_acc = balanced_accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    ll = log_loss(y_true, probs)

    prec = precision_score(y_true, preds, average=None, zero_division=0)
    rec = recall_score(y_true, preds, average=None, zero_division=0)

    return {
        "Accuracy": acc,
        "Balanced_Accuracy": bal_acc,
        "Macro_F1": macro_f1,
        "Log_Loss": ll,
        "BUY_Precision": prec[2] if len(prec) > 2 else 0.0,
        "BUY_Recall": rec[2] if len(rec) > 2 else 0.0,
        "SELL_Recall": rec[0] if len(rec) > 0 else 0.0,
        "Probs": probs,
        "Preds": preds,
    }


def grid_search_lstm_weight(aligned_df):
    """Grid searches LSTM weight allocation while distributing remainder evenly among tree models."""
    print("\n================ Step 3: Grid Searching LSTM Ensemble Weight ================")
    tree_models = ["rf", "xgboost", "lgbm", "catboost"]

    lstm_weights = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.33]
    results = []

    for w_lstm in lstm_weights:
        w_trees = (1.0 - w_lstm) / len(tree_models) if w_lstm < 1.0 else 0.0
        weights = {m: w_trees for m in tree_models}
        weights["lstm"] = w_lstm

        eval_res = evaluate_blend(aligned_df, weights)
        results.append({
            "LSTM_Weight": f"{w_lstm * 100:.0f}%",
            "Tree_Weight_Each": f"{w_trees * 100:.1f}%",
            "Accuracy": eval_res["Accuracy"],
            "Balanced_Accuracy": eval_res["Balanced_Accuracy"],
            "Macro_F1": eval_res["Macro_F1"],
            "Log_Loss": eval_res["Log_Loss"],
            "BUY_Precision": eval_res["BUY_Precision"],
            "BUY_Recall": eval_res["BUY_Recall"],
            "SELL_Recall": eval_res["SELL_Recall"],
        })

    grid_df = pd.DataFrame(results)
    print(grid_df.to_string(index=False))

    return grid_df


def main():
    oof_dict = load_and_audit_all_oofs()
    analyze_sample_mismatch(oof_dict)

    aligned_df = align_datasets(oof_dict)
    print(f"\nFinal Clean Aligned Samples across all 5 models: {len(aligned_df):,}")

    grid_results = grid_search_lstm_weight(aligned_df)

    # Save outputs
    aligned_df.to_csv(REPORTS_DIR / "aligned_ensemble_oof_probabilities.csv", index=False)
    grid_results.to_csv(REPORTS_DIR / "lstm_weight_grid_search.csv", index=False)
    print(f"\nSaved aligned probability dataset to: {REPORTS_DIR / 'aligned_ensemble_oof_probabilities.csv'}")


if __name__ == "__main__":
    main()