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
)


def load_oof_predictions():
    """Loads out-of-fold predictions for all trained models."""
    paths = {
        "rf": "reports/random_forest/rf_oof_predictions.csv",
        "xgb": "reports/xgb_oof_predictions.csv",
        "lgbm": "reports/lgbm_oof_predictions.csv",
        "catboost": "reports/catboost_oof_predictions.csv",
    }

    dfs = {}
    for name, p in paths.items():
        if Path(p).exists():
            df = pd.read_csv(p)

            # Ensure Pred_Class column exists
            prob_cols = ["P_SELL", "P_HOLD", "P_BUY"]
            if "Pred_Class" not in df.columns and "Prediction" in df.columns:
                df["Pred_Class"] = df["Prediction"]
            elif "Pred_Class" not in df.columns and all(
                c in df.columns for c in prob_cols
            ):
                df["Pred_Class"] = np.argmax(df[prob_cols].values, axis=1)

            # Ensure Fold column exists
            if "Fold" not in df.columns and "Test_Year" in df.columns:
                df["Fold"] = df["Test_Year"]

            dfs[name] = df
            print(f"Loaded {name.upper()} OOF predictions ({len(df)} rows).")
        else:
            print(f"⚠️ Warning: OOF predictions file for '{name}' not found at {p}")

    return dfs


def evaluate_ensembles():
    oof_data = load_oof_predictions()
    if not oof_data:
        raise FileNotFoundError("No OOF prediction files found in 'reports/' folder!")

    print("\n" + "=" * 60)
    print("MULTI-MODEL ENSEMBLE EVALUATION")
    print("=" * 60)

    # Use first available dataset for true labels and index alignment
    base_name = list(oof_data.keys())[0]
    base_df = oof_data[base_name].copy()

    target_col = "Target" if "Target" in base_df.columns else "target"
    y_true = base_df[target_col].values

    prob_cols = ["P_SELL", "P_HOLD", "P_BUY"]

    # Calculate individual model Macro F1 scores for weighting
    model_weights = {}
    print("\n--- Individual Model Performances ---")
    for name, df in oof_data.items():
        preds = df["Pred_Class"].values
        f1 = f1_score(y_true, preds, average="macro")
        acc = accuracy_score(y_true, preds)
        model_weights[name] = f1
        print(f"{name.upper():10s} | Macro F1: {f1:.4f} | Accuracy: {acc:.4f}")

    # Normalize weights
    weight_sum = sum(model_weights.values())
    norm_weights = {k: v / weight_sum for k, v in model_weights.items()}

    # ------------------------------------------------------------
    # 1. UNWEIGHTED SOFT VOTING
    # ------------------------------------------------------------
    unweighted_probs = np.zeros((len(base_df), 3))
    for name, df in oof_data.items():
        unweighted_probs += df[prob_cols].values
    unweighted_probs /= len(oof_data)

    # Normalize probabilities to sum strictly to 1.0 per row to suppress warning
    unweighted_probs /= unweighted_probs.sum(axis=1, keepdims=True)

    soft_preds = np.argmax(unweighted_probs, axis=1)

    print("\n" + "=" * 60)
    print("1. UNWEIGHTED SOFT VOTING ENSEMBLE")
    print("=" * 60)
    print(f"Accuracy:          {accuracy_score(y_true, soft_preds):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, soft_preds):.4f}")
    print(f"Macro F1:          {f1_score(y_true, soft_preds, average='macro'):.4f}")
    print(f"Log Loss:          {log_loss(y_true, unweighted_probs):.4f}")

    # ------------------------------------------------------------
    # 2. WEIGHTED SOFT VOTING (F1 WEIGHTS)
    # ------------------------------------------------------------
    weighted_probs = np.zeros((len(base_df), 3))
    for name, df in oof_data.items():
        weighted_probs += df[prob_cols].values * norm_weights[name]

    # Normalize probabilities to sum strictly to 1.0 per row to suppress warning
    weighted_probs /= weighted_probs.sum(axis=1, keepdims=True)

    weighted_preds = np.argmax(weighted_probs, axis=1)

    print("\n" + "=" * 60)
    print("2. WEIGHTED SOFT VOTING ENSEMBLE (F1 WEIGHTS)")
    print("=" * 60)
    print(f"Accuracy:          {accuracy_score(y_true, weighted_preds):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, weighted_preds):.4f}")
    print(f"Macro F1:          {f1_score(y_true, weighted_preds, average='macro'):.4f}")
    print(f"Log Loss:          {log_loss(y_true, weighted_probs):.4f}")

    print("\nClassification Report (Weighted Soft Voting):")
    print(
        classification_report(
            y_true,
            weighted_preds,
            target_names=["SELL(0)", "HOLD(1)", "BUY(2)"],
            zero_division=0,
        )
    )

    # Save output
    cols_to_keep = [
        c
        for c in ["Date", "Ticker", "Target", "Fold", "Test_Year"]
        if c in base_df.columns
    ]
    ensemble_df = base_df[cols_to_keep].copy()
    ensemble_df["Soft_Pred"] = soft_preds
    ensemble_df["Weighted_Pred"] = weighted_preds
    ensemble_df["P_SELL"] = weighted_probs[:, 0]
    ensemble_df["P_HOLD"] = weighted_probs[:, 1]
    ensemble_df["P_BUY"] = weighted_probs[:, 2]

    os.makedirs("reports", exist_ok=True)
    ensemble_df.to_csv("reports/ensemble_oof_predictions.csv", index=False)
    print("\nSaved ensemble predictions to: reports/ensemble_oof_predictions.csv")


if __name__ == "__main__":
    evaluate_ensembles()