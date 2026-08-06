import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)

from configs.config import FEATURE_DATA_PATH
from src.validation.walk_forward import (
    multi_asset_walk_forward_split,
    scale_fold_data,
)


def custom_macro_f1_eval(y_true, y_pred):
    """Custom LightGBM evaluation metric for Macro F1 (Scikit-Learn API)."""
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 3)

    preds = np.argmax(y_pred, axis=1)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    return "macro_f1", macro_f1, True


def train_production_lightgbm():
    DATA_PATH = Path(FEATURE_DATA_PATH) / "final_dataset.csv"
    RESULTS_PATH = "results/lightgbm_walk_forward_results.csv"
    MODEL_PATH = "models/lightgbm_best.pkl"
    OOF_PRED_PATH = "reports/lgbm_oof_predictions.csv"

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found at '{DATA_PATH}'")

    # ============================================================
    # LOAD DATA
    # ============================================================
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

    # Strict chronological multi-asset ordering
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # ============================================================
    # AUTO-DETECT TARGET & DYNAMIC FEATURE SELECTION
    # ============================================================
    target_candidates = ["Target", "Target_Class", "Target_Class_3", "target"]
    TARGET = next((col for col in target_candidates if col in df.columns), None)

    if TARGET is None:
        raise KeyError(
            f"Could not find a valid target column! Available columns in CSV: {list(df.columns)}"
        )

    # Clean missing target rows
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)

    # Explicitly exclude non-feature columns
    EXCLUDE_COLS = [
        "Date",
        "Ticker",
        TARGET,
        "Future_Return",
        "Target_Threshold",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Adj Close",
    ]

    NUMERIC_FEATURES = [
        col for col in df.columns
        if col not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[col])
    ]

    FEATURES = NUMERIC_FEATURES

    # ============================================================
    # INFORMATION
    # ============================================================
    print("=" * 60)
    print("LIGHTGBM PRODUCTION PIPELINE (NATIVE CLASS WEIGHTED)")
    print("=" * 60)
    print(f"Data Path            : {DATA_PATH}")
    print(f"Dataset Shape        : {df.shape}")
    print(f"Target Column        : '{TARGET}'")
    print(f"Number of Features   : {len(FEATURES)}")
    print(f"First 10 Features    : {FEATURES[:10]}")
    print("-" * 60)

    print(
        f"Target Distribution:\n"
        f"{df[TARGET].value_counts(normalize=True).sort_index()}"
    )

    # ============================================================
    # WALK-FORWARD SPLITS
    # ============================================================
    folds = multi_asset_walk_forward_split(df, start_year=2021, test_size_years=1)
    print(f"Number of Walk-Forward Folds: {len(folds)}")

    results = []
    oof_records = []

    best_f1 = -np.inf
    best_model = None

    # ============================================================
    # WALK-FORWARD TRAINING
    # ============================================================
    for idx, fold in enumerate(folds):
        print("\n" + "=" * 60)
        print(f"PROCESSING LIGHTGBM FOLD {idx + 1}")
        print("=" * 60)
        print(
            f"Training through: {fold['train_year_end']} | "
            f"Testing: {fold['test_year_start']}"
        )

        # --------------------------------------------------------
        # SCALE NUMERIC FEATURES
        # --------------------------------------------------------
        X_train_num_scaled, X_test_num_scaled = scale_fold_data(
            df,
            fold["train_idx"],
            fold["test_idx"],
            NUMERIC_FEATURES,
        )

        X_train = pd.DataFrame(X_train_num_scaled, columns=NUMERIC_FEATURES)
        X_test = pd.DataFrame(X_test_num_scaled, columns=NUMERIC_FEATURES)

        y_train = df.iloc[fold["train_idx"]][TARGET].values
        y_test = df.iloc[fold["test_idx"]][TARGET].values

        # ========================================================
        # LIGHTGBM MODEL WITH NATIVE CLASS WEIGHTS
        # ========================================================
        model = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            class_weight="balanced",
            n_estimators=1000,
            learning_rate=0.015,
            num_leaves=31,
            max_depth=5,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

        # ========================================================
        # TRAIN
        # ========================================================
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=custom_macro_f1_eval,
            callbacks=[
                early_stopping(stopping_rounds=150, first_metric_only=True, verbose=False),
                log_evaluation(period=100),
            ],
        )

        # ========================================================
        # PREDICTIONS & OUT-OF-FOLD RECORDING
        # ========================================================
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        test_fold_df = df.iloc[fold["test_idx"]].copy()
        test_fold_df["Pred_Class"] = predictions
        test_fold_df["P_SELL"] = probabilities[:, 0]
        test_fold_df["P_HOLD"] = probabilities[:, 1]
        test_fold_df["P_BUY"] = probabilities[:, 2]
        test_fold_df["Fold"] = idx + 1

        oof_records.append(test_fold_df)

        # ========================================================
        # METRICS
        # ========================================================
        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(y_test, predictions, average="macro")
        balanced_acc = balanced_accuracy_score(y_test, predictions)
        logloss = log_loss(y_test, probabilities, labels=[0, 1, 2])

        print(f"\nFold {idx + 1} Results")
        print(f"Accuracy:           {accuracy:.4f}")
        print(f"Balanced Accuracy:  {balanced_acc:.4f}")
        print(f"Macro F1:           {macro_f1:.4f}")
        print(f"Log Loss:           {logloss:.4f}")
        print(f"Best Iteration:     {model.best_iteration_}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                target_names=["SELL(0)", "HOLD(1)", "BUY(2)"],
                zero_division=0,
            )
        )

        # ========================================================
        # SAVE FOLD RESULTS
        # ========================================================
        results.append(
            {
                "fold": idx + 1,
                "train_until": fold["train_year_end"],
                "test_year": fold["test_year_start"],
                "accuracy": accuracy,
                "balanced_accuracy": balanced_acc,
                "macro_f1": macro_f1,
                "log_loss": logloss,
                "best_iteration": model.best_iteration_,
            }
        )

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model = model

    # ============================================================
    # DIRECTORY ENSURANCE & SAVING
    # ============================================================
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    oof_df = pd.concat(oof_records, axis=0).reset_index(drop=True)
    oof_df.to_csv(OOF_PRED_PATH, index=False)

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("LIGHTGBM WALK-FORWARD SUMMARY")
    print("=" * 60)
    print(f"Average Accuracy          : {results_df['accuracy'].mean():.4f}")
    print(f"Average Balanced Accuracy: {results_df['balanced_accuracy'].mean():.4f}")
    print(f"Average Macro F1          : {results_df['macro_f1'].mean():.4f}")
    print(f"Average Log Loss          : {results_df['log_loss'].mean():.4f}")
    print(f"Best Macro F1             : {results_df['macro_f1'].max():.4f}")

    print(f"\nBest model saved to: {MODEL_PATH}")
    print(f"Walk-forward results saved to: {RESULTS_PATH}")
    print(f"OOF Predictions saved to: {OOF_PRED_PATH}")


if __name__ == "__main__":
    train_production_lightgbm()