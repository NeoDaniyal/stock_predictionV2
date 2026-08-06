import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from configs.config import FEATURE_DATA_PATH

MODEL_PATH = Path("models")
PREDICTION_PATH = Path("reports")

# Default baseline hyperparameters (Fallback)
DEFAULT_XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
}


def load_best_params() -> dict:
    """Loads Optuna best parameters from reports/best_params.json if available."""
    best_params_path = PREDICTION_PATH / "best_params.json"
    if best_params_path.exists():
        print(f"Loading Optuna hyperparameter set from: {best_params_path}")
        with open(best_params_path, "r") as f:
            params = json.load(f)
        return params
    else:
        print("Optuna best parameters not found. Using default baseline parameters.")
        return DEFAULT_XGB_PARAMS


def load_data():
    path = Path(FEATURE_DATA_PATH) / "final_dataset.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def get_features(df: pd.DataFrame) -> list:
    """Dynamically drops metadata and target columns to return all predictor features."""
    exclude_cols = [
        "Date",
        "Ticker",
        "Target",
        "Future_Return",
        "Target_Threshold",
        "Test_Year",
        "Fold",
        "Prediction",
        "Confidence",
        "P_SELL",
        "P_HOLD",
        "P_BUY",
    ]
    features = [col for col in df.columns if col not in exclude_cols]
    return features


def generate_folds(df):
    years = sorted(df["Date"].dt.year.unique())
    folds = []

    for test_year in years:
        if test_year < 2021:
            continue

        train = df[df["Date"].dt.year < test_year].copy()
        test = df[df["Date"].dt.year == test_year].copy()

        if train.empty or test.empty:
            continue

        folds.append((test_year, train, test))

    return folds


def build_model():
    params = load_best_params()
    return XGBClassifier(**params)


def train_model(model, X_train, y_train):
    """
    Trains the XGBoost model.
    Sample weights removed to preserve original signal distribution and accuracy.
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates predictions and computes performance metrics alongside confusion matrix.
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_acc = balanced_accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    cm = confusion_matrix(y_test, predictions)

    metrics = {
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_acc,
        "Macro_F1": macro_f1,
        "Confusion_Matrix": cm.tolist(),
    }

    return metrics, predictions, probabilities


def collect_predictions(test_df, predictions, probabilities, test_year):
    """
    Collects genuine out-of-sample predictions and appends max confidence.
    """
    fold_predictions = test_df[
        [
            "Date",
            "Ticker",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Future_Return",
            "Target",
        ]
    ].copy()

    fold_predictions["Prediction"] = predictions
    fold_predictions["Confidence"] = probabilities.max(axis=1)
    fold_predictions["P_SELL"] = probabilities[:, 0]
    fold_predictions["P_HOLD"] = probabilities[:, 1]
    fold_predictions["P_BUY"] = probabilities[:, 2]
    fold_predictions["Test_Year"] = test_year

    return fold_predictions


def collect_importance(model, features, test_year):
    """
    Records feature importance array per fold.
    """
    return pd.DataFrame(
        {
            "Feature": features,
            "Importance": model.feature_importances_,
            "Year": test_year,
        }
    )


def train_and_evaluate():
    df = load_data()
    features = get_features(df)
    print(f"Loaded {len(features)} features for training:")
    print(features)

    folds = generate_folds(df)

    all_predictions = []
    metrics_list = []
    importance_history = []
    last_model = None

    for year, train_df, test_df in folds:
        print("\n" + "=" * 60)
        print(f"Testing Year: {year}")
        print("=" * 60)

        X_train = train_df[features]
        y_train = train_df["Target"]
        X_test = test_df[features]
        y_test = test_df["Target"]

        classes = np.array([0, 1, 2])
        weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=y_train
        )
        print("Fold Class Weights (Informational):", dict(zip(classes, weights)))

        model = build_model()
        model = train_model(model, X_train, y_train)

        eval_metrics, predictions, probabilities = evaluate_model(
            model, X_test, y_test
        )

        print(f"Accuracy: {eval_metrics['Accuracy']:.4f}")
        print(f"Balanced Accuracy: {eval_metrics['Balanced_Accuracy']:.4f}")
        print(f"Macro F1: {eval_metrics['Macro_F1']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, zero_division=0))

        metrics_list.append(
            {
                "Year": year,
                "Accuracy": eval_metrics["Accuracy"],
                "Balanced_Accuracy": eval_metrics["Balanced_Accuracy"],
                "Macro_F1": eval_metrics["Macro_F1"],
            }
        )

        fold_imp = collect_importance(model, features, year)
        importance_history.append(fold_imp)

        fold_pred = collect_predictions(test_df, predictions, probabilities, year)
        all_predictions.append(fold_pred)

        last_model = model

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_list)

    importance_df = pd.concat(importance_history, ignore_index=True)
    avg_importance = (
        importance_df.groupby("Feature")["Importance"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
    )

    return last_model, predictions_df, metrics_df, avg_importance, df, features


def save_reports(predictions_df, metrics_df, avg_importance, df, features):
    PREDICTION_PATH.mkdir(parents=True, exist_ok=True)

    oof_path = PREDICTION_PATH / "xgb_oof_predictions.csv"
    predictions_df.to_csv(oof_path, index=False)
    print(f"\nOOF predictions saved to: {oof_path}")

    metrics_path = PREDICTION_PATH / "xgb_walk_forward_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Walk-forward metrics saved to: {metrics_path}")

    importance_path = PREDICTION_PATH / "xgb_average_feature_importance.csv"
    avg_importance.to_csv(importance_path)
    print(f"Average feature importance saved to: {importance_path}")

    best_params = load_best_params()

    summary_metadata = {
        "model": "XGBoost (Optuna Optimized)",
        "dataset": "final_dataset.csv",
        "features": len(features),
        "hyperparameters": best_params,
        "training_years": f"{df['Date'].dt.year.min()}-{df['Date'].dt.year.max()}",
        "testing_years": f"{metrics_df['Year'].min()}-{metrics_df['Year'].max()}",
        "average_accuracy": round(float(metrics_df["Accuracy"].mean()), 4),
        "average_balanced_accuracy": round(
            float(metrics_df["Balanced_Accuracy"].mean()), 4
        ),
        "average_macro_f1": round(float(metrics_df["Macro_F1"].mean()), 4),
    }

    summary_path = PREDICTION_PATH / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_metadata, f, indent=4)
    print(f"Experiment metadata saved to: {summary_path}")


def save_model(model):
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    path = MODEL_PATH / "xgboost_v1.pkl"
    joblib.dump(model, path)
    print(f"Final model saved to: {path}")


def main():
    model, predictions_df, metrics_df, avg_importance, raw_df, features = (
        train_and_evaluate()
    )

    print("\n" + "=" * 60)
    print("Walk Forward Summary:")
    print(metrics_df)
    print("\nAverage Metrics:")
    print(metrics_df.mean(numeric_only=True))
    print("=" * 60)

    print("\nTop 15 Average Feature Importances (Mean & Std):")
    print(avg_importance.head(15))

    print("\nPrediction dataset shape:", predictions_df.shape)

    save_reports(predictions_df, metrics_df, avg_importance, raw_df, features)
    save_model(model)


if __name__ == "__main__":
    main()