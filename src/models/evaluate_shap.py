import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from configs.config import FEATURE_DATA_PATH, REPORTS_PATH


def load_data_and_model():
    dataset_path = Path(FEATURE_DATA_PATH) / "final_dataset.csv"
    df = pd.read_csv(dataset_path, parse_dates=["Date"])

    # Exclude metadata and target columns
    meta_cols = [
        "Date",
        "Ticker",
        "Target",
        "Future_Return",
        "Target_Threshold",
    ]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Load trained final model
    model_path = Path("models") / "xgboost_v1.pkl"
    import joblib

    model = joblib.load(model_path)

    return df, feature_cols, model


def run_shap_analysis():
    df, feature_cols, model = load_data_and_model()

    print(
        f"Evaluating SHAP for {len(feature_cols)} features on latest test slice..."
    )

    # Use out-of-sample year (e.g., 2024-2025) for realistic SHAP impact
    df["Year"] = df["Date"].dt.year
    sample_mask = df["Year"] >= 2024
    X_sample = df.loc[sample_mask, feature_cols]

    # Subsample if dataset is very large to speed up SHAP matrix computation
    if len(X_sample) > 5000:
        X_sample = X_sample.sample(n=5000, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # Calculate global importance: mean absolute SHAP across classes and samples
    # For multi-class (3 classes), shap_values shape is (samples, features, classes)
    if len(shap_values.shape) == 3:
        mean_abs_shap = np.abs(shap_values.values).mean(axis=(0, 2))
    else:
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    shap_df = pd.DataFrame(
        {"Feature": feature_cols, "Mean_Abs_SHAP": mean_abs_shap}
    ).sort_values(by="Mean_Abs_SHAP", ascending=False)

    print("\n================ TOP 20 SHAP FEATURES ================")
    print(shap_df.head(20).to_string(index=False))

    print("\n============== BOTTOM 15 SHAP FEATURES ==============")
    print(shap_df.tail(15).to_string(index=False))

    # Save summary table
    reports_dir = Path(REPORTS_PATH)
    reports_dir.mkdir(parents=True, exist_ok=True)
    shap_df.to_csv(reports_dir / "shap_importance_summary.csv", index=False)

    # Save candidates for removal (e.g., bottom 20% or lower SHAP threshold)
    low_impact_threshold = np.percentile(shap_df["Mean_Abs_SHAP"], 20)
    prune_candidates = shap_df[
        shap_df["Mean_Abs_SHAP"] <= low_impact_threshold
    ]["Feature"].tolist()

    with open(reports_dir / "prune_candidates.json", "w") as f:
        json.dump(
            {
                "low_impact_threshold": float(low_impact_threshold),
                "candidates": prune_candidates,
            },
            f,
            indent=4,
        )

    print(
        f"\nIdentified {len(prune_candidates)} candidates for pruning saved to reports/prune_candidates.json"
    )

    # Save summary plot
    plt.figure(figsize=(10, 12))
    shap.summary_plot(
        shap_values[:, :, 1]
        if len(shap_values.shape) == 3
        else shap_values,  # Class 1 (HOLD) or multi-class
        X_sample,
        max_display=25,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(reports_dir / "shap_summary_plot.png", dpi=300)
    plt.close()
    print(f"SHAP summary plot saved to {reports_dir / 'shap_summary_plot.png'}")


if __name__ == "__main__":
    run_shap_analysis()