import os
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,f1_score,log_loss,precision_recall_fscore_support)
from pathlib import Path

if os.path.exists("/content/drive"):
    PROJECT_ROOT = Path("/content/drive/MyDrive/ML_Projects/stock_predictionV2")
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def normalize_probabilities(prob_array):
    """Clips and normalizes probablities to sum to 1.0 across classes."""
    prob_array = np.clip(prob_array, 1e-12, None)
    return prob_array / prob_array.sum(axis=1, keepdims=True)

def load_and_align_oof_predictions():
    file_map = {
        "Random_Forest": "rf_oof_predictions.csv",
        "XGBoost": "xgb_oof_predictions.csv",
        "LightGBM": "lgbm_oof_predictions.csv",
        "CatBoost": "catboost_oof_predictions.csv",
        "LSTM": "lstm_oof_predictions.csv",
        "GRU": "gru_oof_predictions.csv",
        "CNN_LSTM": "cnn_lstm_oof_predictions.csv",
    }
    loaded_dfs = {}
    for model_name, filename in file_map.items():
        filepath = REPORTS_DIR / filename
        if not filepath.exists():
            print(f"⚠️ Warning: Missing {filepath}. Skipping {model_name}.")
            continue

        df = pd.read_csv(filepath, parse_dates=["Date"])
        df["Year"] = df["Date"].dt.year
        
        # Ensure uniform probability columns
        prob_cols = ["P_SELL", "P_HOLD", "P_BUY"]
        probs = df[prob_cols].values
        norm_probs = normalize_probabilities(probs)
        df["P_SELL"], df["P_HOLD"], df["P_BUY"] = norm_probs[:, 0], norm_probs[:, 1], norm_probs[:, 2]
        df[f"Pred_{model_name}"] = np.argmax(norm_probs, axis=1)

        loaded_dfs[model_name] = df

    if not loaded_dfs:
        raise FileNotFoundError("No OOF prediction files found in reports directory.")

    # Find common keys (Date, Ticker) across ALL models
    common_keys = None
    for name, df in loaded_dfs.items():
        keys = set(zip(df["Date"], df["Ticker"]))
        common_keys = keys if common_keys is None else common_keys.intersection(keys)

    print(f"✅ Successfully aligned across {len(common_keys):,} common samples.")

    # Filter each model to common keys
    aligned_data = {}
    sample_df = list(loaded_dfs.values())[0]
    
    # Create master alignment index
    master_keys = pd.MultiIndex.from_tuples(list(common_keys), names=["Date", "Ticker"])

    for model_name, df in loaded_dfs.items():
        df_indexed = df.set_index(["Date", "Ticker"])
        aligned_df = df_indexed.loc[master_keys].reset_index()
        aligned_data[model_name] = aligned_df

    return aligned_data, len(common_keys)


def generate_final_leaderboard():
    aligned_data, sample_count = load_and_align_oof_predictions()

    leaderboard_rows = []
    class_perf_rows = []
    yearly_rows = []

    for model_name, df in aligned_data.items():
        y_true = df["Target"].values
        prob_matrix = df[["P_SELL", "P_HOLD", "P_BUY"]].values
        prob_matrix = normalize_probabilities(prob_matrix)
        y_pred = np.argmax(prob_matrix, axis=1)

        # Global Metrics
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        ll = log_loss(y_true, prob_matrix)

        # Class-Wise Metrics (0: SELL, 1: HOLD, 2: BUY)
        prec, rec, f1_cls, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

        leaderboard_rows.append({
            "Model": model_name,
            "Accuracy": acc,
            "Balanced_Accuracy": bal_acc,
            "Macro_F1": macro_f1,
            "Log_Loss": ll,
        })

        class_perf_rows.append({
            "Model": model_name,
            "SELL_Prec": prec[0], "SELL_Rec": rec[0],
            "HOLD_Prec": prec[1], "HOLD_Rec": rec[1],
            "BUY_Prec": prec[2],  "BUY_Rec": rec[2],
        })

        # Year-by-Year Breakdown
        df["Year"] = pd.to_datetime(df["Date"]).dt.year
        for year, year_df in df.groupby("Year"):
            y_t_yr = year_df["Target"].values
            y_p_yr = year_df[f"Pred_{model_name}"].values
            yr_f1 = f1_score(y_t_yr, y_p_yr, average="macro", zero_division=0)
            yearly_rows.append({"Model": model_name, "Year": year, "Macro_F1": yr_f1})

    # Convert to DataFrames
    df_leaderboard = pd.DataFrame(leaderboard_rows).sort_values("Macro_F1", ascending=False).reset_index(drop=True)
    df_class_perf = pd.DataFrame(class_perf_rows)
    df_yearly = pd.DataFrame(yearly_rows).pivot(index="Model", columns="Year", values="Macro_F1")

    # Add 4-Model Tree Ensemble Evaluation
    tree_models = ["Random_Forest", "XGBoost", "LightGBM", "CatBoost"]
    available_trees = [m for m in tree_models if m in aligned_data]

    if len(available_trees) > 1:
        ensemble_probs = np.zeros_like(aligned_data[available_trees[0]][["P_SELL", "P_HOLD", "P_BUY"]].values)
        for m in available_trees:
            ensemble_probs += aligned_data[m][["P_SELL", "P_HOLD", "P_BUY"]].values
        ensemble_probs /= len(available_trees)
        ensemble_probs = normalize_probabilities(ensemble_probs)

        y_true_ens = aligned_data[available_trees[0]]["Target"].values
        y_pred_ens = np.argmax(ensemble_probs, axis=1)

        ens_acc = accuracy_score(y_true_ens, y_pred_ens)
        ens_bal = balanced_accuracy_score(y_true_ens, y_pred_ens)
        ens_f1 = f1_score(y_true_ens, y_pred_ens, average="macro", zero_division=0)
        ens_ll = log_loss(y_true_ens, ensemble_probs)

        prec_e, rec_e, _, _ = precision_recall_fscore_support(y_true_ens, y_pred_ens, average=None, zero_division=0)

        df_leaderboard.loc[len(df_leaderboard)] = {
            "Model": "4_Tree_Ensemble",
            "Accuracy": ens_acc,
            "Balanced_Accuracy": ens_bal,
            "Macro_F1": ens_f1,
            "Log_Loss": ens_ll,
        }
        df_class_perf.loc[len(df_class_perf)] = {
            "Model": "4_Tree_Ensemble",
            "SELL_Prec": prec_e[0], "SELL_Rec": rec_e[0],
            "HOLD_Prec": prec_e[1], "HOLD_Rec": rec_e[1],
            "BUY_Prec": prec_e[2],  "BUY_Rec": rec_e[2],
        }

    # Print Results
    print(f"\n=================== FINAL ML MODEL LEADERBOARD ({sample_count:,} Common Samples) ===================")
    print(df_leaderboard.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=================== CLASS-WISE PRECISION & RECALL BREAKDOWN ===================")
    print(df_class_perf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=================== YEAR-BY-YEAR MACRO F1 STABILITY ===================")
    print(df_yearly.to_string(float_format=lambda x: f"{x:.4f}"))

    # Save to Reports
    df_leaderboard.to_csv(REPORTS_DIR / "final_ml_leaderboard.csv", index=False)
    df_class_perf.to_csv(REPORTS_DIR / "final_class_performance.csv", index=False)
    df_yearly.to_csv(REPORTS_DIR / "final_yearly_stability.csv", index=False)


if __name__ == "__main__":
    generate_final_leaderboard()