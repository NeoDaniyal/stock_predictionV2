from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def define_feature_subsets(all_features):
    # Model A: Pure Price & Volume
    price_vol_keywords = ["open", "high", "low", "close", "volume", "return_1d", "return_5d", "close_lag", "vol_lag"]
    model_a_cols = [c for c in all_features if any(k in c.lower() for k in price_vol_keywords) and "z_" not in c.lower() and "pct_" not in c.lower()]

    # Model B: Price, Volume + Classic Technicals
    tech_keywords = price_vol_keywords + ["rsi", "macd", "atr", "sma", "ema", "bollinger", "std_14"]
    model_b_cols = [c for c in all_features if any(k in c.lower() for k in tech_keywords) and "market_" not in c.lower() and "kurt" not in c.lower() and "skew" not in c.lower()]

    # Model C: Full Feature Set
    model_c_cols = list(all_features)

    # Model D: Top 20 SHAP Features
    top_20_shap_list = [
        "Market_Vol_20", "Market_Trend_Ratio", "Volatility_30_Pct_252", "Rolling_Kurt_30",
        "Volatility_30_Z_60", "Vol_x_RSI", "Rolling_STD_14", "ATR_14_Pct_252", "ATR_14_Z_60",
        "Rolling_Skew_30", "RSI_14", "MACD_Hist", "Volume_Ratio_20", "SMA_Ratio_50_200",
        "Return_1D", "Return_5D", "Close_lag_1", "Close_lag_5", "ATR_14", "HV_30"
    ]
    model_d_cols = [c for c in top_20_shap_list if c in all_features]

    return {
        "Model_A_Price_Vol": model_a_cols,
        "Model_B_Technicals": model_b_cols,
        "Model_C_Full_69": model_c_cols,
        "Model_D_Top20_SHAP": model_d_cols,
    }


def run_ablation_experiments():
    data_path = DATA_DIR / "processed" / "featured_stock_data.csv"
    df = pd.read_csv(data_path, parse_dates=["Date"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    ignore_cols = ["Date", "Ticker", "Target", "Year", "Future_Return", "Target_Threshold"]
    all_features = [c for c in df.columns if c not in ignore_cols]

    feature_subsets = define_feature_subsets(all_features)
    walk_forward_years = [2021, 2022, 2023, 2024, 2025, 2026]

    print(f"Loaded {len(df):,} rows across {df['Ticker'].nunique()} tickers.")
    for name, cols in feature_subsets.items():
        print(f"  - {name}: {len(cols)} features")

    ablation_summary = []

    for model_name, feature_cols in feature_subsets.items():
        print(f"\n=================== Running Walk-Forward: {model_name} ({len(feature_cols)} features) ===================")
        all_preds, all_probs, all_targets = [], [], []

        for test_year in walk_forward_years:
            train_mask = df["Date"].dt.year < test_year
            test_mask = df["Date"].dt.year == test_year

            X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "Target"]
            X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "Target"]

            if X_train.empty or X_test.empty:
                continue

            xgb = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric="mlogloss",
            )
            xgb.fit(X_train, y_train)

            probs = xgb.predict_proba(X_test)
            probs = np.clip(probs, 1e-12, None)
            probs = probs / probs.sum(axis=1, keepdims=True)

            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_probs.append(probs)
            all_targets.extend(y_test.values)

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_prob = np.vstack(all_probs)

        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        ll = log_loss(y_true, y_prob)

        ablation_summary.append({
            "Feature_Set": model_name,
            "Feature_Count": len(feature_cols),
            "Accuracy": acc,
            "Balanced_Accuracy": bal_acc,
            "Macro_F1": macro_f1,
            "Log_Loss": ll,
        })

    summary_df = pd.DataFrame(ablation_summary)
    print("\n=================== FEATURE ABLATION FINAL RESULTS ===================")
    print(summary_df.to_string(index=False))

    summary_df.to_csv(REPORTS_DIR / "xgboost_feature_ablation_results.csv", index=False)


if __name__ == "__main__":
    run_ablation_experiments()