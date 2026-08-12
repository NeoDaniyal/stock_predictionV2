# export_artifacts.py
from pathlib import Path
import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1] if len(script_dir.parents) > 1 else script_dir
    
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Target directory: {artifacts_dir}")

    # Candidate data paths
    candidate_paths = [
        project_root / "data" /"features" / "final_dataset.csv",
        project_root / "reports" / "processed_data.csv",
        project_root / "data" / "final_dataset.csv",
        project_root / "final_dataset.csv",
    ]

    data_path = next((p for p in candidate_paths if p.exists()), None)

    if data_path is None:
        print("❌ Could not find dataset file.")
        return

    print(f"📊 Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    feature_cols = [col for col in df.columns if col not in ["Date", "Ticker", "Target"]]
    X = df[feature_cols]
    y = df["Target"]

    # 1. Save Feature Column Names
    joblib.dump(feature_cols, artifacts_dir / "feature_names.pkl")
    print(f"✅ Saved feature names ({len(feature_cols)} columns)")

    # 2. Train & Save Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    joblib.dump(scaler, artifacts_dir / "feature_scaler.pkl")
    print(f"✅ Saved scaler")

    # 3. Train & Save CatBoost Model
    model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, verbose=0)
    model.fit(X_scaled_df, y)
    model.save_model(artifacts_dir / "catboost_model.cbm")
    print(f"✅ Saved CatBoost model")

    # 4. Fit & Save Isotonic Calibrator
    raw_probs = model.predict_proba(X_scaled_df)[:, 2]  # Buy probability
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, (y == 2).astype(int))
    joblib.dump(calibrator, artifacts_dir / "isotonic_calibrator.pkl")
    print(f"✅ Saved Isotonic calibrator")

    print("\n🎉 All artifacts exported successfully!")

if __name__ == "__main__":
    main()