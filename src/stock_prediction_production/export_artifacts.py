# export_artifacts.py
from pathlib import Path
import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

def main():
    # 1. Create output artifacts directory
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print("📁 Target directory: artifacts/")

    # 2. Load your prepared training/validation dataset
    # (Adjust path to match your preprocessed data location)
    data_path = Path("features/final_dataset.csv") 
    if not data_path.exists():
        print(f"❌ Error: Could not find dataset at {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # Define features and target (adjust target name if needed)
    feature_cols = [col for col in df.columns if col not in ["Date", "Ticker", "Target"]]
    X = df[feature_cols]
    y = df["Target"]

    # 3. Train & Save Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaler_path = artifacts_dir / "feature_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"✅ Saved scaler to {scaler_path}")

    # 4. Train & Save CatBoost Model
    model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, verbose=0)
    model.fit(X_scaled, y)
    
    model_path = artifacts_dir / "catboost_model.cbm"
    model.save_model(model_path)
    print(f"✅ Saved CatBoost model to {model_path}")

    # 5. Fit & Save Isotonic Calibrator
    # Get raw probabilities on training data (or validation fold)
    raw_probs = model.predict_proba(X_scaled)[:, 2]  # Buy probability
    
    calibrator = IsotonicRegression(out_of_bounds="clip")
    # Binary indicator for target class 2 (BUY)
    calibrator.fit(raw_probs, (y == 2).astype(int))
    
    calibrator_path = artifacts_dir / "isotonic_calibrator.pkl"
    joblib.dump(calibrator, calibrator_path)
    print(f"✅ Saved Isotonic calibrator to {calibrator_path}")
    print("\n🎉 All artifacts exported successfully!")

if __name__ == "__main__":
    main()