from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from catboost import CatBoostClassifier
import joblib
import pandas as pd
import numpy as np

# Global artifact containers
model = None
scaler = None
calibrator = None
feature_names = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load production artifacts on server startup."""
    global model, scaler, calibrator, feature_names
    
    # Resolve artifacts path relative to project root
    project_root = Path(__file__).resolve().parent.parent
    artifacts_dir = project_root / "artifacts"
    
    model_path = artifacts_dir / "catboost_model.cbm"
    scaler_path = artifacts_dir / "feature_scaler.pkl"
    calibrator_path = artifacts_dir / "isotonic_calibrator.pkl"

    if not model_path.exists() or not scaler_path.exists():
        raise RuntimeError(f"Missing deployment artifacts in {artifacts_dir}")

    # Load artifacts
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    scaler = joblib.load(scaler_path)
    calibrator = joblib.load(calibrator_path) if calibrator_path.exists() else None
    
    feature_names = model.feature_names_
    print(f"✅ Loaded CatBoost Model ({len(feature_names)} features)")
    yield
    print("🛑 Shutting down API service")

app = FastAPI(
    title="Stock Prediction V2 - Inference Engine",
    description="Production REST API serving CatBoost + Isotonic Calibration trading signals.",
    version="2.0.0",
    lifespan=lifespan
)

class PredictionRequest(BaseModel):
    ticker: str = Field(..., example="AAPL")
    features: dict[str, float] = Field(
        ..., 
        description="Dictionary mapping 69 engineered feature names to numerical values."
    )

class PredictionResponse(BaseModel):
    ticker: str
    signal: str
    confidence: float
    probabilities: dict[str, float]

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_expected": len(feature_names)
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Generate trade signal for a given ticker and feature set."""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model artifacts are not loaded.")

    # 1. Align payload with expected feature set
    input_dict = request.features
    missing_cols = [col for col in feature_names if col not in input_dict]
    if missing_cols:
        # Default missing features to 0.0 or raise error
        for col in missing_cols:
            input_dict[col] = 0.0

    # Ensure correct column ordering
    df_features = pd.DataFrame([input_dict])[feature_names]

    # 2. Scale features
    scaled_features = scaler.transform(df_features)

    # 3. Predict raw class probabilities [SELL (0), HOLD (1), BUY (2)]
    raw_probs = model.predict_proba(scaled_features)[0]
    p_sell, p_hold, p_buy = float(raw_probs[0]), float(raw_probs[1]), float(raw_probs[2])

    # 4. Apply Isotonic Calibration to BUY probability if available
    if calibrator is not None:
        p_buy = float(calibrator.predict([p_buy])[0])
        # Re-normalize probability vector
        total_p = p_sell + p_hold + p_buy
        p_sell, p_hold, p_buy = p_sell / total_p, p_hold / total_p, p_buy / total_p

    # 5. Signal threshold logic with HOLD zone enforcement (0.50 cutoff)
    threshold = 0.50
    if p_buy >= threshold:
        signal = "BUY"
        confidence = p_buy
    elif p_sell >= threshold:
        signal = "SELL"
        confidence = p_sell
    else:
        signal = "HOLD"
        confidence = p_hold

    return PredictionResponse(
        ticker=request.ticker,
        signal=signal,
        confidence=round(confidence, 4),
        probabilities={
            "SELL": round(p_sell, 4),
            "HOLD": round(p_hold, 4),
            "BUY": round(p_buy, 4)
        }
    )