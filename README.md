# 📈 Stock Prediction V2 // ALPHA-V2 Quant Terminal

An end-to-end quantitative machine learning pipeline and dark-mode trading intelligence terminal. Powered by **CatBoost**, **Isotonic Probability Calibration**, **FastAPI**, and **Streamlit**, containerized with **Docker Compose**.

---

## ⚡ Executive Summary & Key Discovery

During baseline development, initial Deep Learning models (LSTM/GRU/CNN LSTM) achieved deceptively high scores due to **target leakage in temporal feature engineering**.

A complete audit was conducted:

1. **Target Leakage Remediation**: Removed look-ahead bias from rolling technical indicators and forward-looking price returns.
2. **Model Pivot**: Transitioned to a tree-based **CatBoost Classifier** trained across **30,000+ multi-ticker samples** (10 top US equities) using **Walk-Forward Validation (2021–2026)**.
3. **Probability Calibration**: Integrated **Isotonic Regression** to calibrate raw prediction probabilities, enforcing strict **HOLD zones (0.50 threshold)** to prevent over-trading in noisy market conditions.

---

## 🏗️ System Architecture

```text
┌───────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
│  Data & Feature Engine    │ ───► │  Artifacts & Pipeline    │ ───► │   FastAPI Inference API  │
│  (69 Technical Indicators)│      │ (Scaler, CatBoost, Isotonic)   │   (REST Endpoint: :8000) │
└───────────────────────────┘      └──────────────────────────┘      └────────────┬─────────────┘
                                                                                  │
                                                                                  ▼
                                                                     ┌──────────────────────────┐
                                                                     │ Streamlit Quant Terminal │
                                                                     │   (UI Dashboard: :8501)  │
                                                                     └──────────────────────────┘

```

---

## ✨ Features

* **Multi-Ticker Coverage**: Dynamic inference engine supporting `AAPL`, `NVDA`, `TSLA`, `MSFT`, `AMZN`, `GOOGL`, `META`, `AMD`, `JPM`, `NFLX`.
* **69 Engineered Technical Features**: MACD, RSI, Bollinger Bands, ATR, Volatility Ratios, Moving Average Crossovers, and Volume Trend Metrics.
* **Isotonic Probability Calibration**: Converts raw classifier logits into true, calibrated trade confidence probabilities.
* **REST Inference Service**: High-performance FastAPI backend with Pydantic request validation and automated handling for missing feature payloads.
* **Glassmorphic Trading UI**: Streamlit dashboard styled after modern Bloomberg/TradingView terminals with custom CSS, neon signal badges, and Plotly visualizations.
* **One-Command Containerization**: Multi-container setup with Docker Compose.

---

## 📊 Validated Out-of-Fold (OOF) Metrics

Evaluated across leak-free Walk-Forward validation folds:

| Metric | Score | Note |
| --- | --- | --- |
| **Accuracy** | `37.88%` | Realistic performance in 3-class market setup |
| **Balanced Accuracy** | `37.38%` | Uniform handling across Buy, Hold, Sell classes |
| **Macro F1-Score** | `37.16%` | Unbiased metric across imbalanced regimes |
| **Log Loss** | `1.0929` | Evaluated post Isotonic Probability Calibration |

---

## 🛠️ Tech Stack

* **Language**: Python 3.11
* **Machine Learning**: CatBoost, Scikit-learn, Isotonic Regression, Joblib
* **Data Processing**: Pandas, NumPy
* **Backend API**: FastAPI, Uvicorn, Pydantic
* **Frontend UI**: Streamlit, Plotly Express/Objects
* **DevOps & Deployment**: Docker, Docker Compose

---

## 📂 Project Directory Structure

```text
stock_predictionV2/
├── api/
│   └── main.py              # FastAPI REST Inference Engine
├── artifacts/
│   ├── catboost_model.cbm   # Exported CatBoost Model
│   ├── feature_scaler.pkl   # StandardScaler
│   ├── feature_names.pkl    # Feature Column Definitions
│   └── isotonic_calibrator.pkl # Calibration Engine
├── dashboard/
│   └── app.py               # Streamlit Quant Intelligence UI
├── data/
│   └── final_dataset.csv    # Engineered Multi-Ticker Dataset
├── Dockerfile.api           # Docker configuration for FastAPI
├── Dockerfile.dashboard     # Docker configuration for Streamlit
├── docker-compose.yml       # Service Orchestration Configuration
├── export_artifacts.py      # Artifact Export Pipeline Script
├── requirements.txt         # Project Dependencies
└── README.md

```

---

## 🚀 Quickstart Guide

### Option 1: Run with Docker Compose (Recommended)

Ensure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is running:

```powershell
# 1. Clone the repository
git clone https://github.com/your-username/stock_predictionV2.git
cd stock_predictionV2

# 2. Build and launch services in detached mode
docker compose up --build -d

```

Access the interfaces:

* ⚡ **Streamlit Terminal**: [http://localhost:8501](http://localhost:8501)
* ⚙️ **FastAPI Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* 🩺 **API Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

To stop the containers:

```powershell
docker compose down

```

---

### Option 2: Local Python Setup

```powershell
# 1. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Export latest artifacts
python export_artifacts.py

# 4. Start the FastAPI Backend (Terminal 1)
uvicorn api.main:app --reload --port 8000

# 5. Start the Streamlit Dashboard (Terminal 2)
streamlit run dashboard/app.py

```

---

## 📡 API Endpoint Reference

### `POST /predict`

**Request Body:**

```json
{
  "ticker": "AAPL",
  "features": {
    "RSI_14": 58.4,
    "MACD_diff": 0.12
  }
}

```

**Response:**

```json
{
  "ticker": "AAPL",
  "signal": "HOLD",
  "confidence": 0.5421,
  "probabilities": {
    "SELL": 0.1823,
    "HOLD": 0.5421,
    "BUY": 0.2756
  }
}

```

---

## ⚠️ Disclaimer

This system is built for quantitative machine learning demonstration and portfolio presentation purposes only. It does not constitute financial or investment advice.
