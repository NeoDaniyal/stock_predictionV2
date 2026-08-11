from pathlib import Path
import os

# ============================
# Project Information
# ============================

TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "AMD",
    "NFLX",
    "JPM",
]

START_DATE = "2014-01-01"
END_DATE = None

# ============================
# Detect Environment
# ============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if os.path.exists("/content/drive"):
    # Google Colab
    STORAGE_ROOT = Path("/content/drive/MyDrive/ML_Projects/stock_predictionV2")
else:
    # Local Machine
    STORAGE_ROOT = PROJECT_ROOT

# ============================
# Paths
# ============================

DATA_DIR = STORAGE_ROOT / "data"

RAW_DATA_PATH = DATA_DIR / "raw"
PROCESSED_DATA_PATH = DATA_DIR / "processed"
FEATURE_DATA_PATH = DATA_DIR / "features"
DEEP_LEARNING_PATH = DATA_DIR / "deep_learning"

MODELS_PATH = STORAGE_ROOT / "models"
REPORTS_PATH = STORAGE_ROOT / "reports"
RESULTS_PATH = STORAGE_ROOT / "results"

SEQUENCE_LENGTH = 30
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")