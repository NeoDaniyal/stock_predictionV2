import os
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

if os.path.exists("/content/drive"):
    PROJECT_ROOT = Path("/content/drive/MyDrive/ML_Projects/stock_predictionV2")
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 30
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import LSTM architecture
from src.deep_learning.lstm_model import StockLSTM


class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :])
        return logits


def create_sequences_for_df(df, feature_cols, sequence_length=30):
    """Creates 3D sequences (Samples, Seq_Len, Features) per ticker to prevent boundary leakage."""
    X_seq, y_seq, meta_records = [], [], []

    for ticker, group in df.groupby("Ticker"):
        group = group.sort_values("Date").reset_index(drop=True)
        if len(group) <= sequence_length:
            continue

        features = group[feature_cols].values
        targets = group["Target"].values
        dates = group["Date"].values

        for i in range(sequence_length, len(group)):
            X_seq.append(features[i - sequence_length : i])
            y_seq.append(targets[i])
            meta_records.append({
                "Date": dates[i],
                "Ticker": ticker,
                "Target": targets[i],
                "Year": pd.to_datetime(dates[i]).year
            })

    if not X_seq:
        return np.array([]), np.array([]), pd.DataFrame()

    return np.array(X_seq), np.array(y_seq), pd.DataFrame(meta_records)


def train_and_eval_walk_forward():
    # Load dataset
    data_path = DATA_DIR / "features/final_dataset.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Featured dataset not found at {data_path}")

    df = pd.read_csv(data_path, parse_dates=["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    ignore_cols = ["Date", "Ticker", "Target", "Year", "Future_Return", "Target_Threshold"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    tickers = sorted(df["Ticker"].unique())
    print(f"Loaded dataset with {len(df):,} rows across {len(tickers)} tickers: {tickers}")

    all_oof_predictions = []
    walk_forward_years = [2021, 2022, 2023, 2024, 2025, 2026]

    for test_year in walk_forward_years:
        print(f"\n=================== TRAINING LSTM FOR TEST YEAR {test_year} ===================")
        train_df = df[df["Date"].dt.year < test_year].copy()
        test_df = df[df["Date"].dt.year == test_year].copy()

        if train_df.empty or test_df.empty:
            continue

        # Fit Scaler on training features only
        scaler = StandardScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])

        # Create 3D sequences per ticker
        X_train, y_train, _ = create_sequences_for_df(train_df, feature_cols, SEQUENCE_LENGTH)
        X_test, y_test, meta_test = create_sequences_for_df(test_df, feature_cols, SEQUENCE_LENGTH)

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Skipping year {test_year} due to insufficient sequence length.")
            continue

        # DataLoaders
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize model
        model = StockLSTM(input_dim=len(feature_cols)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # Train loop
        model.train()
        for epoch in range(EPOCHS):
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                logits = model(X_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()

        # Inference loop
        model.eval()
        probs_list = []
        with torch.no_grad():
            for X_b, _ in test_loader:
                X_b = X_b.to(DEVICE)
                logits = model(X_b)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)

        probs_all = np.vstack(probs_list)
        preds_all = np.argmax(probs_all, axis=1)

        meta_test["P_SELL"] = probs_all[:, 0]
        meta_test["P_HOLD"] = probs_all[:, 1]
        meta_test["P_BUY"] = probs_all[:, 2]
        meta_test["Prediction"] = preds_all

        acc = accuracy_score(y_test, preds_all)
        bal_acc = balanced_accuracy_score(y_test, preds_all)
        f1 = f1_score(y_test, preds_all, average="macro")

        print(f"Fold {test_year} Samples: {len(meta_test):,} | Acc: {acc:.4f} | BalAcc: {bal_acc:.4f} | F1: {f1:.4f}")
        all_oof_predictions.append(meta_test)

    # Save full OOF prediction CSV
    full_oof_df = pd.concat(all_oof_predictions, ignore_index=True)
    oof_output_path = REPORTS_DIR / "lstm_oof_predictions.csv"
    full_oof_df.to_csv(oof_output_path, index=False)

    print(f"\n✅ Completed full multi-ticker LSTM training!")
    print(f"   Total OOF rows generated: {len(full_oof_df):,}")
    print(f"   Unique tickers covered ({full_oof_df['Ticker'].nunique()}): {sorted(full_oof_df['Ticker'].unique())}")
    print(f"   OOF predictions saved to: {oof_output_path}")


if __name__ == "__main__":
    train_and_eval_walk_forward()