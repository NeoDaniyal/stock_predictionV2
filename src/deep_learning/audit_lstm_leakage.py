import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score

if os.path.exists("/content/drive"):
    PROJECT_ROOT = Path("/content/drive/MyDrive/ML_Projects/stock_predictionV2")
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT /"data"

SEQUENCE_LENGTH = 30
BATCH_SIZE = 64
EPOCH = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
        input_size = input_dim,
        hidden_size = hidden_dim,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :])
        return logits

def audit_pipeline():
    print("=================== 🔍 AUDITING LSTM PIPELINE FOR LEAKAGE ===================")
    data_path = DATA_DIR/"features/final_dataset.csv"
    df = pd.read_csv(data_path, parse_dates=["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    ignore_cols = ["Date", "Ticker", "Target", "Year"]
    features_cols = [c for c in df.columns if c not in ignore_cols]

    print(f"Total Rows: {len(df):,}| Ticker ({df['Ticker'].nunique()}): {sorted(df['Ticker'].unique())}")
    print(f"Feature Count: {len(features_cols)}")

    print("\n--- [CHECK 1 & 4] Sequence & Ticker Isolation Audit ---")
    crossover_found = False
    for ticker, gorup in df.groupby("Ticker"):
        group = group.sort_values("Date").reset_index(drop=True)

        date_diffs = group["Date"].diff().dropna()
        if (date_diffs < pd.Timedelta(day=0)).any():
            print(f"❌ Date ordering error detected in ticker {ticker}!")
            crossover_found = True
    if not crossover_found:
        print("✅ Ticker sequences stricly isolated and chronologically sorted.")


    print("\n--- [CHECK 2] Feature-Target Correlation & Lag Audit ---")
    suspicious_cols = []
    for col in features_cols:
        if "target" in col.lower() or "future" in col.lower() or "lead" in col.lower():
            suspicious_cols.append(col)

    if suspicious_cols:
        print(f"⚠️ Warning: Found potentially future-looking column names in feature set: {suspicious_cols}")
    else:
        print("✅ No target or future-name columns present in feature matrix X.")

    corrs = df[features_cols].apply(lambda x: x.corr(df["Target"]))
    top_corrs = corrs.sort_values(ascending=False).head(5)
    print("Top 5 Feature Correlations with Target:")
    for feat, corr_val in top_corrs.items():
        print(f"  - {feat} :{corr_val: .4f}")

    if top_corrs.iloc[0] > 0.85:
        print("⚠️ HIGH CORRELATION DETECTED: A fature may contain leaked future price data!")
    else:
        print("✅ Feature-target correlations are within normal linear bounds (< 0.85).")

    print("\n--- [CHECK 3 & 5] Scaler Isolation & random Noise Permutation Test ---")
    print("Running Permutation Test: Replacing X with Gussian Random Noise N(0, 1)...")

    test_year = 2024
    train_df = df[df["Date"].dt.year < test_year].copy()
    test_df = df[df["Date"].dt.year == test_year].copy()

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_df[features_cols])
    test_features = scaler.transform(test_df[features_cols])

    train_df[features_cols] = train_features
    test_df[features_cols] = test_features

    X_tr_real, y_tr_real = [], []
    for t, grp in train_df.groupby("Ticker"):
        feats = grp[features_cols].values
        targs = grp["Target"].values
        for i in range(SEQUENCE_LENGTH, len(grp)):
            X_tr_real.append(feats[i - SEQUENCE_LENGTH : i])
            y_tr_real.append(targs[i])

    X_te_real, y_te_real = [], []
    for t, grp in train_df.groupby("Ticker"):
        feats = grp[features_cols].values
        targs = grp["Target"].values
        for i in range(SEQUENCE_LENGTH, len(grp)):
            X_te_real.append(feats[i - SEQUENCE_LENGTH : i])
            y_te_real.append(targs[i])

    X_tr_real, y_tr_real = np.array(X_tr_real), np.array(y_tr_real)
    X_te_real, y_te_real = np.array(X_te_real), np.array(y_te_real)

    X_tr_noise = np.random.normal(size=X_tr_real.shape)
    X_te_noise = np.random.normal(size=X_te_real.shape)

    dataset_noise = TensorDataset(torch.tensor(X_tr_noise, dtype=torch.float32))
    loader_noise = DataLoader(dataset_noise, batch_size=BATCH_SIZE, shuffle=True)

    model_noise = StockLSTM(input_dim=len(features_cols)).to(DEVICE)
    optimizer = torch.optim.Adam(model_noise.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model_noise.train()
    for _ in range(5):
        for Xb, yb in loader_noise:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model_noise(Xb), yb)
            loss.backward()
            optimizer.step()

    model_noise.eval()
    with torch.no_grid():
        logits_noise = model_noise(torch.tensor(X_te_noise, dtype=torch.float32).to(DEVICE))
        pred_noise = torch.argmax(logits_noise, dim=1).cpu().numpy()

    noise_acc = accuracy_score(y_te_real, pred_noise)
    noise_f1 = f1_score(y_te_real, pred_noise, average="macro", zero_division=0)

    print(f"\n  Random Noise Model performance on 2024 Test set: ")
    print(f"\n  ~Accuracy: {noise_acc: .4f} (Expected ~0.33)")
    print(f"\n  ~Macro F1: {noise_f1: .4f} (Expected ~0.33)")

    if noise_acc > 0.45:
        print("❌ CRITICAL LEAKAGE DETECTED: Model scores high even on random noise! Targets or indexing are corrupted.")
    else:
        print("✅ PASS: Random Noise model drops to baseline (~33%). Model architecture does not inherently bleed targets.")

    print("\n--- Target Class Distribution Breakdown ---")
    class_counts = df["Target"].value_counts(normalize=True).sort_index()
    for cls_index, pct in class_counts.items():
        print(f"Class {cls_index} ({'SELL' if cls_index==0 else 'HOLD' if cls_index==1 else 'BUY'}) : {pct * 100: .2f}%")

if __name__ == "__main__":
    audit_pipeline()