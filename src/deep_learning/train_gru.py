from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
import os
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


class StockGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        logits = self.fc(out[:, -1, :])
        return logits


def create_sequences_for_df(df, feature_cols, sequence_length=30):
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


def train_and_eval_gru():
    data_path = DATA_DIR / "features/final_dataset.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Featured dataset not found at {data_path}")

    df = pd.read_csv(data_path, parse_dates=["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # STRICT EXCLUSION: Exclude target metadata and identifiers
    ignore_cols = ["Date", "Ticker", "Target", "Year", "Future_Return", "Target_Threshold"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    print(f"Loaded dataset: {len(df):,} rows across {df['Ticker'].nunique()} tickers.")
    print(f"Using {len(feature_cols)} features for GRU training.")

    all_oof_predictions = []
    walk_forward_years = [2021, 2022, 2023, 2024, 2025, 2026]
    fold_summaries = []

    for test_year in walk_forward_years:
        print(f"\n=================== TRAINING GRU FOR TEST YEAR {test_year} ===================")
        train_df = df[df["Date"].dt.year < test_year].copy()
        test_df = df[df["Date"].dt.year == test_year].copy()

        if train_df.empty or test_df.empty:
            continue

        scaler = StandardScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])

        X_train, y_train, _ = create_sequences_for_df(train_df, feature_cols, SEQUENCE_LENGTH)
        X_test, y_test, meta_test = create_sequences_for_df(test_df, feature_cols, SEQUENCE_LENGTH)

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = StockGRU(input_dim=len(feature_cols)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        model.train()
        for epoch in range(EPOCHS):
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                logits = model(X_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()

        model.eval()
        probs_list = []
        with torch.no_grad():
            for X_b, _ in test_loader:
                X_b = X_b.to(DEVICE)
                logits = model(X_b)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)

        probs_all = np.vstack(probs_list)
        # Strict probability normalization to prevent precision drift
        probs_all = np.clip(probs_all, 1e-12, None)
        probs_all = probs_all / probs_all.sum(axis=1, keepdims=True)

        preds_all = np.argmax(probs_all, axis=1)

        meta_test["P_SELL"] = probs_all[:, 0]
        meta_test["P_HOLD"] = probs_all[:, 1]
        meta_test["P_BUY"] = probs_all[:, 2]
        meta_test["Prediction"] = preds_all

        acc = accuracy_score(y_test, preds_all)
        bal_acc = balanced_accuracy_score(y_test, preds_all)
        f1 = f1_score(y_test, preds_all, average="macro", zero_division=0)
        ll = log_loss(y_test, probs_all)

        print(f"Fold {test_year} Samples: {len(meta_test):,} | Acc: {acc:.4f} | BalAcc: {bal_acc:.4f} | Macro F1: {f1:.4f} | LogLoss: {ll:.4f}")
        
        fold_summaries.append({
            "Test_Year": test_year,
            "Samples": len(meta_test),
            "Accuracy": acc,
            "Balanced_Accuracy": bal_acc,
            "Macro_F1": f1,
            "Log_Loss": ll
        })
        all_oof_predictions.append(meta_test)

    full_oof_df = pd.concat(all_oof_predictions, ignore_index=True)
    oof_output_path = REPORTS_DIR / "gru_oof_predictions.csv"
    full_oof_df.to_csv(oof_output_path, index=False)

    summary_df = pd.DataFrame(fold_summaries)
    print("\n=================== GRU WALK-FORWARD SUMMARY ===================")
    print(summary_df.to_string(index=False))
    print(f"\nAverage Accuracy:        {summary_df['Accuracy'].mean():.4f}")
    print(f"Average Balanced Acc:   {summary_df['Balanced_Accuracy'].mean():.4f}")
    print(f"Average Macro F1:       {summary_df['Macro_F1'].mean():.4f}")
    print(f"Average Log Loss:       {summary_df['Log_Loss'].mean():.4f}")

    print(f"\n✅ GRU training complete! Total OOF rows generated: {len(full_oof_df):,}")
    print(f"   OOF predictions saved to: {oof_output_path}")


if __name__ == "__main__":
    train_and_eval_gru()