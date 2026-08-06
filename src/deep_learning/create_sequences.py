import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import dynamic paths from central config
from configs.config import FEATURE_DATA_PATH, DEEP_LEARNING_PATH

FEATURE_DATASET = FEATURE_DATA_PATH / "final_dataset.csv"
OUTPUT_DIR = DEEP_LEARNING_PATH
SEQUENCE_LENGTH = 30  # Number of time steps in each sequence

META_COLS = ["Date", "Ticker", "Target", "Future_Return", "Target_Threshold"]


def load_and_prep_data():
    """Loads final engineered dataset and clean metadata."""
    print("Loading dataset for Deep Learning sequence creation...")
    df = pd.read_csv(FEATURE_DATASET, parse_dates=["Date"])
    df = df.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)
    df["Year"] = df["Date"].dt.year

    feature_cols = [c for c in df.columns if c not in META_COLS + ["Year"]]
    print(f"Dataset shape: {df.shape}")
    print(f"Features Count: {len(feature_cols)}")
    return df, feature_cols


def create_sequences_by_ticker(df_subset, feature_cols, sequence_length=30):
    """Creates sliding window 3D arrays (samples, sequence_length, features) across all tickers."""
    X_seqs = []
    y_targets = []
    meta_records = []

    for ticker, group in df_subset.groupby("Ticker"):
        group = group.sort_values(by="Date").reset_index(drop=True)

        if len(group) <= sequence_length:
            continue  # Skip tickers with insufficient sequence length

        features_np = group[feature_cols].values
        targets_np = group["Target"].values
        dates_np = group["Date"].values

        for i in range(sequence_length, len(group)):
            X_window = features_np[i - sequence_length : i]
            y_target = targets_np[i]
            date_meta = dates_np[i]

            X_seqs.append(X_window)
            y_targets.append(y_target)
            meta_records.append({
                "Date": date_meta,
                "Ticker": ticker,
                "Target": y_target,
                "Year": pd.to_datetime(date_meta).year,
            })

    # Return after iterating over ALL tickers
    if len(X_seqs) == 0:
        return (
            np.array([]),
            np.array([]),
            pd.DataFrame(columns=["Date", "Ticker", "Target", "Year"]),
        )

    return (
        np.array(X_seqs, dtype=np.float32),
        np.array(y_targets, dtype=np.int64),
        pd.DataFrame(meta_records),
    )


def build_walk_forward_datasets(sequence_length=30):
    """Generate fold-wise scaled train/test 3D sequences for walk-forward evaluation."""
    df, feature_cols = load_and_prep_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_years = [2021, 2022, 2023, 2024, 2025, 2026]

    for test_year in test_years:
        print("=" * 10, f"Processing Fold: Test Year {test_year}", "=" * 10)
        train_df = df[df["Year"] < test_year].copy()
        test_df = df[df["Year"] == test_year].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            print(f"Skipping fold: {test_year} due to insufficient data.")
            continue

        scaler = StandardScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])

        X_train, y_train, meta_train = create_sequences_by_ticker(
            train_df, feature_cols, sequence_length
        )
        X_test, y_test, meta_test = create_sequences_by_ticker(
            test_df, feature_cols, sequence_length
        )

        print(f"Train Sequences: {X_train.shape}, Train Targets: {y_train.shape}")
        print(f"Test Sequences: {X_test.shape}, Test Targets: {y_test.shape}")

        fold_dir = OUTPUT_DIR / f"fold_{test_year}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Save numpy binary arrays for fast PyTorch loading
        np.save(fold_dir / "X_train.npy", X_train)
        np.save(fold_dir / "y_train.npy", y_train)
        np.save(fold_dir / "X_test.npy", X_test)
        np.save(fold_dir / "y_test.npy", y_test)

        meta_train.to_csv(fold_dir / "meta_train.csv", index=False)
        meta_test.to_csv(fold_dir / "meta_test.csv", index=False)
        joblib.dump(scaler, fold_dir / "scaler.pkl")

    # Save feature names correctly
    joblib.dump(feature_cols, OUTPUT_DIR / "feature_names.pkl")
    print(
        f"\nAll walk-forward sequences created and saved successfully to: {OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    build_walk_forward_datasets(sequence_length=SEQUENCE_LENGTH)