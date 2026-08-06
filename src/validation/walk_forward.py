import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def multi_asset_walk_forward_split(
    df: pd.DataFrame,
    start_year: int = 2021,
    test_size_years: int = 1,
) -> list[dict]:
    """Generates expanding-window walk-forward index splits across a multi-asset dataset.

    Ensures strict chronological splits where test years walk forward 1 year at a time
    while the training set expands.

    Args:
        df: Input DataFrame containing a 'Date' column.
        start_year: The first year to be used as the test set.
        test_size_years: Duration of each test fold in years.

    Returns:
        List of dicts containing train/test indices and year boundaries per fold.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year
    min_year = df["Year"].min()
    max_year = df["Year"].max()

    folds = []
    current_test_start = start_year

    while current_test_start <= max_year:
        train_end_year = current_test_start - 1
        test_end_year = current_test_start + test_size_years - 1

        if train_end_year < min_year:
            raise ValueError(
                f"Start year {start_year} leaves no training data prior to min year {min_year}."
            )

        train_mask = df["Year"] <= train_end_year
        test_mask = (df["Year"] >= current_test_start) & (df["Year"] <= test_end_year)

        train_indices = df[train_mask].index.values
        test_indices = df[test_mask].index.values

        if len(train_indices) > 0 and len(test_indices) > 0:
            folds.append(
                {
                    "train_idx": train_indices,
                    "test_idx": test_indices,
                    "train_year_start": int(min_year),
                    "train_year_end": int(train_end_year),
                    "test_year_start": int(current_test_start),
                    "test_year_end": int(min_year if test_end_year > max_year else test_end_year),
                }
            )

        current_test_start += test_size_years

    return folds


def scale_fold_data(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    numeric_features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Fits a StandardScaler solely on the training indices of numeric features and

    transforms both train and test sets to prevent data leakage.

    Args:
        df: Input DataFrame containing feature columns.
        train_idx: Array of integer indices for the training set.
        test_idx: Array of integer indices for the test set.
        numeric_features: List of column names to be scaled.

    Returns:
        tuple (X_train_scaled, X_test_scaled) as NumPy ndarrays.
    """
    scaler = StandardScaler()

    X_train_raw = df.iloc[train_idx][numeric_features].values
    X_test_raw = df.iloc[test_idx][numeric_features].values

    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    return X_train_scaled, X_test_scaled