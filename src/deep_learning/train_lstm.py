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
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Dynamic Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "deep_learning"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

# Import LSTM architecture
from src.deep_learning.lstm_model import StockLSTM


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def train_one_epoch(
    model, dataloader, criterion, optimizer, device, clip_grad=1.0
):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)

    return running_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    return all_preds, all_probs, all_targets


def train_walk_forward(
    epochs=100,
    batch_size=128,
    lr=1e-3,
    patience=15,
):
    set_seed(42)
    device = get_device()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    test_years = [2021, 2022, 2023, 2024, 2025, 2026]
    walk_forward_metrics = []
    all_oof_preds = []

    for test_year in test_years:
        fold_dir = DATA_DIR / f"fold_{test_year}"
        if not fold_dir.exists():
            print(f"Directory {fold_dir} not found. Skipping fold.")
            continue

        print(
            f"\n============================================================"
        )
        print(f"Training LSTM - Testing Year: {test_year}")
        print(
            f"============================================================"
        )

        # Load fold data
        X_train = np.load(fold_dir / "X_train.npy")
        y_train = np.load(fold_dir / "y_train.npy")
        X_test = np.load(fold_dir / "X_test.npy")
        y_test = np.load(fold_dir / "y_test.npy")
        meta_test = pd.read_csv(fold_dir / "meta_test.csv")

        # Class weights calculation
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=y_train
        )
        class_weights_tensor = torch.tensor(
            class_weights, dtype=torch.float32
        ).to(device)
        print(
            f"Fold Class Weights: {dict(zip(classes, class_weights.round(4)))}"
        )

        # Prepare PyTorch DataLoaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        # Model setup
        input_dim = X_train.shape[2]
        model = StockLSTM(input_dim=input_dim, dropout=0.3).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        # Validation split from training data for early stopping
        val_size = int(len(train_dataset) * 0.15)
        train_sub_size = len(train_dataset) - val_size
        train_sub_ds, val_ds = torch.utils.data.random_split(
            train_dataset, [train_sub_size, val_size]
        )

        train_sub_loader = DataLoader(
            train_sub_ds, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Training loop with Early Stopping
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model, train_sub_loader, criterion, optimizer, device
            )

            # Evaluate on validation split
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_v, y_v in val_loader:
                    X_v, y_v = X_v.to(device), y_v.to(device)
                    v_out = model(X_v)
                    v_loss = criterion(v_out, y_v)
                    val_loss += v_loss.item() * X_v.size(0)
            val_loss /= len(val_loader.dataset)

            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping reached at epoch {epoch}. Best Val Loss: {best_loss:.4f}"
                )
                break

        # Load best weights for out-of-sample testing
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Out-of-Sample Test Evaluation
        test_preds, test_probs, test_targets = evaluate(
            model, test_loader, device
        )

        acc = accuracy_score(test_targets, test_preds)
        bal_acc = balanced_accuracy_score(test_targets, test_preds)
        macro_f1 = f1_score(test_targets, test_preds, average="macro")

        print(f"Accuracy: {acc:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print("\nClassification Report:")
        print(
            classification_report(
                test_targets, test_preds, digits=2, zero_division=0
            )
        )

        walk_forward_metrics.append(
            {
                "Year": test_year,
                "Accuracy": acc,
                "Balanced_Accuracy": bal_acc,
                "Macro_F1": macro_f1,
            }
        )

        # Store test predictions
        meta_test["Pred_Class"] = test_preds
        meta_test["Prob_0"] = test_probs[:, 0]
        meta_test["Prob_1"] = test_probs[:, 1]
        meta_test["Prob_2"] = test_probs[:, 2]
        all_oof_preds.append(meta_test)

        # Save final fold model checkpoint
        torch.save(
            model.state_dict(), MODELS_DIR / f"lstm_model_{test_year}.pt"
        )

    # Save summary metrics & OOF predictions
    wf_df = pd.DataFrame(walk_forward_metrics)
    oof_df = pd.concat(all_oof_preds, ignore_index=True)

    wf_df.to_csv(REPORTS_DIR / "lstm_walk_forward_metrics.csv", index=False)
    oof_df.to_csv(REPORTS_DIR / "lstm_oof_predictions.csv", index=False)

    print(
        "\n============================================================"
    )
    print("LSTM Walk Forward Summary:")
    print(wf_df.to_string(index=False))
    print("\nAverage Metrics Across All Folds:")
    print(wf_df[["Accuracy", "Balanced_Accuracy", "Macro_F1"]].mean())
    print(
        "============================================================"
    )


if __name__ == "__main__":
    train_walk_forward(epochs=100, batch_size=128, lr=1e-3, patience=15)