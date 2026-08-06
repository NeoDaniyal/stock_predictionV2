import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
import xgboost as xgb

# Import functions from your existing train_xgboost.py script
from src.models.train_xgboost import generate_folds, get_features, load_data

# Suppress Optuna verbosity to keep console output focused
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 50


def evaluate_walk_forward_optuna(params: dict, folds: list, features: list) -> float:
    """Evaluates XGBoost hyperparameter candidate using your exact walk-forward folds."""
    f1_scores = []

    for _, train_df, test_df in folds:
        X_train, y_train = train_df[features], train_df["Target"]
        X_test, y_test = test_df[features], test_df["Target"]

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro")
        f1_scores.append(f1)

    return float(np.mean(f1_scores)) if f1_scores else 0.0


def objective(trial: optuna.Trial, folds: list, features: list) -> float:
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.05),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    return evaluate_walk_forward_optuna(params, folds, features)


def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data and generating folds from train_xgboost...")
    df = load_data()
    features = get_features()
    folds = generate_folds(df)

    print(
        f"\nStarting Optuna Study ({N_TRIALS} trials) across {len(folds)} walk-forward folds..."
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def print_trial_callback(study, trial):
        print(
            f"Trial {trial.number + 1:>2}/{N_TRIALS} | "
            f"Macro F1: {trial.value:.4f} | "
            f"Best Macro F1: {study.best_value:.4f}"
        )

    study.optimize(
        lambda trial: objective(trial, folds, features),
        n_trials=N_TRIALS,
        callbacks=[print_trial_callback],
    )

    print("\n" + "=" * 60)
    print("OPTUNA OPTIMIZATION COMPLETED")
    print("=" * 60)
    print(f"Best Walk-Forward Macro F1: {study.best_value:.4f}")
    print("\nBest Hyperparameters:")
    for key, val in study.best_params.items():
        print(f"  {key:<20}: {val}")

    # 1. Save best parameters to best_params.json
    best_params_path = reports_dir / "best_params.json"
    best_params_dict = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        **study.best_params,
    }
    with open(best_params_path, "w") as f:
        json.dump(best_params_dict, f, indent=4)
    print(f"\nSaved best parameters to: {best_params_path}")

    # 2. Save optimization history to optuna_history.csv
    history_df = study.trials_dataframe()
    history_path = reports_dir / "optuna_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Saved optimization history to: {history_path}")

    # 3. Save optimization plot to optimization_plot.png
    plt.figure(figsize=(10, 6))
    values = [t.value for t in study.trials if t.value is not None]
    best_values = np.maximum.accumulate(values)

    plt.plot(values, label="Trial Score", alpha=0.6, marker="o", linestyle="")
    plt.plot(best_values, label="Best Score", color="red", linewidth=2)
    plt.title("Optuna Optimization History (Walk-Forward Macro F1)")
    plt.xlabel("Trial Number")
    plt.ylabel("Macro F1 Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plot_path = reports_dir / "optimization_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved optimization history plot to: {plot_path}")


if __name__ == "__main__":
    main()