import json
import os

import optuna

from project.tools.model.xgboost_train import train_xgboost_model


# Define the objective function for Optuna
def objective(trial, data_path="QOL"):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
    }
    model = train_xgboost_model(
        data_path=f"data/{data_path}",
        model_save_path=f"work_dirs/{data_path}/tune/trial_{trial.number}/xgboost_model.joblib",
        **params,
    )
    # Save params
    with open(
        f"work_dirs/{data_path}/tune/trial_{trial.number}/xgboost_model_params.json",
        "w",
    ) as f:
        json.dump(params, f, indent=2)

    # Load the results json
    results_path = (
        f"work_dirs/{data_path}/tune/trial_{trial.number}/xgboost_model_results.json"
    )
    with open(results_path, "r") as f:
        results = json.load(f)
    auc = results["test_results"]["auc"]
    return auc


def main():
    # Ensure models directory exists
    for data_path in ["QOL", "NoQOL"]:
        print(f"Tuning {data_path}...")
        study = optuna.create_study(
            direction="maximize", study_name=f"xgboost_{data_path}_auc"
        )
        study.optimize(lambda trial: objective(trial, data_path), n_trials=30)

        print("Best trial:")
        trial = study.best_trial
        print(f"  AUC: {trial.value}")
        print(f"  Params: {trial.params}")
        print(f"  Trial number: {trial.number}")

        # Make a soft link to the best model
        os.symlink(
            f"work_dirs/{data_path}/tune/trial_{trial.number}",
            f"work_dirs/{data_path}/tune/best",
        )


if __name__ == "__main__":
    main()
