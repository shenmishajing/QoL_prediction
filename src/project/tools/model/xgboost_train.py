"""XGBoost model training for QoL prediction."""

import json
import warnings

import matplotlib.pyplot as plt

from project.model.xgboost import QoLXGBoostModel

warnings.filterwarnings("ignore")


def plot_top_features(top_features, save_path=None, title="Top 10 Feature Importances"):
    """Plot a bar chart of the top features' importance."""
    features = [f for f, _ in top_features]
    importances = [float(i) for _, i in top_features]
    plt.figure(figsize=(10, 6))
    bars = plt.barh(features[::-1], importances[::-1], color="skyblue")
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Top features plot saved to: {save_path}")


def train_xgboost_model(
    data_path: str = "data/QOL",
    model_save_path: str = "models/QOL/xgboost_model.joblib",
    **model_params,
) -> QoLXGBoostModel:
    """Train and save an XGBoost model.

    Args:
        train_data_path: Path to training data
        model_save_path: Path to save the trained model
        **model_params: Additional model parameters

    Returns:
        Trained XGBoost model
    """
    # Create model
    model = QoLXGBoostModel(**model_params)

    # Train model
    results = model.train(data_path)

    # Save model
    model.save_model(model_save_path)

    # Save training results
    results_path = model_save_path.replace(".joblib", "_results.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Training results saved to: {results_path}")

    # Plot top features
    for split in ["train", "test"]:
        plot_path = model_save_path.replace(".joblib", f"_{split}_top_features.png")
        plot_top_features(
            results[f"{split}_results"]["feature_importance"][:10], save_path=plot_path
        )

    return model


if __name__ == "__main__":
    # Example usage
    for data_path in ["NoQOL", "QOL"]:
        model = train_xgboost_model(
            data_path=f"data/{data_path}",
            model_save_path=f"work_dirs/{data_path}/xgboost_model.joblib",
        )
        print("XGBoost model {data_path} training completed!")
