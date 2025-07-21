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
    plt.show()


def train_xgboost_model(
    train_data_path: str = "data/train_data.csv",
    model_save_path: str = "models/xgboost_model.joblib",
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
    results = model.train(train_data_path)

    # Save model
    model.save_model(model_save_path)

    # Save training results
    results_path = model_save_path.replace(".joblib", "_results.json")

    # Create a simplified version of results for JSON serialization
    simplified_results = {
        "cv_scores": [float(score) for score in results["cv_scores"]],
        "mean_cv_accuracy": float(results["mean_cv_accuracy"]),
        "cv_std": float(results["cv_std"]),
        "train_accuracy": float(results["train_accuracy"]),
        "train_f1": float(results["train_f1"]),
        "train_auc": float(results["train_auc"]),
        "train_classification_report": results["train_classification_report"],
        "n_features": int(results["n_features"]),
        "feature_columns": results["feature_columns"],
        "top_features": [
            (name, float(importance)) for name, importance in results["top_features"]
        ],
    }

    with open(results_path, "w") as f:
        json.dump(simplified_results, f, indent=2)

    print(f"Training results saved to: {results_path}")

    # Plot top features
    plot_path = model_save_path.replace(".joblib", "_top_features.png")
    plot_top_features(results["top_features"], save_path=plot_path)

    return model


if __name__ == "__main__":
    # Example usage
    model = train_xgboost_model()
    print("XGBoost model training completed!")
