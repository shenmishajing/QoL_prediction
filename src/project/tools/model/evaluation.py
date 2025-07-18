"""Model evaluation utilities for QoL prediction models."""

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Import our models
from project.model.llm import QoLPredictionLLM
from project.model.xgboost import QoLXGBoostModel

load_dotenv()

warnings.filterwarnings("ignore")


class ModelEvaluator:
    """Comprehensive model evaluation for QoL prediction models."""

    def __init__(self, test_data_path: str = "data/test_data.csv"):
        """Initialize the evaluator.

        Args:
            test_data_path: Path to test data CSV file
        """
        self.test_data_path = test_data_path
        self.test_data = None
        self.results = {}

    def load_test_data(self) -> pd.DataFrame:
        """Load test data.

        Returns:
            Test data DataFrame
        """
        self.test_data = pd.read_csv(self.test_data_path)
        print(f"Test data loaded: {self.test_data.shape}")
        print(
            f"Test target distribution: {self.test_data['OS benefits'].value_counts()}"
        )
        return self.test_data

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model_name: str = "Model",
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            model_name: Name of the model for reporting

        Returns:
            Dictionary containing all metrics
        """
        # Convert string labels to binary if needed
        if isinstance(y_true[0], str):
            y_true_binary = np.array([1 if label == "Y" else 0 for label in y_true])
        else:
            y_true_binary = y_true

        if isinstance(y_pred[0], str):
            y_pred_binary = np.array([1 if label == "Y" else 0 for label in y_pred])
        else:
            y_pred_binary = y_pred

        # Basic metrics
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, average="binary")
        recall = recall_score(y_true_binary, y_pred_binary, average="binary")
        f1 = f1_score(y_true_binary, y_pred_binary, average="binary")

        # Confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()

        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # Same as recall

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }

        # ROC AUC if probabilities are provided
        if y_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true_binary, y_proba)
                metrics["roc_auc"] = roc_auc
            except Exception as e:
                print(f"Could not calculate ROC AUC for {model_name}: {e}")
                metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = None

        return metrics

    def evaluate_llm_model(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        save_predictions: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate LLM model on test data.

        Args:
            api_key: OpenAI API key
            model_name: OpenAI model to use
            save_predictions: Whether to save predictions to file

        Returns:
            Dictionary containing evaluation results
        """
        if self.test_data is None:
            self.load_test_data()

        print(f"Evaluating LLM model ({model_name})...")

        try:
            # Initialize LLM model
            llm_model = QoLPredictionLLM(
                api_key=api_key, model=model_name, temperature=0.1
            )

            # Make predictions
            llm_results = llm_model.evaluate_on_test_set(
                self.test_data_path,
                output_path="results/llm_predictions.csv" if save_predictions else None,
            )

            # Calculate metrics
            y_true = llm_results["true_labels"]
            y_pred = llm_results["predictions"]

            metrics = self.calculate_metrics(
                y_true, y_pred, model_name=f"LLM ({model_name})"
            )

            # Store results
            self.results["llm"] = {
                "model_info": llm_model.get_model_info(),
                "metrics": metrics,
                "predictions": y_pred,
                "true_labels": y_true,
                "patient_ids": llm_results["patient_ids"],
            }

            print("LLM Evaluation Results:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            if metrics["roc_auc"] is not None:
                print(f"  ROC AUC: {metrics['roc_auc']:.4f}")

            return self.results["llm"]

        except Exception as e:
            print(f"LLM evaluation failed: {e}")
            return {"error": str(e)}

    def evaluate_xgboost_model(
        self,
        model_path: str = "models/xgboost_model.joblib",
        save_predictions: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate XGBoost model on test data.

        Args:
            model_path: Path to saved XGBoost model
            save_predictions: Whether to save predictions to file

        Returns:
            Dictionary containing evaluation results
        """
        if self.test_data is None:
            self.load_test_data()

        print("Evaluating XGBoost model...")

        try:
            # Load model
            xgb_model = QoLXGBoostModel()
            xgb_model.load_model(model_path)

            # Make predictions
            y_pred_binary, y_proba = xgb_model.predict(self.test_data)

            # Convert predictions back to Y/N format
            y_pred = xgb_model.target_encoder.inverse_transform(y_pred_binary)
            y_true = self.test_data["OS benefits"].values

            # Calculate metrics
            metrics = self.calculate_metrics(
                y_true, y_pred, y_proba, model_name="XGBoost"
            )

            # Save predictions if requested
            if save_predictions:
                results_df = self.test_data.copy()
                results_df["xgb_prediction"] = y_pred
                results_df["xgb_probability"] = y_proba

                Path("results").mkdir(exist_ok=True)
                results_df.to_csv("results/xgboost_predictions.csv", index=False)
                print("XGBoost predictions saved to: results/xgboost_predictions.csv")

            # Store results
            self.results["xgboost"] = {
                "model_info": xgb_model.get_model_info(),
                "metrics": metrics,
                "predictions": y_pred.tolist(),
                "probabilities": y_proba.tolist(),
                "true_labels": y_true.tolist(),
                "patient_ids": self.test_data["ID"].tolist(),
            }

            print("XGBoost Evaluation Results:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            if metrics["roc_auc"] is not None:
                print(f"  ROC AUC: {metrics['roc_auc']:.4f}")

            return self.results["xgboost"]

        except Exception as e:
            print(f"XGBoost evaluation failed: {e}")
            return {"error": str(e)}

    def compare_models(self) -> Dict[str, Any]:
        """Compare performance of both models.

        Returns:
            Dictionary containing comparison results
        """
        if not self.results:
            print("No models have been evaluated yet.")
            return {}

        comparison = {
            "models_evaluated": list(self.results.keys()),
            "metrics_comparison": {},
        }

        # Extract metrics for comparison
        for model_name, results in self.results.items():
            if "error" not in results:
                comparison["metrics_comparison"][model_name] = results["metrics"]

        # Create comparison table
        if len(comparison["metrics_comparison"]) > 1:
            print("\n" + "=" * 60)
            print("MODEL COMPARISON")
            print("=" * 60)

            metrics_to_compare = [
                "accuracy",
                "f1_score",
                "precision",
                "recall",
                "roc_auc",
            ]

            for metric in metrics_to_compare:
                print(f"\n{metric.upper().replace('_', ' ')}:")
                for model_name, metrics in comparison["metrics_comparison"].items():
                    value = metrics.get(metric, "N/A")
                    if value != "N/A" and value is not None:
                        print(f"  {model_name:12}: {value:.4f}")
                    else:
                        print(f"  {model_name:12}: {value}")

            # Determine best model for each metric
            best_models = {}
            for metric in metrics_to_compare:
                best_score = -1
                best_model = None
                for model_name, metrics in comparison["metrics_comparison"].items():
                    score = metrics.get(metric)
                    if score is not None and score > best_score:
                        best_score = score
                        best_model = model_name
                if best_model:
                    best_models[metric] = {"model": best_model, "score": best_score}

            comparison["best_models"] = best_models

            print("\nBEST PERFORMING MODELS:")
            for metric, info in best_models.items():
                print(
                    f"  {metric.upper().replace('_', ' ')}: {info['model']} ({info['score']:.4f})"
                )

        return comparison

    def plot_confusion_matrices(
        self, save_path: str = "results/confusion_matrices.png"
    ):
        """Plot confusion matrices for all evaluated models.

        Args:
            save_path: Path to save the plot
        """
        if not self.results:
            print("No models have been evaluated yet.")
            return

        # Count valid models
        valid_models = [
            name for name, results in self.results.items() if "error" not in results
        ]

        if not valid_models:
            print("No valid model results to plot.")
            return

        fig, axes = plt.subplots(
            1, len(valid_models), figsize=(6 * len(valid_models), 5)
        )
        if len(valid_models) == 1:
            axes = [axes]

        for idx, model_name in enumerate(valid_models):
            results = self.results[model_name]
            metrics = results["metrics"]

            # Create confusion matrix
            cm = np.array(
                [
                    [metrics["true_negatives"], metrics["false_positives"]],
                    [metrics["false_negatives"], metrics["true_positives"]],
                ]
            )

            # Plot
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Predicted N", "Predicted Y"],
                yticklabels=["Actual N", "Actual Y"],
                ax=axes[idx],
            )
            axes[idx].set_title(f"{model_name.upper()} Confusion Matrix")

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Confusion matrices saved to: {save_path}")

    def save_results(self, output_path: str = "results/evaluation_results.json"):
        """Save evaluation results to JSON file.

        Args:
            output_path: Path to save results
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        serializable_results = {}
        for model_name, results in self.results.items():
            serializable_results[model_name] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[model_name][key] = value.tolist()
                elif isinstance(value, np.integer):
                    serializable_results[model_name][key] = int(value)
                elif isinstance(value, np.floating):
                    serializable_results[model_name][key] = float(value)
                else:
                    serializable_results[model_name][key] = value

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Evaluation results saved to: {output_path}")


def run_full_evaluation(
    test_data_path: str = "data/test_data.csv",
    xgboost_model_path: str = "models/xgboost_model.joblib",
    openai_api_key: Optional[str] = None,
    llm_model_name: str = "gpt-4o",
) -> Dict[str, Any]:
    """Run complete evaluation of both models.

    Args:
        test_data_path: Path to test data
        xgboost_model_path: Path to trained XGBoost model
        openai_api_key: OpenAI API key for LLM evaluation
        llm_model_name: OpenAI model name to use

    Returns:
        Complete evaluation results
    """
    # Initialize evaluator
    evaluator = ModelEvaluator(test_data_path)
    evaluator.load_test_data()

    # Evaluate XGBoost model
    print("Starting XGBoost evaluation...")
    xgb_results = evaluator.evaluate_xgboost_model(xgboost_model_path)

    # Evaluate LLM model (if API key provided)
    if openai_api_key or os.getenv("OPENAI_API_KEY"):
        print("\nStarting LLM evaluation...")
        llm_results = evaluator.evaluate_llm_model(openai_api_key, llm_model_name)
    else:
        print("\nSkipping LLM evaluation (no API key provided)")

    # Compare models
    print("\nComparing models...")
    comparison = evaluator.compare_models()

    # Generate plots and save results
    evaluator.plot_confusion_matrices()
    evaluator.save_results()

    return {"evaluation_results": evaluator.results, "comparison": comparison}


if __name__ == "__main__":
    # Example usage
    results = run_full_evaluation()
    print("\nEvaluation completed!")
