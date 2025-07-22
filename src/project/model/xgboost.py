import os
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler


class QoLXGBoostModel:
    """XGBoost model for Quality of Life prediction."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        """Initialize XGBoost model with hyperparameters.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of features
            random_state: Random seed for reproducibility
        """
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False,
        )

        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False

    def preprocess_data(
        self, df: pd.DataFrame, fit_encoders: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the data for training/prediction.

        Args:
            df: DataFrame containing the data
            fit_encoders: Whether to fit the encoders (True for training, False for prediction)

        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        # Make a copy to avoid modifying original data
        data = df.copy()

        # Get target column
        target_col = "OS benefits"

        # Separate features and target
        if target_col in data.columns:
            y = data[target_col].copy()
            X_data = data.drop(columns=["ID", target_col])
        else:
            y = None
            X_data = data.drop(columns=["ID"])

        # Store feature columns on first fit
        if fit_encoders:
            self.feature_columns = list(X_data.columns)

        # Handle missing values
        # For numerical columns, fill with median
        numerical_cols = X_data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if fit_encoders:
                self.median_values = getattr(self, "median_values", {})
                self.median_values[col] = X_data[col].median()
            X_data[col] = X_data[col].fillna(self.median_values.get(col, 0))

        # For categorical columns, fill with mode
        categorical_cols = X_data.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if fit_encoders:
                self.mode_values = getattr(self, "mode_values", {})
                mode_val = X_data[col].mode()
                self.mode_values[col] = mode_val[0] if len(mode_val) > 0 else "Unknown"
            X_data[col] = X_data[col].fillna(self.mode_values.get(col, "Unknown"))

        # Encode categorical variables
        for col in categorical_cols:
            if fit_encoders:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X_data[col] = self.label_encoders[col].fit_transform(
                        X_data[col].astype(str)
                    )
                else:
                    X_data[col] = self.label_encoders[col].transform(
                        X_data[col].astype(str)
                    )
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    unique_vals = X_data[col].astype(str).unique()
                    known_vals = self.label_encoders[col].classes_

                    # Replace unseen values with the most common one
                    most_common = self.label_encoders[col].classes_[0]
                    X_data[col] = (
                        X_data[col]
                        .astype(str)
                        .apply(lambda x: x if x in known_vals else most_common)
                    )
                    X_data[col] = self.label_encoders[col].transform(X_data[col])
                else:
                    # If encoder not found, use simple label encoding
                    X_data[col] = pd.Categorical(X_data[col]).codes

        # Convert to numpy array
        X = X_data.values.astype(np.float32)

        # Scale features
        if fit_encoders:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        # Encode target if provided
        if y is not None:
            if fit_encoders:
                self.target_encoder = LabelEncoder()
                y = self.target_encoder.fit_transform(y)
            else:
                y = self.target_encoder.transform(y)

        return X, y

    def evaluate(self, data_path: str = "data/test_data.csv") -> Dict[str, Any]:
        """Evaluate the model on the test data.

        Args:
            data_path: Path to test data CSV
        """
        if not os.path.exists(data_path):
            return {}

        df = pd.read_csv(data_path)
        X, y = self.preprocess_data(df, fit_encoders=False)
        predictions = self.model.predict(X)
        proba = self.model.predict_proba(X)[:, 1]

        results = {
            "auc": roc_auc_score(y, proba),
            "classification_report": classification_report(
                y, predictions, output_dict=True
            ),
            "feature_importance": sorted(
                [
                    [feature, float(importance)]
                    for feature, importance in zip(
                        self.feature_columns, self.model.feature_importances_
                    )
                ],
                key=lambda x: x[1],
                reverse=True,
            ),
        }

        return results

    def train(
        self,
        data_path: str = "data",
    ) -> Dict[str, Any]:
        """Train the XGBoost model.

        Args:
            train_data_path: Path to training data CSV
            validation_split: Proportion of training data for validation
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary containing training results
        """
        # Load training data
        train_df = pd.read_csv(os.path.join(data_path, "train_data.csv"))

        print(f"Training data shape: {train_df.shape}")
        print(f"Target distribution: {train_df['OS benefits'].value_counts()}")

        # Preprocess data
        X, y = self.preprocess_data(train_df, fit_encoders=True)

        print(f"Preprocessed data shape: {X.shape}")
        print(f"Number of features: {len(self.feature_columns)}")

        # Train final model
        self.model.fit(X, y)
        self.is_fitted = True

        return {
            "train_results": self.evaluate(os.path.join(data_path, "train_data.csv")),
            "test_results": self.evaluate(os.path.join(data_path, "test_data.csv")),
        }

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data.

        Args:
            data: DataFrame containing features

        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        X, _ = self.preprocess_data(data, fit_encoders=False)

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        return predictions, probabilities

    def save_model(self, model_path: str = "models/xgboost_model.joblib") -> None:
        """Save the trained model and preprocessors.

        Args:
            model_path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")

        # Create directory if it doesn't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        # Save model and preprocessors
        model_data = {
            "model": self.model,
            "label_encoders": self.label_encoders,
            "scaler": self.scaler,
            "target_encoder": self.target_encoder,
            "feature_columns": self.feature_columns,
            "median_values": getattr(self, "median_values", {}),
            "mode_values": getattr(self, "mode_values", {}),
        }

        joblib.dump(model_data, model_path)
        print(f"Model saved to: {model_path}")

    def load_model(self, model_path: str = "models/xgboost_model.joblib") -> None:
        """Load a trained model and preprocessors.

        Args:
            model_path: Path to the saved model
        """
        model_data = joblib.load(model_path)

        self.model = model_data["model"]
        self.label_encoders = model_data["label_encoders"]
        self.scaler = model_data["scaler"]
        self.target_encoder = model_data["target_encoder"]
        self.feature_columns = model_data["feature_columns"]
        self.median_values = model_data.get("median_values", {})
        self.mode_values = model_data.get("mode_values", {})
        self.is_fitted = True

        print(f"Model loaded from: {model_path}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.

        Returns:
            Dictionary containing model information
        """
        return {
            "model_type": "XGBoost",
            "model_params": self.model.get_params() if self.is_fitted else None,
            "n_features": len(self.feature_columns) if self.feature_columns else 0,
            "feature_columns": self.feature_columns,
            "is_fitted": self.is_fitted,
        }
