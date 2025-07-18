"""Unit tests for the XGBoost training module."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.project.tools.model.xgboost_train import QoLXGBoostModel, train_xgboost_model


class TestQoLXGBoostModel:
    """Test cases for QoLXGBoostModel class."""

    @pytest.fixture
    def sample_train_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 50

        data = {
            "ID": range(1, n_samples + 1),
            "OS benefits": np.random.choice(["Y", "N"], n_samples, p=[0.6, 0.4]),
            "Drug Class": np.random.choice(
                ["Targeted therapy", "Immunotherapy"], n_samples
            ),
            "Approval Year": np.random.choice([2018, 2019, 2020], n_samples),
            "Active Arm PFS": np.random.uniform(1, 20, n_samples),
            "Control Arm PFS": np.random.uniform(1, 15, n_samples),
            "PFS p-value": np.random.uniform(0.001, 0.1, n_samples),
            "Active Arm ORR": np.random.uniform(10, 90, n_samples),
            "Control Arm ORR": np.random.uniform(5, 80, n_samples),
            "QoL benefits": np.random.choice(["Y", "N"], n_samples),
        }

        return pd.DataFrame(data)

    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = QoLXGBoostModel()

        assert model.model is not None
        assert isinstance(model.label_encoders, dict)
        assert model.scaler is not None
        assert isinstance(model.feature_columns, list)
        assert model.is_fitted == False

    def test_model_initialization_custom_params(self):
        """Test model initialization with custom parameters."""
        model = QoLXGBoostModel(
            n_estimators=200, max_depth=8, learning_rate=0.05, random_state=123
        )

        params = model.model.get_params()
        assert params["n_estimators"] == 200
        assert params["max_depth"] == 8
        assert params["learning_rate"] == 0.05
        assert params["random_state"] == 123

    def test_preprocess_data_training(self, sample_train_data):
        """Test data preprocessing for training."""
        model = QoLXGBoostModel()

        X, y = model.preprocess_data(sample_train_data, fit_encoders=True)

        # Check shapes
        assert X.shape[0] == len(sample_train_data)
        assert y.shape[0] == len(sample_train_data)

        # Check feature columns were stored
        assert len(model.feature_columns) > 0
        assert "ID" not in model.feature_columns
        assert "OS benefits" not in model.feature_columns

        # Check encoders were fitted
        assert len(model.label_encoders) > 0
        assert hasattr(model, "target_encoder")

    def test_preprocess_data_prediction(self, sample_train_data):
        """Test data preprocessing for prediction."""
        model = QoLXGBoostModel()

        # First fit the encoders
        X_train, y_train = model.preprocess_data(sample_train_data, fit_encoders=True)

        # Create test data (subset of training data)
        test_data = sample_train_data.head(10).drop(columns=["OS benefits"])

        # Test preprocessing for prediction
        X_test, y_test = model.preprocess_data(test_data, fit_encoders=False)

        assert X_test.shape[0] == 10
        assert y_test is None  # No target column in test data
        assert X_test.shape[1] == X_train.shape[1]  # Same number of features

    def test_preprocess_data_missing_values(self):
        """Test handling of missing values during preprocessing."""
        # Create data with missing values
        data = {
            "ID": [1, 2, 3, 4],
            "OS benefits": ["Y", "N", "Y", "N"],
            "Numerical_Feature": [10.0, np.nan, 30.0, 40.0],
            "Categorical_Feature": ["A", "B", None, "A"],
        }
        df = pd.DataFrame(data)

        model = QoLXGBoostModel()
        X, y = model.preprocess_data(df, fit_encoders=True)

        # Check that no NaN values remain
        assert not np.isnan(X).any()

        # Check that median and mode values were stored
        assert hasattr(model, "median_values")
        assert hasattr(model, "mode_values")

    def test_training_basic(self, sample_train_data):
        """Test basic model training functionality."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_train_data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            model = QoLXGBoostModel()
            results = model.train(train_data_path=temp_path, cv_folds=3)

            # Check that model is fitted
            assert model.is_fitted == True

            # Check results structure
            assert isinstance(results, dict)
            required_keys = [
                "cv_scores",
                "mean_cv_accuracy",
                "cv_std",
                "train_accuracy",
                "train_f1",
                "train_auc",
                "n_features",
                "feature_columns",
            ]

            for key in required_keys:
                assert key in results

            # Check that CV scores are reasonable
            assert len(results["cv_scores"]) == 3
            assert 0 <= results["mean_cv_accuracy"] <= 1

        finally:
            os.unlink(temp_path)

    def test_prediction(self, sample_train_data):
        """Test model prediction functionality."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_train_data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            model = QoLXGBoostModel()
            model.train(train_data_path=temp_path, cv_folds=3)

            # Create test data
            test_data = sample_train_data.head(5).drop(columns=["OS benefits"])

            # Make predictions
            predictions, probabilities = model.predict(test_data)

            # Check predictions
            assert len(predictions) == 5
            assert len(probabilities) == 5
            assert all(pred in [0, 1] for pred in predictions)
            assert all(0 <= prob <= 1 for prob in probabilities)

        finally:
            os.unlink(temp_path)

    def test_prediction_without_training(self, sample_train_data):
        """Test that prediction fails when model is not trained."""
        model = QoLXGBoostModel()
        test_data = sample_train_data.head(5).drop(columns=["OS benefits"])

        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict(test_data)

    def test_save_and_load_model(self, sample_train_data):
        """Test model saving and loading functionality."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_train_data.to_csv(f.name, index=False)
            temp_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as model_file:
            model_path = model_file.name

        try:
            # Train and save model
            model1 = QoLXGBoostModel()
            model1.train(train_data_path=temp_path, cv_folds=3)
            model1.save_model(model_path)

            # Load model
            model2 = QoLXGBoostModel()
            model2.load_model(model_path)

            # Check that model2 is properly loaded
            assert model2.is_fitted == True
            assert model2.feature_columns == model1.feature_columns

            # Test that both models make same predictions
            test_data = sample_train_data.head(5).drop(columns=["OS benefits"])
            pred1, prob1 = model1.predict(test_data)
            pred2, prob2 = model2.predict(test_data)

            np.testing.assert_array_equal(pred1, pred2)
            np.testing.assert_allclose(prob1, prob2, rtol=1e-10)

        finally:
            os.unlink(temp_path)
            os.unlink(model_path)

    def test_save_model_without_training(self):
        """Test that saving fails when model is not trained."""
        model = QoLXGBoostModel()

        with pytest.raises(ValueError, match="Model must be trained"):
            model.save_model("dummy_path.joblib")

    def test_get_model_info(self, sample_train_data):
        """Test model info retrieval."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_train_data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            model = QoLXGBoostModel()

            # Before training
            info_before = model.get_model_info()
            assert info_before["model_type"] == "XGBoost"
            assert info_before["is_fitted"] == False
            assert info_before["n_features"] == 0

            # After training
            model.train(train_data_path=temp_path, cv_folds=3)
            info_after = model.get_model_info()
            assert info_after["is_fitted"] == True
            assert info_after["n_features"] > 0
            assert info_after["model_params"] is not None

        finally:
            os.unlink(temp_path)


class TestTrainXGBoostModel:
    """Test cases for train_xgboost_model function."""

    @pytest.fixture
    def sample_train_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 30

        data = {
            "ID": range(1, n_samples + 1),
            "OS benefits": np.random.choice(["Y", "N"], n_samples, p=[0.6, 0.4]),
            "Drug Class": np.random.choice(
                ["Targeted therapy", "Immunotherapy"], n_samples
            ),
            "Approval Year": np.random.choice([2018, 2019, 2020], n_samples),
            "Active Arm PFS": np.random.uniform(1, 20, n_samples),
            "Control Arm PFS": np.random.uniform(1, 15, n_samples),
            "QoL benefits": np.random.choice(["Y", "N"], n_samples),
        }

        return pd.DataFrame(data)

    def test_train_xgboost_model_basic(self, sample_train_data):
        """Test the train_xgboost_model function."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_train_data.to_csv(f.name, index=False)
            temp_data_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as model_file:
            temp_model_path = model_file.name

        try:
            # Train model
            model = train_xgboost_model(
                train_data_path=temp_data_path,
                model_save_path=temp_model_path,
                n_estimators=10,  # Small for fast testing
                random_state=42,
            )

            # Check that model is trained and saved
            assert model.is_fitted == True
            assert os.path.exists(temp_model_path)

            # Check that results file is created
            results_path = temp_model_path.replace(".joblib", "_results.json")
            assert os.path.exists(results_path)

            # Load and check results
            import json

            with open(results_path, "r") as f:
                results = json.load(f)

            assert "cv_scores" in results
            assert "train_accuracy" in results
            assert "n_features" in results

        finally:
            os.unlink(temp_data_path)
            if os.path.exists(temp_model_path):
                os.unlink(temp_model_path)
            results_path = temp_model_path.replace(".joblib", "_results.json")
            if os.path.exists(results_path):
                os.unlink(results_path)

    def test_train_xgboost_model_custom_params(self, sample_train_data):
        """Test training with custom model parameters."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_train_data.to_csv(f.name, index=False)
            temp_data_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as model_file:
            temp_model_path = model_file.name

        try:
            # Train model with custom parameters
            model = train_xgboost_model(
                train_data_path=temp_data_path,
                model_save_path=temp_model_path,
                n_estimators=5,
                max_depth=3,
                learning_rate=0.05,
                random_state=123,
            )

            # Check that custom parameters were used
            params = model.model.get_params()
            assert params["n_estimators"] == 5
            assert params["max_depth"] == 3
            assert params["learning_rate"] == 0.05
            assert params["random_state"] == 123

        finally:
            os.unlink(temp_data_path)
            if os.path.exists(temp_model_path):
                os.unlink(temp_model_path)
            results_path = temp_model_path.replace(".joblib", "_results.json")
            if os.path.exists(results_path):
                os.unlink(results_path)


class TestIntegration:
    """Integration tests for the XGBoost training module."""

    def test_full_training_pipeline(self):
        """Test the complete training pipeline."""
        # Create realistic data
        np.random.seed(42)
        n_samples = 100

        data = {
            "ID": range(1, n_samples + 1),
            "OS benefits": np.random.choice(["Y", "N"], n_samples, p=[0.65, 0.35]),
            "Drug Class": np.random.choice(
                ["Targeted therapy", "Immunotherapy", "Chemotherapy"], n_samples
            ),
            "Approval Year": np.random.choice([2018, 2019, 2020, 2021], n_samples),
            "Endpoint": np.random.choice(["PFS", "OS", "DFS"], n_samples),
            "Active Arm PFS": np.random.uniform(1, 25, n_samples),
            "Control Arm PFS": np.random.uniform(1, 20, n_samples),
            "PFS p-value": np.random.uniform(0.001, 0.2, n_samples),
            "Active Arm ORR": np.random.uniform(10, 90, n_samples),
            "Control Arm ORR": np.random.uniform(5, 85, n_samples),
            "QoL benefits": np.random.choice(["Y", "N"], n_samples, p=[0.4, 0.6]),
        }

        # Add some missing values
        df = pd.DataFrame(data)
        df.loc[5:10, "Active Arm PFS"] = np.nan
        df.loc[15:20, "Drug Class"] = None

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_data_path = f.name

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.joblib")

            try:
                # Train model
                model = train_xgboost_model(
                    train_data_path=temp_data_path,
                    model_save_path=model_path,
                    n_estimators=20,
                    random_state=42,
                )

                # Test prediction on the same data (should work)
                test_data = df.drop(columns=["OS benefits"])
                predictions, probabilities = model.predict(test_data)

                # Basic sanity checks
                assert len(predictions) == n_samples
                assert len(probabilities) == n_samples
                assert all(pred in [0, 1] for pred in predictions)
                assert all(0 <= prob <= 1 for prob in probabilities)

                # Test loading the saved model
                new_model = QoLXGBoostModel()
                new_model.load_model(model_path)

                # Test that loaded model makes same predictions
                new_predictions, new_probabilities = new_model.predict(test_data)
                np.testing.assert_array_equal(predictions, new_predictions)
                np.testing.assert_allclose(probabilities, new_probabilities, rtol=1e-10)

            finally:
                os.unlink(temp_data_path)


if __name__ == "__main__":
    pytest.main([__file__])
