"""Unit tests for the dataset splitting module."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.project.tools.dataset.split_dataset import (
    get_feature_info,
    load_data,
    split_dataset,
)


class TestLoadData:
    """Test cases for load_data function."""

    def test_load_data_valid_file(self):
        """Test loading a valid CSV file."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("ID,OS benefits,Feature1,Feature2\n")
            f.write("1,Y,10,20\n")
            f.write("2,N,15,25\n")
            temp_path = f.name

        try:
            df = load_data(temp_path)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert list(df.columns) == ["ID", "OS benefits", "Feature1", "Feature2"]
        finally:
            os.unlink(temp_path)

    def test_load_data_file_not_found(self):
        """Test loading a non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_data("non_existent_file.csv")


class TestSplitDataset:
    """Test cases for split_dataset function."""

    @pytest.fixture
    def sample_data_file(self):
        """Create a sample data file for testing."""
        # Create sample data similar to the real dataset
        data = {
            "ID": range(1, 21),
            "OS benefits": ["Y"] * 12
            + ["N"] * 8,  # 60-40 split for testing stratification
            "Drug Class": ["Targeted therapy"] * 10 + ["Immunotherapy"] * 10,
            "Approval Year": [2020] * 20,
            "Endpoint": ["PFS"] * 20,
            "Active Arm PFS": np.random.uniform(5, 15, 20),
            "Control Arm PFS": np.random.uniform(3, 12, 20),
            "PFS p-value": np.random.uniform(0.001, 0.1, 20),
        }

        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    def test_split_dataset_basic_functionality(self, sample_data_file):
        """Test basic dataset splitting functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = split_dataset(
                data_path=sample_data_file,
                test_size=0.2,
                random_state=42,
                output_dir=temp_dir,
            )

            # Check return structure
            assert isinstance(result, dict)
            assert "train_data" in result
            assert "test_data" in result
            assert "id_mapping" in result

            # Check data splits
            assert len(result["train_data"]) == 16  # 80% of 20
            assert len(result["test_data"]) == 4  # 20% of 20

            # Check files were created
            assert os.path.exists(result["train_path"])
            assert os.path.exists(result["test_path"])
            assert os.path.exists(result["id_mapping_path"])

    def test_split_dataset_stratification(self, sample_data_file):
        """Test that stratification maintains target distribution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = split_dataset(
                data_path=sample_data_file,
                test_size=0.2,
                random_state=42,
                output_dir=temp_dir,
            )

            # Check stratification worked
            train_dist = result["id_mapping"]["train_target_distribution"]
            test_dist = result["id_mapping"]["test_target_distribution"]

            # Calculate proportions
            train_total = sum(train_dist.values())
            test_total = sum(test_dist.values())

            train_y_prop = train_dist.get("Y", 0) / train_total
            test_y_prop = test_dist.get("Y", 0) / test_total

            # Proportions should be similar (within reasonable tolerance)
            assert abs(train_y_prop - test_y_prop) < 0.2  # 20% tolerance

    def test_split_dataset_reproducibility(self, sample_data_file):
        """Test that splitting is reproducible with same random state."""
        with (
            tempfile.TemporaryDirectory() as temp_dir1,
            tempfile.TemporaryDirectory() as temp_dir2,
        ):
            result1 = split_dataset(
                data_path=sample_data_file,
                test_size=0.2,
                random_state=42,
                output_dir=temp_dir1,
            )

            result2 = split_dataset(
                data_path=sample_data_file,
                test_size=0.2,
                random_state=42,
                output_dir=temp_dir2,
            )

            # Same IDs should be in train/test sets
            assert (
                result1["id_mapping"]["train_ids"] == result2["id_mapping"]["train_ids"]
            )
            assert (
                result1["id_mapping"]["test_ids"] == result2["id_mapping"]["test_ids"]
            )

    def test_split_dataset_id_mapping_content(self, sample_data_file):
        """Test the content of the ID mapping file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = split_dataset(
                data_path=sample_data_file,
                test_size=0.2,
                random_state=42,
                output_dir=temp_dir,
            )

            id_mapping = result["id_mapping"]

            # Check required keys
            required_keys = [
                "train_ids",
                "test_ids",
                "train_size",
                "test_size",
                "target_column",
                "feature_columns",
                "train_target_distribution",
                "test_target_distribution",
            ]

            for key in required_keys:
                assert key in id_mapping

            # Check data types and values
            assert isinstance(id_mapping["train_ids"], list)
            assert isinstance(id_mapping["test_ids"], list)
            assert isinstance(id_mapping["train_size"], int)
            assert isinstance(id_mapping["test_size"], int)
            assert id_mapping["target_column"] == "OS benefits"

    def test_split_dataset_custom_test_size(self, sample_data_file):
        """Test splitting with custom test size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = split_dataset(
                data_path=sample_data_file,
                test_size=0.3,  # 30% test set
                random_state=42,
                output_dir=temp_dir,
            )

            total_samples = len(result["train_data"]) + len(result["test_data"])
            test_proportion = len(result["test_data"]) / total_samples

            # Should be close to 0.3 (within rounding tolerance)
            assert abs(test_proportion - 0.3) < 0.1


class TestGetFeatureInfo:
    """Test cases for get_feature_info function."""

    @pytest.fixture
    def sample_data_file(self):
        """Create a sample data file for testing."""
        data = {
            "ID": [1, 2, 3],
            "OS benefits": ["Y", "N", "Y"],
            "Feature1": [10, 20, None],  # Missing value
            "Feature2": ["A", "B", "A"],
            "Feature3": [1.5, 2.5, 3.5],
        }

        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    def test_get_feature_info_basic(self, sample_data_file):
        """Test basic feature info extraction."""
        info = get_feature_info(sample_data_file)

        # Check return structure
        assert isinstance(info, dict)

        required_keys = [
            "total_samples",
            "total_features",
            "id_column",
            "target_column",
            "feature_columns",
            "target_distribution",
            "missing_values",
            "dtypes",
        ]

        for key in required_keys:
            assert key in info

    def test_get_feature_info_values(self, sample_data_file):
        """Test specific values in feature info."""
        info = get_feature_info(sample_data_file)

        assert info["total_samples"] == 3
        assert info["total_features"] == 3  # Excluding ID and target
        assert info["id_column"] == "ID"
        assert info["target_column"] == "OS benefits"
        assert "Feature1" in info["feature_columns"]
        assert "Feature2" in info["feature_columns"]
        assert "Feature3" in info["feature_columns"]

        # Check target distribution
        assert info["target_distribution"]["Y"] == 2
        assert info["target_distribution"]["N"] == 1

        # Check missing values detection
        assert info["missing_values"]["Feature1"] == 1
        assert info["missing_values"]["Feature2"] == 0


class TestIntegration:
    """Integration tests for the entire module."""

    def test_end_to_end_workflow(self):
        """Test the complete workflow from data loading to splitting."""
        # Create a more realistic dataset
        np.random.seed(42)
        n_samples = 50

        data = {
            "ID": range(1, n_samples + 1),
            "OS benefits": np.random.choice(["Y", "N"], n_samples, p=[0.6, 0.4]),
            "Drug Class": np.random.choice(
                ["Targeted therapy", "Immunotherapy", "Chemotherapy"], n_samples
            ),
            "Approval Year": np.random.choice([2018, 2019, 2020, 2021], n_samples),
            "Active Arm PFS": np.random.uniform(1, 20, n_samples),
            "Control Arm PFS": np.random.uniform(1, 15, n_samples),
            "PFS p-value": np.random.uniform(0.001, 0.1, n_samples),
            "Active Arm ORR": np.random.uniform(10, 90, n_samples),
            "Control Arm ORR": np.random.uniform(5, 80, n_samples),
        }

        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Test the complete workflow
                result = split_dataset(
                    data_path=temp_path,
                    test_size=0.2,
                    random_state=42,
                    output_dir=temp_dir,
                )

                # Verify all files exist and have correct structure
                train_df = pd.read_csv(result["train_path"])
                test_df = pd.read_csv(result["test_path"])

                with open(result["id_mapping_path"], "r") as f:
                    id_mapping = json.load(f)

                # Check data integrity
                assert len(train_df) + len(test_df) == n_samples
                assert set(train_df.columns) == set(test_df.columns)
                assert set(train_df.columns) == set(df.columns)

                # Check ID mappings
                all_train_ids = set(train_df["ID"].tolist())
                all_test_ids = set(test_df["ID"].tolist())
                mapped_train_ids = set(id_mapping["train_ids"])
                mapped_test_ids = set(id_mapping["test_ids"])

                assert all_train_ids == mapped_train_ids
                assert all_test_ids == mapped_test_ids
                assert all_train_ids.isdisjoint(all_test_ids)  # No overlap

        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])
