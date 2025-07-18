"""Dataset splitting utilities for QoL prediction project."""

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(data_path: str) -> pd.DataFrame:
    """Load the dataset from CSV file.

    Args:
        data_path: Path to the CSV data file

    Returns:
        DataFrame containing the loaded data
    """
    return pd.read_csv(data_path)


def split_dataset(
    data_path: str = "data/DataBase_Clean_Zhou_16JUL2025_AImodel.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str = "data",
) -> Dict[str, Any]:
    """Split dataset into train and test sets with 4:1 ratio.

    Args:
        data_path: Path to the CSV data file
        test_size: Proportion of dataset for test set (default 0.2 for 4:1 ratio)
        random_state: Random seed for reproducibility
        output_dir: Directory to save output files

    Returns:
        Dictionary containing split information and file paths
    """
    # Load data
    df = load_data(data_path)

    # Verify data structure
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Check target column (second column - OS benefits)
    target_col = df.columns[1]  # "OS benefits"
    print(f"Target column: {target_col}")
    print(f"Target distribution: {df[target_col].value_counts()}")

    # Split data stratified by target
    train_indices, test_indices = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )

    # Create train and test DataFrames
    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save train and test datasets
    train_path = f"{output_dir}/train_data.csv"
    test_path = f"{output_dir}/test_data.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Save ID mappings
    id_mapping = {
        "train_ids": train_df["ID"].tolist(),
        "test_ids": test_df["ID"].tolist(),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "target_column": target_col,
        "feature_columns": list(df.columns[2:]),  # All columns except ID and target
        "train_target_distribution": train_df[target_col].value_counts().to_dict(),
        "test_target_distribution": test_df[target_col].value_counts().to_dict(),
    }

    id_path = f"{output_dir}/id.json"
    with open(id_path, "w") as f:
        json.dump(id_mapping, f, indent=2)

    print("Data split completed:")
    print(f"  Train set: {len(train_df)} samples")
    print(f"  Test set: {len(test_df)} samples")
    print(f"  Files saved to: {output_dir}/")

    return {
        "train_data": train_df,
        "test_data": test_df,
        "train_path": train_path,
        "test_path": test_path,
        "id_mapping_path": id_path,
        "id_mapping": id_mapping,
    }


def get_feature_info(data_path: str) -> Dict[str, Any]:
    """Get information about features in the dataset.

    Args:
        data_path: Path to the CSV data file

    Returns:
        Dictionary containing feature information
    """
    df = load_data(data_path)

    feature_info = {
        "total_samples": len(df),
        "total_features": len(df.columns) - 2,  # Exclude ID and target
        "id_column": df.columns[0],
        "target_column": df.columns[1],
        "feature_columns": list(df.columns[2:]),
        "target_distribution": df[df.columns[1]].value_counts().to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

    return feature_info


if __name__ == "__main__":
    # Example usage
    result = split_dataset()
    feature_info = get_feature_info("data/DataBase_Clean_Zhou_16JUL2025_AImodel.csv")

    print("\nFeature Information:")
    print(f"Total features: {feature_info['total_features']}")
    print(f"Target distribution: {feature_info['target_distribution']}")
