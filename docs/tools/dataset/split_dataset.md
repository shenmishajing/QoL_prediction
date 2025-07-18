# Dataset Splitting Module

## Overview

The `split_dataset.py` module provides utilities for splitting the QoL prediction dataset into training and testing sets with a 4:1 ratio. It ensures stratified sampling to maintain the distribution of the target variable across both sets.

## Functions

### `load_data(data_path: str) -> pd.DataFrame`

Loads the dataset from a CSV file.

**Parameters:**
- `data_path`: Path to the CSV data file

**Returns:**
- DataFrame containing the loaded data

**Example:**
```python
from src.project.tools.dataset.split_dataset import load_data

df = load_data("data/DataBase_Clean_Zhou_16JUL2025_AImodel.csv")
print(df.shape)
```

### `split_dataset(data_path: str, test_size: float, random_state: int, output_dir: str) -> Dict[str, Any]`

Splits the dataset into training and testing sets with stratified sampling.

**Parameters:**
- `data_path`: Path to the CSV data file (default: "data/DataBase_Clean_Zhou_16JUL2025_AImodel.csv")
- `test_size`: Proportion of dataset for test set (default: 0.2 for 4:1 ratio)
- `random_state`: Random seed for reproducibility (default: 42)
- `output_dir`: Directory to save output files (default: "data")

**Returns:**
- Dictionary containing:
  - `train_data`: Training DataFrame
  - `test_data`: Testing DataFrame
  - `train_path`: Path to saved training CSV
  - `test_path`: Path to saved testing CSV
  - `id_mapping_path`: Path to ID mapping JSON file
  - `id_mapping`: Dictionary with split information

**Example:**
```python
from src.project.tools.dataset.split_dataset import split_dataset

result = split_dataset()
print(f"Train set size: {result['id_mapping']['train_size']}")
print(f"Test set size: {result['id_mapping']['test_size']}")
```

### `get_feature_info(data_path: str) -> Dict[str, Any]`

Provides comprehensive information about the features in the dataset.

**Parameters:**
- `data_path`: Path to the CSV data file

**Returns:**
- Dictionary containing:
  - `total_samples`: Total number of samples
  - `total_features`: Number of features (excluding ID and target)
  - `id_column`: Name of the ID column
  - `target_column`: Name of the target column
  - `feature_columns`: List of feature column names
  - `target_distribution`: Distribution of target classes
  - `missing_values`: Count of missing values per column
  - `dtypes`: Data types of all columns

**Example:**
```python
from src.project.tools.dataset.split_dataset import get_feature_info

info = get_feature_info("data/DataBase_Clean_Zhou_16JUL2025_AImodel.csv")
print(f"Features: {info['total_features']}")
print(f"Target distribution: {info['target_distribution']}")
```

## Data Structure

The dataset contains the following structure:
- **Column 1**: ID - Unique identifier for each patient
- **Column 2**: OS benefits - Target variable (Y/N for binary classification)
- **Columns 3-35**: Clinical features including:
  - Drug Class
  - Approval Year
  - Endpoint measurements
  - PFS (Progression-Free Survival) metrics
  - ORR (Overall Response Rate) data
  - QoL (Quality of Life) measurements

## Output Files

The module generates the following files:

1. **train_data.csv**: Training dataset (80% of original data)
2. **test_data.csv**: Testing dataset (20% of original data)
3. **id.json**: Metadata file containing:
   - List of training and testing patient IDs
   - Dataset sizes
   - Target column name
   - Feature column names
   - Target distribution for both sets

## Usage Notes

- The splitting uses stratified sampling to ensure balanced target distribution
- Random state is set to 42 for reproducible results
- Missing values are handled appropriately for downstream processing
- The module maintains the original data structure and column names

## Command Line Usage

```bash
pixi run python -m src.project.tools.dataset.split_dataset
```

This will execute the splitting with default parameters and display summary statistics. 
