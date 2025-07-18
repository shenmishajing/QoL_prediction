# QoL Prediction Project

## Overview

This project implements a comprehensive machine learning pipeline for predicting Overall Survival (OS) benefits in cancer patients based on clinical trial data. The project compares two different approaches: a traditional XGBoost model and a Large Language Model (LLM) approach using OpenAI's API.

## Project Structure

```
src/project/
├── tools/
│   ├── dataset/
│   │   └── split_dataset.py          # Dataset splitting utilities
│   └── model/
│       ├── xgboost_train.py          # XGBoost model training
│       └── evaluation.py             # Model evaluation framework
└── model/
    └── llm.py                        # LLM model implementation

docs/                                 # Documentation
├── tools/dataset/
│   └── split_dataset.md             # Dataset splitting documentation
└── model/
    └── llm.md                       # LLM model documentation

tests/                               # Unit tests
├── tools/dataset/
│   └── test_split_dataset.py       # Dataset splitting tests
└── tools/model/
    └── test_xgboost_train.py       # XGBoost training tests

data/                                # Data files
├── DataBase_Clean_Zhou_16JUL2025_AImodel.csv  # Original dataset
├── train_data.csv                  # Training split
├── test_data.csv                   # Testing split
└── id.json                         # ID mappings and metadata

models/                              # Trained models
├── xgboost_model.joblib            # Saved XGBoost model
└── xgboost_model_results.json      # Training results

results/                             # Evaluation results
├── xgboost_predictions.csv         # XGBoost predictions
├── llm_predictions.csv             # LLM predictions (if run)
├── evaluation_results.json         # Comprehensive evaluation
└── confusion_matrices.png          # Visualization
```

## Dataset

The dataset contains clinical trial data with:
- **95 patients** total
- **35 features** including clinical measurements, drug information, and QoL metrics
- **Binary target**: OS benefits (Y/N)
- **Train/Test split**: 76/19 samples (4:1 ratio)

### Key Features
- Drug Class (Targeted therapy, Immunotherapy, etc.)
- Approval Year
- Endpoint measurements (PFS, ORR, QoL)
- Clinical trial outcomes and p-values

## Models Implemented

### 1. XGBoost Model (`src/project/tools/model/xgboost_train.py`)

**Features:**
- Gradient boosting classifier optimized for clinical data
- Comprehensive preprocessing pipeline
- Handles missing values and categorical encoding
- Cross-validation for robust evaluation
- Feature importance analysis

**Performance on Test Set:**
- Accuracy: 78.95%
- F1 Score: 86.67%
- Precision: 76.47%
- Recall: 100.00%
- ROC AUC: 67.95%

### 2. LLM Model (`src/project/model/llm.py`)

**Features:**
- OpenAI API integration with medical-specific prompting
- Structured feature presentation for clinical context
- Rate limiting and error handling
- Binary classification with validation

**Configuration:**
- Model: GPT-3.5-turbo
- Temperature: 0.1 (for reproducibility)
- Specialized medical prompt template

## Installation

Check the [User installation document](docs/get_started/installation.md) for how to install the project and all required packages, if you only want to use this project without developing or contributing to it.

Check the [Development Installation document](docs/get_started/contribution.md#installation) for how to install the project and all required packages, if you want to develop based on this project or contribute to it.

For LLM functionality, set your OpenAI API key in the .env file.

## Usage

### 1. Dataset Splitting

```bash
pixi run python -m src.project.tools.dataset.split_dataset
```

This will:
- Load the original dataset
- Split into train/test sets (4:1 ratio)
- Save split datasets and ID mappings
- Display dataset statistics

### 2. Train XGBoost Model

```bash
pixi run python -m src.project.tools.model.xgboost_train
```

This will:
- Train XGBoost model on training data
- Perform cross-validation
- Save trained model and results
- Display performance metrics and feature importance

### 3. Run Evaluation

```python
from src.project.tools.model.evaluation import run_full_evaluation

# Evaluate both models (requires OpenAI API key for LLM)
results = run_full_evaluation(
    openai_api_key="your-api-key"  # Optional
)
```

This will:
- Evaluate XGBoost model on test set
- Evaluate LLM model (if API key provided)
- Compare model performances
- Generate confusion matrices
- Save comprehensive results

## Testing

Run all unit tests:

```bash
pixi run pytest tests/ -v
```

The test suite includes:
- **24 comprehensive tests** covering all modules
- Dataset splitting functionality
- XGBoost training and prediction
- Model serialization and loading
- Error handling and edge cases

## Key Results

### XGBoost Model Performance
- High recall (100%) - catches all positive cases
- Good precision (76.47%) - low false positive rate
- Strong F1 score (86.67%) - balanced performance
- Reasonable accuracy (78.95%) on small test set

### Feature Importance (Top 5)
1. Control arm QoL (6.33%)
2. Active arm QoL (4.74%)
3. Control Arm ORR N (4.40%)
4. Active QoL Variance Upper (4.22%)
5. PFS p-value (3.81%)

## Documentation

Comprehensive documentation is available in the `docs/` directory:
- **Dataset utilities**: `docs/tools/dataset/split_dataset.md`
- **LLM model**: `docs/model/llm.md`

Each module includes:
- Function signatures and parameters
- Usage examples
- Implementation details
- Best practices

## Development

### Code Quality
- Comprehensive unit testing with pytest
- Type hints throughout codebase
- Docstring documentation
- Error handling and validation

### Dependencies
- **pandas**: Data manipulation
- **scikit-learn**: ML utilities and metrics
- **xgboost**: Gradient boosting
- **openai**: LLM API access
- **matplotlib/seaborn**: Visualization

## Future Enhancements

1. **Model Improvements**
   - Hyperparameter tuning
   - Ensemble methods
   - Deep learning approaches

2. **Feature Engineering**
   - Domain-specific feature creation
   - Interaction terms
   - Time-series features

3. **Evaluation**
   - Larger test datasets
   - External validation
   - Clinical significance analysis

4. **Deployment**
   - REST API for predictions
   - Web interface
   - Model monitoring

## License

MIT License - See LICENSE file for details.

## Contribution

See [contribution docs](docs/get_started/contribution.md) for details.

---

**Note**: This project is for research and educational purposes. Clinical decisions should always involve qualified medical professionals. 
