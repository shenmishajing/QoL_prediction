# LLM Model for QoL Prediction

## Overview

The `llm.py` module implements a Large Language Model (LLM) based approach for predicting Overall Survival (OS) benefits in cancer patients using OpenAI's API. The model uses a carefully crafted prompt template to analyze clinical features and make binary predictions.

## Class: QoLPredictionLLM

### Initialization

```python
QoLPredictionLLM(
    api_key: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.1,
    max_tokens: int = 50
)
```

**Parameters:**
- `api_key`: OpenAI API key (if None, looks for OPENAI_API_KEY environment variable)
- `model`: OpenAI model to use (default: "gpt-3.5-turbo")
- `temperature`: Sampling temperature for reproducibility (default: 0.1)
- `max_tokens`: Maximum tokens in response (default: 50)

### Methods

#### `create_prompt_template() -> str`

Creates a specialized prompt template for medical prediction tasks.

**Returns:**
- Formatted prompt template string optimized for clinical decision making

The prompt template includes:
- Medical context and role specification
- Clear instructions for binary classification
- Feature analysis guidelines
- Output format requirements

#### `format_features(row: pd.Series) -> str`

Formats patient clinical features for inclusion in the prompt.

**Parameters:**
- `row`: DataFrame row containing patient features

**Returns:**
- Formatted string with clinical features ready for LLM processing

**Features handled:**
- Missing value replacement with "Not available"
- Proper formatting for medical context
- Exclusion of ID and target columns

#### `predict_single(patient_features: pd.Series) -> str`

Makes a prediction for a single patient.

**Parameters:**
- `patient_features`: Series containing patient clinical features

**Returns:**
- Prediction string ("Y" for OS benefits, "N" for no OS benefits)

**Error handling:**
- API call failures return "N" by default
- Invalid responses are validated and corrected
- Rate limiting considerations

#### `predict(data: pd.DataFrame, delay: float = 1.0) -> List[str]`

Makes predictions for multiple patients with rate limiting.

**Parameters:**
- `data`: DataFrame containing patient data
- `delay`: Delay between API calls in seconds (default: 1.0)

**Returns:**
- List of predictions for all patients

**Features:**
- Progress tracking for large datasets
- Rate limiting to respect API limits
- Batch processing with status updates

#### `evaluate_on_test_set(test_data_path: str, output_path: Optional[str]) -> Dict[str, Any]`

Evaluates the model on a test dataset.

**Parameters:**
- `test_data_path`: Path to test data CSV file
- `output_path`: Optional path to save predictions

**Returns:**
- Dictionary containing:
  - `predictions`: List of model predictions
  - `true_labels`: List of actual labels
  - `patient_ids`: List of patient IDs
  - `results_df`: DataFrame with predictions added
  - `model_info`: Model configuration details

#### `get_model_info() -> Dict[str, Any]`

Returns model configuration information.

**Returns:**
- Dictionary with model type, parameters, and API provider details

## Usage Examples

### Basic Prediction

```python
from src.project.model.llm import QoLPredictionLLM
import pandas as pd

# Initialize model
llm = QoLPredictionLLM(api_key="your-api-key")

# Create sample patient data
patient_data = pd.Series({
    "ID": 999,
    "Drug Class": "Targeted therapy",
    "Approval Year": 2020,
    "Endpoint": "PFS",
    "Active Arm PFS": 13.93,
    "Control Arm PFS": 9.46,
    "PFS p-value": 0.0075
})

# Make prediction
prediction = llm.predict_single(patient_data)
print(f"Prediction: {prediction}")
```

### Batch Evaluation

```python
# Evaluate on test set
results = llm.evaluate_on_test_set(
    test_data_path="data/test_data.csv",
    output_path="results/llm_predictions.csv"
)

print(f"Accuracy: {len([p for p, t in zip(results['predictions'], results['true_labels']) if p == t]) / len(results['predictions'])}")
```

### Multiple Patients

```python
# Load test data
test_df = pd.read_csv("data/test_data.csv")

# Make predictions with rate limiting
predictions = llm.predict(test_df, delay=0.5)
```

## Prompt Engineering

The LLM uses a specialized medical prompt that includes:

1. **Role Definition**: Establishes the AI as a medical assistant
2. **Task Description**: Clear explanation of OS benefits prediction
3. **Analysis Instructions**: Guidance on feature interpretation
4. **Output Format**: Strict binary response requirement
5. **Clinical Context**: Emphasis on established medical knowledge

## API Configuration

### Environment Setup

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Rate Limiting

The model implements rate limiting to respect OpenAI's API limits:
- Default 1-second delay between requests
- Configurable delay parameter
- Progress tracking for long-running evaluations

## Error Handling

- **API Failures**: Default to "N" prediction with error logging
- **Invalid Responses**: Validation and correction of non-binary outputs
- **Missing API Key**: Clear error message with setup instructions
- **Network Issues**: Graceful degradation with informative messages

## Performance Considerations

- **Cost Management**: Token limit control and efficient prompting
- **Speed**: Parallel processing with rate limiting
- **Reliability**: Robust error handling and validation
- **Reproducibility**: Low temperature setting for consistent results

## Model Limitations

1. **API Dependency**: Requires active internet connection and valid API key
2. **Cost Considerations**: Each prediction incurs API usage costs
3. **Rate Limits**: Subject to OpenAI's API rate limiting policies
4. **Model Updates**: Performance may vary with OpenAI model updates
5. **Context Length**: Limited by token context window

## Integration with Evaluation

The LLM model integrates seamlessly with the evaluation framework:

```python
from src.project.tools.model.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
llm_results = evaluator.evaluate_llm_model(
    api_key="your-api-key",
    model_name="gpt-3.5-turbo"
)
``` 
