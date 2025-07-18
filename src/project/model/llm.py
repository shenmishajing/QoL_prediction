"""LLM model implementation using OpenAI API for QoL prediction."""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class QoLPredictionLLM:
    """Large Language Model for Quality of Life prediction using OpenAI API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_tokens: int = 50,
    ):
        """Initialize the LLM model.

        Args:
            api_key: OpenAI API key (if None, will look for OPENAI_API_KEY env var)
            model: OpenAI model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def create_prompt_template(self) -> str:
        """Create the prompt template for QoL prediction.

        Returns:
            Prompt template string
        """
        template = """You are a medical AI assistant specialized in predicting Overall Survival (OS) benefits for cancer patients based on clinical trial data.

Given the following clinical features for a cancer patient, predict whether they will have OS benefits (Overall Survival benefits).

Instructions:
- Analyze each clinical feature carefully
- Consider the drug class, approval year, endpoint measurements, progression-free survival (PFS), overall response rate (ORR), and quality of life (QoL) metrics
- Make a prediction based on established clinical knowledge
- Respond with EXACTLY one word: either "Y" (for OS benefits) or "N" (for no OS benefits)
- Do not provide any explanation or additional text

Clinical Features:
{features}

Prediction (Y or N):"""

        return template

    def format_features(self, row: pd.Series) -> str:
        """Format patient features for the prompt.

        Args:
            row: DataFrame row containing patient features

        Returns:
            Formatted feature string
        """
        features = []

        # Skip ID and target columns, format the rest
        for col, value in row.items():
            if col not in ["ID", "OS benefits"]:
                # Handle missing values
                if pd.isna(value) or value == "":
                    value_str = "Not available"
                else:
                    value_str = str(value)

                features.append(f"- {col}: {value_str}")

        return "\n".join(features)

    def predict_single(self, patient_features: pd.Series) -> str:
        """Predict OS benefits for a single patient.

        Args:
            patient_features: Series containing patient features

        Returns:
            Prediction ("Y" or "N")
        """
        try:
            # Format features for prompt
            formatted_features = self.format_features(patient_features)

            # Create prompt
            prompt = self.create_prompt_template().format(features=formatted_features)

            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Extract prediction
            prediction = response.choices[0].message.content.strip().upper()

            # Validate prediction
            if prediction not in ["Y", "N"]:
                print(f"Invalid prediction '{prediction}', defaulting to 'N'")
                prediction = "N"

            return prediction

        except Exception as e:
            print(f"Error in prediction: {e}")
            return "N"  # Default to "N" on error

    def predict(self, data: pd.DataFrame, delay: float = 1.0) -> List[str]:
        """Predict OS benefits for multiple patients.

        Args:
            data: DataFrame containing patient data
            delay: Delay between API calls to avoid rate limiting

        Returns:
            List of predictions
        """
        predictions = []

        print(f"Making predictions for {len(data)} patients...")

        for idx, (_, row) in enumerate(data.iterrows()):
            if idx > 0 and delay > 0:
                time.sleep(delay)  # Rate limiting

            prediction = self.predict_single(row)
            predictions.append(prediction)

            if (idx + 1) % 10 == 0:
                print(f"Completed {idx + 1}/{len(data)} predictions")

        print("All predictions completed!")
        return predictions

    def evaluate_on_test_set(
        self,
        test_data_path: str = "data/test_data.csv",
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate the LLM on test set.

        Args:
            test_data_path: Path to test data CSV
            output_path: Path to save predictions (optional)

        Returns:
            Dictionary containing predictions and metadata
        """
        # Load test data
        test_data = pd.read_csv(test_data_path)

        # Make predictions
        predictions = self.predict(test_data)

        # Create results DataFrame
        results_df = test_data.copy()
        results_df["llm_prediction"] = predictions

        # Save predictions if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            print(f"Predictions saved to: {output_path}")

        # Prepare results
        results = {
            "predictions": predictions,
            "true_labels": test_data["OS benefits"].tolist(),
            "patient_ids": test_data["ID"].tolist(),
            "results_df": results_df,
            "model_info": {
                "model_name": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        }

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model configuration.

        Returns:
            Dictionary containing model information
        """
        return {
            "model_type": "LLM",
            "model_name": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_provider": "OpenAI",
        }


def demo_prediction():
    """Demonstrate LLM prediction with sample data."""
    # This is just for demonstration - requires API key
    try:
        llm = QoLPredictionLLM()

        # Create sample patient data
        sample_patient = pd.Series(
            {
                "ID": 999,
                "Drug Class": "Targeted therapy",
                "Approval Year": 2020,
                "Endpoint": "PFS",
                "Active Arm PFS": 13.93,
                "Control Arm PFS": 9.46,
                "PFS p-value": 0.0075,
                "Active Arm ORR": 76.4,
                "Control Arm ORR": 62.3,
                "ORR p-value": 0.0012,
                "DFS/PFS benfits": "Y",
                "QoL benefits": "N",
            }
        )

        prediction = llm.predict_single(sample_patient)
        print(f"Sample prediction: {prediction}")

    except Exception as e:
        print(f"Demo failed: {e}")
        print("Make sure to set OPENAI_API_KEY environment variable")


if __name__ == "__main__":
    demo_prediction()
