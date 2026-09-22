import joblib
import pandas as pd
import shap
from pathlib import Path

MODEL_PATH = Path("models/churn_model.pkl")

class ChurnExplainer:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found. Train the model first."
            )

        self.pipeline = joblib.load(MODEL_PATH)

        self.preprocessor = self.pipeline.named_steps[
            "preprocessor"
        ]

        self.model = self.pipeline.named_steps[
            "model"
        ]

        # Create SHAP explainer once
        self.explainer = shap.LinearExplainer(
            self.model,
            self.preprocessor.transform(
                pd.DataFrame(
                    [{
                        "gender": "Male",
                        "SeniorCitizen": 0,
                        "Partner": "No",
                        "Dependents": "No",
                        "tenure": 12,
                        "PhoneService": "Yes",
                        "MultipleLines": "No",
                        "InternetService": "DSL",
                        "OnlineSecurity": "No",
                        "OnlineBackup": "No",
                        "DeviceProtection": "No",
                        "TechSupport": "No",
                        "StreamingTV": "No",
                        "StreamingMovies": "No",
                        "Contract": "Month-to-month",
                        "PaperlessBilling": "Yes",
                        "PaymentMethod": "Electronic check",
                        "MonthlyCharges": 70.0,
                        "TotalCharges": 1000.0
                    }]
                )
            )
        )

    def explain(self, customer_data):

        df = pd.DataFrame([customer_data])

        # Transform raw customer data
        X_transformed = self.preprocessor.transform(df)

        # Calculate SHAP values
        shap_values = self.explainer(
            X_transformed
        )

        # Get feature names
        feature_names = (
            self.preprocessor
            .get_feature_names_out()
        )

        values = shap_values.values[0]

        explanation = pd.DataFrame({
            "feature": feature_names,
            "shap_value": values
        })

        # Absolute contribution
        explanation["importance"] = (
            explanation["shap_value"].abs()
        )

        # Largest contributors first
        explanation = explanation.sort_values(
            "importance",
            ascending=False
        ).reset_index(drop=True)

        return explanation