import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path

MODEL_PATH = Path("models/churn_model.pkl")
DATA_PATH = Path("artifacts/validated_data.csv")

ORIGINAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]

DEFAULT_SAMPLE = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 24,
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
    "MonthlyCharges": 65.0,
    "TotalCharges": 1500.0
}


class ChurnExplainer:
    def __init__(self, data_path=DATA_PATH):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found. Train the model first."
            )

        self.pipeline = joblib.load(MODEL_PATH)
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.model = self.pipeline.named_steps["model"]

        # Prepare a representative background dataset for SHAP
        background_df = self._load_background_data(data_path)
        background_transformed = self.preprocessor.transform(background_df)

        # Initialize SHAP explainer
        try:
            self.explainer = shap.Explainer(self.model, background_transformed)
        except Exception:
            try:
                self.explainer = shap.LinearExplainer(
                    self.model, background_transformed
                )
            except Exception:
                self.explainer = shap.TreeExplainer(self.model)

    def _load_background_data(self, data_path):
        if data_path.exists():
            try:
                df = pd.read_csv(data_path)
                drop_cols = [c for c in ["Churn", "customerID"] if c in df.columns]
                df = df.drop(columns=drop_cols)
                if len(df) > 100:
                    return df.sample(100, random_state=42)
                return df
            except Exception:
                pass
        return pd.DataFrame([DEFAULT_SAMPLE])

    def _map_feature_name(self, raw_name):
        clean = raw_name.replace("numerical__", "").replace("categorical__", "")
        for orig in ORIGINAL_FEATURES:
            if clean == orig or clean.startswith(orig + "_"):
                return orig
        return clean

    def explain(self, customer_data, aggregate=True):
        df = pd.DataFrame([customer_data])
        X_transformed = self.preprocessor.transform(df)

        shap_obj = self.explainer(X_transformed)

        # Handle various output shapes from different SHAP explainers/models
        values = shap_obj.values
        if len(values.shape) == 3:
            # Binary classification [samples, features, classes] -> class 1 (churn)
            raw_shap_values = values[0, :, 1]
        elif len(values.shape) == 2:
            raw_shap_values = values[0]
        else:
            raw_shap_values = np.array(values).flatten()

        feature_names = self.preprocessor.get_feature_names_out()

        if aggregate:
            raw_df = pd.DataFrame({
                "raw_feature": feature_names,
                "shap_value": raw_shap_values
            })
            raw_df["feature"] = raw_df["raw_feature"].apply(self._map_feature_name)

            # Aggregate SHAP contributions per original feature
            explanation = raw_df.groupby("feature", as_index=False)["shap_value"].sum()
            explanation["feature_value"] = explanation["feature"].apply(
                lambda f: customer_data.get(f, "")
            )
        else:
            explanation = pd.DataFrame({
                "feature": feature_names,
                "shap_value": raw_shap_values
            })

        explanation["importance"] = explanation["shap_value"].abs()
        explanation = explanation.sort_values(
            "importance",
            ascending=False
        ).reset_index(drop=True)

        return explanation