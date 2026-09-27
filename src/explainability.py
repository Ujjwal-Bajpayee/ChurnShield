import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path

MODEL_PATH = Path("models/churn_model.pkl")
DATA_PATH = Path("artifacts/validated_data.csv")

ALL_EXPLAINED_FEATURES = [
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
    "TotalCharges",
    "AvgMonthlySpend",
    "ChargeRatio",
    "ServiceCount",
    "CostPerService",
    "DiscountRatio",
    "IsHighRiskTriad",
    "IsShortTenureMonthToMonth",
    "HasStreaming",
    "TenureCohort"
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
            raise FileNotFoundError("Model not found. Train the model first.")

        loaded = joblib.load(MODEL_PATH)
        if isinstance(loaded, dict) and "pipeline" in loaded:
            self.pipeline = loaded["pipeline"]
        else:
            self.pipeline = loaded

        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.model = self.pipeline.named_steps["model"]

        self.background_df = self._load_background_data(data_path)
        self.background_transformed = self.preprocessor.transform(self.background_df)
        self.explainers = self._init_explainers()

    def _load_background_data(self, data_path):
        if data_path.exists():
            try:
                df = pd.read_csv(data_path)
                drop_cols = [c for c in ["Churn", "customerID"] if c in df.columns]
                df = df.drop(columns=drop_cols)
                if len(df) > 50:
                    return df.sample(50, random_state=42)
                return df
            except Exception:
                pass
        return pd.DataFrame([DEFAULT_SAMPLE])

    def _init_explainers(self):
        explainers = []
        estimators = self.model.estimators_ if hasattr(self.model, "estimators_") else [self.model]
        for est in estimators:
            try:
                exp = shap.Explainer(est, self.background_transformed)
                explainers.append(("explainer", exp, est))
            except Exception:
                try:
                    exp = shap.LinearExplainer(est, self.background_transformed)
                    explainers.append(("linear", exp, est))
                except Exception:
                    try:
                        exp = shap.TreeExplainer(est)
                        explainers.append(("tree", exp, est))
                    except Exception:
                        pass
        return explainers

    def _map_feature_name(self, raw_name):
        clean = raw_name.replace("numerical__", "").replace("categorical__", "")
        for orig in ALL_EXPLAINED_FEATURES:
            if clean == orig or clean.startswith(orig + "_"):
                return orig
        return clean

    def explain(self, customer_data, aggregate=True):
        df = pd.DataFrame([customer_data])
        X_transformed = self.preprocessor.transform(df)

        sv_list = []
        for exp_type, exp, est in self.explainers:
            try:
                if exp_type == "tree":
                    sv = exp.shap_values(X_transformed)
                    if isinstance(sv, list):
                        sv_list.append(sv[1][0] if len(sv) > 1 else sv[0][0])
                    elif len(sv.shape) == 3:
                        sv_list.append(sv[0, :, 1])
                    else:
                        sv_list.append(sv[0])
                else:
                    shap_obj = exp(X_transformed)
                    values = shap_obj.values
                    if len(values.shape) == 3:
                        sv_list.append(values[0, :, 1])
                    elif len(values.shape) == 2:
                        sv_list.append(values[0])
                    else:
                        sv_list.append(np.array(values).flatten())
            except Exception:
                pass

        if sv_list:
            raw_shap_values = np.mean(sv_list, axis=0)
        else:
            transformer_step = self.preprocessor.named_steps["transformer"]
            raw_shap_values = np.zeros(len(transformer_step.get_feature_names_out()))

        transformer_step = self.preprocessor.named_steps["transformer"]
        feature_names = transformer_step.get_feature_names_out()

        if aggregate:
            raw_df = pd.DataFrame({
                "raw_feature": feature_names,
                "shap_value": raw_shap_values
            })
            raw_df["feature"] = raw_df["raw_feature"].apply(self._map_feature_name)

            explanation = raw_df.groupby("feature", as_index=False)["shap_value"].sum()

            fe_step = self.preprocessor.named_steps.get("feature_engineer")
            engineered_df = fe_step.transform(df) if fe_step else df

            explanation["feature_value"] = explanation["feature"].apply(
                lambda f: engineered_df[f].iloc[0] if f in engineered_df.columns else customer_data.get(f, "")
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