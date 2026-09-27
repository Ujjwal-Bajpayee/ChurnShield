import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/churn_model.pkl")

class ChurnPredictor:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model not found. Train the model first.")

        loaded = joblib.load(MODEL_PATH)
        if isinstance(loaded, dict) and "pipeline" in loaded:
            self.model = loaded["pipeline"]
            self.threshold = float(loaded.get("threshold", 0.5))
            self.model_name = loaded.get("model_name", "Logistic Regression")
            self.metrics = loaded.get("metrics", {})
        else:
            self.model = loaded
            self.threshold = 0.5
            self.model_name = "Model"
            self.metrics = {}

    def predict(self, customer_data):
        df = pd.DataFrame([customer_data])
        probability = float(self.model.predict_proba(df)[0][1])
        prediction = int(probability >= self.threshold)

        if probability >= 0.7:
            risk = "High"
        elif probability >= 0.4:
            risk = "Medium"
        else:
            risk = "Low"

        return {
            "churn_probability": round(probability, 4),
            "prediction": prediction,
            "risk": risk,
            "threshold": round(self.threshold, 4),
            "model_name": self.model_name
        }