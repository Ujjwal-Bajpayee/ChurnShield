import joblib
import pandas as pd

from pathlib import Path

MODEL_PATH = Path("models/churn_model.pkl")

class ChurnPredictor:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found. Train the model first."
            )

        self.model = joblib.load(MODEL_PATH)

    def predict(self, customer_data):
        df = pd.DataFrame([customer_data])

        probability = self.model.predict_proba(
            df
        )[0][1]

        prediction = int(
            probability >= 0.5
        )

        if probability >= 0.7:
            risk = "High"
        elif probability >= 0.4:
            risk = "Medium"
        else:
            risk = "Low"

        return {
            "churn_probability": round(
                float(probability), 4
            ),
            "prediction": prediction,
            "risk": risk
        }