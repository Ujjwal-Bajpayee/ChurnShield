import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier

from data_transformation import prepare_features


DATA_PATH = Path("artifacts/validated_data.csv")
MODEL_PATH = Path("models/churn_model.pkl")


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(
            y_test,
            predictions
        ),

        "recall": recall_score(
            y_test,
            predictions
        ),

        "f1": f1_score(
            y_test,
            predictions
        ),

        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        ),

        "pr_auc": average_precision_score(
            y_test,
            probabilities
        )
    }

    return metrics


def train():
    df = pd.read_csv(DATA_PATH)

    X, y, preprocessor = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    }

    results = {}

    best_model = None
    best_score = -1
    best_model_name = None

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        print(f"\nTraining {name}...")

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test
        )

        results[name] = metrics

        print(f"\n{name}")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Use F1 as selection metric
        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_model = pipeline
            best_model_name = name

    print("BEST MODEL")
    print(best_model_name)
    print(f"F1 Score: {best_score:.4f}")

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        best_model,
        MODEL_PATH
    )

    print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()