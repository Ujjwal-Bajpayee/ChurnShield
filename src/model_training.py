import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)
from xgboost import XGBClassifier

try:
    from src.data_transformation import prepare_features
except ImportError:
    from data_transformation import prepare_features

DATA_PATH = Path("artifacts/validated_data.csv")
MODEL_PATH = Path("models/churn_model.pkl")

def find_optimal_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        preds = (y_probs >= t).astype(int)
        f1_scores[i] = f1_score(y_true, preds, zero_division=0)
    best_idx = np.argmax(f1_scores) if len(f1_scores) > 0 else 0
    return float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5

def evaluate_model(model, X_test, y_test, threshold=0.5):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "threshold": float(threshold)
    }

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

    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    scale_weight = float(neg_count / max(pos_count, 1))

    base_lr = LogisticRegression(
        C=0.1,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    base_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=42
    )

    base_xgb = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric="logloss",
        random_state=42
    )

    voting_clf = VotingClassifier(
        estimators=[
            ("lr", base_lr),
            ("rf", base_rf),
            ("xgb", base_xgb)
        ],
        voting="soft"
    )

    models = {
        "Logistic Regression": base_lr,
        "Random Forest": base_rf,
        "XGBoost": base_xgb,
        "Voting Ensemble": voting_clf
    }

    results = {}
    best_pipeline = None
    best_score = -1.0
    best_model_name = None
    best_threshold = 0.5
    best_metrics = None

    for name, model_estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model_estimator)
        ])

        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)

        train_probs = pipeline.predict_proba(X_train)[:, 1]
        opt_thresh = find_optimal_threshold(y_train, train_probs)

        metrics = evaluate_model(pipeline, X_test, y_test, threshold=opt_thresh)
        results[name] = metrics

        print(f"{name} (Optimal Threshold: {opt_thresh:.3f})")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_pipeline = pipeline
            best_model_name = name
            best_threshold = opt_thresh
            best_metrics = metrics

    print("\nBEST MODEL SELECTED:")
    print(f"Name: {best_model_name}")
    print(f"F1 Score: {best_score:.4f}")
    print(f"Optimal Decision Threshold: {best_threshold:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": best_pipeline,
        "model_name": best_model_name,
        "threshold": best_threshold,
        "metrics": best_metrics,
        "all_results": results
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"\nModel bundle successfully saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()