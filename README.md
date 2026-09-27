# ChurnShield

An end-to-end customer churn prediction system that predicts churn risk and provides interpretable explanations for individual predictions using SHAP.

## Overview

ChurnShield is built around a production-oriented machine learning and explainability pipeline:

**S3 → Data Ingestion → Data Validation → Preprocessing → Model Training → Evaluation → SHAP Explainability → Streamlit UI**

The project compares multiple classification models and exposes the final prediction through an interactive Streamlit application.

## Architecture

```mermaid
flowchart TD
    subgraph Storage["Data Storage"]
        S3["Amazon S3 (Raw Telco Churn Dataset)"]
        RawCSV["artifacts/raw_data.csv"]
        ValCSV["artifacts/validated_data.csv"]
        ModelPkl["models/churn_model.pkl"]
    end

    subgraph Pipeline["Data Pipeline"]
        Ingest["data_ingestion.py\n(Boto3 S3 Ingestion)"]
        Validate["data_validation.py\n(Schema & Integrity Checks)"]
        Transform["data_transformation.py\n(Imputation, Scaling & OneHotEncoding)"]
    end

    subgraph Modeling["Model Training & Selection"]
        Train["model_training.py"]
        LR["Logistic Regression"]
        RF["Random Forest"]
        XGB["XGBoost"]
        Eval["Model Evaluator\n(F1, ROC-AUC, Precision, Recall, PR-AUC)"]
    end

    subgraph Serving["Serving & Explainability Layer"]
        UI["app.py\n(Streamlit Dashboard)"]
        Predictor["src/prediction.py\n(Probability & Risk Scoring)"]
        Explainer["src/explainability.py\n(SHAP Explainer & Feature Attribution)"]
    end

    S3 --> Ingest --> RawCSV --> Validate --> ValCSV --> Transform
    Transform --> Train
    Train --> LR & RF & XGB --> Eval
    Eval -->|Save Best Pipeline| ModelPkl

    ModelPkl --> Predictor
    ModelPkl --> Explainer
    ValCSV -.->|Background Sample| Explainer

    UI -->|Customer Inputs| Predictor
    UI -->|Customer Inputs| Explainer
    Predictor -->|Risk Score & Probability| UI
    Explainer -->|Aggregated Feature Contributions| UI
```

## Features

- **Automated Data Ingestion**: Secure retrieval from Amazon S3.
- **Dataset Validation & Preprocessing**: Automated schema integrity checks, type coercion, and deduplication.
- **Feature Engineering**: Robust scikit-learn preprocessing pipeline featuring `ColumnTransformer`, `StandardScaler`, and `OneHotEncoder`.
- **Model Benchmarking**: Multi-model comparison across Logistic Regression, Random Forest, and XGBoost.
- **Comprehensive Evaluation**: Benchmarking with Precision, Recall, F1, ROC-AUC, and PR-AUC.
- **Aggregated SHAP Explainability**: Dynamic feature attribution that maps one-hot encoded variables back to original features, eliminating duplicate rows.
- **Interactive Streamlit Interface**: Real-time customer churn risk scoring and side-by-side positive/negative factor analysis.

## Models Comparison

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.504** | **0.783** | **0.614** | **0.841** | **0.633** |
| Random Forest | 0.617 | 0.465 | 0.531 | 0.822 | 0.609 |
| XGBoost | 0.660 | 0.508 | 0.574 | 0.842 | 0.654 |

*Logistic Regression is currently selected as the best deployment model based on F1 score.*

## Project Structure

```text
ChurnShield/
├── src/
│   ├── data_ingestion.py        # S3 data ingestion
│   ├── data_validation.py       # Data verification & cleaning
│   ├── data_transformation.py   # Scikit-learn preprocessing pipelines
│   ├── model_training.py        # Multi-model training and selection
│   ├── prediction.py            # Real-time inference service
│   └── explainability.py        # SHAP explainability engine
├── artifacts/                   # Local raw & validated datasets
├── models/                      # Serialized trained model pipeline
├── app.py                       # Streamlit web application
├── requirements.txt             # Project dependencies
├── .env                         # Environment configurations
├── .gitignore
└── README.md
```

## Setup & Installation

1. Clone repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure AWS credentials using your local AWS configuration or environment variables.

3. Run the end-to-end pipeline:

```bash
python src/data_ingestion.py
python src/data_validation.py
python src/model_training.py
```

4. Launch the interactive dashboard:

```bash
streamlit run app.py
```

## Dataset

The project uses the IBM Telco Customer Churn dataset containing customer demographic, service subscriptions, contract terms, and billing information.

- **Target variable**: `Churn` (1 = Customer left, 0 = Customer stayed)

## Explainability

SHAP (SHapley Additive exPlanations) is used to explain individual predictions by computing the exact marginal contribution of each customer characteristic toward increasing or decreasing the model's estimated churn probability.

- Feature attributions are aggregated back to original customer variables to eliminate redundant one-hot encoding entries.
- Descriptions provide contextual analysis grounded in the customer's actual profile values.
- SHAP explanations describe model behavior and should not be interpreted as proof of causal relationships.

## Tech Stack

Python, Pandas, Scikit-learn, XGBoost, SHAP, Boto3, Joblib, Streamlit, Amazon S3
