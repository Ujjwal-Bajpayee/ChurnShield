# ChurnShield

An end-to-end customer churn prediction system that predicts churn risk and provides interpretable explanations for individual predictions using SHAP.

## Overview

ChurnShield is built around a simple production-oriented machine learning pipeline:

**S3 → Data Ingestion → Data Validation → Preprocessing → Model Training → Evaluation → SHAP Explainability → Streamlit**

The project compares multiple classification models and exposes the final prediction through an interactive Streamlit application.

## Features

- Data ingestion from Amazon S3
- Dataset validation and preprocessing
- One-hot encoding and numerical feature scaling
- Comparison of Logistic Regression, Random Forest, and XGBoost
- Evaluation using Precision, Recall, F1, ROC-AUC, and PR-AUC
- SHAP-based individual prediction explanations
- Interactive Streamlit prediction interface

## Models

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.504 | 0.783 | 0.614 | 0.841 | 0.633 |
| Random Forest | 0.617 | 0.465 | 0.531 | 0.822 | 0.609 |
| XGBoost | 0.660 | 0.508 | 0.574 | 0.842 | 0.654 |

Logistic Regression is selected as the current model based on F1 score.

## Project Structure

```text
ChurnShield/
├── src/
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── data_transformation.py
│   ├── model_training.py
│   ├── prediction.py
│   └── explainability.py
├── artifacts/
├── models/
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Configure AWS credentials using your local AWS configuration or environment variables.

The application expects the Telco Customer Churn dataset in the configured S3 bucket.

## Run the Pipeline

```bash
python src/data_ingestion.py
python src/data_validation.py
cd src
python model_training.py
cd ..
```

Start the Streamlit application:

```bash
streamlit run app.py
```

## Dataset

The project uses the IBM Telco Customer Churn dataset containing customer demographic, service, contract, and billing information.

Target variable:

`Churn`

## Explainability

SHAP is used to explain individual predictions by identifying the customer characteristics that contributed to increasing or decreasing the model's estimated churn probability.

SHAP explanations describe model behavior and should not be interpreted as proof of causal relationships.

## Tech Stack

Python, Pandas, Scikit-learn, XGBoost, SHAP, Boto3, Joblib, Streamlit, Amazon S3

## Future Improvements

- Threshold optimization based on business costs
- Model monitoring
- Experiment tracking
- Automated model retraining
- Cloud deployment
