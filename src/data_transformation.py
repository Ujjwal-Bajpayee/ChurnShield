import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Churn"
DROP_COLUMNS = ["customerID"]

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0.0)

        tenure = X["tenure"].astype(float) if "tenure" in X.columns else pd.Series(0.0, index=X.index)
        monthly = X["MonthlyCharges"].astype(float) if "MonthlyCharges" in X.columns else pd.Series(0.0, index=X.index)
        total = X["TotalCharges"].astype(float) if "TotalCharges" in X.columns else pd.Series(0.0, index=X.index)

        X["AvgMonthlySpend"] = total / (tenure + 1.0)
        X["ChargeRatio"] = monthly / (X["AvgMonthlySpend"] + 1.0)

        services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        X["ServiceCount"] = sum((X[s] == "Yes").astype(int) for s in services if s in X.columns)
        X["CostPerService"] = monthly / (X["ServiceCount"] + 1.0)

        expected_total = monthly * tenure
        X["DiscountRatio"] = (total + 1.0) / (expected_total + 1.0)

        contract = X["Contract"] if "Contract" in X.columns else pd.Series("", index=X.index)
        internet = X["InternetService"] if "InternetService" in X.columns else pd.Series("", index=X.index)
        payment = X["PaymentMethod"] if "PaymentMethod" in X.columns else pd.Series("", index=X.index)

        X["IsHighRiskTriad"] = ((contract == "Month-to-month") & (internet == "Fiber optic") & (payment == "Electronic check")).astype(int)
        X["IsShortTenureMonthToMonth"] = ((contract == "Month-to-month") & (tenure <= 12)).astype(int)

        streaming_tv = X["StreamingTV"] if "StreamingTV" in X.columns else pd.Series("No", index=X.index)
        streaming_movies = X["StreamingMovies"] if "StreamingMovies" in X.columns else pd.Series("No", index=X.index)
        X["HasStreaming"] = ((streaming_tv == "Yes") | (streaming_movies == "Yes")).astype(int)

        tenure_bins = [-1, 6, 12, 24, 48, 100]
        tenure_labels = ["0-6m", "6-12m", "1-2y", "2-4y", "4y+"]
        X["TenureCohort"] = pd.cut(tenure, bins=tenure_bins, labels=tenure_labels).astype(str)

        return X

def get_preprocessor():
    numerical_features = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "AvgMonthlySpend",
        "ChargeRatio",
        "ServiceCount",
        "CostPerService",
        "DiscountRatio",
        "IsHighRiskTriad",
        "IsShortTenureMonthToMonth",
        "HasStreaming"
    ]

    categorical_features = [
        "gender",
        "Partner",
        "Dependents",
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
        "TenureCohort"
    ]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    column_transformer = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("transformer", column_transformer)
    ])

def prepare_features(df):
    df = df.copy()
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
        y = df[TARGET_COLUMN]
        X = df.drop(columns=[TARGET_COLUMN] + [c for c in DROP_COLUMNS if c in df.columns])
    else:
        y = None
        X = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")

    preprocessor = get_preprocessor()
    return X, y, preprocessor