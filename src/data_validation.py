import pandas as pd
from pathlib import Path

RAW_DATA = Path("artifacts/raw_data.csv")
VALIDATED_DATA = Path("artifacts/validated_data.csv")

REQUIRED_COLUMNS = [
    "customerID",
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
    "Churn"
]

def validate_data():
    if not RAW_DATA.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Run data_ingestion.py first."
        )

    df = pd.read_csv(RAW_DATA)
    print(f"Dataset shape: {df.shape}")

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if df["Churn"].isnull().any():
        raise ValueError("Target column contains missing values.")

    duplicates = df["customerID"].duplicated().sum()
    print(f"Duplicate customer IDs: {duplicates}")

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    print(
        f"Missing TotalCharges: "
        f"{df['TotalCharges'].isna().sum()}"
    )

    VALIDATED_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(VALIDATED_DATA, index=False)

    print("Data validation completed.")
    print(f"Validated data saved to: {VALIDATED_DATA}")

if __name__ == "__main__":
    validate_data()