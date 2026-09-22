import boto3
from pathlib import Path

BUCKET_NAME = "churn-shield"
OBJECT_KEY = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

OUTPUT_PATH = Path("artifacts/raw_data.csv")

def download_data():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")

    print("Downloading dataset from S3...")

    s3.download_file(
        BUCKET_NAME,
        OBJECT_KEY,
        str(OUTPUT_PATH)
    )

    print(f"Dataset downloaded to: {OUTPUT_PATH}")

if __name__ == "__main__":
    download_data()