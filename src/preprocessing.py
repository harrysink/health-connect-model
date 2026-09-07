import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

def preprocess_pipeline(data_path, config_path="config.yaml"):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load dataset
    df = pd.read_csv(data_path)

    # === Handle date columns ===
    if "appointment_date" in df.columns and "booking_date" in df.columns:
        df["appointment_date"] = pd.to_datetime(df["appointment_date"], errors="coerce")
        df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")

        # Create numeric feature: lead time in days
        df["lead_time_days"] = (df["appointment_date"] - df["booking_date"]).dt.days

        # Optional: extract day of week and month
        df["appointment_dayofweek"] = df["appointment_date"].dt.dayofweek
        df["appointment_month"] = df["appointment_date"].dt.month

        # Drop original date columns
        df = df.drop(columns=["appointment_date", "booking_date"])

    # === Drop rows with missing target ===
    df = df.dropna(subset=["appointment_outcome"])

    # Separate features and target
    X = df.drop(columns=["appointment_outcome"])
    y = df["appointment_outcome"]

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Preprocessing transformers with imputers
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_cols),
            ("numeric", numeric_transformer, numeric_cols),
        ]
    )

    # Build pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    # Fit and transform
    X_processed = pipeline.fit_transform(X)

    # Save processed dataset
    processed_df = pd.DataFrame(
        X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
    )
    processed_df["appointment_outcome"] = y.values

    # Ensure processed data path exists
    os.makedirs(os.path.dirname(config["processed_data_path"]), exist_ok=True)
    processed_df.to_csv(config["processed_data_path"], index=False)

    # Save pipeline for reuse
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    pipeline_path = os.path.join(models_dir, "preprocessing_pipeline.pkl")
    joblib.dump(pipeline, pipeline_path)

    return processed_df

if __name__ == "__main__":
    # Resolve project root automatically
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    preprocess_pipeline(config["data_path"], config_path=config_path)
