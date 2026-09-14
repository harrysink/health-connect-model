import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

def preprocess_pipeline(data_path, config_path="config.yaml"):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load dataset (already cleaned/encoded by Data Science track)
    df = pd.read_csv(data_path)

    # Drop leakage columns if present
    leakage_cols = ["reminder_sent", "reminder_channel", "waiting_time_minutes"]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors="ignore")

    # Ensure target column exists
    if "target" not in df.columns:
        raise KeyError("Expected 'target' column in dataset")

    # Drop rows with missing target
    df = df.dropna(subset=["target"])

    # Rename target column for consistency
    df.rename(columns={"target": "appointment_outcome"}, inplace=True)

    # Separate features and target
    X = df.drop(columns=["appointment_outcome"])
    y = df["appointment_outcome"]

    # Identify numeric columns only (categoricals already encoded)
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Preprocessing transformers
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_transformer, numeric_cols)],
        remainder="passthrough"  # keep engineered/encoded features
    )

    # Build pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    # Fit and transform
    X_processed = pipeline.fit_transform(X)

    # Save processed dataset
    processed_df = pd.DataFrame(X_processed)
    processed_df["appointment_outcome"] = y.values

    os.makedirs(os.path.dirname(config["processed_data_path"]), exist_ok=True)
    processed_df.to_csv(config["processed_data_path"], index=False)

    # Save pipeline