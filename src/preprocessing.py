import os
import logging
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

logging.basicConfig(level=logging.INFO)

def add_engineered_features(df):
    """Add custom engineered features to the dataset."""

    # 1. No-show rate per patient (based on previous appointments vs no-shows)
    if "previous_appointments" in df.columns and "previous_no_shows" in df.columns:
        df["no_show_rate"] = df["previous_no_shows"] / df["previous_appointments"].replace(0, 1)

    # 2. Weekend flag (based on day_num: 0=Monday … 6=Sunday)
    if "day_num" in df.columns:
        df["is_weekend"] = df["day_num"].apply(lambda x: 1 if x in [5, 6] else 0)

    # 3. Urgent flag (lead days <= 2)
    if "booking_lead_days" in df.columns:
        df["is_urgent"] = df["booking_lead_days"].apply(lambda x: 1 if pd.notnull(x) and x <= 2 else 0)

    return df

def preprocess_pipeline(data_path, config_path):
    """Run preprocessing pipeline on raw dataset."""
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logging.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    logging.info(f"Initial dataset shape: {df.shape}")

    # Drop leakage columns if present
    leakage_cols = ["appointment_id", "scheduled_day"]
    df = df.drop(columns=[col for col in leakage_cols if col in df.columns], errors="ignore")
    logging.info("Leakage columns dropped if present")

    # Ensure target column is named correctly
    if "target" in df.columns:
        df.rename(columns={"target": "appointment_outcome"}, inplace=True)
        logging.info("Renamed 'target' column to 'appointment_outcome'")

    # Drop rows with missing target values
    if "appointment_outcome" in df.columns:
        df = df.dropna(subset=["appointment_outcome"])
        logging.info("Dropped rows with missing target values")

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "appointment_outcome" in numeric_cols:
        numeric_cols.remove("appointment_outcome")

    # Build preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_transformer, numeric_cols)],
        remainder="passthrough"
    )

    logging.info("Fitting preprocessing pipeline")
    processed_array = preprocessor.fit_transform(df)

    # Build dataframe with transformed numeric + passthrough
    processed_df = pd.DataFrame(
        processed_array,
        columns=numeric_cols + [col for col in df.columns if col not in numeric_cols]
    )

    # 👉 Add engineered features using the ORIGINAL df
    engineered_df = add_engineered_features(df)
    for col in ["no_show_rate", "is_weekend", "is_urgent"]:
        if col in engineered_df.columns:
            processed_df[col] = engineered_df[col]

    # Ensure processed folder exists
    os.makedirs(os.path.dirname(config["processed_data_path"]), exist_ok=True)

    # Save processed dataset
    processed_df.to_csv(config["processed_data_path"], index=False)
    logging.info(f"Processed dataset saved to {config['processed_data_path']}")

    # Save pipeline
    os.makedirs(os.path.dirname(config["pipeline_path"]), exist_ok=True)
    joblib.dump(preprocessor, config["pipeline_path"])
    logging.info(f"Preprocessing pipeline saved to {config['pipeline_path']}")

    return processed_df

if __name__ == "__main__":
    preprocess_pipeline("data/raw/Cleaned_Data.csv", "config.yaml")
