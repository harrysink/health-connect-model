import pandas as pd
import os
from preprocessing import preprocess_pipeline

# Paths
raw_path = "data/raw/Cleaned_Data.csv"
processed_path = "data/processed/processed_dataset.csv"
snapshot_dir = "data/test_snapshots"
os.makedirs(snapshot_dir, exist_ok=True)

# === Step 1: Load raw dataset and save snapshot ===
raw_df = pd.read_csv(raw_path)
raw_df.head(10).to_csv(os.path.join(snapshot_dir, "raw_sample.csv"), index=False)
print("✅ Raw snapshot saved:", os.path.join(snapshot_dir, "raw_sample.csv"))

# === Step 2: Run preprocessing pipeline ===
preprocess_pipeline(raw_path, "config.yaml")

# === Step 3: Load processed dataset and save snapshot ===
processed_df = pd.read_csv(processed_path)
processed_df.head(10).to_csv(os.path.join(snapshot_dir, "processed_sample.csv"), index=False)
print("✅ Processed snapshot saved:", os.path.join(snapshot_dir, "processed_sample.csv"))

# === Step 4: Verify engineered features exist ===
engineered_features = ["no_show_rate", "is_weekend", "is_urgent"]
missing_features = [f for f in engineered_features if f not in processed_df.columns]

if missing_features:
    print("⚠️ Missing engineered features:", missing_features)
else:
    print("✅ All engineered features present:", engineered_features)
