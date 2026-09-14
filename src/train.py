import pandas as pd
import yaml
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# === Load config ===
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(project_root, "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# === Load processed dataset ===
df = pd.read_csv(config["processed_data_path"])

# Separate features and target
X = df.drop(columns=["appointment_outcome"])
y = df["appointment_outcome"]

# === Stratified Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config["test_size"],
    random_state=config["random_state"],
    stratify=y
)

# === Handle class imbalance with SMOTE ===
smote = SMOTE(random_state=config["random_state"])
X_train, y_train = smote.fit_resample(X_train, y_train)

models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

# === Train Logistic Regression ===
log_model = LogisticRegression(class_weight="balanced", max_iter=500)
log_model.fit(X_train, y_train)
joblib.dump(log_model, os.path.join(models_dir, "logistic_regression.pkl"))

# === Train Random Forest ===
rf_model = RandomForestClassifier(class_weight="balanced", random_state=config["random_state"])
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, os.path.join(models_dir, "random_forest.pkl"))

# === Train XGBoost ===
xgb_model = XGBClassifier(
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    random_state=config["random_state"]
)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, os.path.join(models_dir, "xgboost.pkl"))

print("Models trained and saved successfully.")
