import pandas as pd
import yaml
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

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

# === Train Logistic Regression ===
log_model = LogisticRegression(class_weight="balanced", max_iter=500)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("=== Logistic Regression Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log, average="weighted", zero_division=0))
print("Recall:", recall_score(y_test, y_pred_log, average="weighted", zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_log, average="weighted", zero_division=0))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log, zero_division=0))

# Save Logistic Regression model
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)
joblib.dump(log_model, os.path.join(models_dir, "logistic_regression.pkl"))

# === Train Random Forest ===
rf_model = RandomForestClassifier(class_weight="balanced", random_state=config["random_state"])
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n=== Random Forest Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, average="weighted", zero_division=0))
print("Recall:", recall_score(y_test, y_pred_rf, average="weighted", zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_rf, average="weighted", zero_division=0))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf, zero_division=0))

# Save Random Forest model
joblib.dump(rf_model, os.path.join(models_dir, "random_forest.pkl"))
