import pandas as pd
import yaml
import os
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import train_test_split
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

# === Random Forest Tuning ===
rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=config["random_state"]),
    rf_param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1
)

print("Tuning Random Forest...")
rf_grid.fit(X_train, y_train)
rf_model = rf_grid.best_estimator_
print("Best RF params:", rf_grid.best_params_)
joblib.dump(rf_model, os.path.join(models_dir, "random_forest_tuned.pkl"))

# === XGBoost Tuning ===
xgb_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

xgb_model = XGBClassifier(
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    random_state=config["random_state"],
    n_jobs=-1
)

xgb_search = RandomizedSearchCV(
    xgb_model,
    xgb_param_grid,
    n_iter=20,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=config["random_state"]
)

print("Tuning XGBoost...")
xgb_search.fit(X_train, y_train)
xgb_model = xgb_search.best_estimator_
print("Best XGB params:", xgb_search.best_params_)
joblib.dump(xgb_model, os.path.join(models_dir, "xgboost_tuned.pkl"))

print("Tuning complete. Best models saved.")
