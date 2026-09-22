import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

# Load processed dataset
df = pd.read_csv("data/processed/processed_dataset.csv")

# Separate features and target
X = df.drop(columns=["appointment_outcome"])
y = df["appointment_outcome"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    cm = confusion_matrix(y_test, y_pred)
    fp = cm[0,1]
    fn = cm[1,0]
    
    return {"Accuracy": acc, "Recall": rec, "ROC AUC": auc, "FP": fp, "FN": fn}

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    results[name] = metrics
    print(f"{name} Results: {metrics}")

# Save models for reuse
joblib.dump(models["Logistic Regression"], "models/logistic_regression.pkl")
joblib.dump(models["Random Forest"], "models/random_forest.pkl")
joblib.dump(models["XGBoost"], "models/xgboost.pkl")

print("\n✅ All models trained and saved.")
