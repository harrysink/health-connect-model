import pandas as pd
import yaml
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

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

# === Load saved models ===
models_dir = os.path.join(project_root, "models")
log_model = joblib.load(os.path.join(models_dir, "logistic_regression.pkl"))
rf_model = joblib.load(os.path.join(models_dir, "random_forest.pkl"))

def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    print(f"\n=== {name} Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="weighted", zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, average="weighted", zero_division=0))
    print("F1 Score:", f1_score(y_test, y_pred, average="weighted", zero_division=0))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_,
                yticklabels=model.classes_)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Save PNG for reproducibility
    filename = f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(filename, bbox_inches="tight")

    # Show inline if running in a notebook
    try:
        get_ipython()  # will raise NameError if not in notebook
        plt.show()
    except NameError:
        plt.close()

# === Evaluate both models ===
evaluate_model(log_model, X_test, y_test, "Logistic Regression")
evaluate_model(rf_model, X_test, y_test, "Random Forest")
