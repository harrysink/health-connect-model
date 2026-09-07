# HealthConnect ML Engineering – Week 5

## 📌 Project Overview
HealthConnect Clinic is facing high rates of missed appointments.  
This project develops a machine learning pipeline to predict patient no-shows, enabling proactive interventions such as reminders, rescheduling, or targeted support.

- **Problem:** Predict whether a patient will attend or miss their appointment.  
- **Target:** Binary classification → Attended (1) vs No-Show (0).  
- **Inputs:** Patient demographics, appointment details, booking info, history, reminders, distance, waiting time.  
- **Outputs:** Baseline predictive models + evaluation metrics.  

---

## 📂 Repository Structure

/data
   HealthConnect_Appointment_Data.csv   # Raw dataset
   /processed
       appointments_clean.csv           # Processed dataset

/models
   logistic_regression.pkl              # Saved Logistic Regression model
   random_forest.pkl                    # Saved RandomForest model
   xgboost.pkl                          # Placeholder for Week 6

/src
   preprocessing.py                     # Data cleaning, encoding, scaling
   train.py                             # Train baseline models
   evaluate.py                          # Evaluate saved models

requirements.txt                        # Dependencies
config.yaml                             # Configurations
README.md                               # Documentation


---

## ⚙️ Setup Instructions
1. Clone the repository and navigate to the project folder.  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run preprocessing:
   python src/preprocessing.py
4. Train models:
   python src/train.py
5. Evaluate models:
   python src/evaluate.py

## 🛠️ Deliverables (Week 5)
Preprocessing Workflow: Modular pipeline for cleaning, encoding, scaling, saving dataset.

Baseline Models: Logistic Regression + RandomForest trained and saved.

Evaluation: Metrics (Accuracy, Precision, Recall, F1, Classification Report).

Configuration: config.yaml for paths, parameters, reproducibility.

Dependencies: requirements.txt for environment setup.

Documentation: README, code comments, risk review.


---

## 📊 Key Findings
Logistic Regression provides a simple, interpretable baseline.

RandomForest captures non-linear relationships and improves recall for No-Shows.

Preprocessing pipeline ensures reproducibility and consistent feature handling.


---

## ⚠️ Assumptions, Limitations, Risks & Dependencies
Relevant: Synthetic dataset, class imbalance, reminder data dependency, fairness concerns.

Resolved: Repo structure, reproducibility, modular workflow.

Changed: Expanded baseline models earlier than planned, partial feature engineering.

New Issues: Dropping “Cancelled” reduced dataset size, categorical encoding expanded feature space, RandomForest training time longer.

Impact: Imbalance may reduce recall, smaller dataset affects robustness, longer training slows iteration.

Mitigation: Apply balancing techniques, optimize encoding, add XGBoost, tune hyperparameters.


---

## 🚀 Next Steps (Week 6)
Add XGBoost baseline model.

Address class imbalance (SMOTE, class weights).

Improve feature engineering (ratios, interaction terms).

Hyperparameter tuning and optimization.

Feature importance analysis.

Prepare deployment-ready pipeline (API/dashboard integration).