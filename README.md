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

## Data Management
**Dataset Location**
The main dataset used in this project, appointments_clean.csv, is too large to be stored directly in the GitHub repository. To keep the repository lightweight and within GitHub’s file size limits, this file has been added to .gitignore.

**Accessing the Dataset**
You can download the dataset from Google Drive using the following link:

[Download appointments_clean.csv](https://drive.google.com/file/d/1W5ltutzV-tOoF7vzWj_jIRCI_dDle5Lp/view?usp=sharing)

After downloading, place the file in the /data directory of the repository:

Code
project-root/
│
├── data/
│   ├── appointments_clean.csv   <-- place file here
│   └── processed/               <-- processed datasets saved here
├── models/
├── src/
├── config.yaml
└── requirements.txt

**Notes**
The dataset is synthetic and anonymized for project purposes.

Scripts (preprocessing.py, train.py, evaluate.py) expect the full dataset to be present in /data.


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


---

# HealthConnect ML Engineering - Week 6 Integration Update
## 🔗 Cross‑Track Collaboration
- Received:

>1. Cleaned dataset (Cleaned_Data.csv) from the Data Science track (Mercy Koech).

>2. Engineered features: no_show_rate, is_weekend, is_urgent.

>3. Candidate baseline model (Logistic Regression) and recommendations for Random Forest/XGBoost.

- Placed in Repository:

>1. /data/raw/Cleaned_Data.csv → raw input from Data Science track.

>2. /data/processed/processed_dataset.csv → pipeline‑ready dataset after numeric scaling/imputation.

- Provided Back:

>1. Integrated ML pipeline capable of running Logistic Regression, Random Forest, and XGBoost on the cleaned dataset.

>2. Validation outputs (confusion matrices, ROC‑AUC, FP/FN counts).

>3. Updated repository structure and documentation for reproducibility.

## ⚙️ Pipeline Changes
- Preprocessing now respects engineered features from Data Science track (no duplication).

- Leakage features (reminder_sent, reminder_channel, waiting_time_minutes) removed.

- Class imbalance handled with SMOTE.

- Added XGBoost model training and evaluation.

- Evaluation extended to include ROC‑AUC and error analysis (false positives/negatives).

## 📄 Evidence of Integration
- Updated config.yaml to point to Mercy’s dataset.

- GitHub commits documenting pipeline changes and integration.

- Confusion matrix PNGs and evaluation logs saved for reproducibility.

- README updated to reflect Week 6 collaboration.

## 🚀 Next Steps (Week 7)
- Broader testing of integrated pipeline.

- Hyperparameter tuning for Random Forest/XGBoost.

- Feature importance analysis to guide clinic interventions.

- Preparation for deployment‑ready pipeline and dashboard integration.