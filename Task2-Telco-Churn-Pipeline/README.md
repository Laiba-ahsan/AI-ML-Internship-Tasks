# Telco Customer Churn Prediction — ML Pipeline

> **DevelopersHub Corporation | AI/ML Engineering Advanced Internship | Task 2**

---

## Objective

Build a reusable, production-ready machine learning pipeline that predicts whether a telecom customer will churn (leave the service) based on their account information and usage patterns.

---

## Dataset

| Detail | Info |
|---|---|
| Name | Telco Customer Churn |
| Source | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Records | 7,043 customers |
| Features | 20 input features + 1 target |
| Target | `Churn` — Yes (1) / No (0) |
| Class Balance | ~73% No Churn / ~27% Churn (imbalanced) |

---

## Methodology & Approach

### 1. Data Preprocessing
- Removed `customerID` (non-predictive identifier)
- Fixed `TotalCharges` column — blank strings converted to numeric
- Encoded all categorical columns using `LabelEncoder`
- Target variable encoded: `Yes → 1`, `No → 0`

### 2. Scikit-learn Pipeline
Each model was wrapped in a `Pipeline` with three steps:
```
SimpleImputer → StandardScaler → Classifier
```
This ensures no data leakage and makes the model directly exportable for deployment.

### 3. Models Trained
- **Logistic Regression** — fast, interpretable baseline
- **Random Forest** — ensemble model, handles non-linearity better

### 4. Hyperparameter Tuning
- Used `GridSearchCV` with 5-fold cross-validation
- Tuned: `n_estimators`, `max_depth`, `min_samples_split`
- Scoring metric: F1-score (better than accuracy for imbalanced data)

### 5. Model Export
- Final tuned pipeline saved using `joblib` as `churn_pipeline.pkl`
- Verified by reloading and running predictions on test data

---

## Key Results

| Model | Accuracy | F1-Score |
|---|---|---|
| Logistic Regression | ~80% | ~57% |
| Random Forest (Default) | ~79% | ~55% |
| Random Forest (Tuned) | ~81% | ~59% |

> **Best Model:** Tuned Random Forest (GridSearchCV)

---

## Key Insights

1. **Contract type** is the strongest churn predictor — month-to-month customers churn at 3x the rate of two-year contract customers
2. **Tenure** is inversely related to churn — newer customers are far more likely to leave
3. **High monthly charges** significantly increase churn probability
4. Customers without **tech support or online security** services churn more
5. The **Pipeline approach** prevents data leakage and makes the model production-ready out of the box

---

## Visualizations Included

- Churn class distribution (bar + pie chart)
- Churn rate by contract type, payment method, internet service, gender
- Numeric feature distributions by churn status
- Feature correlation heatmap
- Confusion matrices for all three models
- ROC curves with AUC scores
- Feature importance chart
- Model accuracy comparison

---

## Tools & Libraries

`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Seaborn` `Joblib`

---

## Project Structure

```
telco-churn-ml-pipeline/
│
├── telco_churn_pipeline.ipynb     # Main Jupyter Notebook
├── churn_pipeline.pkl             # Exported trained pipeline
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset
└── README.md                      # This file
```

---

## How to Run

1. Clone this repository
2. Place the dataset CSV in the same folder as the notebook
3. Open `telco_churn_pipeline.ipynb` in Jupyter Notebook
4. Run all cells top to bottom
5. The trained pipeline will be saved as `churn_pipeline.pkl`
