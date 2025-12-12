📡 Telco Customer Churn Prediction

*End-to-End Machine Learning Project with SHAP Explainability & Streamlit Deployment*

📘 Overview

Customer churn is one of the most significant challenges in the telecommunications industry.
This project predicts whether a customer is likely to churn using machine learning and provides transparent explanations using SHAP.
The final deployed application includes:
1. 🔍 Single-customer churn prediction
2. 📂 Batch predictions (CSV upload)
3. 📊 SHAP waterfall plots for per-customer explanation
4. 🎨 Clean & interactive UI
5. ☁️ Deployed on Streamlit Cloud
This is a complete end-to-end ML system designed for real-world usability.

🎯 Project Goals

1. Build an ML model to predict customer churn
2. Understand key drivers of churn
3. Provide transparent model explanations
4. Package the model into an interactive web application
5. Deploy it for public use

🧠 Machine Learning Workflow

1️⃣ Data Preprocessing
- Handling missing values
- Encoding categorical variables (OneHotEncoder)
- Scaling numerical features (StandardScaler)
- Train-test splitting
- Building a reproducible preprocessing pipeline

2️⃣ Feature Engineering

- Created new features to improve model learning:
- TotalServices
- ContractMonths
- SpendingRate
- TenureGroup

3️⃣ Model Training

Models compared:

Logistic Regression - Random Forest - XGBoost

Random Forest selected due to:

- High ROC-AUC
- Balanced performance
- Robustness with tabular data

Saved as:
telco_final_model.joblib

4️⃣ Evaluation Metrics
| Metric    | Description                            |
| --------- | -------------------------------------- |
| Accuracy  | Overall correctness                    |
| Precision | How many predicted churns were correct |
| Recall    | How many actual churns were detected   |
| F1 Score  | Balance of precision & recall          |
| ROC-AUC   | Ability to separate churn vs non-churn |

Final model achieved ROC-AUC ≈ 0.82.

🧩 SHAP Explainability

SHAP (SHapley Additive exPlanations) explains why a prediction was made.
🔍 Example Insights:
- Month-to-month contracts → major churn driver
- High monthly charges → increases churn
- Low tenure → strong sign of churn
- More services → reduces churn
- Two-year contracts → low churn probability

The Streamlit app uses SHAP waterfall plots for every prediction.

🖥️ Streamlit Web Application

The app has two main modes:

⭐ 1. Single Customer Prediction

- Enter customer details manually
- View churn probability
- View final model prediction
- SHAP waterfall explanation

⭐ 2. Batch Prediction (CSV upload)

- Upload multiple customer records
- Get predictions + probabilities
- Download results as CSV

📂 Project Structure

├── streamlit_app.py               # Complete Streamlit application

├── telco_final_model.joblib       # Saved Random Forest model

├── preprocessor.joblib            # Saved preprocessing pipeline

├── model_metadata.joblib          # Contains numerical/categorical feature names

├── requirements.txt               # Dependencies

└── README.md                      # Project documentation

🧰 Tech Stack

| Component       | Technology                |
| --------------- | ------------------------- |
| Language        | Python                    |
| Data Processing | Pandas, NumPy             |
| ML Model        | Scikit-Learn RandomForest |
| Explainability  | SHAP                      |
| UI              | Streamlit                 |
| Deployment      | Streamlit Cloud           |
| Serialization   | Joblib                    |

🧪 Example Customer Inputs for Testing

| Tenure | Contract       | Monthly Charges | Services | Expected    |
| ------ | -------------- | --------------- | -------- | ----------- |
| 60     | Two year       | 45              | 7        | Not churn   |
| 3      | Month-to-month | 95              | 1        | Churn       |
| 15     | Month-to-month | 70              | 4        | Medium risk |
