📡 Telco Customer Churn Prediction App

An end-to-end Machine Learning project with Streamlit + SHAP Explainability

---

🚀 Overview

This project predicts whether a telecom customer is likely to churn (leave the service provider).
It is built as a full end-to-end ML system, including:

* Data cleaning & preprocessing
* Feature engineering
* Model training
* Evaluation
* Explainability using **SHAP**
* Interactive web app using **Streamlit**

The final deployed app allows users to:

✔ Enter customer details  
✔ Get churn probability & prediction  
✔ View **SHAP waterfall plots** showing feature impact  
✔ Upload CSVs for batch predictions

---

🎯 Goal

To build a real-world machine learning solution that helps telecom companies identify high-risk customers early and reduce churn.

---

🧠 Machine Learning Workflow

1. Data Preprocessing

* Missing value handling
* Numerical scaling
* One-hot encoding of categorical variables
* Feature engineering:
  * `TotalServices`
  * `SpendingRate`
  * `ContractMonths`
  * `TenureGroup`

A full preprocessing pipeline was saved as:
preprocessor.joblib

---

2. Model Training**

Several models were compared:

| Model                     | F1 Score  | ROC-AUC   |
| ------------------------- | --------- | --------- |
| Logistic Regression       | ~0.61     | ~0.84     |
| XGBoost                   | ~0.58     | ~0.81     |
| Random Forest (final)     | ~0.53     | ~0.82     |

The Random Forest model with class balancing and hyperparameter tuning was selected.

Final model saved as:
telco_final_model.joblib

---

3. Explainability with SHAP**

SHAP (SHapley Additive exPlanations) is used to:

* Explain each prediction
* Show which features increase churn risk
* Visualize feature contribution using waterfall plots

This makes the model interpretable and business-friendly.

---

🖥️ Streamlit Web App

The app includes:

🔍 Single Prediction Mode**

* Input customer details
* Model outputs:
  * Churn Probability
  * Will Churn / Will Not Churn
* SHAP waterfall plot to explain the prediction

📂 Batch Prediction Mode

* Upload a CSV file
* Receive predictions for all customers
* Download the results

---

🧩 Project Structure

├── streamlit_app.py               # Streamlit application
├── telco_final_model.joblib       # Trained Random Forest model
├── preprocessor.joblib            # Preprocessing pipeline
├── model_metadata.joblib          # Feature metadata
├── requirements.txt               # Dependencies for deployment
└── README.md                      # Project documentation

---

🚀 Deployment

The app is deployed on Streamlit Cloud.

🔗 Live App: *[Add your URL here]*  
🔗 GitHub Repo: *[Add link here]*

To deploy your own version:

1. Push your files to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Select your repository
4. Set main file → `streamlit_app.py`
5. Deploy 🚀

---

📦 Installation (Run Locally)

git clone <repo-url>
cd telco-churn-predictor
pip install -r requirements.txt
streamlit run streamlit_app.py

---

🧪 Example Usage

Sample Inputs for Testing:

| Tenure | Contract       | Monthly Charges | Services | Expected Churn?  |
| ------ | -------------- | --------------- | -------- | ---------------  |
| 60     | Two year       | 45              | 7        | ❌ No           |
| 3      | Month-to-month | 95              | 1        | ✔️ Yes          |
| 15     | Month-to-month | 70              | 4        | ⚠️ Medium       |

---

📊 SHAP Insights

SHAP explanations reveal:

* Month-to-month contract → increases churn
* Low tenure → strongest churn indicator
* High monthly charges → increases risk
* More services → decreases churn
* Long-term contracts → reduce churn

This helps business teams understand why a customer may leave.

---

🛠️ Technologies Used

* **Python**
* **Pandas**, **NumPy**
* **Scikit-Learn**
* **RandomForestClassifier**
* **SHAP**
* **Matplotlib**
* **Streamlit**

---

🌟 Key Features

✔ End-to-end ML workflow  
✔ Interactive Streamlit UI  
✔ SHAP-powered explainability  
✔ Batch predictions  
✔ Clean modular code  
✔ Deployment-ready

-------------------------