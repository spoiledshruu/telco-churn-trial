import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

# --------------------------------------------------------
# Load Model + Preprocessor + Metadata
# --------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("telco_final_model.joblib")
    preprocessor = joblib.load("preprocessor.joblib")
    metadata = joblib.load("model_metadata.joblib")
    return model, preprocessor, metadata["num_features"], metadata["cat_features"]

model, preprocessor, num_features, cat_features = load_artifacts()

classifier = model.named_steps["clf"]
explainer = shap.TreeExplainer(classifier)

# --------------------------------------------------------
# UI
# --------------------------------------------------------
st.title("📡 Telco Customer Churn Predictor")
st.write("Predict churn & view SHAP explanations.")

mode = st.sidebar.selectbox("Choose Mode", ["Single Prediction", "Batch Prediction"])

# --------------------------------------------------------
# SINGLE PREDICTION MODE
# --------------------------------------------------------
if mode == "Single Prediction":
    st.header("🔍 Single Customer Prediction")

    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    tenure = st.number_input("Tenure (Months)", 0, 72, 12)
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    TotalServices = st.slider("Total Services Used", 0, 9, 4)

    ContractMonths = 1 if Contract == "Month-to-month" else (12 if Contract == "One year" else 24)
    SpendingRate = MonthlyCharges / (tenure + 1)
    TenureGroup = (
        "0–1 yr" if tenure <= 12 else
        "1–2 yrs" if tenure <= 24 else
        "2–4 yrs" if tenure <= 48 else
        "4–5 yrs" if tenure <= 60 else
        "5–6 yrs"
    )

    # Build customer data
    user = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": MonthlyCharges * tenure,
        "TotalServices": TotalServices,
        "Contract": Contract,
        "InternetService": InternetService,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "Partner": "No",
        "Dependents": "No",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "ContractMonths": ContractMonths,
        "SpendingRate": SpendingRate,
        "TenureGroup": TenureGroup
    }])

    transformed = preprocessor.transform(user)

    prob = model.predict_proba(user)[0][1]
    pred = model.predict(user)[0]

    st.subheader("📌 Prediction")
    st.write(f"**Churn Probability:** {prob:.2f}")
    st.write(f"**Prediction:** {'🔴 Will Churn' if pred == 1 else '🟢 Will Not Churn'}")

    # --------------------------------------------------------
    # SHAP EXPLANATION — FINAL FIXED VERSION
    # --------------------------------------------------------
    st.subheader("📊 SHAP Explanation")

    # Try unified API first
    try:
        shap_output = explainer(transformed)
        shap_values = np.array(shap_output.values)
        base_values = np.array(shap_output.base_values)
    except Exception:
        shap_output = None
        shap_values = None
        base_values = None

    # If unified API failed, use legacy API
    if shap_values is None:
        try:
            legacy = explainer.shap_values(transformed)
        except:
            st.error("SHAP could not be computed.")
            st.stop()

        # If legacy returns list → pick class 1
        if isinstance(legacy, list):
            shap_row = legacy[1][0]      # (53,)
            base_value = explainer.expected_value[1]
        else:
            arr = np.array(legacy)
            if arr.ndim == 3 and arr.shape[2] == 2:
                shap_row = arr[0, :, 1]  # pick class 1
            elif arr.ndim == 2:
                shap_row = arr[0]
            else:
                shap_row = arr.reshape(-1)
            base_value = (explainer.expected_value[1]
                           if isinstance(explainer.expected_value, (list, np.ndarray))
                           else explainer.expected_value)
    else:
        # unified API path
        vals = shap_values  # shape may be (1,53,2) or (1,53)
        if vals.ndim == 3 and vals.shape[2] == 2:
            shap_row = vals[0, :, 1]      # class 1
        elif vals.ndim == 2:
            shap_row = vals[0]
        else:
            shap_row = vals.reshape(-1)

        # base value
        if base_values.ndim == 2 and base_values.shape[1] >= 2:
            base_value = base_values[0, 1]
        elif base_values.ndim == 1 and len(base_values) >= 2:
            base_value = base_values[1]
        else:
            base_value = base_values.flatten()[0]

    feature_names = preprocessor.get_feature_names_out()
    exp = shap.Explanation(values=shap_row, base_values=base_value, feature_names=feature_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(exp, max_display=10)
    st.pyplot(fig)


# --------------------------------------------------------
# BATCH MODE
# --------------------------------------------------------
elif mode == "Batch Prediction":
    st.header("📂 Batch Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]

        df["Prediction"] = preds
        df["Churn Probability"] = probs

        st.write("📊 Results:")
        st.dataframe(df)

        st.download_button("Download Results", df.to_csv(index=False), "predictions.csv")
