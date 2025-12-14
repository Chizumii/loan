import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Credit Risk Prediction System",
    page_icon="💳",
    layout="wide"
)

# ===============================
# LOAD ARTIFACTS
# ===============================
@st.cache_resource
def load_artifacts():
    logreg = joblib.load("logreg_model.pkl")
    xgb = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("feature_list.pkl")
    return logreg, xgb, scaler, features

logreg_model, xgb_model, scaler, FEATURE_LIST = load_artifacts()

EXCHANGE_RATE = 16000  # USD → IDR (training consistency)

# ===============================
# PREPROCESS INPUT
# ===============================
def preprocess_input(
    annual_inc,
    loan_amount,
    term,
    int_rate,
    dti,
    home_ownership,
    emp_length
):
    data = {}

    # Numeric
    data["annual_inc"] = np.log1p(annual_inc / EXCHANGE_RATE)
    data["loan_amnt"] = loan_amount / EXCHANGE_RATE
    data["int_rate"] = int_rate
    data["dti"] = min(dti, 100)
    data["term"] = 36 if term == "36 months" else 60

    emp_map = {
        "< 1 year": 0,
        "1-3 years": 2,
        "4-7 years": 5,
        "8-10 years": 9,
        "10+ years": 10
    }
    data["emp_length"] = emp_map.get(emp_length, 0)

    # Initialize all features = 0
    for col in FEATURE_LIST:
        if col not in data:
            data[col] = 0

    # One-hot home ownership
    if f"home_ownership_{home_ownership}" in FEATURE_LIST:
        data[f"home_ownership_{home_ownership}"] = 1

    df_input = pd.DataFrame([data])[FEATURE_LIST]
    return df_input

# ===============================
# UI
# ===============================
st.title("💳 Credit Risk Prediction System")
st.markdown("Predict **loan repayment probability** using Machine Learning.")

model_choice = st.radio(
    "Choose Model:",
    ["Logistic Regression (Explainable)", "XGBoost (High Performance)"]
)

st.markdown("---")

tab1, tab2 = st.tabs(["🧑 Manual Input", "📁 Upload CSV"])

# ===============================
# TAB 1 — MANUAL INPUT
# ===============================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        annual_inc = st.number_input("Annual Income (IDR)", 0, value=120_000_000)
        loan_amount = st.number_input("Loan Amount (IDR)", 0, value=50_000_000)
        term = st.selectbox("Loan Term", ["36 months", "60 months"])

    with col2:
        int_rate = st.number_input("Interest Rate (%)", 0.0, 40.0, 11.5)
        dti = st.number_input("Debt-to-Income Ratio (%)", 0.0, 100.0, 18.0)
        home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
        emp_length = st.selectbox(
            "Employment Length",
            ["< 1 year", "1-3 years", "4-7 years", "8-10 years", "10+ years"]
        )

    if st.button("🔍 Predict Risk"):
        with st.spinner("Running model..."):
            time.sleep(0.8)

            X_input = preprocess_input(
                annual_inc,
                loan_amount,
                term,
                int_rate,
                dti,
                home_ownership,
                emp_length
            )

            X_scaled = scaler.transform(X_input)

            model = logreg_model if "Logistic" in model_choice else xgb_model
            prob_good = model.predict_proba(X_scaled)[0][1]
            prob_default = 1 - prob_good

        st.subheader("📊 Prediction Result")
        st.metric("Probability Fully Paid", f"{prob_good:.2%}")
        st.metric("Probability Default", f"{prob_default:.2%}")

        if prob_good >= 0.6:
            st.success("✅ Recommended: APPROVE")
        elif prob_good >= 0.45:
            st.warning("⚠️ Recommended: REVIEW")
        else:
            st.error("❌ Recommended: REJECT")

# ===============================
# TAB 2 — CSV UPLOAD
# ===============================
with tab2:
    uploaded_file = st.file_uploader("Upload CSV (same format as training features)", type="csv")

    if uploaded_file:
        df_test = pd.read_csv(uploaded_file)

        st.write("Preview:")
        st.dataframe(df_test.head())

        if st.button("Run Batch Prediction"):
            model = logreg_model if "Logistic" in model_choice else xgb_model

            df_test_scaled = scaler.transform(df_test[FEATURE_LIST])
            probs = model.predict_proba(df_test_scaled)[:, 1]

            df_test["prob_fully_paid"] = probs
            df_test["decision"] = np.where(probs >= 0.6, "APPROVE",
                                  np.where(probs >= 0.45, "REVIEW", "REJECT"))

            st.success("Prediction completed")
            st.dataframe(df_test.head())
