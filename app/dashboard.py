import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Import reusable plot functions
from plots import (
    churn_by_contract,
    monthly_charges_vs_churn,
    tenure_distribution,
    feature_importance_chart
)

# --- Load dataset (for charts only) ---
df = pd.read_csv("data/churn_clean.csv")

# --- Load model and feature list ---
model = joblib.load("app/model.pkl")
model_features = joblib.load("app/model_features.pkl")

# --- Streamlit page setup ---
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📊 Customer Churn Dashboard")

# =======================
# KPI SECTION
# =======================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Customers", len(df))
with col2:
    st.metric("Churned Customers", int(df["Churn"].sum()))
with col3:
    churn_rate = round(df["Churn"].mean() * 100, 2)
    st.metric("Churn Rate (%)", churn_rate)

st.markdown("---")

# =======================
# VISUALIZATIONS
# =======================
st.subheader("Churn by Contract Type")
st.plotly_chart(churn_by_contract(df), use_container_width=True)

st.subheader("Monthly Charges vs Churn")
st.plotly_chart(monthly_charges_vs_churn(df), use_container_width=True)

st.subheader("Tenure Distribution by Churn")
st.plotly_chart(tenure_distribution(df), use_container_width=True)

st.markdown("---")

# =======================
# PREDICTION TOOL
# =======================
st.header("🔮 Predict Customer Churn")

# --- Quick Demo High-Risk Profile ---
st.markdown("### ⚡ Quick Demo")
use_demo = st.checkbox("Load Example High-Risk Profile")

if use_demo:
    # High-risk profile values
    demo_profile = {
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 100,
        "TotalCharges": 200
    }

    st.info("⚡ Using Example High-Risk Profile")
    st.dataframe(pd.DataFrame([demo_profile]))

    # Assign values for prediction
    input_df = pd.DataFrame([demo_profile])

else:
    # --- Multi-column layout for form ---
    st.subheader("👤 Demographics & Account Info")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    with col2:
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
    with col3:
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    st.subheader("🔌 Services")
    col1, col2, col3 = st.columns(3)

    with col1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
    with col3:
        device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

    st.subheader("💳 Billing & Charges")
    col1, col2, col3 = st.columns(3)

    with col1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with col2:
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    with col3:
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

    monthly_charges = st.slider("Monthly Charges", 0, 120, 50)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=100.0, step=10.0)

    # Assemble Input
    input_df = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

# One-hot encode input
input_encoded = pd.get_dummies(input_df)

# Align with training features
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

# --- Prediction ---
prediction = model.predict(input_encoded)[0]
probability = model.predict_proba(input_encoded)[0][1]

st.subheader("Prediction Result")
st.write("Prediction:", "❌ Churn" if prediction == 1 else "✅ No Churn")
st.write(f"Churn Probability: {probability:.2%}")

# =======================
# FEATURE IMPORTANCE
# =======================
st.markdown("---")
st.subheader("📈 Feature Importance (Global)")

if hasattr(model, "feature_importances_"):
    importance = model.feature_importances_
else:
    importance = np.abs(model.coef_[0])

feat_imp = pd.DataFrame({
    "Feature": model_features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False).head(15)

fig_imp = feature_importance_chart(feat_imp)
st.plotly_chart(fig_imp, use_container_width=True)
