import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

lr_model = joblib.load("lr_model.pkl")
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_fraud_model.pkl")

scaler = joblib.load("scaler.pkl")

data = pd.read_csv("creditcard_sample.csv")

st.title("Credit Card Fraud Detection App")
st.write("Enter transaction details and detect if it is fraudulent.")

st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose ML Model", 
    ["Logistic Regression", "Random Forest", "XGBoost"]
)

if model_choice == "Logistic Regression":
    model = lr_model
elif model_choice == "Random Forest":
    model = rf_model
else:
    model = xgb_model

st.header("Fraud Distribution")
fig, ax = plt.subplots()
sns.countplot(data=data, x="Class", ax=ax)
st.pyplot(fig)

st.header("Enter Manual Transaction Details")

amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=50000.0, value=100.0)
time = st.number_input(
    "Time (seconds from first transaction)", min_value=0.0, max_value=200000.0, value=50000.0)

v_features = np.random.normal(0, 1, 28)

if st.button("Predict Fraud"):
    input_data = np.array([time] + list(v_features) + [amount]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error(f"Fraud Detected! (Model: {model_choice})")
    else:
        st.success(f"Transaction Safe (Model: {model_choice})")

st.write("---")

if model_choice in ["Random Forest", "XGBoost"]:
    st.header("Feature Importance")

    importances = model.feature_importances_

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances, y=np.arange(len(importances)), ax=ax2)
    ax2.set_title(f"Model Feature Importance - {model_choice}")
    st.pyplot(fig2)

elif model_choice == "Logistic Regression":
    st.header("Feature Importance (Coefficients)")

    importances = np.abs(model.coef_).flatten()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances, y=np.arange(len(importances)), ax=ax2)
    ax2.set_title("Logistic Regression Coefficient Magnitudes")
    st.pyplot(fig2)

st.header("Upload a CSV File for Bulk Fraud Detection")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    scaled_user_data = scaler.transform(user_df)
    bulk_preds = model.predict(scaled_user_data)
    user_df["Prediction"] = bulk_preds

    st.write("Bulk Predictions:")
    st.dataframe(user_df)

    st.download_button(
        label="Download Results as CSV",
        data=user_df.to_csv(index=False),
        file_name="fraud_predictions.csv",
        mime="text/csv",
    )




st.info("""
    V1-V28 are anonymized PCA components from the bank to protect user privacy.
    They represent patterns in the transaction but cannot be interpreted directly.
""")


# Step 9: Advaned Streamlit Visualizations + CSV Upload System
