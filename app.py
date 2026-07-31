import streamlit as st
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

st.title("Customer Churn Prediction")

st.write("Enter customer details below.")

age = st.number_input("Age", 18, 100, 30)

gender = st.selectbox("Gender", ["Male", "Female"])

tenure = st.number_input("Tenure", 0, 120, 12)

monthly_charge = st.number_input("Monthly Charges", 0.0, 200.0, 60.0)

if st.button("Predict"):

    gender_value = 1 if gender == "Female" else 0

    data = np.array([[age, gender_value, tenure, monthly_charge]])

    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is not likely to Churn")