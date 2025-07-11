import streamlit as st
import numpy as np
import joblib

# Load your trained XGBoost model
model = joblib.load('xgb_loan_model.pkl')

st.title("🏦 Loan Approval Prediction App")

st.write("""
Enter the applicant details below and then click **Predict** to see if the loan will be **Approved ✅** or **Rejected ❌**.
""")

# 📌 INPUT FIELDS — adjust min/max as needed
no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income (in ₹)", min_value=0)
loan_amount = st.number_input("Loan Amount (in ₹)", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=0)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
residential_assets_value = st.number_input("Residential Assets Value (in ₹)", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value (in ₹)", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value (in ₹)", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value (in ₹)", min_value=0)

# Encode categorical variables
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# Prepare input for model
input_data = np.array([[no_of_dependents, education, self_employed, income_annum,
                        loan_amount, loan_term, cibil_score, residential_assets_value,
                        commercial_assets_value, luxury_assets_value, bank_asset_value]])

# Predict when button is clicked
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_data)
    result = "✅ **Approved**" if prediction[0] == 1 else "❌ **Rejected**"
    st.subheader(f"Loan Status: {result}")
