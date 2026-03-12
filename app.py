import streamlit as st
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open("credit_risk_model.pkl", "rb"))

st.title("Credit Risk Prediction System")

st.info("Fill the borrower details below. The system will estimate whether the loan is safe or has a high risk of default.")

# User inputs
age = st.number_input(
    "Customer Age",
    min_value=18,
    max_value=100,
    help="Enter the age of the borrower."
)

income = st.number_input(
    "Customer Income (Annual Income in ₹)",
    help="Enter the borrower's total yearly income before tax."
)

employment = st.number_input(
    "Employment Duration (Years at Current Job)",
    help="How many years the borrower has been working in their current job."
)

loan_amount = st.number_input(
    "Loan Amount (₹)",
    help="Total loan amount the borrower wants to take."
)

interest = st.number_input(
    "Loan Interest Rate (%)",
    help="Annual interest rate applied to the loan."
)

term = st.number_input(
    "Loan Term (Years)",
    help="Number of years over which the borrower will repay the loan."
)

history = st.selectbox(
    "Past Loan Default",
    [0,1],
    format_func=lambda x: "No previous default" if x==0 else "Has defaulted before",
    help="Select if the borrower has ever failed to repay a loan in the past."
)

cred_length = st.number_input(
    "Credit History Length (Years)",
    help="How many years the borrower has been using credit products such as credit cards or loans."
)

# Predict button
if st.button("Predict Risk"):

    # Create dataframe with all features used during training
    sample = pd.DataFrame(columns=model.feature_names_in_)

    # Fill numeric values
    sample.loc[0,"customer_age"] = age
    sample.loc[0,"customer_income"] = income
    sample.loc[0,"employment_duration"] = employment
    sample.loc[0,"loan_amnt"] = loan_amount
    sample.loc[0,"loan_int_rate"] = interest
    sample.loc[0,"term_years"] = term
    sample.loc[0,"historical_default"] = history
    sample.loc[0,"cred_hist_length"] = cred_length

    # Fill remaining columns with 0
    sample = sample.fillna(0)

    # Predict
    prediction = model.predict(sample)

    if prediction[0] == 0:
        st.success("Loan is SAFE (No Default Expected)")
    else:
        st.error("High Risk of DEFAULT")
