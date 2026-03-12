import streamlit as st
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open("credit_risk_model.pkl", "rb"))

st.title("Credit Risk Prediction System")

st.write("Enter borrower details to predict loan default risk")

# User inputs
age = st.number_input("Customer Age", min_value=18, max_value=100)
income = st.number_input("Customer Income")
employment = st.number_input("Employment Duration (years)")
loan_amount = st.number_input("Loan Amount")
interest = st.number_input("Loan Interest Rate")
term = st.number_input("Loan Term (years)")
history = st.selectbox("Historical Default", [0,1])
cred_length = st.number_input("Credit History Length")

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

    # Fill remaining columns with 0 (for encoded features)
    sample = sample.fillna(0)

    # Predict
    prediction = model.predict(sample)

    if prediction[0] == 0:
        st.success("Loan is SAFE (No Default Expected)")
    else:
        st.error("High Risk of DEFAULT")