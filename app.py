import streamlit as st
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open("credit_risk_model.pkl", "rb"))

st.title("Credit Risk Prediction System")

st.info("Fill the borrower details below. The system will estimate whether the loan is safe or has a high risk of default.")

# User inputs with placeholder
age = st.text_input(
    "Customer Age",
    placeholder="18"
)

income = st.text_input(
    "Customer Income (Annual Income in ₹)",
    placeholder="0.00"
)

employment = st.text_input(
    "Employment Duration (Years at Current Job)",
    placeholder="0.00"
)

loan_amount = st.text_input(
    "Loan Amount (₹)",
    placeholder="0.00"
)

interest = st.text_input(
    "Loan Interest Rate (%)",
    placeholder="0.00"
)

term = st.text_input(
    "Loan Term (Years)",
    placeholder="0.00"
)

history = st.selectbox(
    "Past Loan Default",
    [0,1],
    format_func=lambda x: "No previous default" if x==0 else "Has defaulted before"
)

cred_length = st.text_input(
    "Credit History Length (Years)",
    placeholder="0.00"
)

# Predict button
if st.button("Predict Risk"):

    # Convert inputs to numeric
    age = float(age or 0)
    income = float(income or 0)
    employment = float(employment or 0)
    loan_amount = float(loan_amount or 0)
    interest = float(interest or 0)
    term = float(term or 0)
    cred_length = float(cred_length or 0)

    # Create dataframe
    sample = pd.DataFrame(columns=model.feature_names_in_)

    sample.loc[0,"customer_age"] = age
    sample.loc[0,"customer_income"] = income
    sample.loc[0,"employment_duration"] = employment
    sample.loc[0,"loan_amnt"] = loan_amount
    sample.loc[0,"loan_int_rate"] = interest
    sample.loc[0,"term_years"] = term
    sample.loc[0,"historical_default"] = history
    sample.loc[0,"cred_hist_length"] = cred_length

    sample = sample.fillna(0)

    prediction = model.predict(sample)
    probability = model.predict_proba(sample)[0][1]

    if prediction[0] == 0:
        st.success("Loan is SAFE (No Default Expected)")
        st.write(f"Estimated Default Risk: {probability*100:.2f}%")
        st.info("This borrower appears financially stable based on the provided financial information.")

    else:
        st.error("High Risk of DEFAULT")
        st.write(f"Estimated Default Risk: {probability*100:.2f}%")
        st.warning("The borrower may face difficulty repaying the loan based on the provided financial information.")
