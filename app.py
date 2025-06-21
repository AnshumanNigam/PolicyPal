import streamlit as st
import numpy as np
import joblib

# Load trained models
linear_model = joblib.load('insurance_model.pkl')         # Linear Regression
logistic_model = joblib.load('insurance_log_model.pkl')   # Logistic Regression

st.title("💸 Insurance Cost & Advice App")
st.markdown("Estimate your annual medical charges and get advice on whether to buy insurance.")

# User input
age = st.slider("Age", 18, 100, 30)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
children = st.selectbox("Number of Children", list(range(0, 6)))
smoker = st.selectbox("Smoker?", ["Yes", "No"])
smoker_val = 1 if smoker == "Yes" else 0

# Feature engineering
age_smoker = age * smoker_val
bmi_smoker = bmi * smoker_val
age2 = age ** 2
bmi2 = bmi ** 2

# Final input array
log_bmi = np.log(bmi)

X_base = np.array([[age, log_bmi, children, smoker_val, age_smoker, bmi_smoker, age2, bmi2]], dtype=float)


# ------------------- Predictions -------------------

# 1️⃣ Linear Regression: Predict log_charges → actual charges
log_pred = linear_model.predict(X_base)[0]
charges = np.exp(log_pred)

# 2️⃣ Logistic Regression: Predict insurance buy probability
X_logistic = np.append(X_base, [[log_pred]], axis=1)
buy_prob = logistic_model.predict_proba(X_logistic)[0][1]
advice = (
    "✅ **My advice:** You **should** consider buying insurance."
    if buy_prob >= 0.5 else
    "❌ **My advice:** You **might not need** insurance right now."
)

# ------------------- Output -------------------

st.subheader("🧾 Estimated Annual Charges:")
import locale
def format_inr(amount):
    return f"₹ {amount:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
formatted_amount = locale.format_string("%0.2f", charges, grouping=True)
st.success(format_inr(charges))

st.subheader("🤖 Insurance Advice:")
st.info(f"Probability of buying insurance: **{buy_prob:.2%}**")
st.markdown(advice)
from fpdf import FPDF
import tempfile

# Function to create a PDF summary
def create_pdf(age, bmi, children, smoker, charges, buy_prob, advice_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Insurance Cost & Advice Report", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.cell(200, 10, txt=f"BMI: {bmi}", ln=True)
    pdf.cell(200, 10, txt=f"Children: {children}", ln=True)
    pdf.cell(200, 10, txt=f"Smoker: {smoker}", ln=True)
    pdf.ln(5)

    pdf.cell(200, 10, txt=f"Estimated Charges: Rs. {charges:,.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Probability of Buying Insurance: {buy_prob:.2%}", ln=True)
    pdf.ln(5)

    pdf.multi_cell(0, 10, f"Advice: {advice_text.replace('✅','').replace('❌','')}")
    pdf.ln(10)

    pdf.cell(200, 10, txt="Thank you for using our service!", ln=True)

    # Save to a temporary file and return the path
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name
# Create PDF and offer download
pdf_path = create_pdf(age, bmi, children, smoker, charges, buy_prob, advice)

with open(pdf_path, "rb") as file:
    st.download_button(
        label="📄 Download Report as PDF",
        data=file,
        file_name="insurance_report.pdf",
        mime="application/pdf"
    )
import shap
import matplotlib.pyplot as plt

import pandas as pd
import shap

# Build input row with column names
X_log_df = pd.DataFrame(X_logistic, columns=[
    'age', 'log_bmi', 'children', 'smoker',
    'age_smoker', 'bmi_smoker', 'age2', 'bmi2', 'log_charges'
])

# Explain prediction using SHAP
import pandas as pd
import shap

# Recreate the input row as a DataFrame with correct feature names (used in logistic model)
X_log_df = pd.DataFrame(X_logistic, columns=[
    'age', 'bmi', 'children', 'smoker',
    'age_smoker', 'bmi_smoker', 'age2', 'bmi2', 'log_charges'
])

# Create SHAP explainer and compute SHAP values
explainer = shap.Explainer(logistic_model, X_log_df)
shap_values = explainer(X_log_df)

# Extract SHAP values and feature names for the first prediction
shap_vals = shap_values[0].values
feature_names = shap_values[0].feature_names
shap_impact = pd.Series(shap_vals, index=feature_names)

# Keep feature order logical
shap_impact = shap_impact.reindex([
    'age', 'log_bmi', 'children', 'smoker',
    'age_smoker', 'bmi_smoker', 'age2', 'bmi2', 'log_charges'
])

# Sort features by absolute impact (top 4)
top_features = shap_impact.abs().sort_values(ascending=False).index[:4]

# Convert to user-friendly explanation
explanation = []
for feat in top_features:
    val = shap_impact[feat]
    direction = "increased" if val > 0 else "decreased"
    phrase = ""

    if feat == 'smoker':
        phrase = f"Being a **{'smoker' if smoker_val == 1 else 'non-smoker'}** has {direction} your likelihood of needing insurance."
    elif feat == 'log_bmi':
        phrase = f"Your **BMI of {log_bmi}** has {direction} the predicted need."
    elif feat == 'age':
        phrase = f"Your **age ({age})** has {direction} the prediction."
    elif feat == 'children':
        phrase = f"Having **{children} children** has {direction} your expected insurance need."
    elif feat == 'age_smoker':
        phrase = f"The combined effect of **being a smoker and your age** has {direction} your need."
    elif feat == 'bmi_smoker':
        phrase = f"The combination of **your BMI and smoker status** has {direction} your predicted cost."
    elif feat == 'age2':
        phrase = f"The squared age effect has {direction} the estimate slightly."
    elif feat == 'bmi2':
        phrase = f"The squared BMI effect has {direction} the estimate slightly."
    elif feat == 'log_charges':
        phrase = f"The predicted medical charges influenced this advice."

    if phrase:
        explanation.append(f"- {phrase}")

# Display in Streamlit
st.subheader("🧠 Why this advice?")
st.markdown("Your predicted insurance need was based on the following factors:")
st.markdown("\n".join(explanation))


st.write("Input features:", X_base)
st.write("Predicted log charges:", log_pred)
