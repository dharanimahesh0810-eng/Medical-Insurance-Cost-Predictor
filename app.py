"""
app.py
-------
Streamlit web interface for the Insurance Cost Prediction model.

Before running this, run train_model.py once to generate:
    insurance_model.pkl, encoders.pkl, feature_columns.pkl

Then start the app with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="💰",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("insurance_model.pkl")
    encoders = joblib.load("encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, encoders, feature_columns


try:
    model, encoders, feature_columns = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Please run `python train_model.py` first "
        "to generate insurance_model.pkl, encoders.pkl and feature_columns.pkl."
    )
    st.stop()

st.title("💰 Insurance Cost Prediction")
st.write("Enter the details below to estimate medical insurance charges.")

with st.form("insurance_form"):
    age = st.slider("Age", min_value=18, max_value=65, value=25)
    bmi = st.slider("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
    children = st.slider("Children", min_value=0, max_value=5, value=0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoker = st.selectbox("Smoker", ["Yes", "No"])
    region = st.selectbox("Region", ["Northwest", "Northeast", "Southwest", "Southeast"])

    submitted = st.form_submit_button("Predict Insurance Cost")

if submitted:
    # Encode categorical inputs using the exact encoders fit during training
    sex_encoded = encoders["sex"].transform([gender.lower()])[0]
    smoker_encoded = encoders["smoker"].transform([smoker.lower()])[0]
    region_encoded = encoders["region"].transform([region.lower()])[0]

    input_df = pd.DataFrame(
        [[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]],
        columns=feature_columns,
    )

    prediction = model.predict(input_df)[0]

    st.subheader("Result")
    st.success(f"💵 Estimated Insurance Charges: **₹ {prediction:,.2f}**")
