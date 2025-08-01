import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# ========== Must be FIRST Streamlit command ==========
st.set_page_config(
    page_title="Stroke Risk Predictor",
    layout="centered",
    page_icon="🧠"
)

# ========== Load model ==========
@st.cache_resource
def load_model():
    return joblib.load("stroke_prediction_model.pkl")

model = load_model()

# ========== Page Header ==========
st.title("🧠 Stroke Risk Prediction App")
st.markdown("Predict whether a patient is at risk of stroke based on key health indicators.")
st.markdown(
    "<p style='font-size: 14px; color: gray;'>Developed by <b>Your Name</b></p>",
    unsafe_allow_html=True
)

# ========== Sidebar Input ==========
st.sidebar.header("Input Patient Data")

gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
age = st.sidebar.slider("Age", 1, 100, 30)
hypertension = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.sidebar.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.sidebar.selectbox("Ever Married", ["No", "Yes"])
work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.sidebar.slider("Average Glucose Level", 50.0, 250.0, 100.0)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 22.0)
smoking_status = st.sidebar.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# ========== Predict Button ==========
if st.sidebar.button("Predict"):
    try:
        # Create dataframe from user input
        input_data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [1 if hypertension == "Yes" else 0],
            'heart_disease': [1 if heart_disease == "Yes" else 0],
            'ever_married': [1 if ever_married == "Yes" else 0],
            'work_type': [work_type],
            'Residence_type': [residence_type],
            'avg_glucose_level': [min(avg_glucose_level, 200)],
            'bmi': [min(max(bmi, 18), 45)],
            'smoking_status': [np.nan if smoking_status == "Unknown" else smoking_status]
        })

        # Predict
        prediction = model.predict(input_data)[0]
        result = "🔴 High Risk of Stroke" if prediction == 1 else "🟢 Low Risk of Stroke"
        st.success(f"Prediction: **{result}**")

    except Exception as e:
        st.error(f"Error in prediction: {e}")
