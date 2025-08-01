# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import traceback

# ========== Configuration ==========
MODEL_PATH = 'stroke_prediction_model.pkl'
DEVELOPER_NAME = "Jibraan Attar"  # Change if needed

# ========== Load model ==========
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model = load_model()

# ========== Page Setup ==========
st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")
st.title("🧠 Stroke Risk Prediction App")
st.markdown("Predict whether a patient is at risk of stroke based on key health attributes.")
st.markdown(f"<p style='font-size: 14px; color: gray;'>Developed by <b>{DEVELOPER_NAME}</b></p>", unsafe_allow_html=True)

# ========== User Input ==========
st.header("📝 Enter Patient Information")

with st.form("input_form"):
    age = st.slider("Age", 1, 100, 45)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
    bmi = st.number_input("BMI", 10.0, 60.0, 24.0)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    submitted = st.form_submit_button("Predict Stroke Risk")

# ========== Prediction ==========
if submitted:
    try:
        input_df = pd.DataFrame([{
            'age': age,
            'gender': gender,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': 1 if ever_married == "Yes" else 0,
            'work_type': work_type,
            'Residence_type': Residence_type,
            'avg_glucose_level': min(avg_glucose_level, 200),  # cap as per training
            'bmi': np.clip(bmi, 18, 45),                       # clip as per training
            'smoking_status': smoking_status if smoking_status != "Unknown" else np.nan
        }])

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.success(f"✅ Stroke Risk Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}")
        st.metric(label="Probability of Stroke", value=f"{proba:.2%}")

    except Exception as e:
        st.error("❌ Error in prediction. Check input format or model compatibility.")
        st.code(traceback.format_exc(), language="python")
