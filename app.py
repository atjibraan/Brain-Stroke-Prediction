import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

import plotly.graph_objects as go

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

st.title("🧠 Stroke Risk Prediction App")
st.markdown("Predict whether a patient is at risk of stroke based on key health indicators.")
st.markdown("<p style='font-size: 14px; color: gray;'>Developed by <b>Your Name</b></p>", unsafe_allow_html=True)

# ========== Centered layout ==========
with st.form("input_form"):
    st.markdown("### 🧾 Input Patient Data")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.slider("Age", 1, 100, 30)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])

    with col2:
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        avg_glucose_level = st.slider("Average Glucose Level", 50.0, 250.0, 100.0)
        bmi = st.slider("BMI", 10.0, 60.0, 22.0)
        smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

    submit = st.form_submit_button("Predict")

if submit:
    try:
        # Input DataFrame
        input_df = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [1 if hypertension == "Yes" else 0],
            'heart_disease': [1 if heart_disease == "Yes" else 0],
            'ever_married': [1 if ever_married == "Yes" else 0],
            'work_type': [work_type],
            'Residence_type': [residence_type],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'smoking_status': [np.nan if smoking_status == "Unknown" else smoking_status]
        })

        # Prediction
        prediction = model.predict(input_df)[0]
        result = "🔴 High Risk of Stroke" if prediction == 1 else "🟢 Low Risk of Stroke"
        st.success(f"Prediction: **{result}**")

        # ========== 🔍 Visualizations ==========
        st.markdown("### 📊 Visual Summary")

        # Bar Chart
        st.subheader("Patient's Vital Stats")
        st.bar_chart(pd.DataFrame({
            'Metric': ['Glucose Level', 'BMI', 'Age'],
            'Value': [avg_glucose_level, bmi, age]
        }).set_index('Metric'))

        # Radar chart for risk indicators
        st.subheader("Risk Factor Radar Chart")
        categories = ['Hypertension', 'Heart Disease', 'Age > 60', 'BMI > 30', 'Glucose > 140']
        risk_values = [
            1 if hypertension == "Yes" else 0,
            1 if heart_disease == "Yes" else 0,
            1 if age > 60 else 0,
            1 if bmi > 30 else 0,
            1 if avg_glucose_level > 140 else 0
        ]

        fig = go.Figure(data=go.Scatterpolar(
            r=risk_values + [risk_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            line_color='crimson'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False
        )
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
