import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="🧠 Stroke Risk Predictor",
    layout="centered",
    page_icon="🧠"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("stroke_prediction_model.pkl")

model = load_model()

# === Title and Description ===
st.title("🧠 Stroke Risk Prediction App")
st.markdown("""
Welcome to the Stroke Risk Prediction App.  
Enter basic patient information and health metrics below to estimate stroke risk.  
The app uses a trained RandomForest model and offers interpretable visualizations for decision support.
""")
st.markdown("<p style='font-size: 14px; color: gray;'>Developed by <b>Your Name</b></p>", unsafe_allow_html=True)
st.markdown("---")

# === Input Form ===
with st.form("input_form"):
    st.markdown("## 👤 Patient Details")

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

    submit = st.form_submit_button("🔍 Predict Stroke Risk")

# === Prediction & Visual Explanation ===
if submit:
    try:
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

        # Predict
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] * 100
        result = "🔴 High Risk of Stroke" if prediction == 1 else "🟢 Low Risk of Stroke"

        st.markdown("## 🧾 Prediction Result")
        st.success(f"**Prediction:** {result}")
        st.metric(label="Predicted Stroke Probability", value=f"{prob:.2f} %")

        st.markdown("---")
        st.markdown("## 📊 Visual Risk Explanation")

        # Bar chart of patient values
        bar_df = pd.DataFrame({
            'Metric': ['Age', 'BMI', 'Glucose'],
            'Value': [age, bmi, avg_glucose_level]
        })
        fig_bar = px.bar(bar_df, x='Metric', y='Value', title="Health Indicators")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### 🧠 Interpreting These Indicators")
        st.markdown("""
        - **Age > 60** increases stroke risk significantly.
        - **BMI > 30** is associated with obesity-related complications.
        - **Glucose > 140** can indicate diabetes or pre-diabetic conditions.
        """)

        # Radar chart of binary risk factors
        radar_categories = ['Hypertension', 'Heart Disease', 'Age > 60', 'BMI > 30', 'Glucose > 140']
        radar_values = [
            1 if hypertension == "Yes" else 0,
            1 if heart_disease == "Yes" else 0,
            1 if age > 60 else 0,
            1 if bmi > 30 else 0,
            1 if avg_glucose_level > 140 else 0
        ]
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=radar_categories + [radar_categories[0]],
            fill='toself',
            line_color='indigo'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            title="Patient Risk Profile (Radar Chart)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Pie chart of smoking status impact
        st.markdown("### 🚬 Smoking Status Breakdown")
        pie_fig = px.pie(
            names=["Smokes", "Never Smoked", "Formerly Smoked", "Unknown"],
            values=[1 if smoking_status == status else 0.25 for status in
                    ["smokes", "never smoked", "formerly smoked", "Unknown"]],
            title="Smoking Category Impact",
            hole=0.4
        )
        st.plotly_chart(pie_fig, use_container_width=True)

        st.markdown("### 📌 Summary")
        st.markdown(f"""
        This patient is predicted to have a **{"high" if prediction == 1 else "low"} risk** of stroke with a 
        probability of **{prob:.2f}%**.  
        Use this prediction alongside clinical judgment and further diagnostics.
        """)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
