import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

st.set_page_config(
    page_title="Stroke Risk Predictor",
    layout="wide",
    page_icon="🧠"
)

# ========== Load model ==========
@st.cache_resource

def load_model():
    return joblib.load("stroke_prediction_model.pkl")

model = load_model()

# ========== Title and Info ==========
st.title("🧠 Stroke Risk Prediction App")
st.markdown("""
This application predicts the likelihood of a stroke using patient information and visualizes the associated health indicators.

**Instructions:** Fill in the patient details on the left panel or upload a CSV file. Predictions and visualizations will appear on the right.
""")
st.markdown("<hr>", unsafe_allow_html=True)

# ========== Sidebar Input ==========
st.sidebar.header("🧾 Input Patient Data")

input_mode = st.sidebar.radio("Select Input Mode", ("Manual Input", "CSV Upload"))

if input_mode == "Manual Input":
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.sidebar.slider("Age", 1, 100, 30)
    hypertension = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.sidebar.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.sidebar.selectbox("Ever Married", ["No", "Yes"])
    work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.sidebar.slider("Average Glucose Level", 50.0, 250.0, 100.0)
    bmi = st.sidebar.slider("BMI", 10.0, 60.0, 22.0)

    input_df = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [1 if hypertension == 'Yes' else 0],
        'heart_disease': [1 if heart_disease == 'Yes' else 0],
        'ever_married': [1 if ever_married == 'Yes' else 0],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': ['never smoked']
    })

    center_col = st.columns([1, 2, 1])[1]

    with center_col:
        st.markdown("## 🔍 Prediction Result")
        prediction = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0][1]

        st.success(f"**Prediction:** {'Stroke Risk' if prediction == 1 else 'No Stroke Risk'}")
        st.info(f"**Probability:** {pred_proba:.2f}")

        # === Visualizations ===
        st.markdown("## 📊 Visual Analysis")

        fig1, ax1 = plt.subplots()
        sns.barplot(x=['No Stroke', 'Stroke'], y=model.predict_proba(input_df)[0], ax=ax1)
        ax1.set_title("Stroke Probability Distribution")
        st.pyplot(fig1)

        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_proba * 100,
            title={'text': "Stroke Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ]
            }
        ))
        st.plotly_chart(fig2, use_container_width=True)

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="Max 100MB, 2 million rows")

    if uploaded_file is not None:
        try:
            if uploaded_file.size > 100 * 1024 * 1024:
                st.error("File size exceeds 100MB limit.")
            else:
                df = pd.read_csv(uploaded_file, nrows=2000000)
                st.markdown("## 📄 Uploaded Data Preview")
                st.dataframe(df.head())

                preds = model.predict(df)
                df['Prediction'] = np.where(preds == 1, 'Stroke Risk', 'No Stroke Risk')

                st.markdown("## 📋 Prediction Summary")
                st.write(df[['Prediction']].value_counts().rename("Count"))

                st.markdown("## 📈 Prediction Distribution")
                fig, ax = plt.subplots()
                sns.countplot(data=df, x='Prediction', palette='coolwarm', ax=ax)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {str(e)}")
