#!/usr/bin/env python
# coding: utf-8

import streamlit as st

# ✅ This must be the first Streamlit command
st.set_page_config(
    page_title="Stroke Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import pickle
import matplotlib.pyplot as plt
from fpdf import FPDF
import os

# Detect if running in Streamlit Cloud
IN_STREAMLIT_CLOUD = os.environ.get("STREAMLIT_SERVER_PORT") is not None

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return pickle.load(open('stroke_prediction_model.pkl', 'rb'))
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model = load_model()

# Explanations for key risk factors
RISK_FACTORS = {
    'age': "Risk doubles every decade after 55",
    'avg_glucose_level': ">140 mg/dL = diabetic risk range",
    'bmi': ">30 BMI = obesity-related risks",
    'hypertension': "High blood pressure damages vessels",
    'smoking_status': "Smoking doubles stroke risk",
    'heart_disease': "Cardiac issues increase clot risk"
}

# Inject custom CSS
st.markdown("""
<style>
    .header { color: #1f77b4; font-weight: 700; font-size: 28px; }
    .subheader { color: #2ca02c; font-weight: 600; }
    .risk-high { background-color: #ffcccc; padding: 10px; border-radius: 5px; }
    .risk-moderate { background-color: #ffe6cc; padding: 10px; border-radius: 5px; }
    .risk-low { background-color: #ccffcc; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# PDF Report Generator
def create_pdf_report(data, probability, factors):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Stroke Risk Assessment Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Patient Information", ln=True)
    pdf.set_font("Arial", size=12)

    patient_data = data.iloc[0]
    info = f"""
    Age: {patient_data['age']}
    Gender: {patient_data['gender']}
    Hypertension: {'Yes' if patient_data['hypertension'] == 1 else 'No'}
    Heart Disease: {'Yes' if patient_data['heart_disease'] == 1 else 'No'}
    Marital Status: {patient_data['ever_married']}
    Work Type: {patient_data['work_type']}
    Residence: {patient_data['Residence_type']}
    Avg. Glucose: {patient_data['avg_glucose_level']} mg/dL
    BMI: {patient_data['bmi']}
    Smoking Status: {patient_data['smoking_status']}
    """
    pdf.multi_cell(0, 8, info)
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Risk Assessment", ln=True)
    pdf.set_font("Arial", size=12)

    risk_level = "High" if probability >= 70 else "Moderate" if probability >= 40 else "Low"
    pdf.cell(0, 10, f"10-Year Stroke Risk Probability: {probability:.1f}%", ln=True)
    pdf.cell(0, 10, f"Risk Classification: {risk_level}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Recommendations", ln=True)
    pdf.set_font("Arial", size=12)

    if risk_level == "High":
        rec = "1. Urgent consultation with a cardiologist\n2. Immediate blood pressure monitoring\n3. Lifestyle modification program"
    elif risk_level == "Moderate":
        rec = "1. Schedule preventive care visit within 30 days\n2. Dietary assessment and modification\n3. Regular exercise program"
    else:
        rec = "1. Annual health check-ups\n2. Maintain healthy lifestyle\n3. Regular blood pressure monitoring"

    pdf.multi_cell(0, 8, rec)
    pdf.ln(10)

    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, "This tool does not replace medical advice. Consult your physician for personal evaluation.", ln=True)

    return pdf.output(dest='S').encode('latin1')

# Main App
def main():
    st.title("🧠 Stroke Risk Prediction Tool")
    st.markdown("This clinical assessment tool estimates your 10-year stroke risk using medically-validated predictive factors.")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Patient Information")
        with st.form("patient_form"):
            age = st.slider("Age", 18, 100, 45)
            gender = st.radio("Gender", ["Male", "Female"])
            hypertension = st.radio("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            heart_disease = st.radio("Heart Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            ever_married = st.radio("Marital Status", ["No", "Yes"])
            work_type = st.selectbox("Employment Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
            Residence_type = st.radio("Residence Type", ["Urban", "Rural"])
            avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", 50, 300, 100)
            bmi = st.number_input("BMI", 15, 50, 25)
            smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])
            submitted = st.form_submit_button("Assess Stroke Risk")

    with col2:
        st.subheader("Risk Assessment")
        if submitted:
            input_data = pd.DataFrame([{
                'gender': gender,
                'age': age,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'ever_married': ever_married,
                'work_type': work_type,
                'Residence_type': Residence_type,
                'avg_glucose_level': avg_glucose_level,
                'bmi': bmi,
                'smoking_status': smoking_status
            }])

            try:
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1] * 100

                if probability >= 70:
                    st.markdown(f"<div class='risk-high'><h3>🚨 High Risk: {probability:.1f}%</h3>Urgent clinical evaluation recommended</div>", unsafe_allow_html=True)
                elif probability >= 40:
                    st.markdown(f"<div class='risk-moderate'><h3>⚠️ Moderate Risk: {probability:.1f}%</h3>Preventive care consultation advised</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='risk-low'><h3>✅ Low Risk: {probability:.1f}%</h3>Maintain healthy lifestyle monitoring</div>", unsafe_allow_html=True)

                st.markdown("#### Risk Factor Contribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                factors = {
                    'Age': min(age / 100 * 40, 40),
                    'Glucose Level': min((max(avg_glucose_level - 100, 0) / 100) * 25, 25),
                    'BMI': min((max(bmi - 25, 0) / 10) * 15, 15),
                    'Hypertension': 5 if hypertension == 1 else 0,
                    'Smoking': 8 if smoking_status == "smokes" else (4 if smoking_status == "formerly smoked" else 0),
                    'Heart Disease': 4 if heart_disease == 1 else 0
                }
                sorted_factors = dict(sorted(factors.items(), key=lambda item: item[1], reverse=True))
                ax.bar(sorted_factors.keys(), sorted_factors.values(), color='#ff7f0e')
                ax.set_ylabel('Risk Contribution (%)')
                ax.set_title('Top Risk Factors')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("#### Key Risk Factors")
                for factor, explanation in RISK_FACTORS.items():
                    st.markdown(f"🔹 **{factor.replace('_', ' ').title()}**: {explanation}")

                st.markdown("### Download Full Report")
                pdf = create_pdf_report(input_data, probability, factors)
                st.download_button(
                    label="Download PDF Report",
                    data=pdf,
                    file_name=f"Stroke_Risk_Assessment_{age}_{gender}.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
        else:
            st.info("Please fill the form and click 'Assess Stroke Risk'")
            st.image("https://images.unsplash.com/photo-1576091160550-2173dba999ef", caption="Early stroke prevention saves lives")

if __name__ == "__main__":
    main()


