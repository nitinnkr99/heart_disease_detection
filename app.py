import streamlit as st
import numpy as np
import pandas as pd
import joblib

model  = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Heart Disease Detection", page_icon="❤️", layout="wide")

st.title("❤️ Heart Disease Detection System")
st.markdown("Enter the patient's clinical details below to predict the likelihood of heart disease.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Patient Info")
    age      = st.slider("Age", 20, 80, 50)
    sex      = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp       = st.selectbox("Chest Pain Type", options=[0,1,2,3],
                             format_func=lambda x: {0:"Typical Angina", 1:"Atypical Angina",
                                                     2:"Non-Anginal Pain", 3:"Asymptomatic"}[x])
    fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0,1],
                             format_func=lambda x: "No" if x == 0 else "Yes")
    exang    = st.selectbox("Exercise Induced Angina", options=[0,1],
                             format_func=lambda x: "No" if x == 0 else "Yes")

with col2:
    st.subheader("Vitals")
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
    thalach  = st.slider("Max Heart Rate Achieved", 70, 210, 150)
    oldpeak  = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)

with col3:
    st.subheader("ECG & Tests")
    restecg  = st.selectbox("Resting ECG Results", options=[0,1,2],
                             format_func=lambda x: {0:"Normal", 1:"ST-T Abnormality",
                                                     2:"Left Ventricular Hypertrophy"}[x])
    slope    = st.selectbox("Slope of Peak Exercise ST", options=[0,1,2],
                             format_func=lambda x: {0:"Upsloping", 1:"Flat", 2:"Downsloping"}[x])
    ca       = st.selectbox("Major Vessels Coloured by Fluoroscopy", options=[0,1,2,3,4])
    thal     = st.selectbox("Thalassemia", options=[0,1,2,3],
                             format_func=lambda x: {0:"Unknown", 1:"Normal",
                                                     2:"Fixed Defect", 3:"Reversible Defect"}[x])

st.divider()

if st.button("🔍 Predict", use_container_width=True, type="primary"):
    features = np.array([[age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak, slope, ca, thal]])
    features_scaled = scaler.transform(features)
    prediction      = model.predict(features_scaled)[0]
    probability     = model.predict_proba(features_scaled)[0]

    st.divider()
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        if prediction == 1:
            st.error("**Result: Heart Disease Detected**")
        else:
            st.success("**Result: No Heart Disease**")

    with res_col2:
        st.metric("Heart Disease Probability", f"{probability[1]*100:.1f}%")

    with res_col3:
        st.metric("No Disease Probability", f"{probability[0]*100:.1f}%")

    st.progress(float(probability[1]))

    if prediction == 1:
        st.warning("⚠️ High risk indicators detected. Please consult a cardiologist.")
    else:
        st.info("✅ Low risk profile based on provided parameters.")

    st.caption("⚠️ This tool is for educational purposes only and is not a substitute for medical diagnosis.")

st.divider()
with st.expander("ℹ️ About this model"):
    st.markdown("""
    **Dataset:** Heart Disease Dataset (1,025 patients)  
    **Best Model:** Support Vector Machine (SVM)  
    **Accuracy:** 92.7% | **ROC-AUC:** 0.977 | **CV Score:** 91.2%  
    
    | Model | Accuracy | ROC-AUC |
    |---|---|---|
    | Logistic Regression | 81.0% | 0.930 |
    | Random Forest (tuned) | 92.2% | 0.973 |
    | SVM | 92.7% | 0.977 |
    """)