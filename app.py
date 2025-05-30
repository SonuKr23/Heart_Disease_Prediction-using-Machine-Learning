import streamlit as st
import numpy as np
import joblib
import re  # for safer number extraction

# Load trained model
try:
    model = joblib.load('heart_disease_model.pkl')
except:
    st.error("Model file not found. Please ensure 'heart_disease_model.pkl' exists in this directory.")
    st.stop()

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("💓 Heart Disease Prediction App")
st.markdown("**Enter patient data to predict the risk of heart disease**")

# Input fields with help text
age = st.number_input("Age", 18, 100, 30, help="Age of the patient (in years)")
sex = st.selectbox("Sex", ["Male", "Female"], help="Biological sex of the patient")
cp_options = {
    "No pain": 0,
    "Mild pain with exercise": 1,
    "Unrelated to heart": 2,
    "Severe or frequent chest pain": 3
}
cp_label = st.selectbox("Chest Pain Type", list(cp_options.keys()), help="Describe the chest pain, if any")
cp_value = cp_options[cp_label]

trestbps = st.number_input(
    "Resting Blood Pressure (mm Hg)", 
    90, 200, 120, 
    help="Normal resting BP is around 120 mm Hg. Above 130 may be considered high."
)
chol = st.number_input(
    "Cholesterol (mg/dL)", 
    100, 600, 200, 
    help="Normal cholesterol is under 200 mg/dL. 200–239 is borderline high; 240+ is high."
)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"], help="Is fasting blood sugar > 120 mg/dL?")
restecg = st.selectbox("Rest ECG Results", [
    "Normal (0)", 
    "ST-T wave abnormality (1)", 
    "Left ventricular hypertrophy (2)"
], help="Results of resting electrocardiogram")
expected_hr = 220 - age
thalach = st.number_input(
    "Max Heart Rate Achieved",
    70,
    220,
    150,
    help=f"Typical max heart rate for your age ({age}) is around {expected_hr} bpm"
)

exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"], help="Chest pain triggered by exercise?")
oldpeak = st.number_input("Stress (oldpeak)", 0.0, 6.0, 1.0, help="ST depression induced by exercise relative to rest")
slope_options = {
    "Rising after exercise": 0,
    "Flat (no change)": 1,
    "Falling during exercise": 2
}
slope_label = st.selectbox("ST Segment Slope", list(slope_options.keys()), help="Trend of heart's ST segment during exercise")
slope_value = slope_options[slope_label]

ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3], help="Number of major vessels colored by fluoroscopy")
thal = st.selectbox("Thalassemia", [
    "Normal (1)", 
    "Fixed Defect (2)", 
    "Reversible Defect (3)"
], help="Type of thalassemia blood disorder")

# Helper to extract number from string
def extract_number(label):
    match = re.search(r'\((\d+)\)', label)
    return int(match.group(1)) if match else 0

# Format input for model
input_data = np.array([[
    age,
    1 if sex == "Male" else 0,
    cp_value,
    trestbps,
    chol,
    1 if fbs == "Yes" else 0,
    extract_number(restecg),
    thalach,
    1 if exang == "Yes" else 0,
    oldpeak,
    slope_value,
    ca,
    extract_number(thal)
]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0]  # Get probability for both classes

    result = "🚨 Heart Disease Detected" if prediction[0] == 1 else "✅ No Heart Disease"
    
    st.success(f"Prediction: {result}")
    st.write(f"Probabilities → No Disease: **{proba[0]:.2f}**, Disease: **{proba[1]:.2f}**")
