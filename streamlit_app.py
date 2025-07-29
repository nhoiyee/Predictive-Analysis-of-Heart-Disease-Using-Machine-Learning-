import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('model.pkl')

st.title("❤ Heart Disease Predictor")
st.write("Enter the patient’s medical information below:")

# Inputs matching the 8 features
age = st.slider("Age", 20, 80, 45)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)

# Prepare input DataFrame
input_data = pd.DataFrame([[
    age, sex, trestbps, chol, fbs, thalach, exang, oldpeak
]], columns=['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak'])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "has **Heart Disease** 💔" if prediction == 1 else "does **NOT** have Heart Disease ❤️"
    st.markdown(f"### Result: The patient {result}")
