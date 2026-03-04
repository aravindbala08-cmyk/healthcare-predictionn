# =========================================
# HEALTHCARE DISEASE PREDICTION
# Dataset Written Manually Inside Code
# =========================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Healthcare Prediction", layout="wide")
st.title("🏥 Healthcare Disease Prediction System")

# ------------------------------------------------
# 1️⃣ DATASET WRITTEN DIRECTLY INSIDE THE CODE
# ------------------------------------------------

data = pd.DataFrame({
    "age": [25, 45, 60, 35, 50, 65, 40, 70, 55, 30],
    "blood_pressure": [120, 140, 160, 130, 150, 170, 135, 180, 155, 110],
    "cholesterol": [180, 220, 250, 200, 240, 260, 210, 280, 230, 190],
    "max_heart_rate": [150, 140, 130, 160, 145, 120, 155, 110, 135, 165],
    "blood_sugar": [90, 110, 150, 100, 140, 160, 105, 170, 130, 95],
    "disease": [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
})

st.subheader("📊 Healthcare Dataset")
st.dataframe(data)

# ------------------------------------------------
# 2️⃣ TRAIN MODEL
# ------------------------------------------------

X = data.drop("disease", axis=1)
y = data["disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.sidebar.header("Model Accuracy")
st.sidebar.write(f"Accuracy: **{accuracy:.2f}**")

# ------------------------------------------------
# 3️⃣ VISUALIZATION
# ------------------------------------------------

st.subheader("📈 Age Distribution")
fig = px.histogram(data, x="age", color="disease",
                   title="Age vs Disease")
st.plotly_chart(fig)

# ------------------------------------------------
# 4️⃣ USER INPUT FOR PREDICTION
# ------------------------------------------------

st.subheader("🔍 Enter Patient Details")

age = st.number_input("Age", 20, 80, 40)
bp = st.number_input("Blood Pressure", 90, 200, 130)
chol = st.number_input("Cholesterol", 150, 300, 200)
hr = st.number_input("Max Heart Rate", 60, 200, 140)
sugar = st.number_input("Blood Sugar", 70, 200, 110)

if st.button("Predict"):

    input_data = np.array([[age, bp, chol, hr, sugar]])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠ High Risk of Disease (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk (Probability: {probability:.2f})")
