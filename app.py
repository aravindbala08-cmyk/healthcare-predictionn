# =========================================
# BREAST CANCER PREDICTION WEB APP
# Streamlit Version
# =========================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Healthcare Prediction App", layout="wide")

st.title("🏥 Breast Cancer Prediction System")
st.write("Machine Learning-based Healthcare Prediction Web Application")

# -----------------------------
# Load Dataset
# -----------------------------
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# -----------------------------
# Train Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.sidebar.header("Model Information")
st.sidebar.write(f"Model Accuracy: **{accuracy:.2f}**")

# -----------------------------
# Data Visualization Section
# -----------------------------
st.subheader("📊 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(X, x="mean radius", title="Distribution of Mean Radius")
    st.plotly_chart(fig)

with col2:
    target_df = pd.DataFrame({"Diagnosis": y})
    fig2 = px.histogram(target_df, x="Diagnosis",
                        title="Cancer Diagnosis Distribution")
    st.plotly_chart(fig2)

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("🔍 Enter Patient Details for Prediction")

input_data = []

for feature in data.feature_names:
    value = st.number_input(f"{feature}", float(X[feature].min()),
                            float(X[feature].max()),
                            float(X[feature].mean()))
    input_data.append(value)

if st.button("Predict"):

    input_array = np.array(input_data).reshape(1, -1)
    input_array = scaler.transform(input_array)

    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][prediction]

    st.subheader("Prediction Result")

    if prediction == 0:
        st.error(f"⚠ Malignant Tumor (High Risk)\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Benign Tumor (Low Risk)\nProbability: {probability:.2f}")

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("📈 Top Important Features")

importances = model.feature_importances_

feat_df = pd.DataFrame({
    "Feature": data.feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(10)

fig3 = px.bar(feat_df, x="Importance", y="Feature",
              orientation="h",
              title="Top 10 Important Features")

st.plotly_chart(fig3)
