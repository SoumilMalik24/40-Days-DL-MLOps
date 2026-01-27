"""
Day 35 — Streamlit Interface
User-facing UI for ML inference API
"""

import streamlit as st
import requests

st.title("ML Sentiment Prediction Demo")

API_URL = "https://your-deployment-url/predict"

st.write("Enter feature values to get a prediction from the ML model.")

features = st.text_input(
    "Input features (comma-separated)",
    value="0.038,0.050,0.062,0.022,-0.044,-0.035,-0.043,-0.003,0.020,-0.018"
)

if st.button("Predict"):
    try:
        feature_list = [float(x.strip()) for x in features.split(",")]
        response = requests.post(API_URL, json={"features": feature_list})
        result = response.json()
        st.success(f"Prediction: {result['prediction']}")
    except Exception as e:
        st.error(f"Error: {e}")
