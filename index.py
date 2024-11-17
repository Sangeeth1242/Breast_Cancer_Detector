import streamlit as st
import numpy as np
import joblib

rfc_model = joblib.load('rfc_model.pkl')

st.title("Breast Cancer Prediction App")

mean_radius = st.number_input("Mean Radius")
mean_texture = st.number_input("Mean Texture")
mean_perimeter = st.number_input("Mean Perimeter")
mean_area = st.number_input("Mean Area")
mean_smoothness = st.number_input("Mean Smoothness")

if st.button("Predict"):
    input_data = np.array([mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]).reshape(1, -1)
    prediction = rfc_model.predict(input_data)
    result = "Malignant" if prediction[0] == 1 else "Benign"
    st.success(f"Diagnosis: {result}")
