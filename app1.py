import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the scaler and model
scaler = joblib.load('scaler.pkl')
rfc_model = joblib.load('rfc_model.pkl')

st.title("Breast Cancer Prediction App")

# Input fields for user data
mean_radius = st.number_input("Mean Radius", min_value=0.0)
mean_texture = st.number_input("Mean Texture", min_value=0.0)
mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0)
mean_area = st.number_input("Mean Area", min_value=0.0)
mean_smoothness = st.number_input("Mean Smoothness", min_value=0.0)

# Predict button
if st.button("Predict"):
    # Ensure all inputs are filled and valid
    if any(v == 0 for v in [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]):
        st.error("Please fill in all fields with valid values.")
    else:
        # Prepare the input data for prediction
        input_data = np.array([mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]).reshape(1, -1)

        # Scale the input data using the saved scaler
        input_data = scaler.transform(input_data)

        # Check the shape of the input data after scaling
        st.write(f"Input Data after Scaling: {input_data}, Shape: {input_data.shape}")

        # Make the prediction
        try:
            prediction = rfc_model.predict(input_data)
            result = "Malignant" if prediction[0] == 1 else "Benign"
            st.success(f"Diagnosis: {result}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
