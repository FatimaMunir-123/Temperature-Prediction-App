import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("temperature_prediction_model.pkl", "rb"))

st.title("🌡️ Temperature Prediction App")

st.write("Enter the weather details below:")

humidity = st.number_input("Humidity", value=0.89)
pressure = st.number_input("Pressure", value=1.0286)
cloud_cover = st.number_input("Cloud Cover", value=8.0)
global_radiation = st.number_input("Global Radiation", value=0.20)
sunshine = st.number_input("Sunshine", value=0.0)

if st.button("Predict Temperature"):
    data = np.array([[humidity, pressure, cloud_cover, global_radiation, sunshine]])
    prediction = model.predict(data)

    st.success(f"Predicted Temperature: {prediction[0]:.2f} °C")