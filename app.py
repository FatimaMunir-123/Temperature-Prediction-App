import streamlit as st
import pickle
import numpy as np

# Page Settings
st.set_page_config(
    page_title="Weather Temperature Predictor",
    page_icon="🌤️",
    layout="centered"
)

# Load Model
model = pickle.load(open("temperature_prediction_model.pkl", "rb"))

# Title
st.title("🌤️ Weather Temperature Predictor")
st.markdown("### Predict temperature using weather conditions")

st.divider()

# Inputs
humidity = st.number_input("💧 Humidity", value=0.89)
pressure = st.number_input("🌡️ Pressure", value=1.0286)
cloud_cover = st.number_input("☁️ Cloud Cover", value=8.0)
global_radiation = st.number_input("☀️ Global Radiation", value=0.20)
sunshine = st.number_input("🌞 Sunshine", value=0.0)

st.divider()

# Prediction
if st.button("🔮 Predict Temperature", use_container_width=True):
    data = np.array([[humidity, pressure, cloud_cover, global_radiation, sunshine]])
    prediction = model.predict(data)

    st.success(f"🌡️ Predicted Temperature: **{prediction[0]:.2f} °C**")

st.markdown("---")
st.caption("Developed by Fatima Munir | Machine Learning Project")
