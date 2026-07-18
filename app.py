import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Temperature Predictor AI",
    page_icon="🌤️",
    layout="centered"
)

# Background Styling
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1534088568595-a066f410bcda");
    background-size: cover;
    background-position: center;
}

h1 {
    text-align: center;
    font-size: 32px;
}

.stButton button {
    width: 100%;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# Load Model
model = pickle.load(open("temperature_prediction_model.pkl", "rb"))


# Title
st.title("🌤️ Temperature Predictor AI")
st.markdown("### AI-based temperature prediction using weather data")

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
