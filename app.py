import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pkg_resources
import subprocess
import sys

def install_missing_requirements(requirements_file="requirements.txt"):
    with open(requirements_file, "r") as file:
        dependencies = file.readlines()

    for dependency in dependencies:
        try:
            pkg_resources.require([dependency.strip()])
        except pkg_resources.DistributionNotFound:
            print(f"{dependency.strip()} is missing. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dependency.strip()])
    print("All requirements satisfied.")

install_missing_requirements()

# Placeholder for training data (replace with actual data loading if needed)
def load_and_train_model():
    # Sample dataset for training (use the actual dataset here)
    weather_data = pd.DataFrame({
        "TempMax": np.random.uniform(20, 40, 1000),
        "TempMin": np.random.uniform(10, 30, 1000),
        "TempAvg": np.random.uniform(15, 35, 1000),
        "FeelsLikeMax": np.random.uniform(21, 41, 1000),
        "FeelsLikeMin": np.random.uniform(11, 31, 1000),
        "FeelsLikeAvg": np.random.uniform(16, 36, 1000),
        "DewPoint": np.random.uniform(5, 25, 1000),
        "Humidity": np.random.uniform(30, 100, 1000),
        "WindGust": np.random.uniform(0, 50, 1000),
        "WindSpeed": np.random.uniform(0, 20, 1000),
        "Visibility": np.random.uniform(5, 15, 1000),
        "SolarRadiation": np.random.uniform(100, 500, 1000),
        "SolarEnergy": np.random.uniform(10, 25, 1000),
        "UVIndex": np.random.randint(0, 11, 1000),
        "RainTomorrow": np.random.randint(0, 2, 1000)
    })

    features = [
        "TempMax", "TempMin", "TempAvg", "FeelsLikeMax", "FeelsLikeMin", "FeelsLikeAvg",
        "DewPoint", "Humidity", "WindGust", "WindSpeed", "Visibility", "SolarRadiation",
        "SolarEnergy", "UVIndex"
    ]

    X = weather_data[features]
    y = weather_data["RainTomorrow"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X, y

# Load the trained model and data
rf_model, X, y = load_and_train_model()

st.title("Weather Prediction: Rain Tomorrow?")
st.write("""
This application predicts whether it will rain tomorrow based on today's weather data.
Input today's weather conditions to get a prediction.
""")

# Sidebar title
st.sidebar.title("Input Today's Weather (you can enter the real data or you can enter weather's situation that you felt like!)")

# Sidebar inputs for today's weather
temp_max = st.sidebar.number_input("Maximum Temperature (°C):", value=30.0)
temp_min = st.sidebar.number_input("Minimum Temperature (°C):", value=20.0)
temp_avg = st.sidebar.number_input("Average Temperature (°C):", value=25.0)
feels_like_max = st.sidebar.number_input("Feels Like Max (°C):", value=32.0)
feels_like_min = st.sidebar.number_input("Feels Like Min (°C):", value=22.0)
feels_like_avg = st.sidebar.number_input("Feels Like Avg (°C):", value=27.0)
dew_point = st.sidebar.number_input("Dew Point (°C):", value=18.0)
humidity = st.sidebar.slider("Humidity (%):", min_value=0, max_value=100, value=60)
wind_gust = st.sidebar.number_input("Wind Gust (km/h):", value=20.0)
wind_speed = st.sidebar.number_input("Wind Speed (km/h):", value=10.0)
visibility = st.sidebar.number_input("Visibility (km):", value=10.0)
solar_radiation = st.sidebar.number_input("Solar Radiation (W/m^2):", value=200.0)
solar_energy = st.sidebar.number_input("Solar Energy (MJ/m^2):", value=15.0)
uv_index = st.sidebar.slider("UV Index:", min_value=0, max_value=11, value=5)

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    "TempMax": [temp_max],
    "TempMin": [temp_min],
    "TempAvg": [temp_avg],
    "FeelsLikeMax": [feels_like_max],
    "FeelsLikeMin": [feels_like_min],
    "FeelsLikeAvg": [feels_like_avg],
    "DewPoint": [dew_point],
    "Humidity": [humidity],
    "WindGust": [wind_gust],
    "WindSpeed": [wind_speed],
    "Visibility": [visibility],
    "SolarRadiation": [solar_radiation],
    "SolarEnergy": [solar_energy],
    "UVIndex": [uv_index]
})

# Prediction
if st.sidebar.button("Predict"):
    prediction = rf_model.predict(input_data)
    result = "Rain" if prediction[0] == 1 else "No Rain"
    color = "red" if result == "Rain" else "blue"

    st.subheader("Prediction Result")
    st.markdown(f"""
    <h2 style='text-align: center;'>
        Based on today's weather data, it is predicted that there will be 
        <span style='color: {color};'><b>{result}</b></span> tomorrow.
    </h2>
    """, unsafe_allow_html=True)


