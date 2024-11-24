import streamlit as st
import requests

# Set up the Streamlit app
st.title("Weather Information")

# Input city name
city = st.text_input("Enter the name of a city:", "")

# Fetch and display weather data
if city:
    # API setup
    api_key = "b484d8848a0046a5f2fd5142a1e9397e"  # Replace with your OpenWeatherMap API key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    # Fetch the data
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]

        st.write(f"**Weather**: {weather.capitalize()}")
        st.write(f"**Temperature**: {temp}°C")
        st.write(f"**Feels Like**: {feels_like}°C")
    else:
        st.error("City not found or invalid API response.")
    