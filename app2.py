import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.title("🌱 Greenhouse Temperature Prediction")

# Load dataset
file_path = "small_data.csv"
df = pd.read_csv(file_path)

df.columns = df.columns.str.lower().str.strip()
df = df.dropna()

X = df[['humidity', 'pressure', 'lux']]
y = df['temperature']

# Train model inside app
model = RandomForestRegressor(n_estimators=20, random_state=42)
model.fit(X, y)

st.write("Enter values:")

humidity = st.number_input("Humidity", 0.0, 100.0, 50.0)
pressure = st.number_input("Pressure", 900.0, 1100.0, 1013.0)
lux = st.number_input("Lux", 0.0, 10000.0, 500.0)

if st.button("Predict"):
    prediction = model.predict([[humidity, pressure, lux]])
    st.success(f"Temperature: {prediction[0]:.2f} °C")
