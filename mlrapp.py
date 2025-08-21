# app.py
import streamlit as st
import pickle
import numpy as np

# Load saved model
with open(r'C:\Users\DELL\Desktop\VS_code\Machine learning\house_price_model_mlr.pkl', "rb") as f:
    model = pickle.load(f)

st.title("🏠 House Price Prediction App")

st.write("Enter house details below:")

sqft_living = st.number_input("Living Area (sqft)", min_value=500, max_value=10000, step=50)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)

if st.button("Predict Price"):
    features = np.array([[sqft_living, bedrooms, bathrooms]])
    prediction = model.predict(features)
    st.success(f"Estimated Price: ${prediction[0]:,.2f}")
