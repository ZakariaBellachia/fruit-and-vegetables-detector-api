import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "https://fruit-backend-1-0-1.onrender.com/predict")

st.title("Fruits and Vegetables Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if st.button("Predict"):
    if uploaded_file is None:
        st.warning("Please upload an image first!")
    else:
        image_bytes = uploaded_file.read()
        st.image(uploaded_file, caption="Uploaded Image", width=400)

        files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
        
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()

            st.success("Prediction completed successfully!")

            label = result["predicted_label"].upper()
            confidence = result["confidence"]

            st.subheader("Result")
            st.write(f"Prediction: {label}")
            st.write(f"Confidence: {confidence}%")

        else:
            st.error("Backend error. Is the API running?")
            