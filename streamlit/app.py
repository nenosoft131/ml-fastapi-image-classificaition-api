import streamlit as st
import requests
from PIL import Image
import io

st.title("Offline Image Classification")

uploaded_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width='stretch')
    if st.button("Classify"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post("http://127.0.0.1:8002/classify", files=files)
        
        if response.status_code == 200:
            predictions = response.json()["predictions"]
            st.subheader("Top Predictions:")
            for pred in predictions:
                st.write(f"{pred['label']} → Probability: {pred['probability']}")
        else:
            st.error("Error classifying image")