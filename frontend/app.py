import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="MDC-Net Steel Defect Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 MDC-Net Steel Defect Detection")
st.write("Upload a steel surface image to predict the defect type, location, and caption.")

uploaded_file = st.file_uploader(
    "Upload a steel surface image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=400
    )

    if st.button("Predict Defect"):

        with st.spinner("Running MDC-Net..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            response = requests.post(
                API_URL,
                files=files
            )

        if response.status_code == 200:

            result = response.json()

            st.success("Prediction completed!")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Prediction")
                st.write("**Defect:**", result["defect"])
                st.write("**Location:**", result["location"])

            with col2:
                st.subheader("Generated Caption")
                st.write(result["caption"])

        else:
            st.error("Prediction failed.")
            st.write(response.text)