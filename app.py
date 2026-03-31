

import streamlit as st
import pickle
import requests
import os
import numpy as np
import tensorflow_hub as hub

# =========================
# 🖥️ Page Config
# =========================
st.set_page_config(
    page_title="Fault Prediction System",
    page_icon="🔧",
    layout="centered"
)

st.title(" Fault Prediction System")
st.markdown("Enter symptoms and get the predicted fault")

# =========================
# 🔗 Hugging Face Links
# =========================
MODEL_URL = "https://huggingface.co/philemonndayi/roadside-files/resolve/main/RandomForest_fault_model.pkl"
ENCODER_URL = "https://huggingface.co/philemonndayi/roadside-files/resolve/main/le_fault.pkl"

# =========================
#  Download Function
# =========================
@st.cache_resource
def download_file(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename}..."):
            response = requests.get(url)
            with open(filename, "wb") as f:
                f.write(response.content)
    return filename

# =========================
# 📦 Load Models
# =========================
@st.cache_resource
def load_models():
    model_path = download_file(MODEL_URL, "model.pkl")
    encoder_path = download_file(ENCODER_URL, "encoder.pkl")

    model = pickle.load(open(model_path, "rb"))
    encoder = pickle.load(open(encoder_path, "rb"))

    # Load embedding model (same as training)
    embed_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    return model, encoder, embed_model

# =========================
# ⏳ Loading
# =========================
with st.spinner("Loading models... please wait ⏳"):
    model, encoder, embed_model = load_models()

# =========================
# 🧠 Input Section
# =========================
symptoms = st.text_area(
    "Enter symptoms",
    placeholder="e.g. engine overheating, battery not charging"
)

# =========================
# ⚡ Prediction
# =========================
if st.button("Predict Fault"):
    if symptoms.strip() == "":
        st.warning("⚠️ Please enter symptoms")
    else:
        with st.spinner("Analyzing symptoms..."):
            # Convert text → embedding
            features = embed_model([symptoms]).numpy()

            # Predict
            prediction = model.predict(features)

            # Convert number → fault name
            fault = encoder.inverse_transform(prediction)[0]

        st.success(f"🔧 Predicted Fault: {fault}")