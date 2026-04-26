import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

st.set_page_config(page_title="Plant Disease Detector", layout="centered")

# ------------------ FULL CUSTOM UI ------------------
st.markdown("""
<style>

/* HIDE STREAMLIT HEADER (Deploy button, menu) */
header {visibility: hidden;}
footer {visibility: hidden;}

/* DARK BACKGROUND */
.stApp {
background: linear-gradient(135deg, #0d3b2e, #1b5e20, #2e7d32);
                color: white;
}

/* MAIN CARD */
.block-container {
    background-color: rgba(0, 0, 0, 0.2);
    padding: 55px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    text-align: center;
    margin-top: 40px;
}

/* TITLE */
.title {
    font-size: 45px;
    font-weight: bold;
    color: #00e676;
    text-align: center;
}

/* SLOGANS */
.slogan {
    font-size: 18px;
    color: #b2ff59;
    text-align: center;
}

/* CENTER UPLOAD */
section[data-testid="stFileUploader"] {
    display: flex;
    justify-content: center;
}

/* UPLOAD BOX */
div[data-testid="stFileUploader"] {
    border: 2px dashed #00e676;
    padding: 20px;
    border-radius: 12px;
    background-color: rgba(255,255,255,0.05);
}

/* RESULT BOX */
.result-box {
    padding: 20px;
    border-radius: 10px;
    background: #00e676;
    color: green;
    margin-top: 20px;
    font-weight: bold;
    box-shadow: 0 0 15px #00e676;
}

/* TEXT FIX */
label, .css-1cpxqw2, .css-1v0mbdj {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
model = load_model("plant_model.h5")
class_names = sorted(os.listdir("dataset"))

# ------------------ HEADER ------------------
st.markdown('<div class="title">🌿 AgriScan AI</div>', unsafe_allow_html=True)

st.markdown('<div class="slogan"> AI Powered Farming</div>', unsafe_allow_html=True)
st.markdown('<div class="slogan"> Smart Crop Protection</div>', unsafe_allow_html=True)
st.markdown('<div class="slogan"> Detect Diseases Instantly</div>', unsafe_allow_html=True)

# ------------------ UPLOAD ------------------
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "png", "jpeg"])

# ------------------ PREDICTION ------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((224,224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    clean_name = predicted_class.replace("___", " ").replace("_", " ")

    st.markdown(f"""
    <div class="result-box">
        🌱 {clean_name}<br>
        Confidence: {confidence:.2f}%
    </div>
    """, unsafe_allow_html=True)

    if "healthy" in clean_name.lower():
        st.success("🟢 Plant is Healthy 🌱")
    else:
        st.error("🔴 Disease Detected! Take action ⚠️")