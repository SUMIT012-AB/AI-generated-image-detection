import sys
import os

# TensorFlow installed to short path to work around Windows Long Path limitation.
# Remove this line if TensorFlow is installed in the default site-packages.
_TF_PATH = r"C:\tf_pkg"
if os.path.isdir(_TF_PATH) and _TF_PATH not in sys.path:
    sys.path.insert(0, _TF_PATH)

import json
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# ==========================
# Constants
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "my_model.keras")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "class_indices.json")
MAX_FILE_SIZE_MB = 10
IMG_SIZE = (224, 224)

# ==========================
# Streamlit Page Settings
# ==========================
st.set_page_config(page_title="🕵‍♂ Fake & Real Images Detection", page_icon="🖼", layout="centered")
st.title("🕵‍♂ Fake & Real Images Detection")
st.write("📤 Upload an image (jpg, jpeg, or png) to predict whether it's Real Art or AI-Generated Art 🤖")

# ==========================
# Load Class Indices
# ==========================
@st.cache_data
def load_class_indices():
    """Load class-to-index mapping from the JSON file produced during training."""
    if not os.path.exists(CLASS_INDICES_PATH):
        st.warning("⚠ class_indices.json not found — using default mapping.")
        return {"Fake": 0, "Real": 1}
    with open(CLASS_INDICES_PATH, "r") as f:
        return json.load(f)

class_indices = load_class_indices()
# Build index-to-class reverse mapping
index_to_class = {v: k for k, v in class_indices.items()}

# ==========================
# Load Model (Cached)
# ==========================
@st.cache_resource
def load_model_cached():
    """Load the trained Keras model from disk."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.stop()
    # compile=False prevents optimizer-related compatibility errors
    return load_model(MODEL_PATH, compile=False, safe_mode=True)

model = load_model_cached()

# ==========================
# Prediction Function
# ==========================
def predict_image(img, model, img_size=IMG_SIZE):
    """
    Preprocess the image and run inference.

    Returns:
        label (str): Human-readable prediction label.
        confidence (float): Confidence score (0-1).
    """
    # Ensure image is in RGB format
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize and preprocess the image
    img_resized = img.resize(img_size)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    prob = model.predict(img_array, verbose=0)[0][0]

    # Use class_indices to determine the label
    # The model outputs a sigmoid probability for the positive class (index 1)
    positive_class = index_to_class.get(1, "Real")
    negative_class = index_to_class.get(0, "Fake")

    if prob > 0.5:
        label = f"🖼 {positive_class}  ✅"
        confidence = prob
    else:
        label = f"🤖 {negative_class}  ❌"
        confidence = 1 - prob

    return label, confidence

# ==========================
# File Upload & Display
# ==========================
uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- Input validation: file size ---
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"⚠ File too large ({file_size_mb:.1f} MB). Maximum allowed size is {MAX_FILE_SIZE_MB} MB.")
    else:
        try:
            img = Image.open(uploaded_file)
            # Validate that the image can actually be read
            img.verify()
            # Re-open after verify (verify() makes the object unusable)
            uploaded_file.seek(0)
            img = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"⚠ The uploaded file is not a valid image: {e}")
            st.stop()

        display_width = 350

        # Split screen layout
        col1, col2 = st.columns([1.2, 1])

        with col1:
            # Display uploaded image
            st.image(img, caption="🖼 Uploaded Image", width=display_width)

        with col2:
            # Make prediction
            try:
                label, confidence = predict_image(img, model)
            except Exception as e:
                st.error(f"⚠ Prediction failed: {e}")
                st.stop()

            if label is not None:
                # Choose colors based on result
                is_real = "Real" in label
                bg_color = "#d4edda" if is_real else "#f8d7da"
                text_color = "#155724" if is_real else "#721c24"
                border_color = "#c3e6cb" if is_real else "#f5c6cb"

                # Display result box
                st.markdown(
                    f"""
                    <div style='border-radius: 15px; padding: 20px; margin-top: 30px;
                                text-align: left; font-size: 20px; font-weight: bold;
                                background-color: {bg_color};
                                color: {text_color};
                                border: 2px solid {border_color};
                                box-shadow: 0px 4px 10px rgba(0,0,0,0.1);'>
                        🔎 Prediction: {label}<br><br>
                        <span style='font-size:16px; font-weight:normal;'>Confidence: {confidence:.2f} 🔹</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
