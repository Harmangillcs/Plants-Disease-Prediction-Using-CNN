from pathlib import Path
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from huggingface_hub import hf_hub_download
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CLASS_INDICES_PATH = BASE_DIR / "app" / "class_indices.json"

# Hugging Face model details
HF_MODEL_REPO = "harmancs/Plants_Disease_Prediction"
HF_MODEL_FILENAME = "plant_disease_prediction_model.h5"


HF_TOKEN = os.getenv("HF_TOKEN")  
MODEL_PATH = hf_hub_download(
    repo_id="harmancs/Plants_Disease_Prediction",
    filename="plant_disease_prediction_model.h5",
    use_auth_token=HF_TOKEN
)
model = tf.keras.models.load_model(MODEL_PATH)
st.success("Model loaded successfully!")

# Load class indices
with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)


def load_and_preprocess_image(image, target_size=(224, 224)):
    if not isinstance(image, Image.Image):
        image = Image.open(image)
        img = image.resize(target_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0
        return img_array


def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


st.title("🌿 Plant Disease Classifier")

uploaded_image = st.file_uploader(
    "Upload a plant leaf image (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image.resize((150, 150)), caption="Uploaded Image")

    with col2:
        if st.button("Classify"):
            with st.spinner("Predicting..."):
                prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f"Prediction: **{prediction}**")
