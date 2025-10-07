import streamlit as st
from PIL import Image
import numpy as np
import pickle
import json
import base64

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.title("Bovine Breed Classifier")

st.markdown("""
    <style>
    .stAppViewContainer {
        background-image: url('bg.jpg');
    }
    </style>""",
    unsafe_allow_html=True
    )

def set_bg_image(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()  # encode as base64

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: auto;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image file
set_bg_image("bg.jpg")

# Load ML model and labels once
@st.cache_resource
def load_model():
    with open("data/models/cattle_breed_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("data/models/breed_labels.pkl", "rb") as f:
        categories = pickle.load(f)
    return clf, categories

clf, categories = load_model()

# Load CNN feature extractor once
@st.cache_resource
def load_feature_extractor():
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

base_model = load_feature_extractor()

# Feature extraction
def extract_features(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = base_model.predict(x, verbose=0)
    return features.flatten()

# Predict breed
def get_breed(bovine_img):
    features = extract_features(bovine_img)
    pred = clf.predict([features])[0]
    return categories[pred]

# Upload and predict
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns([1,1])

    with col1:
        st.image(img, caption="Uploaded image", use_container_width=True)
    with col2:
        breed = get_breed(img)
        with open("data/json_files/breed_details.json") as f:
            breed_json = json.load(f)
        
        info = breed_json[breed.lower()]

        st.markdown(
            f"""
            <div style="
                padding: 8px;
                margin: 0;
                ">
                <h3 style="margin:0">{breed}</h3>
                <p style="margin:2px 0"><b>Weight:</b> {info['weight']}</p>
                <p style="margin:2px 0"><b>Milk production:</b> {info['milk_production']}</p>
                <p style="margin:2px 0"><b>Max age:</b> {info['max_age']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
