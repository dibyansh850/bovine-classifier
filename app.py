import streamlit as st
from PIL import Image
import numpy as np
import pickle

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.title("Bovine Breed Classifier")

# Load ML model and labels once
@st.cache_resource
def load_model():
    with open("models/cattle_breed_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("models/breed_labels.pkl", "rb") as f:
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
    st.image(img, caption="Uploaded image", use_container_width=True)
    st.text(f"Predicted breed: {get_breed(img)}")
