import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from gtts import gTTS
from io import BytesIO

# Page Configuration
st.set_page_config(page_title="Leaf Disease Classifier", page_icon="🌿", layout="centered")

# Load model and class indices
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# CSS Styling
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        color: #2e8b57;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .footer {
        position: fixed;
        bottom: 10px;
        width: 100%;
        text-align: center;
        font-size: 14px;
        color: #aaaaaa;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🌿 Leaf Disease Classifier 🌿</div>', unsafe_allow_html=True)

# Language selection
st.subheader("🌐 Select Language for Voice Output")
lang_option = st.selectbox("Choose a language:", [
    "English", "Hindi", "Telugu", "Tamil", "Kannada", "French", "Spanish", "German"
])

language_map = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "French": "fr",
    "Spanish": "es",
    "German": "de"
}

# Function to Load and Preprocess Image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Prediction Function
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    Accuracy = float(np.max(predictions)) * 100
    return predicted_class_name, Accuracy

# Upload Image
uploaded_image = st.file_uploader("📄 Upload an image (JPG/PNG):", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="🖼 Uploaded Leaf Image", width=300)

    st.markdown("---")
    st.write("Click the button below to classify the image:")

    if st.button('🚀 Classify'):
        with st.spinner("Analyzing image..."):
            prediction, Accuracy = predict_image_class(model, uploaded_image, class_indices)
            st.success(f"✅ *Prediction:* {prediction}")
            st.info(f"📊 *Accuracy:* {Accuracy:.2f}%")

            if Accuracy < 60:
                st.warning("⚠ The Accuracy is relatively low. Please try with a clearer image.")

            # Voice Feedback
            lang_code = language_map[lang_option]
            speak_text = f"The predicted disease is {prediction} with {Accuracy:.1f} percent Accuracy."
            tts = gTTS(text=speak_text, lang=lang_code)
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            st.audio(audio_bytes.getvalue(), format='audio/mp3')
# Sample Disease Gallery
st.subheader("🖼 Sample Disease Gallery")

sample_dir = os.path.join(working_dir, "sample_images")

col1, col2, col3 = st.columns(3)

with col1:
    try:
        st.image(os.path.join(sample_dir, "tomato.jpg"), caption="Tomato Yellow Leaf Disease", width=200)
    except:
        st.warning("Tomato image not found.")

with col2:
    try:
        st.image(os.path.join(sample_dir, "corn.jpg"), caption="Corn Blight", width=200)
    except:
        st.warning("Corn image not found.")

with col3:
    try:
        st.image(os.path.join(sample_dir, "apple.jpg"), caption="Apple Scab", width=200)
    except:
        st.warning("Apple image not found.")

# Footer
st.markdown('<div class="footer">Built with ❤ using Streamlit, TensorFlow, and gTTS</div>', unsafe_allow_html=True)