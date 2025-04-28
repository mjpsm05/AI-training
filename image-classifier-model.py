import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Load your trained model and processor
model_name = "mjpsm/confidence-image-classifier"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Label mapping
id2label = {
    0: "Confident",
    1: "No Confidence",
    2: "Somewhat Confident"
}

# Streamlit app title
st.title("Confidence Detector 📸")

st.write("Take a picture or upload one, and the AI will predict your confidence level!")

# Upload or capture an image
uploaded_file = st.camera_input("Take a picture")  # Opens your webcam
# You could also use st.file_uploader("Upload a picture", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
    # Map prediction
    predicted_label = id2label[predicted_class_idx]
    
    # Show result
    st.markdown(f"## Prediction: **{predicted_label}** 🎯")
