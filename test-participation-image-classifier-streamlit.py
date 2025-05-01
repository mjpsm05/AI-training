import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import torch.nn.functional as F


# App title
st.title("📸 Participation Classifier")
st.write("Take a picture of yourself and the AI will classify whether you're participating or not!")

# Load model and processor
model_path = "mjpsm/mazzy-specified-participation-image-classifier-updated"
model = AutoModelForImageClassification.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)
model.eval()

# Take photo using webcam
uploaded_file = st.camera_input("Take a picture")

if uploaded_file is not None:
    # Load image and display it
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your Image", use_container_width=True)

    # Preprocess and predict
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Prediction
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class_idx]


    # Add this after logits
    probs = F.softmax(logits, dim=1)
    confidence = probs[0][predicted_class_idx].item()

    st.success(f"🧠 Predicted Label: **{label}** with {confidence:.2%} confidence")
