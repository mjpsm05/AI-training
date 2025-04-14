import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Map model labels to human-readable labels
LABEL_MAP = {
    "LABEL_0": "Bad",
    "LABEL_1": "Mediocre",
    "LABEL_2": "Good"
}

# Load model and tokenizer (force CPU usage)
@st.cache_resource
def load_model():
    # Check for CUDA (GPU) availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("mjpsm/check-ins-classifier")
    model = AutoModelForSequenceClassification.from_pretrained("mjpsm/check-ins-classifier")
    model.to(device)  # Move model to the available device
    return tokenizer, model, device

tokenizer, model, device = load_model()

st.title("Check-In Classifier")
st.write("Enter your check-in so I can see if it's **Good**, **Mediocre**, or **Bad**.")

# User input
user_input = st.text_area("💬 Your Check-In Message:")

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

        # Move input tensors to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

        # Get prediction
        predicted_class = torch.argmax(probs, dim=1).item()
        label_key = model.config.id2label[predicted_class]
        human_label = LABEL_MAP.get(label_key, label_key)
        confidence = torch.max(probs).item()

        st.success(f"🧾 Prediction: **{human_label}** (Confidence: {confidence:.2%})")

        # Show all class probabilities with human-readable labels
        st.subheader("📊 Class Probabilities:")
        for idx, prob in enumerate(probs[0]):
            label_key = model.config.id2label.get(idx, f"LABEL_{idx}")
            label_name = LABEL_MAP.get(label_key, label_key)
            st.write(f"{label_name}: {prob:.2%}")



