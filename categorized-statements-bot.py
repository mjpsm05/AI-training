import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
model_name = 'mjpsm/Categorized-Interview-statements'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Move model to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the mapping from numerical labels to categories
id_to_category = {  # Replace with actual mapping
    0: "Education",
    1: "Early-life",
    2: "Experience",
}

# Function to predict the category of a statement
def predict_category(statement):
    # Tokenize the input statement
    inputs = tokenizer(statement, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Move inputs to device (GPU if available)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted label (highest logit)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Convert numerical label back to category
    category = id_to_category[predicted_class]
    return category

# Streamlit interface
st.title("Categorized Interview Statement Classifier")
st.write("Enter a statement below to classify it into 3 categories: (Education, Early-life, or Experience).")

# User input (text area for the statement)
statement = st.text_area("Statement")

# If the user submits a statement
if st.button("Classify Statement") and statement:
    # Call the prediction function
    category = predict_category(statement)
    
    # Display the result
    st.write(f"Predicted Category: {category}")
