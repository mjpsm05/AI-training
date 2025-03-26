import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
model_name = "mjpsm/Confidence-Statement-Model-final"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to make predictions
def predict_statement(statement: str):
    # Tokenize the input statement
    inputs = tokenizer(statement, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the logits and apply softmax to get the probabilities
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get the predicted class (0: lack of self-confidence, 1: self-confidence)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    # Map the predicted class back to the label
    label_mapping = {0: 'lack of self-confidence', 1: 'self-confidence'}
    
    # Return the predicted label and probability for self-confidence
    return label_mapping[predicted_class], probabilities[0][predicted_class].item()

# Streamlit app interface
st.title("Self-Confidence Chatbot")

# Input box for user to chat with the bot
user_input = st.text_input("Give me a statement and I will tell if you have self-confidence or lack of self-confidence:", "")

# Display bot's response when the user enters a statement
if user_input:
    # Get prediction from the model
    label, probability = predict_statement(user_input)
    
    # Display the result
    st.write(f"**Bot's Response**: The statement indicates the user has **{label}** ")
    
    
    # Display the chat history
    for message in st.session_state.history:
        st.write(message)

