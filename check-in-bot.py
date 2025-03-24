import streamlit as st
from transformers import pipeline

# Define the label mapping
id2label = {"LABEL_0": "Bad", "LABEL_1": "Mediocre", "LABEL_2": "Good"}

# Load the pipeline with the correct model
pipe = pipeline("text-classification", model="mjpsm/check-ins-classifier")

# Streamlit UI
st.title("Check-in Sentiment Classifier 🤖")
st.write("Enter a check-in message below and get its classification!")

# User input
user_input = st.text_area("Enter your check-in text:", "")

if st.button("Classify"):
    if user_input.strip():  # Check if input is not empty
        result = pipe(user_input)

        # Convert model output to human-readable labels
        for r in result:
            r["label"] = id2label[r["label"]]

        # Display the result
        st.write(f"**Predicted Rating:** {result[0]['label']} 🎯")
        st.write(f"**Confidence Score:** {round(result[0]['score'] * 100, 2)}%")
    else:
        st.warning("Please enter some text before classifying!")

