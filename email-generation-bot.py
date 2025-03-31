import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model and tokenizer
model_name = "mjpsm/Email_generation_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure the pad token is set (this is necessary if it's not already)
tokenizer.pad_token = tokenizer.eos_token

# Function to generate email response
def generate_email_response(prompt, max_length=150):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Generate a response using the model
    outputs = model.generate(inputs["input_ids"], max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    
    # Decode the generated output and remove the prompt from it
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the response if it appears at the beginning
    response_text = generated_text.replace(prompt, "").strip()
    
    return response_text

# Streamlit app interface
st.title("Email Generation Chatbot")

# User input (free-form text)
user_query = st.text_area("Enter your query:")

# When the user submits the input
if st.button("Generate Email Response"):
    # Generate the email response based on user input
    response = generate_email_response(user_query)
    
    # Display the generated email response
    st.subheader("Generated Email Response")
    st.write(response)
