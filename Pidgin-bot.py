import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model and tokenizer
model_name = "mjpsm/gpt2-vernacular-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize Streamlit app
st.title("Nigerian Pidgin Translator")
st.write("Chat with the bot in regular english to get a reponse back in pidgin, responses may be limited due to small dataset")

# Function to generate responses
def generate_response(user_input):
    # Encode the user input
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    # Generate the model's response
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    # Decode the generated response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the user input part from the response
    if user_input.lower() in response.lower():
        response = response.lower().replace(user_input.lower(), "").strip()

    return response

# Input text box for the user to type their message
user_input = st.text_input("You: ", "")

# Display response when the user enters a message
if user_input:
    response = generate_response(user_input)
    print(response)
    st.write(f"Chatbot: {response}")
