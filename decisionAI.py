import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return "".join(chunk.text for chunk in response)

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")
st.title("Welcome to the AI Assistant")
st.write("What would you like to do today?")

options = [
    "Analyze a resume",
    "View applicants",
    "View prospects",
    "View job openings",
    "Other (type your request)"
]

choice = st.radio("Please select an option:", options)

if choice == "Other (type your request)":
    user_input = st.text_input("Please describe what you want to do:")
    if user_input:
        with st.spinner("DecisionAI is thinking..."):
            response = get_gemini_response(user_input)
        st.success("DecisionAI's response:")
        st.write(response)
else:
    st.success(f"You selected: {choice}")
