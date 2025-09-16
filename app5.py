from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os
from dotenv import load_dotenv
import time

# Load .env file
load_dotenv()

# ✅ Setup LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "ZubiGPT")

# ✅ Get Gemini API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY not found in .env file")
    st.stop()

# Create LLM client with Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user queries."),
    ("user", "Question: {question}")
])

# Streamlit Page Config
st.set_page_config(page_title="ZubiGPT Chatbot", page_icon="🤖", layout="centered")

# Custom CSS for Beautiful Chat Bubbles
st.markdown("""
    <style>
        body { background-color: #f4f6f9; }
        .main-title { font-size: 40px !important; color: #4CAF50; text-align: center; margin-bottom: 20px; font-weight: bold; }
        .user-msg { background-color: #DCF8C6; color: #000; padding: 12px; border-radius: 15px; margin: 10px 0; text-align: right; max-width: 75%; margin-left: auto; font-size: 16px; }
        .bot-msg { background-color: #E8E8E8; color: #000; padding: 12px; border-radius: 15px; margin: 10px 0; text-align: left; max-width: 75%; margin-right: auto; font-size: 16px; }
        .footer { text-align: center; margin-top: 40px; color: gray; font-size: 14px; }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-title'>🤖 ZubiGPT</h1>", unsafe_allow_html=True)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.text_input("💬 Ask me anything:")

if user_input:
    # Save user message first
    st.session_state.chat_history.append(("user", user_input))
    
    # Placeholder for bot response (typing effect)
    bot_placeholder = st.empty()

    # Run chain (this will be traced by LangSmith ✅)
    chain = prompt | llm
    response = chain.invoke({"question": user_input})
    bot_response = response.content

    # Typing effect
    displayed_text = ""
    for char in bot_response:
        displayed_text += char
        bot_placeholder.markdown(f"<div class='bot-msg'> ZubiGPT: {displayed_text}</div>", unsafe_allow_html=True)
        time.sleep(0.02)

    # Save full bot response to history
    st.session_state.chat_history.append(("bot", bot_response))

# Display full chat history (previous messages)
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"<div class='user-msg'> You: {msg}</div>", unsafe_allow_html=True)
    elif role == "bot" and msg != bot_response:  # skip currently typing message
        st.markdown(f"<div class='bot-msg'> ZubiGPT: {msg}</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>Built By Muhammad Zubair</div>", unsafe_allow_html=True)
