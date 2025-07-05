"""
Streamlit Chatbot App with OpenAI, LangGraph, and MongoDB

Setup Instructions:
1. Install dependencies: pip install -r requirements.txt
2. Set environment variables (see below) in a .env file or your shell:
   - OPENAI_API_KEY=your_openai_api_key
   - MONGODB_ATLAS_URI=your_mongodb_atlas_uri (optional)
3. Run the app: streamlit run main.py

Environment Variables:
- OPENAI_API_KEY: Your OpenAI API key (required)
- MONGODB_ATLAS_URI: MongoDB Atlas connection string (optional; uses mock DB if not set)

File Structure:
- main.py: Streamlit UI and app entry point
- chatbot.py: Chatbot logic (OpenAI, LangGraph)
- db.py: Database logic (MongoDB/mocking)

"""
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env if present
load_dotenv()

# Import app logic
from chatbot import Chatbot
from db import get_db

# Initialize DB and Chatbot
if 'db' not in st.session_state:
    st.session_state['db'] = get_db()
if 'chatbot' not in st.session_state:
    st.session_state['chatbot'] = Chatbot(db=st.session_state['db'])

st.title("💬 Local Chatbot with OpenAI, LangGraph, and MongoDB")

# Display conversation turn counter
chatbot = st.session_state['chatbot']
turns = chatbot.get_conversation_turns()
st.sidebar.metric("Conversation Turns", turns)
st.sidebar.metric("Turns until next question", 3 - (turns % 3) if turns % 3 != 0 else 0)

# Display chat history
conversation = chatbot.get_conversation()

for i, (role, msg) in enumerate(conversation):
    if role == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")

# Handle follow-up question if there is one
last_question = chatbot.get_last_question()
if last_question:
    st.markdown("---")
    st.markdown(f"**Follow-up Question:** {last_question}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes", key="yes_btn"):
            result = chatbot.process_user_message("yes")
            st.rerun()
    with col2:
        if st.button("No", key="no_btn"):
            result = chatbot.process_user_message("no")
            st.rerun()

# User input (only show if no follow-up question is pending)
if not last_question:
    user_input = st.text_input("Type your message:", key="user_input")
    if st.button("Send") and user_input:
        result = chatbot.process_user_message(user_input)
        st.rerun()

# Optionally, show rejected questions for debugging
with st.expander("Show rejected questions (debug)"):
    st.write(list(chatbot.rejected_questions)) 