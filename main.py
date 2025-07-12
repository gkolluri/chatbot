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

# Initialize DB
if 'db' not in st.session_state:
    st.session_state['db'] = get_db()

# User authentication/session management
if 'user_authenticated' not in st.session_state:
    st.session_state['user_authenticated'] = False

if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None

if 'user_name' not in st.session_state:
    st.session_state['user_name'] = None

st.title("💬 Local Chatbot with OpenAI, LangGraph, and MongoDB")

# User authentication section
if not st.session_state['user_authenticated']:
    st.markdown("### 👋 Welcome! Please enter your name to start chatting.")
    
    with st.form("user_auth"):
        user_name = st.text_input("Your Name:", placeholder="Enter your name here...")
        submit_button = st.form_submit_button("Start Chatting")
        
        if submit_button and user_name.strip():
            # Get or create user
            user_id, name = st.session_state['db'].get_or_create_user(user_name.strip())
            
            # Store user info in session state
            st.session_state['user_id'] = user_id
            st.session_state['user_name'] = name
            st.session_state['user_authenticated'] = True
            
            # Initialize chatbot for this user
            st.session_state['chatbot'] = Chatbot(
                db=st.session_state['db'],
                user_id=user_id,
                user_name=name
            )
            
            st.success(f"Welcome back, {name}! Your session has been loaded.")
            st.rerun()
        elif submit_button and not user_name.strip():
            st.error("Please enter your name to continue.")

else:
    # User is authenticated - show chat interface
    chatbot = st.session_state['chatbot']
    user_info = chatbot.get_user_info()
    
    # Display user info in sidebar
    st.sidebar.markdown(f"### 👤 User: {user_info['user_name']}")
    st.sidebar.markdown(f"**User ID:** `{user_info['user_id'][:8]}...`")
    
    # Display conversation turn counter and question statistics
    turns = chatbot.get_conversation_turns()
    stats = chatbot.get_question_stats()

    st.sidebar.metric("Conversation Turns", turns)
    st.sidebar.metric("Turns until next question", 3 - (turns % 3) if turns % 3 != 0 else 0)

    # Display question statistics in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Question Statistics")
    st.sidebar.metric("Accepted Questions", stats['accepted_count'])
    st.sidebar.metric("Rejected Questions", stats['rejected_count'])
    st.sidebar.metric("Total Questions", stats['total_questions'])

    # Logout button
    if st.sidebar.button("🚪 Logout"):
        st.session_state['user_authenticated'] = False
        st.session_state['user_id'] = None
        st.session_state['user_name'] = None
        st.session_state['chatbot'] = None
        st.rerun()

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

    # Debug section with both accepted and rejected questions
    with st.expander("🔍 Debug Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Accepted Questions")
            accepted_questions = chatbot.get_accepted_questions()
            if accepted_questions:
                for i, question in enumerate(accepted_questions, 1):
                    st.write(f"{i}. {question}")
            else:
                st.write("No accepted questions yet.")
        
        with col2:
            st.markdown("### ❌ Rejected Questions")
            rejected_questions = chatbot.get_rejected_questions()
            if rejected_questions:
                for i, question in enumerate(rejected_questions, 1):
                    st.write(f"{i}. {question}")
            else:
                st.write("No rejected questions yet.") 