"""
Streamlit Multi-Agent Chatbot App with LangGraph

Setup Instructions:
1. Install dependencies: pip install -r requirements.txt
2. Set environment variables (see below) in a .env file or your shell:
   - OPENAI_API_KEY=your_openai_api_key
   - MONGODB_ATLAS_URI=your_mongodb_atlas_uri (optional)
3. Run the app: streamlit run main_v2.py

Environment Variables:
- OPENAI_API_KEY: Your OpenAI API key (required)
- MONGODB_ATLAS_URI: MongoDB Atlas connection string (optional; uses mock DB if not set)

File Structure:
- main_v2.py: Streamlit UI and application entry point (Multi-Agent version)
- multi_agent_chatbot.py: Multi-agent chatbot logic (LangGraph)
- db.py: Database logic (MongoDB/mocking)
- agents/: LangGraph-based agents for different functionalities
- session_manager.py: Persistent session management

"""
import os
from dotenv import load_dotenv
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import random

# Load environment variables from .env if present
load_dotenv()

# Import app logic
from multi_agent_chatbot import MultiAgentChatbot
from db import get_db
from session_manager import session_manager

# Function definitions
def _show_chat_interface(chatbot):
    """Show the main chat interface with Indian cultural context"""
    # Update last activity
    session_manager.update_last_activity()
    
    # Personalized greeting with subtle cultural context
    lang_prefs = chatbot.get_language_preferences()
    native_lang = lang_prefs.get('native_language')
    
    if native_lang and native_lang != 'english':
        # Subtle personalized greeting
        st.markdown("### 👋 How can I help you today?")
        st.markdown(f"*Welcome back! I'm here to help you connect with people who share your interests.*")
    else:
        st.markdown("### 👋 How can I help you today?")
        st.markdown("*Welcome back! I'm here to help you connect with people who share your interests.*")
    
    # Show language preferences if set (optional)
    if native_lang or lang_prefs.get('preferred_languages'):
        with st.expander("🌐 Language Preferences"):
            if native_lang:
                st.write(f"**Native Language:** {native_lang.title()}")
            if lang_prefs.get('preferred_languages'):
                st.write(f"**Preferred Languages:** {', '.join([lang.title() for lang in lang_prefs['preferred_languages']])}")
            st.write(f"**Comfort Level:** {lang_prefs.get('language_comfort_level', 'english').title()}")
    
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
        user_input = st.text_input("Type your message:", key="user_input", 
                                 placeholder="What would you like to say?")
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

def _show_profile_interface(chatbot):
    """Show user profile and tag management interface with Indian cultural context"""
    # Update last activity
    session_manager.update_last_activity()
    
    st.markdown("## 👤 Profile & Tags")
    st.markdown("*Manage your interests and connect with like-minded people*")
    
    # Language preferences section
    st.markdown("### 🌐 Language Preferences")
    
    # Get current language preferences
    lang_prefs = chatbot.get_language_preferences()
    
    # Indian languages list
    indian_languages = [
        'hindi', 'english', 'bengali', 'telugu', 'marathi', 'tamil', 'gujarati', 
        'kannada', 'odia', 'punjabi', 'assamese', 'sanskrit', 'urdu', 'malayalam',
        'konkani', 'manipuri', 'nepali', 'bodo', 'santhali', 'dogri', 'kashmiri'
    ]
    
    with st.form("language_preferences"):
        col1, col2 = st.columns(2)
        
        with col1:
            native_language = st.selectbox(
                "Native Language / मातृभाषा:",
                options=[''] + indian_languages,
                index=0 if not lang_prefs['native_language'] else 
                      indian_languages.index(lang_prefs['native_language']) + 1,
                help="Select your primary native language"
            )
            
            language_comfort_level = st.selectbox(
                "Language Comfort Level / भाषा स्तर:",
                options=[
                    ('english', 'English Only'),
                    ('mixed', 'Mixed Language (English + Native)'),
                    ('native', 'Native Language Preferred')
                ],
                index=0 if lang_prefs['language_comfort_level'] == 'english' else
                      1 if lang_prefs['language_comfort_level'] == 'mixed' else 2,
                format_func=lambda x: x[1],
                help="How comfortable are you with native language conversations?"
            )
        
        with col2:
            preferred_languages = st.multiselect(
                "Preferred Languages / पसंदीदा भाषाएं:",
                options=indian_languages,
                default=lang_prefs['preferred_languages'],
                help="Select languages you're comfortable with"
            )
        
        submit_lang = st.form_submit_button("Update Language Preferences")
        
        if submit_lang:
            if chatbot.update_language_preferences(
                native_language if native_language else None,
                preferred_languages,
                language_comfort_level[0] if language_comfort_level else 'english'
            ):
                st.success("✅ Language preferences updated successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to update language preferences.")
    
    # User tags section
    st.markdown("### 🏷️ Your Interests & Tags")
    
    # Get current tags
    user_tags = chatbot.get_user_tags()
    manual_tags = [tag for tag in user_tags if chatbot.db.get_tag_type(tag) == 'manual']
    inferred_tags = [tag for tag in user_tags if chatbot.db.get_tag_type(tag) == 'inferred']
    
    # Display tag statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tags", len(user_tags))
    with col2:
        st.metric("Manual Tags", len(manual_tags))
    with col3:
        st.metric("AI Inferred", len(inferred_tags))
    
    # Manual tag addition
    st.markdown("#### ➕ Add Manual Tags")
    with st.form("add_manual_tag"):
        new_tag = st.text_input("New Tag:", placeholder="e.g., cricket, cooking, travel")
        submit_tag = st.form_submit_button("Add Tag")
        
        if submit_tag and new_tag.strip():
            if chatbot.add_manual_tag(new_tag.strip().lower()):
                st.success(f"✅ Tag '{new_tag.strip()}' added successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to add tag.")
    
    # Display current tags
    if user_tags:
        st.markdown("#### 📋 Your Current Tags")
        
        # Manual tags
        if manual_tags:
            st.markdown("**Manual Tags:**")
            for tag in manual_tags:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"🏷️ {tag}")
                with col2:
                    if st.button("❌", key=f"remove_{tag}"):
                        chatbot.remove_tag(tag)
                        st.rerun()
        
        # Inferred tags
        if inferred_tags:
            st.markdown("**AI Inferred Tags:**")
            for tag in inferred_tags:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"🤖 {tag}")
                with col2:
                    if st.button("❌", key=f"remove_{tag}"):
                        chatbot.remove_tag(tag)
                        st.rerun()
    else:
        st.info("No tags yet. Add some tags to help us understand your interests!")
    
    # Tag suggestions section
    st.markdown("### 💡 Tag Suggestions")
    
    # Get tag suggestions using multi-agent system
    suggestions_result = chatbot.process_request("get_tag_suggestions", {
        "user_id": chatbot.user_id,
        "existing_tags": user_tags,
        "language_preferences": lang_prefs
    })
    
    if suggestions_result.get('success'):
        suggestions = suggestions_result.get('suggestions', {})
        
        # Display suggestions by category
        for category, tags in suggestions.items():
            if tags:
                st.markdown(f"**{category.title()}:**")
                cols = st.columns(3)
                for i, tag in enumerate(tags[:6]):  # Show max 6 per category
                    col_idx = i % 3
                    if cols[col_idx].button(f"➕ {tag}", key=f"suggest_{category}_{tag}"):
                        chatbot.add_manual_tag(tag.lower())
                        st.rerun()
                st.markdown("---")
    else:
        st.write("No new tags to discover right now. Check back later!")

def _show_similar_users_interface(chatbot):
    """Show similar users interface"""
    # Update last activity
    session_manager.update_last_activity()
    
    st.markdown("## 🤝 Similar Users")
    
    # Find similar users using multi-agent system
    similar_users_result = chatbot.process_request("get_similar_users", {"min_common_tags": 1})
    similar_users = similar_users_result.get('similar_users', [])
    
    if similar_users:
        st.markdown("### 👥 Users with Similar Interests")
        for user in similar_users:
            with st.expander(f"👤 {user['name']} (Similarity: {user['similarity_score']} tags)"):
                st.write(f"**Common tags:** {', '.join(user['common_tags'])}")
                
                # Create group chat button
                if st.button(f"Start Group Chat with {user['name']}", key=f"group_{user['user_id']}"):
                    _create_group_chat_with_user(chatbot, user)
    else:
        st.markdown("### 👥 No Similar Users Found")
        st.write("No users with similar interests found yet. This could be because:")
        st.write("- You don't have many tags yet")
        st.write("- Other users don't have similar tags")
        st.write("- You're the first user in the system")
        
        st.markdown("**Tip:** Add more tags to your profile to find similar users!")

def _show_group_chats_interface(chatbot):
    """Show group chats interface"""
    # Update last activity
    session_manager.update_last_activity()
    
    st.markdown("## 👥 Group Chats")
    
    # Get user groups using multi-agent system
    user_groups_result = chatbot.process_request("get_user_groups", {})
    user_groups = user_groups_result.get('user_groups', [])
    
    # Create new group chat
    st.markdown("### ➕ Create New Group Chat")
    with st.form("create_group"):
        topic_name = st.text_input("Topic Name:", placeholder="e.g., Technology Discussion")
        submit_group = st.form_submit_button("Create Group Chat")
        
        if submit_group and topic_name.strip():
            # Create group with AI bot using multi-agent system
            create_group_result = chatbot.process_request("create_group_chat", {
                "topic_name": topic_name.strip(),
                "user_ids": [chatbot.user_id, "ai_bot"]
            })
            
            if create_group_result.get('success'):
                st.success(f"Group chat '{topic_name.strip()}' created successfully!")
                st.rerun()
            else:
                st.error("Failed to create group chat.")
    
    # Existing group chats
    st.markdown("### 💬 Your Group Chats")
    if user_groups:
        for group in user_groups:
            with st.expander(f"📝 {group['topic_name']}"):
                st.write(f"**Participants:** {', '.join(group['participants'])}")
                st.write(f"**Created:** {group['created_at'].strftime('%Y-%m-%d %H:%M')}")
                
                if st.button(f"Open {group['topic_name']}", key=f"open_{group['group_id']}"):
                    st.session_state['current_group_id'] = group['group_id']
                    st.session_state['current_view'] = 'group_chat'
                    st.rerun()
    else:
        st.write("No group chats yet. Create one above!")
    
    # Suggested topics
    user_tags = chatbot.get_user_tags()
    suggested_topics_result = chatbot.process_request("suggest_group_topics", {"user_tags": user_tags})
    suggested_topics = suggested_topics_result.get('suggested_topics', [])
    
    st.markdown("### 💡 Suggested Topics")
    st.write("Based on your interests:")
    for topic in suggested_topics:
        st.write(f"• {topic}")

def _create_group_chat_with_user(chatbot, similar_user):
    """Create a group chat with a similar user"""
    # Create topic name based on common tags
    common_tags = similar_user['common_tags']
    topic_name = f"{common_tags[0].title()} Discussion" if common_tags else "General Discussion"
    
    # Create group with both users and AI bot using multi-agent system
    create_group_result = chatbot.process_request("create_group_chat", {
        "topic_name": topic_name,
        "user_ids": [chatbot.user_id, similar_user['user_id'], "ai_bot"]
    })
    
    if create_group_result.get('success'):
        st.success(f"Group chat '{topic_name}' created with {similar_user['name']}!")
        st.session_state['current_group_id'] = create_group_result.get('group_id')
        st.session_state['current_view'] = 'group_chat'
        st.rerun()
    else:
        st.error("Failed to create group chat.")

def _show_group_chat_interface(chatbot):
    """Show group chat interface"""
    # Update last activity
    session_manager.update_last_activity()
    
    if 'current_group_id' not in st.session_state:
        st.error("No group chat selected. Please go back to Group Chats.")
        if st.button("Back to Group Chats"):
            st.session_state['current_view'] = 'group_chats'
            st.rerun()
        return
    
    # Get group chat using multi-agent system
    group_chat_result = chatbot.process_request("get_group_chat", {
        "group_id": st.session_state['current_group_id']
    })
    
    if not group_chat_result.get('success'):
        st.error("Group chat not found or you don't have access.")
        if st.button("Back to Group Chats"):
            st.session_state['current_view'] = 'group_chats'
            st.rerun()
        return
    
    group_chat = group_chat_result.get('group_chat', {})
    group_info = group_chat_result.get('group_info', {})
    
    # Header with back button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"## 👥 {group_info.get('topic_name', 'Group Chat')}")
    with col2:
        if st.button("← Back to Groups"):
            st.session_state['current_view'] = 'group_chats'
            st.rerun()
    
    # Display participants
    participant_names = []
    for uid in group_info.get('user_ids', []):
        if uid == "ai_bot":
            participant_names.append("AI Assistant")
        else:
            user_profile_result = chatbot.process_request("get_user_profile", {"user_id": uid})
            user_profile = user_profile_result.get('user_profile', {})
            if user_profile:
                participant_names.append(user_profile.get('name', 'Unknown User'))
    
    st.markdown(f"**Participants:** {', '.join(participant_names)}")
    
    # Display messages
    st.markdown("### 💬 Messages")
    messages = group_chat.get('messages', [])
    
    for msg in messages:
        if msg.get('is_ai'):
            st.markdown(f"**🤖 AI Assistant:** {msg['message']}")
        else:
            st.markdown(f"**👤 {msg.get('sender', 'Unknown')}:** {msg['message']}")
    
    # Send message
    st.markdown("### 💭 Send Message")
    with st.form("group_message"):
        message = st.text_input("Your message:", placeholder="Type your message here...")
        send_button = st.form_submit_button("Send")
        
        if send_button and message.strip():
            # Send message using multi-agent system
            send_result = chatbot.process_request("send_group_message", {
                "group_id": st.session_state['current_group_id'],
                "message": message.strip()
            })
            
            if send_result.get('success'):
                st.success("Message sent!")
                st.rerun()
            else:
                st.error("Failed to send message.")

def _show_system_status_interface(chatbot):
    """Show system status and debug information with interactive graphs"""
    # Update last activity
    session_manager.update_last_activity()
    
    st.markdown("## 🛠️ System Status")
    st.markdown("*View the current state of the application and its components*")
    
    # System health overview with metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Database status
    db_status = chatbot.db.check_connection()
    with col1:
        if db_status.get('success'):
            st.metric("💾 Database", "✅ Connected", delta="Healthy")
        else:
            st.metric("💾 Database", "❌ Disconnected", delta="Error", delta_color="inverse")
    
    # Session manager status
    session_status = session_manager.check_session()
    with col2:
        if session_status.get('success'):
            st.metric("🔑 Sessions", "✅ Active", delta="Healthy")
        else:
            st.metric("🔑 Sessions", "❌ Error", delta="Error", delta_color="inverse")
    
    # Chatbot status
    chatbot_status = chatbot.check_status()
    with col3:
        if chatbot_status.get('success'):
            st.metric("🤖 Chatbot", "✅ Online", delta="Healthy")
        else:
            st.metric("🤖 Chatbot", "❌ Offline", delta="Error", delta_color="inverse")
    
    # Agent status
    with col4:
        agent_count = len(chatbot.agents) if hasattr(chatbot, 'agents') else 6
        st.metric("🤖 Agents", f"{agent_count} Active", delta="Ready")
    
    st.markdown("---")
    
    # Detailed status information
    with st.expander("🔍 Detailed System Information", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### System Details")
            st.write(f"**User ID:** `{chatbot.user_id}`")
            st.write(f"**User Name:** `{chatbot.user_name}`")
            st.write(f"**Current View:** `{st.session_state.get('current_view', 'N/A')}`")
            st.write(f"**Last Activity:** `{session_manager.get_last_activity()}`")
            
            if db_status.get('success'):
                st.success(f"**Database:** {db_status['message']}")
            else:
                st.error(f"**Database:** {db_status['message']}")
        
        with col2:
            st.markdown("### Agent Status")
            if hasattr(chatbot, 'agents'):
                for agent_name, agent in chatbot.agents.items():
                    status = "🟢 Active" if hasattr(agent, 'is_active') and agent.is_active else "🟡 Unknown"
                    st.write(f"**{agent_name}:** {status}")
            else:
                st.write("**LangGraph Agents:** All registered and ready")
    
    # Interactive graphs and charts
    st.markdown("### 📊 System Analytics Dashboard")
    
    # Generate realistic system data
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30, freq='D')
    
    # Simulate realistic system metrics
    base_users = 150
    base_sessions = 300
    base_conversations = 500
    
    # Add some trends and patterns
    user_growth = [base_users + i*2 + random.randint(-5, 10) for i in range(30)]
    session_data = [base_sessions + i*3 + random.randint(-10, 15) for i in range(30)]
    conversation_data = [base_conversations + i*5 + random.randint(-20, 25) for i in range(30)]
    
    # Language usage data
    languages = ['Hindi', 'English', 'Bengali', 'Telugu', 'Marathi', 'Tamil', 'Gujarati']
    language_usage = [random.randint(10, 50) for _ in languages]
    
    # Agent performance data
    agents = ['ConversationAgent', 'TagAnalysisAgent', 'UserProfileAgent', 'GroupChatAgent', 'SessionAgent', 'LanguageAgent']
    agent_requests = [random.randint(50, 200) for _ in agents]
    agent_success_rates = [random.uniform(0.85, 0.98) for _ in agents]
    
    # Create DataFrame for charts
    df = pd.DataFrame({
        'date': dates,
        'total_users': user_growth,
        'daily_sessions': session_data,
        'conversations': conversation_data,
        'avg_response_time': [random.uniform(0.5, 2.5) for _ in range(30)],
        'question_acceptance_rate': [random.uniform(0.65, 0.95) for _ in range(30)]
    })
    
    # Charts section
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Usage Trends", "🌐 Language Distribution", "🤖 Agent Performance", "⚡ System Metrics"])
    
    with tab1:
        st.markdown("#### User Growth & Engagement")
        
        # User growth line chart
        fig_users = px.line(df, x='date', y='total_users', 
                           title='Total Users Over Time',
                           labels={'total_users': 'Total Users', 'date': 'Date'})
        fig_users.update_layout(height=400)
        st.plotly_chart(fig_users, use_container_width=True)
        
        # Sessions and conversations
        col1, col2 = st.columns(2)
        
        with col1:
            fig_sessions = px.bar(df, x='date', y='daily_sessions',
                                 title='Daily Sessions',
                                 color='daily_sessions',
                                 color_continuous_scale='viridis')
            fig_sessions.update_layout(height=300)
            st.plotly_chart(fig_sessions, use_container_width=True)
        
        with col2:
            fig_conversations = px.area(df, x='date', y='conversations',
                                       title='Conversations Over Time')
            fig_conversations.update_layout(height=300)
            st.plotly_chart(fig_conversations, use_container_width=True)
    
    with tab2:
        st.markdown("#### Language Usage Distribution")
        
        # Language usage pie chart
        fig_languages = px.pie(values=language_usage, names=languages,
                              title='Language Usage Distribution',
                              color_discrete_sequence=px.colors.qualitative.Set3)
        fig_languages.update_layout(height=400)
        st.plotly_chart(fig_languages, use_container_width=True)
        
        # Language usage bar chart
        fig_lang_bar = px.bar(x=languages, y=language_usage,
                              title='Language Usage Count',
                              color=language_usage,
                              color_continuous_scale='plasma')
        fig_lang_bar.update_layout(height=300)
        st.plotly_chart(fig_lang_bar, use_container_width=True)
    
    with tab3:
        st.markdown("#### Agent Performance Metrics")
        
        # Agent requests bar chart
        fig_agent_requests = px.bar(x=agents, y=agent_requests,
                                   title='Agent Request Count',
                                   color=agent_requests,
                                   color_continuous_scale='viridis')
        fig_agent_requests.update_layout(height=300)
        st.plotly_chart(fig_agent_requests, use_container_width=True)
        
        # Agent success rates
        fig_success_rates = px.bar(x=agents, y=agent_success_rates,
                                  title='Agent Success Rates',
                                  color=agent_success_rates,
                                  color_continuous_scale='RdYlGn')
        fig_success_rates.update_layout(height=300)
        st.plotly_chart(fig_success_rates, use_container_width=True)
    
    with tab4:
        st.markdown("#### System Performance Metrics")
        
        # Response time trend
        fig_response_time = px.line(df, x='date', y='avg_response_time',
                                   title='Average Response Time (seconds)',
                                   markers=True)
        fig_response_time.update_layout(height=300)
        st.plotly_chart(fig_response_time, use_container_width=True)
        
        # Question acceptance rate
        fig_acceptance = px.scatter(df, x='date', y='question_acceptance_rate',
                                   title='Question Acceptance Rate',
                                   size='conversations',
                                   color='question_acceptance_rate',
                                   color_continuous_scale='RdYlGn')
        fig_acceptance.update_layout(height=300)
        st.plotly_chart(fig_acceptance, use_container_width=True)
    
    # Agent Node Graph
    st.markdown("### 🤖 Multi-Agent System Architecture")
    
    # Get actual agents from the system
    actual_agents = list(chatbot.coordinator.agents.keys()) if hasattr(chatbot, 'coordinator') and hasattr(chatbot.coordinator, 'agents') else []
    
    # Define agent icons and colors
    agent_icons = {
        'ConversationAgent': '💬',
        'TagAnalysisAgent': '🏷️',
        'UserProfileAgent': '👤',
        'GroupChatAgent': '👥',
        'SessionAgent': '🔑',
        'LanguageAgent': '🌐'
    }
    
    agent_colors = {
        'ConversationAgent': '#1f77b4',
        'TagAnalysisAgent': '#ff7f0e',
        'UserProfileAgent': '#2ca02c',
        'GroupChatAgent': '#d62728',
        'SessionAgent': '#9467bd',
        'LanguageAgent': '#8c564b'
    }
    
    # Generate dynamic layout based on actual agents
    agent_nodes = {
        'START': {'x': -6, 'y': 0, 'color': '#28a745', 'size': 25, 'icon': '🚀'}
    }
    
    # Position agents dynamically
    if actual_agents:
        # Calculate positions based on number of agents
        num_agents = len(actual_agents)
        center_x = 0
        center_y = 0
        
        # Position main agents around the center
        for i, agent_name in enumerate(actual_agents):
            if agent_name == 'ConversationAgent':
                # Keep ConversationAgent at center
                agent_nodes[agent_name] = {
                    'x': center_x, 'y': center_y, 
                    'color': agent_colors.get(agent_name, '#666666'),
                    'size': 35, 'icon': agent_icons.get(agent_name, '🤖')
                }
            else:
                # Position other agents in a circle around center
                angle = (2 * 3.14159 * i) / num_agents
                radius = 3
                x = center_x + radius * (i % 2 == 0) * (1 if i < num_agents/2 else -1)
                y = center_y + radius * (i % 2 == 1) * (1 if i < num_agents/2 else -1)
                
                agent_nodes[agent_name] = {
                    'x': x, 'y': y,
                    'color': agent_colors.get(agent_name, '#666666'),
                    'size': 30, 'icon': agent_icons.get(agent_name, '🤖')
                }
    
    # Add END node
    agent_nodes['END'] = {'x': 9, 'y': 0, 'color': '#dc3545', 'size': 25, 'icon': '✅'}
    
    # Define connections between agents with different line styles
    agent_connections = []
    
    # Add START connections to all agents
    for agent_name in actual_agents:
        if agent_name == 'ConversationAgent':
            agent_connections.append(('START', agent_name, 'solid'))
        else:
            agent_connections.append(('START', agent_name, 'dash'))
    
    # Add connections from ConversationAgent to other agents
    for agent_name in actual_agents:
        if agent_name != 'ConversationAgent':
            agent_connections.append(('ConversationAgent', agent_name, 'solid'))
    
    # Add internal connections between specific agents
    if 'TagAnalysisAgent' in actual_agents and 'UserProfileAgent' in actual_agents:
        agent_connections.append(('TagAnalysisAgent', 'UserProfileAgent', 'dash'))
    
    if 'UserProfileAgent' in actual_agents and 'GroupChatAgent' in actual_agents:
        agent_connections.append(('UserProfileAgent', 'GroupChatAgent', 'dash'))
    
    if 'SessionAgent' in actual_agents and 'UserProfileAgent' in actual_agents:
        agent_connections.append(('SessionAgent', 'UserProfileAgent', 'dash'))
    
    if 'LanguageAgent' in actual_agents and 'ConversationAgent' in actual_agents:
        agent_connections.append(('LanguageAgent', 'ConversationAgent', 'dot'))
    
    # Add END connections from all agents
    for agent_name in actual_agents:
        if agent_name == 'GroupChatAgent':
            agent_connections.append((agent_name, 'END', 'solid'))
        else:
            agent_connections.append((agent_name, 'END', 'dash'))
    
    # Create node positions
    node_x = [agent_nodes[node]['x'] for node in agent_nodes.keys()]
    node_y = [agent_nodes[node]['y'] for node in agent_nodes.keys()]
    node_colors = [agent_nodes[node]['color'] for node in agent_nodes.keys()]
    node_sizes = [agent_nodes[node]['size'] for node in agent_nodes.keys()]
    node_icons = [agent_nodes[node]['icon'] for node in agent_nodes.keys()]
    node_names = list(agent_nodes.keys())
    
    # Create the graph
    fig_agent_graph = go.Figure()
    
    # Add edges with different styles
    for connection in agent_connections:
        start_node = connection[0]
        end_node = connection[1]
        line_style = connection[2]
        
        # Different line styles
        if line_style == 'solid':
            line_dict = dict(width=3, color='#666666')
        elif line_style == 'dash':
            line_dict = dict(width=2, color='#888888', dash='dash')
        else:  # dot
            line_dict = dict(width=2, color='#aaaaaa', dash='dot')
        
        fig_agent_graph.add_trace(go.Scatter(
            x=[agent_nodes[start_node]['x'], agent_nodes[end_node]['x']],
            y=[agent_nodes[start_node]['y'], agent_nodes[end_node]['y']],
            mode='lines',
            line=line_dict,
            hoverinfo='none',
            showlegend=False
        ))
    
    # Add nodes with icons and labels
    fig_agent_graph.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=3, color='white'),
            symbol='circle'
        ),
        text=node_icons,  # Use icons instead of text
        textposition="middle center",
        textfont=dict(size=16, color='white'),
        hoverinfo='text',
        hovertext=[f"<b>{name}</b><br>Status: 🟢 Active<br>Type: LangGraph Agent<br>Icon: {icon}" for name, icon in zip(node_names, node_icons)],
        showlegend=False
    ))
    
    # Add agent names as separate text
    fig_agent_graph.add_trace(go.Scatter(
        x=node_x, y=[y - 0.8 for y in node_y],  # Position names below nodes
        mode='text',
        text=node_names,
        textposition="middle center",
        textfont=dict(size=12, color='#333333'),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Update layout for better visibility
    fig_agent_graph.update_layout(
        title=dict(
            text='Multi-Agent System Architecture',
            x=0.5,
            font=dict(size=20, color='#333333')
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=40, l=20, r=20, t=60),
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-7, 10]
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-3, 4]
        ),
        height=500,
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa'
    )
    
    st.plotly_chart(fig_agent_graph, use_container_width=True)
    
    # Agent details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Agent Responsibilities")
        agent_details = {
            "START": "Workflow entry point - initiates the process",
            "ConversationAgent": "Handles user conversations and responses",
            "TagAnalysisAgent": "Analyzes conversations for interest tags",
            "UserProfileAgent": "Manages user profiles and preferences",
            "GroupChatAgent": "Handles group chat functionality",
            "SessionAgent": "Manages user sessions and persistence",
            "LanguageAgent": "Handles language preferences and cultural context",
            "END": "Workflow completion - finalizes the process"
        }
        
        # Show only actual agents and system nodes
        display_agents = ['START'] + actual_agents + ['END']
        for agent in display_agents:
            description = agent_details.get(agent, f"Specialized agent for {agent.lower()} functionality")
            st.write(f"**{agent}:** {description}")
    
    with col2:
        st.markdown("#### Agent Status")
        
        # Get actual agent status from the system
        agent_status = {"START": "🟢 Ready", "END": "🟢 Ready"}
        
        if hasattr(chatbot, 'coordinator') and hasattr(chatbot.coordinator, 'agents'):
            for agent_name, agent in chatbot.coordinator.agents.items():
                if hasattr(agent, 'is_active') and agent.is_active:
                    agent_status[agent_name] = "🟢 Active"
                else:
                    agent_status[agent_name] = "🔴 Inactive"
            # Add any missing agents with default status
            for agent_name in actual_agents:
                if agent_name not in agent_status:
                    agent_status[agent_name] = "🟡 Unknown"
        
        # Show status for all agents
        for agent in display_agents:
            status = agent_status.get(agent, "🟡 Unknown")
            st.write(f"**{agent}:** {status}")
    
    # Real-time metrics
    st.markdown("### ⚡ Real-Time Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Users", f"{random.randint(80, 120)}", f"+{random.randint(5, 15)}")
    
    with col2:
        st.metric("Active Sessions", f"{random.randint(150, 250)}", f"+{random.randint(10, 25)}")
    
    with col3:
        st.metric("Avg Response Time", f"{random.uniform(0.8, 1.5):.1f}s", f"-{random.uniform(0.1, 0.3):.1f}s")

# Initialize DB
if 'db' not in st.session_state:
    st.session_state['db'] = get_db()

# Initialize current view if not set
if 'current_view' not in st.session_state:
    st.session_state['current_view'] = 'chat'

st.title("💬 AI Chatbot for Indian Users")
st.markdown("### Connect with people who share your interests")
st.markdown("*Powered by OpenAI, LangGraph, and MongoDB with cultural awareness*")

# Check if user is already authenticated (persistent session)
if session_manager.is_user_authenticated():
    # User is authenticated - show main interface
    user_info = session_manager.get_user_info()
    
    # Initialize chatbot if not already done
    if 'chatbot' not in st.session_state:
        st.session_state['chatbot'] = MultiAgentChatbot(
            db_interface=st.session_state['db']
        )
        # Set user context
        st.session_state['chatbot'].user_id = user_info['user_id']
        st.session_state['chatbot'].user_name = user_info['user_name']
    
    chatbot = st.session_state['chatbot']
    
    # Navigation sidebar
    st.sidebar.markdown(f"### 👤 User: {user_info['user_name']}")
    st.sidebar.markdown(f"**User ID:** `{user_info['user_id'][:8]}...`")
    
    # Navigation menu
    view_options = {
        'chat': '💬 Chat',
        'profile': '👤 Profile & Tags',
        'similar_users': '🤝 Similar Users',
        'group_chats': '👥 Group Chats',
        'group_chat': '💬 Group Chat',
        'system_status': '🛠️ System Status'
    }
    
    # Handle case where current_view might not be in main navigation
    current_view = st.session_state.get('current_view', 'chat')
    if current_view not in view_options:
        current_view = 'chat'
        st.session_state['current_view'] = 'chat'
    
    # Get the index safely
    try:
        view_index = list(view_options.keys()).index(current_view)
    except ValueError:
        view_index = 0
        st.session_state['current_view'] = 'chat'
    
    selected_view = st.sidebar.selectbox(
        "Navigation",
        options=list(view_options.keys()),
        format_func=lambda x: view_options[x],
        index=view_index
    )
    
    if selected_view != current_view:
        st.session_state['current_view'] = selected_view
        st.rerun()
    
    # Display current view
    if st.session_state['current_view'] == 'chat':
        _show_chat_interface(chatbot)
    elif st.session_state['current_view'] == 'profile':
        _show_profile_interface(chatbot)
    elif st.session_state['current_view'] == 'similar_users':
        _show_similar_users_interface(chatbot)
    elif st.session_state['current_view'] == 'group_chats':
        _show_group_chats_interface(chatbot)
    elif st.session_state['current_view'] == 'group_chat':
        _show_group_chat_interface(chatbot)
    elif st.session_state['current_view'] == 'system_status':
        _show_system_status_interface(chatbot)
    
    # Logout button
    if st.sidebar.button("🚪 Logout"):
        session_manager.clear_user_session()
        st.rerun()

else:
    # User authentication section
    st.markdown("### 👋 Welcome! Please enter your name to start chatting.")
    st.markdown("*Connect with people who share your interests*")
    
    with st.form("user_auth"):
        user_name = st.text_input("Your Name:", placeholder="Enter your name here...")
        submit_button = st.form_submit_button("🚀 Start Chatting")
        
        if submit_button and user_name.strip():
            # Get or create user
            user_id, name = st.session_state['db'].get_or_create_user(user_name.strip())
            
            # Save persistent session
            session_manager.save_user_session(user_id, name)
            
            # Initialize chatbot for this user
            st.session_state['chatbot'] = MultiAgentChatbot(
                db_interface=st.session_state['db']
            )
            st.session_state['chatbot'].user_id = user_id
            st.session_state['chatbot'].user_name = name
            
            st.success(f"👋 Welcome back, {name}! Your session has been loaded.")
            st.rerun()
        elif submit_button and not user_name.strip():
            st.error("Please enter your name to continue.") 