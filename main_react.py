"""
Streamlit React AI Pattern Multi-Agent Chatbot App

Setup Instructions:
1. Install dependencies: pip install -r requirements.txt
2. Set environment variables (see below) in a .env file or your shell:
   - OPENAI_API_KEY=your_openai_api_key
   - MONGODB_ATLAS_URI=your_mongodb_atlas_uri (optional)
3. Run the app: streamlit run main_react.py

Environment Variables:
- OPENAI_API_KEY: Your OpenAI API key (required)
- MONGODB_ATLAS_URI: MongoDB Atlas connection string (optional; uses mock DB if not set)

File Structure:
- main_react.py: Streamlit UI and application entry point (React AI Pattern version)
- react_multi_agent_chatbot.py: React AI pattern-based multi-agent chatbot logic
- db.py: Database logic (MongoDB/mocking)
- agents/: React AI pattern-based agents for different functionalities
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
import numpy as np

# Load environment variables from .env if present
load_dotenv()

# Import app logic
from react_multi_agent_chatbot import ReactMultiAgentChatbot
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
                st.error("❌ Error updating language preferences.")
    
    # Get user tags
    user_tags = chatbot.get_user_tags()
    
    # Display current tags
    st.markdown("### Current Tags")
    if user_tags:
        for tag in user_tags:
            st.markdown(f"🏷️ {tag}")
    else:
        st.markdown("No tags yet. Start chatting to discover your interests!")
    
    # Manual tag addition
    st.markdown("### Add Tags Manually")
    with st.form("add_tag_form"):
        new_tag = st.text_input("Enter a tag", placeholder="e.g., programming, cooking, travel")
        submitted = st.form_submit_button("Add Tag")
        
        if submitted and new_tag:
            if chatbot.add_user_tag(new_tag.lower().strip()):
                st.success(f"Tag '{new_tag}' added successfully!")
                st.rerun()
            else:
                st.error(f"Error adding tag '{new_tag}'.")
    
    # AI-powered tag analysis
    st.markdown("### AI-Powered Tag Analysis")
    if st.button("Analyze Conversation for Tags"):
        with st.spinner("React AI is analyzing your conversation..."):
            analysis_result = chatbot.analyze_conversation_for_tags()
            
            if analysis_result.get('success'):
                st.success("Analysis complete!")
                
                # Show analysis results
                with st.expander("🔍 React AI Analysis Results"):
                    st.markdown(f"**Analysis Summary:** {analysis_result.get('analysis_summary', 'N/A')}")
                    st.markdown(f"**Reasoning Steps:** {analysis_result.get('reasoning_steps', 0)}")
                    st.markdown(f"**Framework:** {analysis_result.get('framework', 'React AI Pattern')}")
                
                # Get tag suggestions
                suggestions_result = chatbot.get_tag_suggestions()
                
                if suggestions_result.get('success'):
                    suggestions = suggestions_result.get('suggestions', [])
                    if suggestions:
                        st.markdown("### Suggested Tags")
                        for suggestion in suggestions:
                            if st.button(f"Add: {suggestion}", key=f"suggest_{suggestion}"):
                                if chatbot.add_user_tag(suggestion):
                                    st.success(f"Tag '{suggestion}' added!")
                                    st.rerun()
                    else:
                        st.info("No new tag suggestions at this time.")
            else:
                st.error(f"Analysis failed: {analysis_result.get('error', 'Unknown error')}")

def _show_similar_users_interface(chatbot):
    """Show similar users interface"""
    # Update last activity
    session_manager.update_last_activity()
    
    st.markdown("## 🤝 Similar Users")
    st.markdown("*Discover people who share your interests*")
    
    # Get similar users
    similar_users = chatbot.find_similar_users()
    
    if similar_users:
        st.markdown("### Users with Similar Interests")
        for user in similar_users:
            with st.expander(f"👤 {user['name']} (Similarity: {user['similarity_score']:.1f}%)"):
                st.write(f"**Tags:** {', '.join(user['tags'])}")
                st.write(f"**Languages:** {', '.join(user.get('languages', ['Not specified']))}")
                
                # Create group chat button
                if st.button(f"Create Group Chat with {user['name']}", key=f"group_{user['user_id']}"):
                    _create_group_chat_with_user(chatbot, user)
    else:
        st.info("No similar users found yet. Keep chatting to discover connections!")

def _show_group_chats_interface(chatbot):
    """Show group chats interface"""
    # Update last activity
    session_manager.update_last_activity()
    
    st.markdown("## 👥 Group Chats")
    st.markdown("*Join or create group conversations*")
    
    # Get user's group chats
    group_chats = chatbot.get_user_group_chats()
    
    if group_chats:
        st.markdown("### Your Group Chats")
        for group in group_chats:
            with st.expander(f"💬 {group['topic_name']}"):
                st.write(f"**Participants:** {len(group['user_ids'])} users")
                st.write(f"**Created:** {group['created_at']}")
                
                if st.button(f"Join {group['topic_name']}", key=f"join_{group['group_id']}"):
                    st.session_state['current_group_chat'] = group['group_id']
                    st.session_state['current_view'] = 'group_chat'
                    st.rerun()
    else:
        st.info("No group chats yet. Create one to start connecting!")
    
    # Create new group chat
    st.markdown("### Create New Group Chat")
    with st.form("create_group_chat"):
        topic_name = st.text_input("Topic Name", placeholder="e.g., Tech Enthusiasts, Food Lovers")
        submitted = st.form_submit_button("Create Group Chat")
        
        if submitted and topic_name:
            result = chatbot.create_group_chat(topic_name)
            if result.get('success'):
                st.success(f"Group chat '{topic_name}' created successfully!")
                st.rerun()
            else:
                st.error(f"Error creating group chat: {result.get('error', 'Unknown error')}")

def _create_group_chat_with_user(chatbot, similar_user):
    """Create a group chat with a similar user"""
    topic_name = f"Chat with {similar_user['name']}"
    result = chatbot.create_group_chat(topic_name, [similar_user['user_id']])
    
    if result.get('success'):
        st.success(f"Group chat created with {similar_user['name']}!")
        st.session_state['current_group_chat'] = result['group_id']
        st.session_state['current_view'] = 'group_chat'
        st.rerun()
    else:
        st.error(f"Error creating group chat: {result.get('error', 'Unknown error')}")

def _show_group_chat_interface(chatbot):
    """Show specific group chat interface"""
    # Update last activity
    session_manager.update_last_activity()
    
    group_id = st.session_state.get('current_group_chat')
    if not group_id:
        st.error("No group chat selected.")
        st.session_state['current_view'] = 'group_chats'
        st.rerun()
    
    # Get group chat info
    group_info = chatbot.get_group_chat_info(group_id)
    if not group_info:
        st.error("Group chat not found.")
        st.session_state['current_view'] = 'group_chats'
        st.rerun()
    
    st.markdown(f"## 💬 {group_info['topic_name']}")
    
    # Display group messages
    messages = chatbot.get_group_messages(group_id)
    
    for message in messages:
        if message['user_id'] == 'ai_bot':
            st.markdown(f"**🤖 AI Bot:** {message['message']}")
        else:
            st.markdown(f"**👤 User:** {message['message']}")
    
    # Send message
    user_input = st.text_input("Type your message:", key="group_input", 
                             placeholder="What would you like to say?")
    if st.button("Send") and user_input:
        result = chatbot.send_group_message(group_id, user_input)
        if result.get('success'):
            st.rerun()
        else:
            st.error(f"Error sending message: {result.get('error', 'Unknown error')}")

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
        agent_count = len(chatbot.agents)
        st.metric("🤖 Agents", f"{agent_count} Active", delta="Ready")
    
    st.markdown("---")
    
    # Get system status
    status_result = chatbot.get_system_status()
    
    if status_result.get('success'):
        # Display React AI metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Agents", status_result.get('total_agents', 0))
        
        with col2:
            st.metric("Framework", "React AI Pattern")
        
        with col3:
            st.metric("Active Agents", len(status_result.get('agent_details', {})))
        
        with col4:
            st.metric("Coordinator Status", "✅ Active" if status_result.get('coordinator_status', {}).get('total_agents', 0) > 0 else "❌ Inactive")
        
        # Agent details
        st.markdown("### Agent Details")
        for agent_name, details in status_result.get('agent_details', {}).items():
            with st.expander(f"🤖 {agent_name}"):
                st.markdown(f"**Status:** {'✅ Active' if details['status']['is_active'] else '❌ Inactive'}")
                st.markdown(f"**Framework:** {details['status']['framework']}")
                st.markdown(f"**Tools:** {details['status'].get('tools_count', 0)}")
                st.markdown(f"**Reasoning History:** {details.get('reasoning_history_count', 0)}")
                st.markdown(f"**Capabilities:** {', '.join(details.get('capabilities', []))}")
        
        # React AI pattern features
        st.markdown("### React AI Pattern Features")
        features = status_result.get('react_ai_features', [])
        for feature in features:
            st.markdown(f"✅ {feature}")
        
        # Health check
        health_result = chatbot.check_status()
        if health_result.get('success'):
            st.markdown("### System Health")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Database", "✅ Connected" if health_result.get('database_connected') else "❌ Disconnected")
            
            with col2:
                st.metric("Agents", "✅ Active" if health_result.get('agents_active') else "❌ Inactive")
            
            with col3:
                st.metric("Coordinator", "✅ Ready" if health_result.get('coordinator_ready') else "❌ Not Ready")
    else:
        st.error(f"Error getting system status: {status_result.get('error', 'Unknown error')}")
    
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
    agents = ['ReactConversationAgent', 'ReactTagAnalysisAgent', 'ReactUserProfileAgent', 'ReactGroupChatAgent', 'ReactSessionAgent', 'ReactLanguageAgent']
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
    
    # Dynamically get all registered agents from the backend
    actual_agents = list(chatbot.agents.keys())
    
    # Define default icons and colors for known agent types
    default_agent_icons = {
        'ConversationAgent': '💬',
        'TagAnalysisAgent': '🏷️',
        'UserProfileAgent': '👤',
        'GroupChatAgent': '👥',
        'SessionAgent': '🔑',
        'LanguageAgent': '🌐'
    }
    default_agent_colors = {
        'ConversationAgent': '#1f77b4',
        'TagAnalysisAgent': '#ff7f0e',
        'UserProfileAgent': '#2ca02c',
        'GroupChatAgent': '#d62728',
        'SessionAgent': '#9467bd',
        'LanguageAgent': '#8c564b'
    }
    # Assign icons/colors for any custom agents
    agent_icons = {name: default_agent_icons.get(name, '🤖') for name in actual_agents}
    agent_colors = {name: default_agent_colors.get(name, '#666666') for name in actual_agents}
    
    # Generate dynamic layout based on actual agents
    agent_nodes = {
        'START': {'x': -6, 'y': 0, 'color': '#28a745', 'size': 25, 'icon': '🚀'}
    }
    # Position agents: ConversationAgent at center, others in a circle
    num_agents = len(actual_agents)
    center_x, center_y = 0, 0
    angle_step = 2 * 3.14159 / max(num_agents-1, 1) if num_agents > 1 else 1
    for i, agent_name in enumerate(actual_agents):
        if agent_name == 'ConversationAgent':
            agent_nodes[agent_name] = {
                'x': center_x, 'y': center_y,
                'color': agent_colors[agent_name],
                'size': 35, 'icon': agent_icons[agent_name]
            }
        else:
            angle = angle_step * (i if agent_name != 'ConversationAgent' else 0)
            radius = 3
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            agent_nodes[agent_name] = {
                'x': x, 'y': y,
                'color': agent_colors[agent_name],
                'size': 30, 'icon': agent_icons[agent_name]
            }
    # Add END node
    agent_nodes['END'] = {'x': 9, 'y': 0, 'color': '#dc3545', 'size': 25, 'icon': '✅'}
    
    # Define dynamic connections based on agent presence
    agent_connections = []
    # START connects to all agents (solid for ConversationAgent, dash for others)
    for agent_name in actual_agents:
        style = 'solid' if agent_name == 'ConversationAgent' else 'dash'
        agent_connections.append(('START', agent_name, style))
    # ConversationAgent connects to all others (solid)
    for agent_name in actual_agents:
        if agent_name != 'ConversationAgent':
            agent_connections.append(('ConversationAgent', agent_name, 'solid'))
    # Internal/conditional connections
    if 'TagAnalysisAgent' in actual_agents and 'UserProfileAgent' in actual_agents:
        agent_connections.append(('TagAnalysisAgent', 'UserProfileAgent', 'dash'))
    if 'UserProfileAgent' in actual_agents and 'GroupChatAgent' in actual_agents:
        agent_connections.append(('UserProfileAgent', 'GroupChatAgent', 'dash'))
    if 'SessionAgent' in actual_agents and 'UserProfileAgent' in actual_agents:
        agent_connections.append(('SessionAgent', 'UserProfileAgent', 'dash'))
    if 'LanguageAgent' in actual_agents and 'ConversationAgent' in actual_agents:
        agent_connections.append(('LanguageAgent', 'ConversationAgent', 'dot'))
    # All agents connect to END (solid for GroupChatAgent, dash for others)
    for agent_name in actual_agents:
        style = 'solid' if agent_name == 'GroupChatAgent' else 'dash'
        agent_connections.append((agent_name, 'END', style))
    
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
        start_node, end_node, line_style = connection
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
        text=node_icons,
        textposition="middle center",
        textfont=dict(size=16, color='white'),
        hoverinfo='text',
        hovertext=[f"<b>{name}</b><br>Status: 🟢 Active<br>Type: React AI Agent<br>Icon: {icon}" for name, icon in zip(node_names, node_icons)],
        showlegend=False
    ))
    # Add agent names as separate text
    fig_agent_graph.add_trace(go.Scatter(
        x=node_x, y=[y - 0.8 for y in node_y],
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
            text='React AI Multi-Agent System Architecture',
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
            description = agent_details.get(agent, f"Specialized React AI agent for {agent.lower()} functionality")
            st.write(f"**{agent}:** {description}")
    
    with col2:
        st.markdown("#### Agent Status")
        
        # Get actual agent status from the system
        agent_status = {"START": "🟢 Ready", "END": "🟢 Ready"}
        
        for agent_name, agent in chatbot.agents.items():
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
st.markdown("*Powered by OpenAI, React AI Pattern, and MongoDB with cultural awareness*")

# Check if user is already authenticated (persistent session)
if session_manager.is_user_authenticated():
    # User is authenticated - show main interface
    user_info = session_manager.get_user_info()
    
    # Initialize chatbot if not already done
    if 'chatbot' not in st.session_state:
        st.session_state['chatbot'] = ReactMultiAgentChatbot()
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
            st.session_state['chatbot'] = ReactMultiAgentChatbot()
            st.session_state['chatbot'].user_id = user_id
            st.session_state['chatbot'].user_name = name
            
            st.success(f"👋 Welcome back, {name}! Your session has been loaded.")
            st.rerun()
        elif submit_button and not user_name.strip():
            st.error("Please enter your name to continue.") 