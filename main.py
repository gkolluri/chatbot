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
- main.py: Streamlit UI and application entry point
- chatbot.py: Chatbot logic (OpenAI, LangGraph)
- db.py: Database logic (MongoDB/mocking)
- tag_analyzer.py: Tag inference and analysis
- group_chat.py: Group chat functionality
- session_manager.py: Persistent session management

"""
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env if present
load_dotenv()

# Import app logic
from chatbot import Chatbot
from db import get_db
from group_chat import GroupChatManager
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
    st.markdown("### 🏷️ Your Tags")
    
    user_tags = chatbot.get_user_tags()
    manual_tags = [tag for tag in user_tags if chatbot.db.get_user_tags(chatbot.user_id, "manual")]
    inferred_tags = [tag for tag in user_tags if tag not in manual_tags]
    
    # Manual tags
    st.markdown("#### Manual Tags")
    if manual_tags:
        for tag in manual_tags:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"🏷️ {tag}")
            with col2:
                if st.button(f"Remove {tag}", key=f"remove_{tag}"):
                    chatbot.remove_tag(tag)
                    st.rerun()
    else:
        st.write("No manual tags yet.")
    
    # Inferred tags
    st.markdown("#### Inferred Tags")
    if inferred_tags:
        for tag in inferred_tags:
            st.write(f"🤖 {tag} (AI inferred)")
    else:
        st.write("No inferred tags yet.")
    
    # Interactive Tag Discovery
    st.markdown("### 🎯 Discover New Tags")
    st.markdown("*Swipe through tags to find interests that match you!*")
    
    # Get popular and suggested tags for swiping
    popular_tags = chatbot.tag_analyzer.get_popular_tags(chatbot.db)
    current_user_tags = set(chatbot.get_user_tags())
    
    # Filter out tags user already has
    available_tags = [tag for tag in popular_tags if tag not in current_user_tags]
    
    if available_tags:
        # Create a session state for current card index
        if 'card_index' not in st.session_state:
            st.session_state.card_index = 0
        
        if 'swiped_tags' not in st.session_state:
            st.session_state.swiped_tags = {'liked': [], 'disliked': []}
        
        # Display current card
        if st.session_state.card_index < len(available_tags):
            current_tag = available_tags[st.session_state.card_index]
            
            # Card styling with emojis and icons
            tag_icons = {
                # Technology & Digital
                'technology': '💻', 'programming': '⌨️', 'ai': '🤖', 'startup': '🚀', 'digital': '📱',
                'mobile apps': '📲', 'web development': '🌐', 'data science': '📊',
                
                # Entertainment & Media
                'music': '🎵', 'movies': '🎬', 'bollywood': '🎭', 'gaming': '🎮', 'streaming': '📺',
                'podcasts': '🎧', 'comedy': '😄', 'dance': '💃',
                
                # Sports & Fitness
                'sports': '⚽', 'cricket': '🏏', 'fitness': '💪', 'yoga': '🧘', 'gym': '🏋️',
                'running': '🏃', 'swimming': '🏊', 'badminton': '🏸',
                
                # Food & Cuisine
                'food': '🍕', 'cooking': '👨‍🍳', 'indian food': '🍛', 'street food': '🌮',
                'biryani': '🍚', 'desserts': '🍰', 'healthy eating': '🥗',
                
                # Travel & Adventure
                'travel': '✈️', 'adventure': '🗺️', 'hiking': '🥾', 'photography': '📸',
                'backpacking': '🎒', 'road trips': '🚗', 'international travel': '🌍',
                
                # Arts & Culture
                'art': '🎨', 'culture': '🏺', 'classical music': '🎼', 'folk art': '🎪',
                'traditional crafts': '🛠️', 'painting': '🖼️',
                
                # Business & Career
                'business': '💼', 'entrepreneurship': '💡', 'career': '📈', 'finance': '💰',
                'investing': '📈', 'marketing': '📢', 'consulting': '🤝',
                
                # Education & Learning
                'education': '🎓', 'learning': '📚', 'online courses': '💻', 'languages': '🗣️',
                'reading': '📖', 'writing': '✍️', 'research': '🔬',
                
                # Health & Wellness
                'health': '🏥', 'wellness': '🌿', 'meditation': '🧘‍♀️', 'ayurveda': '🌱',
                'mental health': '🧠', 'nutrition': '🥑', 'fitness': '💪',
                
                # Lifestyle & Personal
                'fashion': '👗', 'beauty': '💄', 'lifestyle': '🌟', 'self-improvement': '📈',
                'motivation': '💪', 'productivity': '⚡', 'minimalism': '📦',
                
                # Social & Community
                'community': '👥', 'volunteering': '🤝', 'social work': '❤️', 'networking': '🌐',
                'mentoring': '👨‍🏫', 'leadership': '👑',
                
                # Creative & Hobbies
                'photography': '📸', 'writing': '✍️', 'poetry': '📝', 'music production': '🎹',
                'gardening': '🌱', 'diy': '🔧', 'crafts': '🎨',
                
                # Regional & Cultural
                'regional cinema': '🎬', 'classical dance': '💃', 'folk music': '🎵',
                'traditional festivals': '🎉', 'heritage': '🏛️',
                
                # Contemporary
                'sustainability': '♻️', 'environment': '🌱', 'social media': '📱', 'influencer': '⭐',
                'content creation': '📹', 'digital nomad': '💻'
            }
            
            icon = tag_icons.get(current_tag, '🏷️')
            
            # Card container
            with st.container():
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    st.markdown(f"""
                    <div style="
                        border: 2px solid #e0e0e0;
                        border-radius: 15px;
                        padding: 20px;
                        text-align: center;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        margin: 10px 0;
                    ">
                        <h2 style="font-size: 48px; margin: 10px 0;">{icon}</h2>
                        <h3 style="font-size: 24px; margin: 10px 0; text-transform: capitalize;">{current_tag}</h3>
                        <p style="font-size: 16px; opacity: 0.9;">Discover this interest?</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Progress indicator
                progress = (st.session_state.card_index + 1) / len(available_tags)
                st.progress(progress)
                st.caption(f"Card {st.session_state.card_index + 1} of {len(available_tags)}")
                
                # Swipe buttons with better styling
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                
                with col2:
                    if st.button("👎 Pass", key=f"dislike_{current_tag}", use_container_width=True, 
                               help="Skip this interest"):
                        st.session_state.swiped_tags['disliked'].append(current_tag)
                        st.session_state.card_index += 1
                        st.rerun()
                
                with col3:
                    if st.button("👍 Like", key=f"like_{current_tag}", use_container_width=True,
                               help="Add this interest to your profile"):
                        st.session_state.swiped_tags['liked'].append(current_tag)
                        chatbot.add_manual_tag(current_tag)
                        st.session_state.card_index += 1
                        st.rerun()
                
                # Skip all button
                with col4:
                    if st.button("⏭️ Skip All", key=f"skip_all_{current_tag}", use_container_width=True,
                               help="Skip remaining cards"):
                        st.session_state.card_index = len(available_tags)
                        st.rerun()
        
        # Show completion message
        else:
            st.markdown("### 🎉 Tag Discovery Complete!")
            
            # Summary cards
            col1, col2 = st.columns(2)
            
            with col1:
                if st.session_state.swiped_tags['liked']:
                    st.markdown("""
                    <div style="
                        border: 2px solid #4CAF50;
                        border-radius: 10px;
                        padding: 15px;
                        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                        color: white;
                        text-align: center;
                    ">
                        <h3>✅ Tags Added</h3>
                        <h2>{}</h2>
                    </div>
                    """.format(len(st.session_state.swiped_tags['liked'])), unsafe_allow_html=True)
                    
                    st.write("**Your new interests:**")
                    for tag in st.session_state.swiped_tags['liked']:
                        st.write(f"• {tag}")
            
            with col2:
                if st.session_state.swiped_tags['disliked']:
                    st.markdown("""
                    <div style="
                        border: 2px solid #f44336;
                        border-radius: 10px;
                        padding: 15px;
                        background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
                        color: white;
                        text-align: center;
                    ">
                        <h3>👎 Tags Passed</h3>
                        <h2>{}</h2>
                    </div>
                    """.format(len(st.session_state.swiped_tags['disliked'])), unsafe_allow_html=True)
                    
                    st.write("**Not your interests:**")
                    for tag in st.session_state.swiped_tags['disliked']:
                        st.write(f"• {tag}")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Discover More Tags", use_container_width=True):
                    st.session_state.card_index = 0
                    st.session_state.swiped_tags = {'liked': [], 'disliked': []}
                    st.rerun()
            
            with col2:
                if st.button("👥 Find Similar Users", use_container_width=True):
                    st.session_state['current_view'] = 'similar_users'
                    st.rerun()
            
            with col3:
                if st.button("💬 Start Chatting", use_container_width=True):
                    st.session_state['current_view'] = 'chat'
                    st.rerun()
    
    else:
        st.info("🎯 You've already added all the popular tags! Try adding custom tags below.")
    
    # Manual tag addition
    st.markdown("### ➕ Add Custom Tag")
    with st.form("add_tag"):
        new_tag = st.text_input("Enter a new tag:", placeholder="e.g., technology, music, travel, cricket, bollywood, yoga")
        submit_tag = st.form_submit_button("Add Tag")
        
        if submit_tag and new_tag.strip():
            if chatbot.add_manual_tag(new_tag.strip()):
                st.success(f"Tag '{new_tag.strip()}' added successfully!")
                st.rerun()
            else:
                st.error("Invalid tag. Please use only letters, numbers, spaces, hyphens, and underscores.")
    
    # Enhanced AI-powered tag suggestions
    st.markdown("### 💡 AI-Powered Tag Suggestions")
    
    # Auto-add option
    auto_add = st.checkbox("🚀 Automatically add all suggestions as tags", value=False, 
                          help="When enabled, all AI suggestions will be automatically added to your profile")
    
    if user_tags:
        # Get conversation for context
        conversation = chatbot.get_conversation()
        
        # Generate different types of suggestions
        with st.spinner("Generating intelligent tag suggestions..."):
            # Get language preferences for context-aware suggestions
            language_preferences = chatbot.get_language_preferences()
            
            # Dynamic suggestions using AI
            ai_suggestions = chatbot.tag_analyzer.generate_dynamic_tag_suggestions(user_tags, conversation, language_preferences)
            
            # Category-based suggestions
            category_suggestions = chatbot.tag_analyzer.generate_category_suggestions(user_tags)
            
            # Synonym suggestions
            synonym_suggestions = chatbot.tag_analyzer.generate_synonym_suggestions(user_tags)
            
            # Related concept suggestions
            related_suggestions = chatbot.tag_analyzer.generate_related_concept_suggestions(user_tags)
        
        # Collect all suggestions and track duplicates
        all_suggestions = []
        seen_suggestions = set()
        suggestion_sources = {}  # Track which category each suggestion came from
        
        # Helper function to add suggestions with source tracking
        def add_suggestions_with_source(suggestions, source):
            for tag in suggestions:
                if tag not in seen_suggestions:
                    all_suggestions.append(tag)
                    seen_suggestions.add(tag)
                    suggestion_sources[tag] = source
                elif tag in suggestion_sources:
                    # If tag appears in multiple categories, keep the first source
                    pass
        
        # Add suggestions from each category
        if ai_suggestions:
            add_suggestions_with_source(ai_suggestions, "ai")
        if category_suggestions:
            add_suggestions_with_source(category_suggestions, "category")
        if synonym_suggestions:
            add_suggestions_with_source(synonym_suggestions, "synonym")
        if related_suggestions:
            add_suggestions_with_source(related_suggestions, "related")
        
        # Auto-add suggestions if enabled
        if auto_add and all_suggestions:
            added_count = 0
            for tag in all_suggestions:
                if chatbot.add_manual_tag(tag):
                    added_count += 1
            if added_count > 0:
                st.success(f"🚀 Automatically added {added_count} new tags to your profile!")
                st.rerun()
        
        # Add all suggestions button (only show if auto-add is disabled)
        if all_suggestions and not auto_add:
            if st.button("🚀 Add All Suggestions", key="add_all_suggestions"):
                added_count = 0
                for tag in all_suggestions:
                    if chatbot.add_manual_tag(tag):
                        added_count += 1
                st.success(f"Added {added_count} new tags to your profile!")
                st.rerun()
        
        # Show consolidated unique suggestions
        if all_suggestions:
            st.markdown("#### 📋 All Unique Suggestions")
            st.write(f"Found {len(all_suggestions)} unique suggestions across all categories:")
            for i, tag in enumerate(all_suggestions):
                source = suggestion_sources.get(tag, "unknown")
                source_emoji = {"ai": "🎯", "category": "📂", "synonym": "🔄", "related": "🔗"}.get(source, "❓")
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"{source_emoji} {tag}")
                with col2:
                    st.write(f"({source})")
                with col3:
                    if st.button(f"Add", key=f"consolidated_{i}_{tag}"):
                        chatbot.add_manual_tag(tag)
                        st.rerun()
        
        # Display suggestions in organized sections with unique keys
        if ai_suggestions:
            st.markdown("#### 🎯 Smart Suggestions")
            st.write("Based on your interests and conversation:")
            for i, tag in enumerate(ai_suggestions):
                if st.button(f"Add '{tag}'", key=f"ai_suggest_{i}_{tag}"):
                    chatbot.add_manual_tag(tag)
                    st.rerun()
        
        if category_suggestions:
            st.markdown("#### 📂 Category Suggestions")
            st.write("Broader categories you might be interested in:")
            for i, tag in enumerate(category_suggestions):
                if st.button(f"Add '{tag}'", key=f"category_{i}_{tag}"):
                    chatbot.add_manual_tag(tag)
                    st.rerun()
        
        if synonym_suggestions:
            st.markdown("#### 🔄 Synonym Suggestions")
            st.write("Alternative ways to express your interests:")
            for i, tag in enumerate(synonym_suggestions):
                if st.button(f"Add '{tag}'", key=f"synonym_{i}_{tag}"):
                    chatbot.add_manual_tag(tag)
                    st.rerun()
        
        if related_suggestions:
            st.markdown("#### 🔗 Related Concepts")
            st.write("Closely related topics and emerging trends:")
            for i, tag in enumerate(related_suggestions):
                if st.button(f"Add '{tag}'", key=f"related_{i}_{tag}"):
                    chatbot.add_manual_tag(tag)
                    st.rerun()
        
        # Show recently added tags (if auto-add was used)
        if auto_add and all_suggestions:
            st.markdown("#### 📝 Recently Added Tags")
            st.write("The following tags were automatically added:")
            for i, tag in enumerate(all_suggestions):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"✅ {tag}")
                with col2:
                    if st.button(f"Remove {tag}", key=f"remove_auto_{i}_{tag}"):
                        chatbot.remove_tag(tag)
                        st.rerun()
        
        # Refresh suggestions button
        if st.button("🔄 Refresh Suggestions"):
            st.rerun()
            
    else:
        st.write("Add some tags to your profile to get personalized AI suggestions!")
        st.info("💡 **Tip**: Start by adding a few tags that interest you, then AI will suggest related categories, synonyms, and concepts!")
    
    # Tag statistics
    if user_tags:
        st.markdown("### 📊 Tag Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tags", len(user_tags))
        with col2:
            st.metric("Manual Tags", len(manual_tags))
        with col3:
            st.metric("AI Inferred", len(inferred_tags))

def _show_similar_users_interface(chatbot):
    """Show similar users interface"""
    # Update last activity
    session_manager.update_last_activity()
    
    st.markdown("## 🤝 Similar Users")
    
    # Find similar users
    similar_users = chatbot.get_similar_users(min_common_tags=1)
    
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
    
    group_manager = GroupChatManager(chatbot.db)
    user_groups = group_manager.get_user_groups(chatbot.user_id)
    
    # Create new group chat
    st.markdown("### ➕ Create New Group Chat")
    with st.form("create_group"):
        topic_name = st.text_input("Topic Name:", placeholder="e.g., Technology Discussion")
        submit_group = st.form_submit_button("Create Group Chat")
        
        if submit_group and topic_name.strip():
            # Create group with AI bot
            user_ids = [chatbot.user_id, "ai_bot"]
            group_id = group_manager.create_group_chat(topic_name.strip(), user_ids, chatbot.user_id)
            st.success(f"Group chat '{topic_name.strip()}' created successfully!")
            st.rerun()
    
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
    suggested_topics = group_manager.suggest_group_topics(user_tags)
    
    st.markdown("### 💡 Suggested Topics")
    st.write("Based on your interests:")
    for topic in suggested_topics:
        st.write(f"• {topic}")

def _create_group_chat_with_user(chatbot, similar_user):
    """Create a group chat with a similar user"""
    group_manager = GroupChatManager(chatbot.db)
    
    # Create topic name based on common tags
    common_tags = similar_user['common_tags']
    topic_name = f"{common_tags[0].title()} Discussion" if common_tags else "General Discussion"
    
    # Create group with both users and AI bot
    user_ids = [chatbot.user_id, similar_user['user_id'], "ai_bot"]
    group_id = group_manager.create_group_chat(topic_name, user_ids, chatbot.user_id)
    
    st.success(f"Group chat '{topic_name}' created with {similar_user['name']}!")
    st.session_state['current_group_id'] = group_id
    st.session_state['current_view'] = 'group_chat'
    st.rerun()

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
    
    group_manager = GroupChatManager(chatbot.db)
    group_chat = group_manager.get_group_chat(st.session_state['current_group_id'], chatbot.user_id)
    
    if not group_chat:
        st.error("Group chat not found or you don't have access.")
        if st.button("Back to Group Chats"):
            st.session_state['current_view'] = 'group_chats'
            st.rerun()
        return
    
    # Get group info
    group_info = chatbot.db.get_group_info(st.session_state['current_group_id'])
    
    # Header with back button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"## 👥 {group_info['topic_name']}")
    with col2:
        if st.button("← Back to Groups"):
            st.session_state['current_view'] = 'group_chats'
            st.rerun()
    
    # Display participants
    participant_names = []
    for uid in group_info['user_ids']:
        if uid == "ai_bot":
            participant_names.append("AI Assistant")
        else:
            user_profile = chatbot.db.get_user_profile(uid)
            if user_profile:
                participant_names.append(user_profile['name'])
    
    st.markdown(f"**Participants:** {', '.join(participant_names)}")
    
    # Display messages
    st.markdown("### 💬 Messages")
    messages = group_chat.get_messages()
    
    for msg in messages:
        if msg['is_ai']:
            st.markdown(f"**🤖 AI Assistant:** {msg['message']}")
        else:
            st.markdown(f"**👤 {msg['sender']}:** {msg['message']}")
    
    # Send message
    st.markdown("### 💭 Send Message")
    with st.form("group_message"):
        message = st.text_input("Your message:", placeholder="Type your message here...")
        send_button = st.form_submit_button("Send")
        
        if send_button and message.strip():
            # Send message and get AI response
            ai_response = group_chat.send_message(message.strip())
            st.success("Message sent!")
            st.rerun()

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
        st.session_state['chatbot'] = Chatbot(
            db=st.session_state['db'],
            user_id=user_info['user_id'],
            user_name=user_info['user_name']
        )
    
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
        'group_chat': '💬 Group Chat'
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
            st.session_state['chatbot'] = Chatbot(
                db=st.session_state['db'],
                user_id=user_id,
                user_name=name
            )
            
            st.success(f"👋 Welcome back, {name}! Your session has been loaded.")
            st.rerun()
        elif submit_button and not user_name.strip():
            st.error("Please enter your name to continue.") 