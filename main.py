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
    
    # Location preferences section
    st.markdown("### 📍 Location Preferences")
    
    # Get current location preferences
    location_prefs = chatbot.get_location_preferences()
    
    # Indian states and major cities
    indian_states = {
        'Andhra Pradesh': ['Hyderabad', 'Visakhapatnam', 'Vijayawada', 'Guntur', 'Nellore', 'Kurnool', 'Rajahmundry', 'Tirupati'],
        'Arunachal Pradesh': ['Itanagar', 'Naharlagun', 'Pasighat', 'Tezpur', 'Bomdila'],
        'Assam': ['Guwahati', 'Silchar', 'Dibrugarh', 'Jorhat', 'Nagaon', 'Tinsukia', 'Tezpur'],
        'Bihar': ['Patna', 'Gaya', 'Bhagalpur', 'Muzaffarpur', 'Purnia', 'Darbhanga', 'Bihar Sharif'],
        'Chhattisgarh': ['Raipur', 'Bhilai', 'Korba', 'Bilaspur', 'Durg', 'Rajnandgaon'],
        'Goa': ['Panaji', 'Vasco da Gama', 'Margao', 'Mapusa', 'Ponda'],
        'Gujarat': ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot', 'Bhavnagar', 'Jamnagar', 'Gandhinagar', 'Anand'],
        'Haryana': ['Gurugram', 'Faridabad', 'Panipat', 'Ambala', 'Yamunanagar', 'Rohtak', 'Hisar', 'Karnal'],
        'Himachal Pradesh': ['Shimla', 'Dharamshala', 'Solan', 'Mandi', 'Kullu', 'Manali', 'Kasauli'],
        'Jharkhand': ['Ranchi', 'Jamshedpur', 'Dhanbad', 'Bokaro', 'Deoghar', 'Hazaribagh'],
        'Karnataka': ['Bangalore', 'Mysore', 'Hubli', 'Mangalore', 'Belgaum', 'Gulbarga', 'Davangere', 'Bellary'],
        'Kerala': ['Thiruvananthapuram', 'Kochi', 'Kozhikode', 'Thrissur', 'Kollam', 'Palakkad', 'Alappuzha', 'Kannur'],
        'Madhya Pradesh': ['Bhopal', 'Indore', 'Gwalior', 'Jabalpur', 'Ujjain', 'Sagar', 'Dewas', 'Satna'],
        'Maharashtra': ['Mumbai', 'Pune', 'Nagpur', 'Thane', 'Nashik', 'Aurangabad', 'Solapur', 'Amravati', 'Kolhapur'],
        'Manipur': ['Imphal', 'Thoubal', 'Bishnupur', 'Churachandpur'],
        'Meghalaya': ['Shillong', 'Tura', 'Jowai', 'Nongstoin'],
        'Mizoram': ['Aizawl', 'Lunglei', 'Saiha', 'Champhai'],
        'Nagaland': ['Kohima', 'Dimapur', 'Mokokchung', 'Tuensang'],
        'Odisha': ['Bhubaneswar', 'Cuttack', 'Rourkela', 'Brahmapur', 'Sambalpur', 'Puri', 'Balasore'],
        'Punjab': ['Chandigarh', 'Ludhiana', 'Amritsar', 'Jalandhar', 'Patiala', 'Bathinda', 'Mohali', 'Pathankot'],
        'Rajasthan': ['Jaipur', 'Jodhpur', 'Kota', 'Bikaner', 'Ajmer', 'Udaipur', 'Bhilwara', 'Alwar'],
        'Sikkim': ['Gangtok', 'Namchi', 'Gyalshing', 'Mangan'],
        'Tamil Nadu': ['Chennai', 'Coimbatore', 'Madurai', 'Tiruchirappalli', 'Salem', 'Tirunelveli', 'Erode', 'Vellore'],
        'Telangana': ['Hyderabad', 'Warangal', 'Nizamabad', 'Khammam', 'Karimnagar', 'Ramagundam'],
        'Tripura': ['Agartala', 'Dharmanagar', 'Udaipur', 'Kailashahar'],
        'Uttar Pradesh': ['Lucknow', 'Kanpur', 'Ghaziabad', 'Agra', 'Varanasi', 'Meerut', 'Allahabad', 'Bareilly', 'Moradabad'],
        'Uttarakhand': ['Dehradun', 'Haridwar', 'Roorkee', 'Haldwani', 'Rudrapur', 'Kashipur', 'Rishikesh'],
        'West Bengal': ['Kolkata', 'Howrah', 'Durgapur', 'Asansol', 'Siliguri', 'Malda', 'Bardhaman', 'Kharagpur'],
        'Delhi': ['New Delhi', 'Delhi', 'Noida', 'Ghaziabad', 'Faridabad', 'Gurugram'],
        'Jammu and Kashmir': ['Srinagar', 'Jammu', 'Anantnag', 'Baramulla', 'Udhampur'],
        'Ladakh': ['Leh', 'Kargil']
    }
    
    # Privacy levels
    privacy_levels = {
        'exact': '🎯 Exact Location (GPS coordinates)',
        'city_only': '🏙️ City Only',
        'state_only': '🗺️ State Only', 
        'country_only': '🌍 Country Only',
        'private': '🔒 Private (No location sharing)'
    }
    
    with st.form("location_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Country selection
            country = st.selectbox(
                "Country",
                options=['India', 'United States', 'United Kingdom', 'Canada', 'Australia', 'Other'],
                index=0 if not location_prefs.get('country') else 
                      (['India', 'United States', 'United Kingdom', 'Canada', 'Australia', 'Other'].index(location_prefs['country']) 
                       if location_prefs['country'] in ['India', 'United States', 'United Kingdom', 'Canada', 'Australia', 'Other'] else 5)
            )
            
            # State selection (dynamic based on country)
            if country == 'India':
                state = st.selectbox(
                    "State/Union Territory",
                    options=[''] + list(indian_states.keys()),
                    index=0 if not location_prefs.get('state') else 
                          (list(indian_states.keys()).index(location_prefs['state']) + 1 
                           if location_prefs.get('state') in indian_states else 0)
                )
            else:
                state = st.text_input(
                    "State/Province",
                    value=location_prefs.get('state', '') or '',
                    placeholder="Enter your state/province"
                )
        
        with col2:
            # City selection (dynamic based on state for India)
            if country == 'India' and state and state in indian_states:
                city = st.selectbox(
                    "City",
                    options=[''] + indian_states[state],
                    index=0 if not location_prefs.get('city') else 
                          (indian_states[state].index(location_prefs['city']) + 1 
                           if location_prefs.get('city') in indian_states[state] else 0)
                )
            else:
                city = st.text_input(
                    "City",
                    value=location_prefs.get('city', '') or '',
                    placeholder="Enter your city"
                )
            
            # Privacy level
            privacy_level = st.selectbox(
                "Privacy Level",
                options=list(privacy_levels.keys()),
                format_func=lambda x: privacy_levels[x],
                index=list(privacy_levels.keys()).index(location_prefs.get('privacy_level', 'city_only'))
            )
        
        # Optional: GPS coordinates input
        st.markdown("#### 📍 GPS Coordinates (Optional)")
        col3, col4 = st.columns(2)
        
        current_coords = location_prefs.get('coordinates', {})
        with col3:
            latitude = st.number_input(
                "Latitude",
                value=current_coords.get('latitude') or 0.0,
                format="%.6f",
                help="Enter latitude for precise location (optional)"
            )
        
        with col4:
            longitude = st.number_input(
                "Longitude", 
                value=current_coords.get('longitude') or 0.0,
                format="%.6f",
                help="Enter longitude for precise location (optional)"
            )
        
        # Timezone (auto-detect based on location)
        timezone_options = [
            'Asia/Kolkata', 'America/New_York', 'America/Los_Angeles', 'Europe/London',
            'Australia/Sydney', 'Asia/Tokyo', 'Europe/Paris', 'Asia/Dubai'
        ]
        
        timezone = st.selectbox(
            "Timezone",
            options=timezone_options,
            index=0 if not location_prefs.get('timezone') else 
                  (timezone_options.index(location_prefs['timezone']) 
                   if location_prefs.get('timezone') in timezone_options else 0)
        )
        
        submit_location = st.form_submit_button("💾 Update Location")
        
        if submit_location:
            coordinates = None
            if latitude != 0.0 or longitude != 0.0:
                coordinates = {'latitude': latitude, 'longitude': longitude}
            
            if chatbot.update_location_preferences(
                city=city if city else None,
                state=state if state else None,
                country=country if country else None,
                timezone=timezone,
                coordinates=coordinates,
                privacy_level=privacy_level
            ):
                st.success("✅ Location preferences updated successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to update location preferences.")
    
    # Display current location info
    if any(location_prefs.values()):
        st.markdown("#### 📍 Current Location")
        location_display = []
        if location_prefs.get('city'):
            location_display.append(f"🏙️ **City:** {location_prefs['city']}")
        if location_prefs.get('state'):
            location_display.append(f"🗺️ **State:** {location_prefs['state']}")
        if location_prefs.get('country'):
            location_display.append(f"🌍 **Country:** {location_prefs['country']}")
        if location_prefs.get('timezone'):
            location_display.append(f"🕐 **Timezone:** {location_prefs['timezone']}")
        if location_prefs.get('privacy_level'):
            location_display.append(f"🔒 **Privacy:** {privacy_levels.get(location_prefs['privacy_level'], 'Unknown')}")
        
        for item in location_display:
            st.markdown(item)
    
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
    
    # Location-based features
    if any(location_prefs.values()):
        st.markdown("### 🌍 Location-Based Features")
        
        # Get location-based recommendations
        location_recommendations = []
        if location_prefs.get('city') or location_prefs.get('state'):
            # Get regional tag suggestions
            regional_tags = chatbot.tag_analyzer.get_location_based_tag_suggestions(location_prefs)
            if regional_tags:
                location_recommendations.extend(regional_tags[:5])
        
        # Display location-based tag suggestions
        if location_recommendations:
            st.markdown("#### 🏷️ Regional Interest Suggestions")
            st.write("Based on your location, you might be interested in:")
            
            # Create columns for regional tags
            cols = st.columns(min(len(location_recommendations), 3))
            for i, tag in enumerate(location_recommendations):
                with cols[i % 3]:
                    if st.button(f"Add '{tag}'", key=f"regional_{i}_{tag}"):
                        if chatbot.add_manual_tag(tag):
                            st.success(f"Added '{tag}' to your interests!")
                            st.rerun()
        
        # Nearby users section
        st.markdown("#### 👥 People Near You")
        col1, col2 = st.columns(2)
        
        with col1:
            nearby_users = chatbot.find_users_in_city()
            if nearby_users:
                st.metric("Users in Your Area", len(nearby_users))
                if st.button("🔍 Find Local Connections", use_container_width=True):
                    st.session_state['current_view'] = 'similar_users'
                    st.rerun()
            else:
                st.info("No users found in your area yet")
        
        with col2:
            # GPS-based nearby users (if coordinates available)
            if location_prefs.get('coordinates', {}).get('latitude'):
                gps_users = chatbot.find_nearby_users(max_distance_km=25)
                if gps_users:
                    st.metric("Users Within 25km", len(gps_users))
                    if st.button("📍 Find Nearby Users", use_container_width=True):
                        st.session_state['current_view'] = 'similar_users'
                        st.rerun()
                else:
                    st.info("No users found within 25km")
            else:
                st.info("Add GPS coordinates to find nearby users")
        
        # Local events and activities (placeholder)
        st.markdown("#### 🎉 Local Recommendations")
        location_display = []
        if location_prefs.get('city'):
            location_display.append(location_prefs['city'])
        if location_prefs.get('state'):
            location_display.append(location_prefs['state'])
        
        location_text = ", ".join(location_display) if location_display else "your area"
        
        local_activities = [
            f"🎭 Cultural events in {location_text}",
            f"🍽️ Local cuisine and restaurants",
            f"🏛️ Historical sites and museums",
            f"🌳 Parks and recreational areas",
            f"🎪 Festivals and celebrations"
        ]
        
        for activity in local_activities:
            st.write(f"• {activity}")
        
        # Location privacy reminder
        st.markdown("#### 🔒 Privacy Settings")
        privacy_level = location_prefs.get('privacy_level', 'city_only')
        privacy_descriptions = {
            'exact': 'Your exact location is shared with other users',
            'city_only': 'Only your city is visible to other users',
            'state_only': 'Only your state is visible to other users',
            'country_only': 'Only your country is visible to other users',
            'private': 'Your location is completely private'
        }
        
        st.info(f"🔒 **Current Privacy Level**: {privacy_descriptions.get(privacy_level, 'Unknown')}")
        
        if st.button("⚙️ Update Privacy Settings", use_container_width=True):
            st.info("Scroll up to the Location Preferences section to update your privacy settings.")
    
    else:
        st.markdown("### 📍 Add Your Location")
        st.info("Add your location to discover local users, events, and region-specific interests!")
        if st.button("📍 Set Up Location", use_container_width=True):
            st.info("Scroll up to the Location Preferences section to add your location.")

def _show_similar_users_interface(chatbot):
    """Show similar users interface with location-based features"""
    # Update last activity
    session_manager.update_last_activity()
    
    st.markdown("## 🤝 Similar Users")
    
    # User's location for context
    user_location = chatbot.get_location_preferences()
    
    # Search options
    st.markdown("### 🔍 Search Options")
    col1, col2 = st.columns(2)
    
    with col1:
        search_type = st.selectbox(
            "Search Type",
            options=["interests", "location", "both"],
            format_func=lambda x: {
                "interests": "🏷️ By Interests Only",
                "location": "📍 By Location Only", 
                "both": "🎯 By Interests + Location"
            }[x],
            index=2
        )
    
    with col2:
        min_tags = st.slider("Minimum Common Tags", 1, 5, 1)
    
    # Location-based search options
    if search_type in ["location", "both"]:
        st.markdown("#### 📍 Location Search")
        location_search_type = st.radio(
            "Location Search Type",
            options=["nearby", "same_city", "same_state"],
            format_func=lambda x: {
                "nearby": "🎯 Nearby (GPS-based)",
                "same_city": "🏙️ Same City",
                "same_state": "🗺️ Same State"
            }[x],
            horizontal=True
        )
        
        if location_search_type == "nearby":
            max_distance = st.slider("Max Distance (km)", 5, 200, 50)
    
    # Find users based on search type
    if search_type == "interests":
        similar_users = chatbot.get_similar_users(min_common_tags=min_tags, include_location=False)
    elif search_type == "location":
        if location_search_type == "nearby":
            similar_users = chatbot.find_nearby_users(max_distance_km=max_distance)
            # Add dummy similarity score for consistency
            for user in similar_users:
                user['similarity_score'] = 0
                user['common_tags'] = []
                user['total_score'] = 0
        else:
            similar_users = chatbot.find_users_in_city()
            # Add dummy similarity score for consistency
            for user in similar_users:
                user['similarity_score'] = 0
                user['common_tags'] = []
                user['total_score'] = 0
    else:  # both
        similar_users = chatbot.get_similar_users(min_common_tags=min_tags, include_location=True)
        
        # Also get location-based users to combine results
        if location_search_type == "nearby":
            nearby_users = chatbot.find_nearby_users(max_distance_km=max_distance)
        else:
            nearby_users = chatbot.find_users_in_city()
        
        # Merge location-based users with interest-based users
        user_ids_seen = {user['user_id'] for user in similar_users}
        for nearby_user in nearby_users:
            if nearby_user['user_id'] not in user_ids_seen:
                # Get tags for this user to calculate similarity
                other_tags = set(chatbot.db.get_user_tags(nearby_user['user_id']))
                user_tags = set(chatbot.get_user_tags())
                common_tags = user_tags.intersection(other_tags)
                
                similar_users.append({
                    'user_id': nearby_user['user_id'],
                    'name': nearby_user['name'],
                    'common_tags': list(common_tags),
                    'similarity_score': len(common_tags),
                    'location_bonus': 2 if location_search_type == "nearby" else 1,
                    'total_score': len(common_tags) + (2 if location_search_type == "nearby" else 1),
                    'location_info': {
                        'city': nearby_user.get('city'),
                        'state': nearby_user.get('state'),
                        'distance_km': nearby_user.get('distance_km'),
                        'privacy_level': nearby_user.get('privacy_level', 'city_only')
                    }
                })
        
        # Re-sort by total score
        similar_users.sort(key=lambda x: x.get('total_score', x['similarity_score']), reverse=True)
    
    # Display results
    if similar_users:
        st.markdown(f"### 👥 Found {len(similar_users)} Users")
        
        for user in similar_users:
            # Create user card with location info
            location_info = user.get('location_info', {})
            
            # Build location display
            location_display = []
            if location_info.get('city'):
                location_display.append(f"🏙️ {location_info['city']}")
            if location_info.get('state'):
                location_display.append(f"🗺️ {location_info['state']}")
            if location_info.get('distance_km'):
                location_display.append(f"📍 {location_info['distance_km']} km away")
            
            location_text = " • ".join(location_display) if location_display else "📍 Location not shared"
            
            # Build score display
            score_parts = []
            if user.get('similarity_score', 0) > 0:
                score_parts.append(f"🏷️ {user['similarity_score']} common tags")
            if user.get('location_bonus', 0) > 0:
                score_parts.append(f"📍 +{user['location_bonus']} location bonus")
            
            score_text = " • ".join(score_parts) if score_parts else "No common interests"
            
            with st.expander(f"👤 {user['name']} • {score_text}"):
                # Location information
                st.markdown(f"**📍 Location:** {location_text}")
                
                # Common interests
                if user.get('common_tags'):
                    st.markdown(f"**🏷️ Common Interests:** {', '.join(user['common_tags'])}")
                else:
                    st.markdown("**🏷️ Common Interests:** None found")
                
                # Additional location details
                if location_info:
                    details = []
                    if location_info.get('same_city'):
                        details.append("🏙️ Same city")
                    elif location_info.get('same_state'):
                        details.append("🗺️ Same state")
                    elif location_info.get('same_country'):
                        details.append("🌍 Same country")
                    
                    if details:
                        st.markdown(f"**🎯 Location Match:** {', '.join(details)}")
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"💬 Start Group Chat", key=f"group_{user['user_id']}"):
                        _create_group_chat_with_user(chatbot, user)
                
                with col2:
                    if st.button(f"👤 View Profile", key=f"profile_{user['user_id']}"):
                        st.info("Profile viewing feature coming soon!")
        
        # Summary statistics
        st.markdown("### 📊 Search Results Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Users Found", len(similar_users))
        
        with col2:
            same_city_count = sum(1 for user in similar_users 
                                if user.get('location_info', {}).get('same_city'))
            st.metric("Same City", same_city_count)
        
        with col3:
            with_common_tags = sum(1 for user in similar_users 
                                 if user.get('similarity_score', 0) > 0)
            st.metric("With Common Interests", with_common_tags)
    
    else:
        st.markdown("### 👥 No Users Found")
        st.write("No users found matching your criteria. This could be because:")
        st.write("- You don't have enough tags yet")
        st.write("- No users in your area have similar interests")
        st.write("- Your location privacy settings are too restrictive")
        st.write("- You're among the first users in the system")
        
        st.markdown("**Tips:**")
        st.write("- 🏷️ Add more tags to your profile")
        st.write("- 📍 Update your location preferences")
        st.write("- 🔍 Try different search criteria")
        st.write("- 👥 Invite friends to join the platform")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏷️ Add More Tags", use_container_width=True):
            st.session_state['current_view'] = 'profile'
            st.rerun()
    
    with col2:
        if st.button("📍 Update Location", use_container_width=True):
            st.session_state['current_view'] = 'profile'
            st.rerun()
    
    with col3:
        if st.button("🔄 Refresh Search", use_container_width=True):
            st.rerun()

def _show_nearby_users_interface(chatbot):
    """Show nearby users interface focused on location-based discovery"""
    # Update last activity
    session_manager.update_last_activity()
    
    st.markdown("## 📍 Nearby Users")
    
    # Get user's location
    user_location = chatbot.get_location_preferences()
    
    if not any(user_location.values()):
        st.warning("⚠️ Location not set up yet!")
        st.markdown("### 📍 Set Up Your Location")
        st.write("To find nearby users, please add your location information first.")
        
        if st.button("📍 Go to Profile Settings", use_container_width=True):
            st.session_state['current_view'] = 'profile'
            st.rerun()
        return
    
    # Display current location
    st.markdown("### 📍 Your Location")
    location_display = []
    if user_location.get('city'):
        location_display.append(f"🏙️ {user_location['city']}")
    if user_location.get('state'):
        location_display.append(f"🗺️ {user_location['state']}")
    if user_location.get('country'):
        location_display.append(f"🌍 {user_location['country']}")
    
    if location_display:
        st.write(" • ".join(location_display))
    
    # Search options
    st.markdown("### 🔍 Search Options")
    col1, col2 = st.columns(2)
    
    with col1:
        search_radius = st.selectbox(
            "Search Area",
            options=["same_city", "same_state", "nearby_gps"],
            format_func=lambda x: {
                "same_city": "🏙️ Same City",
                "same_state": "🗺️ Same State",
                "nearby_gps": "📍 GPS Radius"
            }[x]
        )
    
    with col2:
        if search_radius == "nearby_gps":
            max_distance = st.slider("Max Distance (km)", 5, 200, 50)
        else:
            max_distance = None
    
    # Find nearby users
    if search_radius == "nearby_gps" and max_distance:
        nearby_users = chatbot.find_nearby_users(max_distance_km=max_distance)
        search_type = f"within {max_distance}km"
    elif search_radius == "same_city":
        nearby_users = chatbot.find_users_in_city()
        search_type = "in your city"
    else:  # same_state
        nearby_users = chatbot.find_users_in_city(state=user_location.get('state'))
        search_type = "in your state"
    
    # Display results
    if nearby_users:
        st.markdown(f"### 👥 Found {len(nearby_users)} Users {search_type}")
        
        for user in nearby_users:
            with st.expander(f"👤 {user['name']} • {user.get('city', 'Unknown city')}"):
                # Location info
                location_info = []
                if user.get('city'):
                    location_info.append(f"🏙️ **City:** {user['city']}")
                if user.get('state'):
                    location_info.append(f"🗺️ **State:** {user['state']}")
                if user.get('distance_km'):
                    location_info.append(f"📍 **Distance:** {user['distance_km']} km")
                
                for info in location_info:
                    st.markdown(info)
                
                # Privacy level
                privacy_level = user.get('privacy_level', 'city_only')
                privacy_emoji = {
                    'exact': '🎯',
                    'city_only': '🏙️',
                    'state_only': '🗺️',
                    'country_only': '🌍',
                    'private': '🔒'
                }.get(privacy_level, '❓')
                
                st.markdown(f"**🔒 Privacy:** {privacy_emoji} {privacy_level.replace('_', ' ').title()}")
                
                # Get common interests
                other_tags = set(chatbot.db.get_user_tags(user['user_id']))
                user_tags = set(chatbot.get_user_tags())
                common_tags = user_tags.intersection(other_tags)
                
                if common_tags:
                    st.markdown(f"**🏷️ Common Interests:** {', '.join(common_tags)}")
                else:
                    st.markdown("**🏷️ Common Interests:** None found")
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"💬 Start Chat", key=f"chat_{user['user_id']}"):
                        # Create a user object for group chat creation
                        user_obj = {
                            'user_id': user['user_id'],
                            'name': user['name'],
                            'common_tags': list(common_tags)
                        }
                        _create_group_chat_with_user(chatbot, user_obj)
                
                with col2:
                    if st.button(f"👤 View Details", key=f"details_{user['user_id']}"):
                        st.info("Detailed profile viewing coming soon!")
        
        # Summary statistics
        st.markdown("### 📊 Location Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Users", len(nearby_users))
        
        with col2:
            same_city_count = sum(1 for user in nearby_users 
                                if user.get('city', '').lower() == user_location.get('city', '').lower())
            st.metric("Same City", same_city_count)
        
        with col3:
            if search_radius == "nearby_gps":
                avg_distance = sum(user.get('distance_km', 0) for user in nearby_users) / len(nearby_users)
                st.metric("Avg Distance", f"{avg_distance:.1f} km")
            else:
                same_state_count = sum(1 for user in nearby_users 
                                     if user.get('state', '').lower() == user_location.get('state', '').lower())
                st.metric("Same State", same_state_count)
    
    else:
        st.markdown(f"### 👥 No Users Found {search_type}")
        st.write("No users found in your search area. This could be because:")
        st.write("- You're among the first users in your area")
        st.write("- Other users have more restrictive privacy settings")
        st.write("- Try expanding your search radius")
        
        st.markdown("**Tips:**")
        st.write("- 🔍 Try different search options")
        st.write("- 🏷️ Add more interests to find like-minded people")
        st.write("- 👥 Invite friends from your area to join")
        st.write("- 📍 Check your location privacy settings")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏷️ Find by Interests", use_container_width=True):
            st.session_state['current_view'] = 'similar_users'
            st.rerun()
    
    with col2:
        if st.button("📍 Update Location", use_container_width=True):
            st.session_state['current_view'] = 'profile'
            st.rerun()
    
    with col3:
        if st.button("🔄 Refresh Search", use_container_width=True):
            st.rerun()

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
        'nearby_users': '📍 Nearby Users',
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
    elif st.session_state['current_view'] == 'nearby_users':
        _show_nearby_users_interface(chatbot)
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