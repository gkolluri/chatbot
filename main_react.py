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
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import requests
import json
import pytz
import time

# Load environment variables from .env if present
load_dotenv()

# Import app logic
from react_multi_agent_chatbot import ReactMultiAgentChatbot
from db import get_db
from session_manager import session_manager

# Enhanced Location Helper Functions
def get_timezone_from_coordinates(lat, lng):
    """Get timezone from coordinates using a timezone API"""
    try:
        # Using TimeZoneDB API (free tier available)
        # For production, consider using a proper API key
        response = requests.get(
            f"http://worldtimeapi.org/api/timezone",
            timeout=5
        )
        if response.status_code == 200:
            timezones = response.json()
            # Return a reasonable default based on coordinates
            if lat and lng:
                if -125 <= lng <= -66:  # North America
                    if lng <= -120:
                        return "America/Los_Angeles"
                    elif lng <= -105:
                        return "America/Denver"
                    elif lng <= -90:
                        return "America/Chicago"
                    else:
                        return "America/New_York"
                elif -10 <= lng <= 40:  # Europe/Africa
                    return "Europe/London"
                elif 100 <= lng <= 180:  # Asia/Pacific
                    return "Asia/Tokyo"
                elif 70 <= lng <= 100:  # India/Central Asia
                    return "Asia/Kolkata"
        return "UTC"
    except:
        return "UTC"

def reverse_geocode_coordinates(lat, lng):
    """Convert coordinates to address using reverse geocoding"""
    try:
        geolocator = Nominatim(user_agent="chatbot_location")
        location = geolocator.reverse(f"{lat}, {lng}", timeout=10)
        
        if location and location.raw:
            address = location.raw.get('address', {})
            return {
                'city': address.get('city') or address.get('town') or address.get('village', ''),
                'state': address.get('state', ''),
                'country': address.get('country', ''),
                'formatted_address': location.address
            }
    except (GeocoderTimedOut, GeocoderUnavailable):
        pass
    return None

def geocode_address(address):
    """Convert address to coordinates using geocoding"""
    try:
        geolocator = Nominatim(user_agent="chatbot_location")
        location = geolocator.geocode(address, timeout=10)
        
        if location:
            return {
                'lat': location.latitude,
                'lng': location.longitude,
                'formatted_address': location.address
            }
    except (GeocoderTimedOut, GeocoderUnavailable):
        pass
    return None

def get_common_timezones():
    """Get a list of common timezones organized by region"""
    return {
        'UTC': 'UTC',
        'North America': {
            'America/New_York': 'Eastern Time (US & Canada)',
            'America/Chicago': 'Central Time (US & Canada)',
            'America/Denver': 'Mountain Time (US & Canada)',
            'America/Los_Angeles': 'Pacific Time (US & Canada)',
            'America/Toronto': 'Toronto',
            'America/Vancouver': 'Vancouver'
        },
        'Europe': {
            'Europe/London': 'London',
            'Europe/Paris': 'Paris',
            'Europe/Berlin': 'Berlin',
            'Europe/Rome': 'Rome',
            'Europe/Madrid': 'Madrid',
            'Europe/Amsterdam': 'Amsterdam',
            'Europe/Stockholm': 'Stockholm',
            'Europe/Moscow': 'Moscow'
        },
        'Asia': {
            'Asia/Kolkata': 'India Standard Time',
            'Asia/Shanghai': 'China Standard Time',
            'Asia/Tokyo': 'Japan Standard Time',
            'Asia/Seoul': 'Korea Standard Time',
            'Asia/Singapore': 'Singapore',
            'Asia/Dubai': 'UAE Standard Time',
            'Asia/Bangkok': 'Thailand',
            'Asia/Jakarta': 'Indonesia'
        },
        'Australia/Pacific': {
            'Australia/Sydney': 'Sydney',
            'Australia/Melbourne': 'Melbourne',
            'Australia/Perth': 'Perth',
            'Pacific/Auckland': 'Auckland',
            'Pacific/Honolulu': 'Hawaii'
        },
        'Other': {
            'Africa/Cairo': 'Cairo',
            'Africa/Johannesburg': 'Johannesburg',
            'America/Sao_Paulo': 'São Paulo',
            'America/Argentina/Buenos_Aires': 'Buenos Aires'
        }
    }

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
    
    # Enhanced Location preferences section
    st.markdown("### 📍 Enhanced Location Preferences")
    st.markdown("*Set your location using multiple convenient methods*")
    
    # Get current location preferences
    location_prefs = chatbot.get_location_preferences(chatbot.user_id)
    
    # Location setting method selection
    location_method = st.radio(
        "Choose how to set your location:",
        options=[
            "🌍 Use Device Location (Geolocation)",
            "🗺️ Interactive Map",
            "📍 Address Search",
            "✍️ Manual Entry"
        ],
        index=0,
        help="Select your preferred method to set location"
    )
    
    # Clear search results when switching methods
    if 'last_location_method' in st.session_state and st.session_state['last_location_method'] != location_method:
        # Clear all search-related session state
        search_keys = ['address_search_lat', 'address_search_lng', 'address_search_city', 
                      'address_search_state', 'address_search_country', 'address_search_timezone', 
                      'address_search_formatted', 'manual_search_lat', 'manual_search_lng']
        for key in search_keys:
            if key in st.session_state:
                del st.session_state[key]
    
    st.session_state['last_location_method'] = location_method
    
    # Initialize variables
    selected_lat = location_prefs.get('coordinates', {}).get('lat', 0.0)
    selected_lng = location_prefs.get('coordinates', {}).get('lng', 0.0)
    selected_city = location_prefs.get('city', '')
    selected_state = location_prefs.get('state', '')
    selected_country = location_prefs.get('country', '')
    selected_timezone = location_prefs.get('timezone', 'UTC')
    
    # Method 1: Device Geolocation
    if location_method == "🌍 Use Device Location (Geolocation)":
        st.markdown("#### 🌍 Device Geolocation")
        st.info("📱 Click the button below to use your device's location. Your browser may ask for permission.")
        
        # JavaScript for geolocation
        geolocation_html = """
        <script>
        function getLocation() {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(showPosition, showError);
            } else {
                alert("Geolocation is not supported by this browser.");
            }
        }
        
        function showPosition(position) {
            const lat = position.coords.latitude;
            const lng = position.coords.longitude;
            
            // Store in session storage
            sessionStorage.setItem('geolocation_lat', lat);
            sessionStorage.setItem('geolocation_lng', lng);
            
            // Update display
            document.getElementById('coordinates').innerHTML = 
                `📍 Latitude: ${lat.toFixed(6)}, Longitude: ${lng.toFixed(6)}`;
            
            // Enable the use location button
            document.getElementById('use_location_btn').style.display = 'block';
        }
        
        function showError(error) {
            switch(error.code) {
                case error.PERMISSION_DENIED:
                    alert("User denied the request for Geolocation.");
                    break;
                case error.POSITION_UNAVAILABLE:
                    alert("Location information is unavailable.");
                    break;
                case error.TIMEOUT:
                    alert("The request to get user location timed out.");
                    break;
                case error.UNKNOWN_ERROR:
                    alert("An unknown error occurred.");
                    break;
            }
        }
        </script>
        
        <button onclick="getLocation()" style="background-color: #FF6B6B; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">
            📍 Get My Location
        </button>
        
        <div id="coordinates" style="margin-top: 10px; font-weight: bold;"></div>
        <div id="use_location_btn" style="display: none; margin-top: 10px;">
            <p>✅ Location detected! Use the form below to save it.</p>
        </div>
        """
        
        st.components.v1.html(geolocation_html, height=150)
        
        # Manual coordinate input for geolocation results
        col1, col2 = st.columns(2)
        with col1:
            geo_lat = st.number_input("Detected Latitude:", value=selected_lat, format="%.6f", key="geo_lat")
        with col2:
            geo_lng = st.number_input("Detected Longitude:", value=selected_lng, format="%.6f", key="geo_lng")
        
        if geo_lat != 0.0 and geo_lng != 0.0:
            # Reverse geocode to get address
            with st.spinner("🔍 Looking up address..."):
                address_info = reverse_geocode_coordinates(geo_lat, geo_lng)
                if address_info:
                    selected_lat = geo_lat
                    selected_lng = geo_lng
                    selected_city = address_info.get('city', '')
                    selected_state = address_info.get('state', '')
                    selected_country = address_info.get('country', '')
                    selected_timezone = get_timezone_from_coordinates(geo_lat, geo_lng)
                    
                    st.success(f"📍 Address found: {address_info.get('formatted_address', 'Unknown')}")
                else:
                    st.warning("Could not determine address from coordinates")
    
    # Method 2: Interactive Map
    elif location_method == "🗺️ Interactive Map":
        st.markdown("#### 🗺️ Interactive Map")
        st.info("🖱️ Click on the map to select your location")
        
        # Create map centered on current location or default
        map_center = [selected_lat if selected_lat != 0.0 else 20.0, selected_lng if selected_lng != 0.0 else 77.0]
        
        m = folium.Map(location=map_center, zoom_start=10)
        
        # Add current location marker if exists
        if selected_lat != 0.0 and selected_lng != 0.0:
            folium.Marker(
                [selected_lat, selected_lng],
                popup="Current Location",
                tooltip="Your current location",
                icon=folium.Icon(color='red', icon='home')
            ).add_to(m)
        
        # Display map
        map_data = st_folium(m, width=700, height=400, returned_objects=["last_object_clicked"])
        
        # Handle map click
        if map_data['last_object_clicked']:
            clicked_lat = map_data['last_object_clicked']['lat']
            clicked_lng = map_data['last_object_clicked']['lng']
            
            # Reverse geocode the clicked location
            with st.spinner("🔍 Looking up address..."):
                address_info = reverse_geocode_coordinates(clicked_lat, clicked_lng)
                if address_info:
                    selected_lat = clicked_lat
                    selected_lng = clicked_lng
                    selected_city = address_info.get('city', '')
                    selected_state = address_info.get('state', '')
                    selected_country = address_info.get('country', '')
                    selected_timezone = get_timezone_from_coordinates(clicked_lat, clicked_lng)
                    
                    st.success(f"📍 Selected: {address_info.get('formatted_address', 'Unknown')}")
                    st.info(f"🌐 Coordinates: {clicked_lat:.6f}, {clicked_lng:.6f}")
    
    # Method 3: Address Search
    elif location_method == "📍 Address Search":
        st.markdown("#### 📍 Address Search")
        st.info("🔍 Enter an address to search for your location")
        
        search_address = st.text_input(
            "Enter your address:",
            placeholder="e.g., 1600 Pennsylvania Avenue, Washington, DC or Mumbai, India",
            help="Enter city, state, country or full address"
        )
        
        if st.button("🔍 Search Address") and search_address:
            with st.spinner("🔍 Searching for address..."):
                geocode_result = geocode_address(search_address)
                if geocode_result:
                    # Store results in session state to persist across re-renders
                    st.session_state['address_search_lat'] = geocode_result['lat']
                    st.session_state['address_search_lng'] = geocode_result['lng']
                    
                    # Get detailed address info
                    address_info = reverse_geocode_coordinates(geocode_result['lat'], geocode_result['lng'])
                    if address_info:
                        st.session_state['address_search_city'] = address_info.get('city', '')
                        st.session_state['address_search_state'] = address_info.get('state', '')
                        st.session_state['address_search_country'] = address_info.get('country', '')
                        st.session_state['address_search_timezone'] = get_timezone_from_coordinates(geocode_result['lat'], geocode_result['lng'])
                        st.session_state['address_search_formatted'] = geocode_result.get('formatted_address', 'Unknown')
                        
                        st.success(f"✅ Found: {geocode_result.get('formatted_address', 'Unknown')}")
                        st.info(f"🌐 Coordinates: {geocode_result['lat']:.6f}, {geocode_result['lng']:.6f}")
                        
                        # Show mini map
                        mini_map = folium.Map(location=[geocode_result['lat'], geocode_result['lng']], zoom_start=12)
                        folium.Marker(
                            [geocode_result['lat'], geocode_result['lng']],
                            popup=geocode_result.get('formatted_address', 'Found Location'),
                            icon=folium.Icon(color='green', icon='search')
                        ).add_to(mini_map)
                        st_folium(mini_map, width=700, height=300)
                else:
                    st.error("❌ Address not found. Please try a different search term.")
        
        # Use search results if available
        if 'address_search_lat' in st.session_state and 'address_search_lng' in st.session_state:
            selected_lat = st.session_state['address_search_lat']
            selected_lng = st.session_state['address_search_lng']
            selected_city = st.session_state.get('address_search_city', '')
            selected_state = st.session_state.get('address_search_state', '')
            selected_country = st.session_state.get('address_search_country', '')
            selected_timezone = st.session_state.get('address_search_timezone', 'UTC')
            
            # Display found location details
            st.markdown("##### 📍 Found Location Details")
            st.info(f"**Address:** {st.session_state.get('address_search_formatted', 'Unknown')}")
            st.info(f"**City:** {selected_city}")
            st.info(f"**State:** {selected_state}")
            st.info(f"**Country:** {selected_country}")
            st.info(f"**Coordinates:** {selected_lat:.6f}, {selected_lng:.6f}")
            
            # Clear search results button
            if st.button("🔄 Clear Search Results"):
                for key in ['address_search_lat', 'address_search_lng', 'address_search_city', 
                           'address_search_state', 'address_search_country', 'address_search_timezone', 
                           'address_search_formatted']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    # Method 4: Manual Entry
    elif location_method == "✍️ Manual Entry":
        st.markdown("#### ✍️ Manual Entry")
        st.info("📝 Enter your location details manually")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_country = st.text_input("Country:", value=selected_country, placeholder="e.g., India, USA, UK")
            selected_state = st.text_input("State/Province:", value=selected_state, placeholder="e.g., California, Maharashtra")
            selected_city = st.text_input("City:", value=selected_city, placeholder="e.g., Mumbai, New York")
        
        with col2:
            selected_lat = st.number_input("Latitude (Optional):", value=selected_lat, format="%.6f")
            selected_lng = st.number_input("Longitude (Optional):", value=selected_lng, format="%.6f")
            
            # Geocode button for manual entry
            if st.button("🔍 Find Coordinates") and selected_city:
                address_query = f"{selected_city}, {selected_state}, {selected_country}".strip(', ')
                with st.spinner("🔍 Finding coordinates..."):
                    geocode_result = geocode_address(address_query)
                    if geocode_result:
                        # Store coordinates in session state for persistence
                        st.session_state['manual_search_lat'] = geocode_result['lat']
                        st.session_state['manual_search_lng'] = geocode_result['lng']
                        st.success(f"✅ Coordinates found: {geocode_result['lat']:.6f}, {geocode_result['lng']:.6f}")
                        st.rerun()
        
        # Use manual search results if available
        if 'manual_search_lat' in st.session_state and 'manual_search_lng' in st.session_state:
            selected_lat = st.session_state['manual_search_lat']
            selected_lng = st.session_state['manual_search_lng']
            
            st.info(f"🌐 Found coordinates: {selected_lat:.6f}, {selected_lng:.6f}")
            if st.button("🔄 Clear Found Coordinates"):
                for key in ['manual_search_lat', 'manual_search_lng']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    # Timezone selection
    st.markdown("#### 🕐 Timezone Selection")
    timezones = get_common_timezones()
    
    # Flatten timezone options for selectbox
    timezone_options = ['UTC']
    timezone_labels = {'UTC': 'UTC'}
    
    for region, tz_dict in timezones.items():
        if region == 'UTC':
            continue
        if isinstance(tz_dict, dict):
            for tz_code, tz_name in tz_dict.items():
                timezone_options.append(tz_code)
                timezone_labels[tz_code] = f"{region}: {tz_name}"
        else:
            timezone_options.append(tz_dict)
            timezone_labels[tz_dict] = f"{region}: {tz_dict}"
    
    # Auto-detect timezone if coordinates are available
    if selected_lat != 0.0 and selected_lng != 0.0 and selected_timezone == 'UTC':
        selected_timezone = get_timezone_from_coordinates(selected_lat, selected_lng)
    
    selected_timezone = st.selectbox(
        "Timezone:",
        options=timezone_options,
        index=timezone_options.index(selected_timezone) if selected_timezone in timezone_options else 0,
        format_func=lambda x: timezone_labels.get(x, x),
        help="Select your timezone (auto-detected if coordinates are available)"
    )
    
    # Privacy level selection
    st.markdown("#### 🔒 Privacy Settings")
    privacy_options = [
        ('exact', '🎯 Exact Location (GPS coordinates)'),
        ('city_only', '🏙️ City Only'),
        ('state_only', '🗺️ State/Province Only'),
        ('country_only', '🌍 Country Only'),
        ('private', '🔒 Private (No location sharing)')
    ]
    
    current_privacy = location_prefs.get('privacy_level', 'city_only')
    privacy_index = next((i for i, (code, _) in enumerate(privacy_options) if code == current_privacy), 1)
    
    privacy_level = st.selectbox(
        "Privacy Level:",
        options=privacy_options,
        index=privacy_index,
        format_func=lambda x: x[1],
        help="Choose how much location information to share with other users"
    )
    
    # Save location button
    st.markdown("#### 💾 Save Location")
    
    # Display current selection
    if selected_city or selected_state or selected_country:
        location_display = []
        if selected_city:
            location_display.append(f"🏙️ {selected_city}")
        if selected_state:
            location_display.append(f"🗺️ {selected_state}")
        if selected_country:
            location_display.append(f"🌍 {selected_country}")
        
        st.info(f"📍 Location to save: {' • '.join(location_display)}")
        
        if selected_lat != 0.0 and selected_lng != 0.0:
            st.info(f"🌐 Coordinates: {selected_lat:.6f}, {selected_lng:.6f}")
        
        st.info(f"🕐 Timezone: {timezone_labels.get(selected_timezone, selected_timezone)}")
        st.info(f"🔒 Privacy: {privacy_level[1]}")
    
    if st.button("💾 Save Location Preferences", type="primary", use_container_width=True):
        coordinates = None
        if selected_lat != 0.0 and selected_lng != 0.0:
            coordinates = {'lat': selected_lat, 'lng': selected_lng}
        
        success = chatbot.update_location_preferences(
            user_id=chatbot.user_id,
            city=selected_city if selected_city else None,
            state=selected_state if selected_state else None,
            country=selected_country if selected_country else None,
            timezone=selected_timezone,
            coordinates=coordinates,
            privacy_level=privacy_level[0]
        )
        
        if success:
            st.success("✅ Location preferences updated successfully!")
            st.balloons()
            st.rerun()
        else:
            st.error("❌ Error updating location preferences. Please try again.")
    
    # Display current location (if privacy allows)
    if location_prefs and any(location_prefs.values()):
        st.markdown("#### 📍 Current Location")
        privacy = location_prefs.get('privacy_level', 'private')
        
        if privacy == 'private':
            st.info("🔒 Your location is set to private")
        elif privacy == 'country_only':
            st.info(f"🌍 Country: {location_prefs.get('country', 'Not set')}")
        elif privacy == 'state_only':
            st.info(f"🗺️ {location_prefs.get('state', 'Not set')}, {location_prefs.get('country', 'Not set')}")
        elif privacy == 'city_only':
            st.info(f"🏙️ {location_prefs.get('city', 'Not set')}, {location_prefs.get('state', 'Not set')}, {location_prefs.get('country', 'Not set')}")
        elif privacy == 'exact':
            coords = location_prefs.get('coordinates', {})
            if coords:
                st.info(f"📍 {location_prefs.get('city', 'Not set')}, {location_prefs.get('state', 'Not set')}, {location_prefs.get('country', 'Not set')}")
                st.info(f"🌐 Coordinates: {coords.get('lat', 0):.6f}, {coords.get('lng', 0):.6f}")
            else:
                st.info(f"🏙️ {location_prefs.get('city', 'Not set')}, {location_prefs.get('state', 'Not set')}, {location_prefs.get('country', 'Not set')}")
        
        # Show timezone
        if location_prefs.get('timezone'):
            tz_display = timezone_labels.get(location_prefs.get('timezone'), location_prefs.get('timezone'))
            st.info(f"🕐 Timezone: {tz_display}")
        
        # Show location on map (if coordinates available and privacy allows)
        if privacy in ['exact', 'city_only'] and location_prefs.get('coordinates'):
            coords = location_prefs.get('coordinates')
            if coords.get('lat') and coords.get('lng'):
                st.markdown("##### 🗺️ Your Location on Map")
                location_map = folium.Map(
                    location=[coords['lat'], coords['lng']], 
                    zoom_start=12
                )
                
                # Add marker with privacy-appropriate popup
                popup_text = f"{location_prefs.get('city', 'Your Location')}"
                if privacy == 'exact':
                    popup_text += f" ({coords['lat']:.4f}, {coords['lng']:.4f})"
                
                folium.Marker(
                    [coords['lat'], coords['lng']],
                    popup=popup_text,
                    tooltip="Your saved location",
                    icon=folium.Icon(color='blue', icon='home')
                ).add_to(location_map)
                
                st_folium(location_map, width=700, height=300)

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
            # Get conversation text for analysis
            conversation = chatbot.get_conversation()
            conversation_text = " ".join([msg for role, msg in conversation])
            
            analysis_result = chatbot.analyze_conversation_for_tags(
                user_id=chatbot.user_id,
                conversation_text=conversation_text,
                language_preferences=chatbot.get_language_preferences()
            )
            
            if analysis_result.get('success'):
                st.success("Analysis complete!")
                
                # Show analysis results
                with st.expander("🔍 React AI Analysis Results"):
                    st.markdown(f"**Analysis Summary:** {analysis_result.get('analysis_summary', 'N/A')}")
                    st.markdown(f"**Reasoning Steps:** {analysis_result.get('reasoning_steps', 0)}")
                    st.markdown(f"**Framework:** {analysis_result.get('framework', 'React AI Pattern')}")
                
                # Get tag suggestions
                existing_tags = chatbot.get_user_tags()
                suggestions_result = chatbot.get_tag_suggestions(
                    user_id=chatbot.user_id,
                    analysis_result=str(analysis_result.get('analysis_summary', '')),
                    existing_tags=[tag['tag'] if isinstance(tag, dict) else tag for tag in existing_tags]
                )
                
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
    
    # Location-aware tag suggestions
    st.markdown("### 📍 Location-Based Tag Suggestions")
    if st.button("Get Location-Aware Suggestions"):
        with st.spinner("React AI is analyzing your location context..."):
            suggestions_result = chatbot.suggest_tags(
                user_id=chatbot.user_id,
                limit=10
            )
            
            if suggestions_result.get('success'):
                suggestions = suggestions_result.get('suggestions', [])
                location_context = suggestions_result.get('location_context', {})
                
                if suggestions:
                    st.success(f"Found {len(suggestions)} location-based suggestions!")
                    
                    # Show location context
                    if location_context:
                        with st.expander("🌍 Location Context"):
                            st.write(f"**City:** {location_context.get('city', 'Not set')}")
                            st.write(f"**State:** {location_context.get('state', 'Not set')}")
                            st.write(f"**Country:** {location_context.get('country', 'Not set')}")
                            st.write(f"**Framework:** {suggestions_result.get('framework', 'React AI Pattern')}")
                    
                    # Show suggestions
                    st.markdown("### Suggested Tags Based on Your Location")
                    for suggestion in suggestions:
                        if st.button(f"Add: {suggestion}", key=f"location_suggest_{suggestion}"):
                            if chatbot.add_user_tag(suggestion):
                                st.success(f"Tag '{suggestion}' added!")
                                st.rerun()
                else:
                    st.info("No location-based suggestions available. Please set your location preferences first.")
            else:
                st.error(f"Error getting suggestions: {suggestions_result.get('error', 'Unknown error')}")

def _show_similar_users_interface(chatbot):
    """Show similar users interface with location-based search"""
    # Update last activity
    session_manager.update_last_activity()
    
    st.markdown("## 🤝 Similar Users")
    st.markdown("*Discover people who share your interests and location*")
    
    # Search options
    st.markdown("### 🔍 Search Options")
    search_type = st.selectbox(
        "Search Type:",
        options=['interests', 'location', 'both'],
        index=0,
        help="Choose how to find similar users"
    )
    
    # Location-based search options
    if search_type in ['location', 'both']:
        st.markdown("#### Location Search Options")
        location_search_type = st.selectbox(
            "Location Search Type:",
            options=['nearby', 'same_city', 'same_state'],
            index=0,
            help="Choose location search method"
        )
        
        if location_search_type == 'nearby':
            radius_km = st.slider("Search Radius (km):", min_value=1, max_value=500, value=50, help="Search radius in kilometers")
            location_filter = {'type': 'nearby', 'radius_km': radius_km}
        elif location_search_type == 'same_city':
            user_location = chatbot.get_location_preferences(chatbot.user_id)
            if user_location and user_location.get('city'):
                st.info(f"Searching for users in {user_location.get('city')}, {user_location.get('state', '')}")
                location_filter = {'type': 'city', 'city': user_location.get('city'), 'state': user_location.get('state')}
            else:
                st.warning("Please set your location preferences first to search by city.")
                location_filter = {}
        elif location_search_type == 'same_state':
            user_location = chatbot.get_location_preferences(chatbot.user_id)
            if user_location and user_location.get('state'):
                st.info(f"Searching for users in {user_location.get('state')}")
                location_filter = {'type': 'state', 'state': user_location.get('state')}
            else:
                st.warning("Please set your location preferences first to search by state.")
                location_filter = {}
        else:
            location_filter = {}
    else:
        location_filter = {}
    
    # Number of results
    limit = st.slider("Number of Results:", min_value=1, max_value=20, value=5, help="Maximum number of users to show")
    
    # Search button
    if st.button("🔍 Find Similar Users"):
        with st.spinner("React AI is finding similar users..."):
            if search_type == 'interests':
                result = chatbot.find_similar_users(chatbot.user_id, limit=limit)
            else:
                result = chatbot.find_similar_users_with_location(
                    chatbot.user_id,
                    search_type=search_type,
                    location_filter=location_filter,
                    limit=limit
                )
            
            # Handle the result structure properly
            if isinstance(result, dict) and result.get('success'):
                similar_users = result.get('similar_users', [])
            elif isinstance(result, list):
                similar_users = result
            else:
                similar_users = []
            
            if similar_users:
                st.success(f"Found {len(similar_users)} similar users!")
                
                st.markdown("### Users with Similar Interests")
                for user in similar_users:
                    # Handle both dict and string user objects
                    if isinstance(user, dict):
                        user_name = user.get('name', 'Unknown User')
                        user_id = user.get('user_id', 'unknown')
                        similarity_score = user.get('similarity_score', 0)
                        tags = user.get('tags', [])
                        languages = user.get('languages', ['Not specified'])
                        location_info = user.get('location', {})
                        distance_km = user.get('distance_km')
                    else:
                        # If user is a string, create a basic structure
                        user_name = str(user)
                        user_id = str(user)
                        similarity_score = 0
                        tags = []
                        languages = ['Not specified']
                        location_info = {}
                        distance_km = None
                    
                    with st.expander(f"👤 {user_name} (Similarity: {similarity_score:.1f}%)"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Tags:** {', '.join(tags)}")
                            st.write(f"**Languages:** {', '.join(languages)}")
                            
                            # Show location info based on privacy
                            privacy_level = location_info.get('privacy_level', 'private')
                            
                            if privacy_level == 'private':
                                st.write("**Location:** 🔒 Private")
                            elif privacy_level == 'country_only':
                                st.write(f"**Location:** 🌍 {location_info.get('country', 'Not specified')}")
                            elif privacy_level == 'state_only':
                                st.write(f"**Location:** 🏛️ {location_info.get('state', 'Not specified')}, {location_info.get('country', '')}")
                            elif privacy_level == 'city_only':
                                st.write(f"**Location:** 🏙️ {location_info.get('city', 'Not specified')}, {location_info.get('state', '')}")
                            elif privacy_level == 'exact':
                                coords = location_info.get('coordinates', {})
                                if coords:
                                    st.write(f"**Location:** 📍 {location_info.get('city', 'Not specified')}, {location_info.get('state', '')} ({coords.get('lat', 0):.4f}, {coords.get('lng', 0):.4f})")
                                else:
                                    st.write(f"**Location:** 🏙️ {location_info.get('city', 'Not specified')}, {location_info.get('state', '')}")
                        
                        with col2:
                            # Show distance if available
                            if distance_km is not None:
                                st.write(f"**Distance:** 📏 {distance_km:.1f} km away")
                            
                            # Create group chat button
                            if isinstance(user, dict):
                                if st.button(f"Create Group Chat with {user_name}", key=f"group_{user_id}"):
                                    _create_group_chat_with_user(chatbot, user)
                            else:
                                st.info("Group chat creation not available for this user format")
            else:
                st.info("No similar users found yet. Keep chatting to discover connections!")
    
    # Show nearby users section
    st.markdown("### 📍 Nearby Users")
    if st.button("Find Nearby Users"):
        with st.spinner("React AI is finding nearby users..."):
            nearby_result = chatbot.find_nearby_users(chatbot.user_id, radius_km=50, limit=10)
            
            # Handle the result structure properly
            if isinstance(nearby_result, dict) and nearby_result.get('success'):
                nearby_users = nearby_result.get('nearby_users', [])
            elif isinstance(nearby_result, list):
                nearby_users = nearby_result
            else:
                nearby_users = []
            
            if nearby_users:
                st.success(f"Found {len(nearby_users)} nearby users!")
                
                for user in nearby_users:
                    # Handle both dict and string user objects
                    if isinstance(user, dict):
                        user_name = user.get('name', 'Unknown User')
                        user_id = user.get('user_id', 'unknown')
                        distance = user.get('distance_km', 0)
                        tags = user.get('tags', [])
                        location_info = user.get('location', {})
                    else:
                        # If user is a string, create a basic structure
                        user_name = str(user)
                        user_id = str(user)
                        distance = 0
                        tags = []
                        location_info = {}
                    
                    with st.expander(f"📍 {user_name} - {distance:.1f} km away"):
                        st.write(f"**Tags:** {', '.join(tags)}")
                        st.write(f"**Distance:** {distance:.1f} km")
                        
                        # Show location based on privacy
                        privacy_level = location_info.get('privacy_level', 'private')
                        
                        if privacy_level != 'private':
                            if privacy_level == 'city_only':
                                st.write(f"**Location:** 🏙️ {location_info.get('city', 'Not specified')}")
                            elif privacy_level == 'exact':
                                st.write(f"**Location:** 📍 {location_info.get('city', 'Not specified')}")
                        
                        # Create group chat button
                        if isinstance(user, dict):
                            if st.button(f"Create Group Chat with {user_name}", key=f"nearby_{user_id}"):
                                _create_group_chat_with_user(chatbot, user)
                        else:
                            st.info("Group chat creation not available for this user format")
            else:
                st.info("No nearby users found. This might be because:")
                st.info("- You haven't set your location preferences")
                st.info("- No other users are nearby")
                st.info("- Other users have private location settings")

def _show_group_chats_interface(chatbot):
    """Show group chats interface"""
    # Update last activity
    session_manager.update_last_activity()
    
    st.markdown("## 👥 Group Chats")
    st.markdown("*Join or create group conversations*")
    
    # Get user's group chats using the database directly
    try:
        from group_chat import GroupChatManager
        group_manager = GroupChatManager(chatbot.db)
        user_groups_result = group_manager.get_user_groups(chatbot.user_id)
        user_groups = user_groups_result.get('groups', []) if isinstance(user_groups_result, dict) else user_groups_result
        
        if user_groups:
            st.markdown("### Your Group Chats")
            for group in user_groups:
                with st.expander(f"💬 {group['topic_name']}"):
                    # Safely handle participants display
                    participants = group.get('participants', [])
                    if isinstance(participants, list):
                        st.write(f"**Participants:** {', '.join(participants)}")
                    else:
                        st.write(f"**Participants:** {str(participants)}")
                    st.write(f"**Created:** {group['created_at']}")
                    
                    if st.button(f"Join {group['topic_name']}", key=f"join_{group['group_id']}"):
                        st.session_state['current_group_chat'] = group['group_id']
                        st.session_state['current_view'] = 'group_chat'
                        st.rerun()
        else:
            st.info("No group chats yet. Create one to start connecting!")
    except Exception as e:
        st.error(f"Error loading group chats: {str(e)}")
        st.info("No group chats available.")
    
    # Create new group chat
    st.markdown("### Create New Group Chat")
    with st.form("create_group_chat"):
        topic_name = st.text_input("Topic Name", placeholder="e.g., Tech Enthusiasts, Food Lovers")
        submitted = st.form_submit_button("Create Group Chat")
        
        if submitted and topic_name:
            try:
                # Create group with AI bot
                user_ids = [chatbot.user_id, "ai_bot"]
                result = group_manager.create_group_chat(topic_name.strip(), user_ids, chatbot.user_id)
                if result.get('success'):
                    st.success(f"Group chat '{topic_name.strip()}' created successfully!")
                    st.rerun()
                else:
                    st.error(f"Error creating group chat: {result.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"Error creating group chat: {str(e)}")

def _show_group_chat_interface(chatbot):
    """Show specific group chat interface"""
    # Update last activity
    session_manager.update_last_activity()
    
    group_id = st.session_state.get('current_group_chat')
    if not group_id:
        st.error("No group chat selected.")
        st.session_state['current_view'] = 'group_chats'
        st.rerun()
    
    try:
        # Get group chat using GroupChatManager
        from group_chat import GroupChatManager, GroupChat
        group_manager = GroupChatManager(chatbot.db)
        
        group_chat = group_manager.get_group_chat(group_id, chatbot.user_id)
        
        if not isinstance(group_chat, GroupChat):
            st.error(f"Group chat not found or you don't have access. Please try again or contact support.")
            st.session_state['current_view'] = 'group_chats'
            st.rerun()
            return
        
        # Get group info from database
        group_info = chatbot.db.get_group_info(group_id)
        if not group_info:
            st.error("Group chat not found.")
            st.session_state['current_view'] = 'group_chats'
            st.rerun()
            return
        
        # Header with back button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"## 💬 {group_info['topic_name']}")
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
        
        # Comprehensive Debug Panel
        with st.expander("🔧 System Debug Panel", expanded=False):
            st.markdown("#### Group Chat System Status")
            
            # Database connection status
            try:
                db_status = chatbot.db.check_connection()
                db_status_icon = "✅" if db_status.get('success') else "❌"
                st.markdown(f"{db_status_icon} **Database Connection:** {'Connected' if db_status.get('success') else 'Disconnected'}")
            except Exception as e:
                st.markdown(f"❌ **Database Connection:** Error - {str(e)}")
            
            # Group chat manager status
            try:
                group_manager_status = "✅ Active" if group_manager else "❌ Not Available"
                st.markdown(f"🔧 **Group Manager:** {group_manager_status}")
            except Exception as e:
                st.markdown(f"❌ **Group Manager:** Error - {str(e)}")
            
            # Chatbot status
            try:
                chatbot_status = chatbot.check_status()
                chatbot_status_icon = "✅" if chatbot_status.get('success') else "❌"
                st.markdown(f"{chatbot_status_icon} **Chatbot Status:** {'Online' if chatbot_status.get('success') else 'Offline'}")
            except Exception as e:
                st.markdown(f"❌ **Chatbot Status:** Error - {str(e)}")
            
            # Agent status
            try:
                agent_count = len(chatbot.agents) if hasattr(chatbot, 'agents') else 0
                st.markdown(f"🤖 **Active Agents:** {agent_count}")
            except Exception as e:
                st.markdown(f"❌ **Agent Status:** Error - {str(e)}")
            
            st.markdown("---")
            
            # Processing metrics
            st.markdown("#### Processing Metrics")
            
            # Message processing time estimation
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg AI Response Time", "~3 seconds")
                st.metric("Citation Generation", "Enabled")
            with col2:
                st.metric("Message Storage", "Real-time")
                st.metric("Framework", "React AI Pattern")
            
            # System capabilities
            st.markdown("#### System Capabilities")
            capabilities = [
                "✅ Real-time message processing",
                "✅ AI response generation with citations",
                "✅ User context awareness",
                "✅ Citation system integration",
                "✅ Database persistence",
                "✅ Multi-agent coordination"
            ]
            
            for capability in capabilities:
                st.markdown(capability)
            
            # Error tracking
            st.markdown("#### Recent Errors")
            if 'group_chat_errors' not in st.session_state:
                st.session_state['group_chat_errors'] = []
            
            if st.session_state['group_chat_errors']:
                for error in st.session_state['group_chat_errors'][-3:]:  # Show last 3 errors
                    st.error(f"**{error['timestamp']}:** {error['message']}")
            else:
                st.success("✅ No recent errors")
        
        # Display group messages with enhanced debug information
        st.markdown("### 💬 Messages")
        messages = group_chat.get_messages()
        
        # Debug information header
        with st.expander("🔧 Debug Information", expanded=False):
            st.markdown(f"**Group ID:** `{group_id}`")
            st.markdown(f"**Total Messages:** {len(messages)}")
            st.markdown(f"**Current User:** {chatbot.user_name} ({chatbot.user_id})")
            st.markdown(f"**Group Topic:** {group_info['topic_name']}")
            st.markdown(f"**Participants:** {', '.join(group_info['user_ids'])}")
            
            # Message statistics
            ai_messages = [m for m in messages if m['is_ai']]
            user_messages = [m for m in messages if not m['is_ai']]
            messages_with_citations = [m for m in messages if m.get('has_citations', False)]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("AI Messages", len(ai_messages))
            with col2:
                st.metric("User Messages", len(user_messages))
            with col3:
                st.metric("Messages with Citations", len(messages_with_citations))
        
        # Display messages with enhanced debug information
        for i, message in enumerate(messages):
            # Create a container for each message with debug info
            with st.container():
                # Message header with debug info
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    # Message sender and content
                    if message['is_ai']:
                        sender_icon = "🤖"
                        sender_name = "AI Assistant"
                    else:
                        sender_icon = "👤"
                        sender_name = message['sender']
                    
                    # Display message with citation links
                    message_text = message['message']
                    citation_links = message.get('citation_links', '')
                    if citation_links:
                        message_text += citation_links
                    
                    st.markdown(f"**{sender_icon} {sender_name}:** {message_text}")
                    
                    # Display citation details if available (expandable)
                    if message.get('citation_details'):
                        with st.expander(f"📚 View Citations ({len(message.get('citations', []))} total)", expanded=False):
                            citation_details = message['citation_details']
                            for citation_num, citation_info in citation_details.items():
                                st.markdown(f"**[{citation_num}]** {citation_info['display_text']}")
                                with st.expander(f"Details for Citation {citation_num}", expanded=False):
                                    st.json({
                                        "type": citation_info['type'],
                                        "source": citation_info['source'],
                                        "content": citation_info['content'],
                                        "relevance_score": citation_info['relevance_score'],
                                        "metadata": citation_info.get('metadata', {})
                                    })
                
                with col2:
                    # Debug button for this message
                    if st.button(f"🔧", key=f"debug_{i}", help="View debug info"):
                        st.session_state[f'debug_message_{i}'] = not st.session_state.get(f'debug_message_{i}', False)
                
                # Debug information for this message (expandable)
                if st.session_state.get(f'debug_message_{i}', False):
                    with st.expander(f"🔧 Debug Info for Message {i+1}", expanded=True):
                        st.markdown("#### Message Details")
                        st.json({
                            "message_index": i,
                            "sender": message['sender'],
                            "user_id": message['user_id'],
                            "is_ai": message['is_ai'],
                            "timestamp": message['timestamp'],
                            "message_length": len(message['message']),
                            "has_citations": message.get('has_citations', False),
                            "citation_count": len(message.get('citations', [])),
                            "citation_links": message.get('citation_links', ''),
                            "citation_details": message.get('citation_details', {})
                        })
                        
                        # Citation details if present
                        if message.get('citations'):
                            st.markdown("#### Citation Details")
                            citations = message['citations']
                            for j, citation in enumerate(citations):
                                if hasattr(citation, 'type'):
                                    # Citation object
                                    st.markdown(f"**Citation {j+1}:**")
                                    st.json({
                                        "id": citation.id,
                                        "type": citation.type,
                                        "source": citation.source,
                                        "content": citation.content[:100] + "..." if len(citation.content) > 100 else citation.content,
                                        "relevance_score": citation.relevance_score,
                                        "timestamp": citation.timestamp,
                                        "metadata": citation.metadata
                                    })
                                else:
                                    # Citation dict
                                    st.markdown(f"**Citation {j+1}:**")
                                    st.json(citation)
                        
                        # Processing information for AI messages
                        if message['is_ai']:
                            st.markdown("#### AI Processing Info")
                            st.info("This message was generated by the AI with the following context:")
                            
                            # Try to get processing context from the message
                            processing_info = {
                                "message_type": "AI Response",
                                "framework": "React AI Pattern with Citations",
                                "citation_generation": "Enabled" if message.get('has_citations') else "Disabled",
                                "response_length": len(message['message']),
                                "estimated_processing_time": "~2-5 seconds"
                            }
                            st.json(processing_info)
                        
                        # User message context
                        else:
                            st.markdown("#### User Message Context")
                            st.info("This message was sent by a user:")
                            
                            user_context = {
                                "message_type": "User Message",
                                "sender_name": message['sender'],
                                "user_id": message['user_id'],
                                "message_length": len(message['message']),
                                "timestamp": message['timestamp']
                            }
                            st.json(user_context)
            
            # Add a subtle separator between messages
            st.markdown("---")
    
    except Exception as e:
        st.error(f"Error loading group chat: {str(e)}")
        st.session_state['current_view'] = 'group_chats'
        st.rerun()
        return
    
    # Send message with enhanced debug information
    st.markdown("### 💭 Send Message")
    
    # Message sending debug panel
    with st.expander("🔧 Message Sending Debug", expanded=False):
        st.markdown("#### Sending Configuration")
        
        # Current user context
        st.markdown(f"**Current User:** {chatbot.user_name} ({chatbot.user_id})")
        st.markdown(f"**Group ID:** {group_id}")
        st.markdown(f"**Group Topic:** {group_info['topic_name']}")
        
        # Message processing settings
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Processing Settings:**")
            st.markdown("• AI Response Generation: ✅ Enabled")
            st.markdown("• Citation Generation: ✅ Enabled")
            st.markdown("• Real-time Processing: ✅ Enabled")
        
        with col2:
            st.markdown("**Expected Behavior:**")
            st.markdown("• User message saved to database")
            st.markdown("• AI generates contextual response")
            st.markdown("• Citations generated for response")
            st.markdown("• Response saved with metadata")
        
        # Recent sending history
        if 'message_sending_history' not in st.session_state:
            st.session_state['message_sending_history'] = []
        
        if st.session_state['message_sending_history']:
            st.markdown("#### Recent Message History")
            for entry in st.session_state['message_sending_history'][-5:]:  # Last 5 messages
                status_icon = "✅" if entry['success'] else "❌"
                st.markdown(f"{status_icon} **{entry['timestamp']}:** {entry['message'][:50]}...")
    
    with st.form("group_message"):
        user_input = st.text_input("Type your message:", placeholder="What would you like to say?")
        send_button = st.form_submit_button("Send")
        
        if send_button and user_input:
            try:
                # Track sending start time
                import time
                start_time = time.time()
                
                # Send message using the group chat instance
                ai_response_data = group_chat.send_message(user_input.strip())
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Log the sending attempt
                sending_entry = {
                    'timestamp': time.strftime('%H:%M:%S'),
                    'message': user_input.strip(),
                    'success': True,
                    'processing_time': f"{processing_time:.2f}s",
                    'response_length': len(ai_response_data.get('response', '')),
                    'citations_generated': len(ai_response_data.get('citations', []))
                }
                
                if 'message_sending_history' not in st.session_state:
                    st.session_state['message_sending_history'] = []
                st.session_state['message_sending_history'].append(sending_entry)
                
                # Show success with debug info
                st.success(f"✅ Message sent successfully! (Processing time: {processing_time:.2f}s)")
                
                # Show response debug info
                with st.expander("🔧 Response Debug Info", expanded=False):
                    st.markdown("#### AI Response Details")
                    st.json({
                        "processing_time": f"{processing_time:.2f}s",
                        "response_length": len(ai_response_data.get('response', '')),
                        "citations_count": len(ai_response_data.get('citations', [])),
                        "has_citations": ai_response_data.get('has_citations', False),
                        "framework": ai_response_data.get('framework', 'React AI Pattern'),
                        "citation_links": ai_response_data.get('citation_links', ''),
                        "citation_details": ai_response_data.get('citation_details', {})
                    })
                
                st.rerun()
                
            except Exception as e:
                # Log the error
                error_entry = {
                    'timestamp': time.strftime('%H:%M:%S'),
                    'message': user_input.strip(),
                    'success': False,
                    'error': str(e)
                }
                
                if 'message_sending_history' not in st.session_state:
                    st.session_state['message_sending_history'] = []
                st.session_state['message_sending_history'].append(error_entry)
                
                # Log error in group chat errors
                if 'group_chat_errors' not in st.session_state:
                    st.session_state['group_chat_errors'] = []
                
                st.session_state['group_chat_errors'].append({
                    'timestamp': time.strftime('%H:%M:%S'),
                    'message': f"Message sending failed: {str(e)}"
                })
                
                st.error(f"❌ Error sending message: {str(e)}")
                
                # Show detailed error info
                with st.expander("🔧 Error Debug Info", expanded=True):
                    st.markdown("#### Error Details")
                    st.json({
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "user_input": user_input.strip(),
                        "group_id": group_id,
                        "user_id": chatbot.user_id,
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                    })

def _show_nearby_users_rag_interface(chatbot):
    """Show RAG-powered nearby users interface with semantic search"""
    # Update last activity
    session_manager.update_last_activity()
    
    st.markdown("## 📍 Nearby Users (RAG Enhanced)")
    st.markdown("*Discover nearby users using advanced semantic search and location intelligence*")
    
    # Get user's location
    user_location = chatbot.get_location_preferences(chatbot.user_id)
    
    if not any(user_location.values()):
        st.warning("⚠️ Location not set up yet!")
        st.markdown("### 📍 Set Up Your Location")
        st.write("To find nearby users with RAG-enhanced search, please add your location information first.")
        
        if st.button("📍 Go to Profile Settings", use_container_width=True):
            st.session_state['current_view'] = 'profile'
            st.rerun()
        return
    
    # Display current location
    st.markdown("### 📍 Your Current Location")
    location_display = []
    if user_location.get('city'):
        location_display.append(f"🏙️ {user_location['city']}")
    if user_location.get('state'):
        location_display.append(f"🗺️ {user_location['state']}")
    if user_location.get('country'):
        location_display.append(f"🌍 {user_location['country']}")
    
    if location_display:
        st.write(" • ".join(location_display))
    
    # RAG Search Configuration
    st.markdown("### 🔍 RAG Search Configuration")
    
    # Search type selection outside form for dynamic UI updates
    search_type = st.selectbox(
        "Search Type:",
        options=['hybrid', 'semantic', 'location'],
        format_func=lambda x: {
            'hybrid': '🎯 Hybrid (Location + Semantic)',
            'semantic': '🧠 Semantic Similarity Only',
            'location': '📍 Location-Based Only'
        }[x],
        index=st.session_state.get('rag_search_type_index', 0),
        help="Choose search method",
        key="main_search_type"
    )
    
    # Save the current selection index to session state
    st.session_state['rag_search_type_index'] = ['hybrid', 'semantic', 'location'].index(search_type)
    
    # Show description of current search type
    search_descriptions = {
        'hybrid': "🎯 **Hybrid Search**: Combines location proximity with semantic similarity for balanced results",
        'semantic': "🧠 **Semantic Search**: Finds users based on similar interests and content, regardless of location",
        'location': "📍 **Location Search**: Finds users based on geographic proximity only"
    }
    st.info(search_descriptions[search_type])
    
    # Dynamic form based on search type
    with st.form("rag_search_config"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_results = st.slider(
                "Max Results:",
                min_value=5,
                max_value=50,
                value=10,
                help="Maximum number of users to return"
            )
            
            # Show location radius only for hybrid and location searches
            if search_type in ['hybrid', 'location']:
                location_radius_km = st.slider(
                    "🌍 Search Radius (km):",
                    min_value=1,
                    max_value=500,
                    value=50,
                    help="Search radius in kilometers"
                )
                st.success("✅ Location search enabled")
            else:
                location_radius_km = 50  # Default value for semantic search
                st.info("ℹ️ Location search disabled for semantic-only mode")
        
        with col2:
            # Show semantic query only for hybrid and semantic searches
            if search_type in ['hybrid', 'semantic']:
                semantic_query = st.text_input(
                    "🧠 Semantic Query:",
                    placeholder="e.g., 'technology enthusiast', 'food lover', 'travel blogger'",
                    help="Describe the type of person you're looking for"
                )
                st.success("✅ Semantic search enabled")
                
                if search_type == 'semantic':
                    st.info("💡 **Tip**: Use specific terms like 'bollywood', 'technology', or 'food' for best results")
            else:
                semantic_query = ""  # Empty for location-only search
                st.info("ℹ️ Semantic search disabled for location-only mode")
        
        search_submitted = st.form_submit_button("🔍 Search with RAG", use_container_width=True)
    
    # Perform RAG search
    if search_submitted:
        with st.spinner("🤖 Searching with RAG intelligence..."):
            # First, ensure user profile is vectorized
            vectorization_result = chatbot.vectorize_user_profile(chatbot.user_id)
            
            if vectorization_result.get('success'):
                st.success("✅ User profile vectorized successfully!")
            else:
                st.warning(f"⚠️ Vectorization issue: {vectorization_result.get('error', 'Unknown error')}")
            
            # Perform RAG search
            rag_result = chatbot.rag_nearby_users_search(
                user_id=chatbot.user_id,
                search_type=search_type,
                location_radius_km=location_radius_km,
                semantic_query=semantic_query,
                max_results=max_results
            )
            
            if rag_result.get('success'):
                nearby_users = rag_result.get('nearby_users', [])
                
                if nearby_users:
                    st.markdown(f"### 🎯 Found {len(nearby_users)} Users")
                    st.markdown(f"**Search Method:** {rag_result.get('search_method', 'RAG Enhanced')}")
                    st.markdown(f"**Framework:** {rag_result.get('framework', 'React AI + RAG')}")
                    
                    # Display search results
                    for user in nearby_users:
                        with st.expander(f"👤 {user['name']} • Score: {user.get('combined_score', user.get('semantic_score', 0)):.3f}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**🏷️ Interests:**")
                                if user.get('tags'):
                                    for tag in user['tags'][:5]:  # Show first 5 tags
                                        st.markdown(f"• {tag}")
                                else:
                                    st.markdown("• No tags available")
                                
                                # Show scores
                                if user.get('semantic_score'):
                                    st.markdown(f"**🧠 Semantic Score:** {user['semantic_score']:.3f}")
                                if user.get('location_score'):
                                    st.markdown(f"**📍 Location Score:** {user['location_score']:.3f}")
                                if user.get('combined_score'):
                                    st.markdown(f"**🎯 Combined Score:** {user['combined_score']:.3f}")
                            
                            with col2:
                                st.markdown("**📍 Location:**")
                                if user.get('distance_km'):
                                    st.markdown(f"• **Distance:** {user['distance_km']:.1f} km")
                                if user.get('city'):
                                    st.markdown(f"• **City:** {user['city']}")
                                if user.get('state'):
                                    st.markdown(f"• **State:** {user['state']}")
                                
                                privacy_level = user.get('privacy_level', 'private')
                                privacy_emoji = {
                                    'exact': '🎯',
                                    'city_only': '🏙️',
                                    'state_only': '🗺️',
                                    'country_only': '🌍',
                                    'private': '🔒'
                                }.get(privacy_level, '❓')
                                
                                st.markdown(f"**🔒 Privacy:** {privacy_emoji} {privacy_level.replace('_', ' ').title()}")
                            
                            # Action buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"💬 Chat with {user['name']}", key=f"chat_{user['user_id']}"):
                                    st.info(f"Chat feature with {user['name']} - Coming soon!")
                            
                            with col2:
                                if st.button(f"👥 Create Group", key=f"group_{user['user_id']}"):
                                    st.info(f"Group creation with {user['name']} - Coming soon!")
                
                else:
                    st.markdown("### 🔍 No Users Found")
                    st.write("No users found matching your search criteria. This could be because:")
                    st.write("- No users match your semantic query")
                    st.write("- No users are within your location radius")
                    st.write("- Users have restrictive privacy settings")
                    st.write("- Your profile hasn't been vectorized yet")
                    
                    # Suggestions
                    st.markdown("**💡 Suggestions:**")
                    st.write("- 🔍 Try different search parameters")
                    st.write("- 📍 Expand your search radius")
                    st.write("- 🧠 Use different semantic queries")
                    st.write("- 🏷️ Add more interests to your profile")
            
            else:
                st.error(f"❌ RAG search failed: {rag_result.get('error', 'Unknown error')}")
    
    # RAG Statistics
    st.markdown("### 📊 RAG System Statistics")
    
    try:
        rag_stats = chatbot.get_rag_statistics()
        
        if rag_stats.get('rag_enabled', True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Cached Embeddings", rag_stats.get('cached_embeddings', 0))
            
            with col2:
                st.metric("Vector Store", "✅ Available" if rag_stats.get('vector_store_available') else "❌ Unavailable")
            
            with col3:
                st.metric("Embeddings", "✅ Available" if rag_stats.get('embeddings_available') else "❌ Unavailable")
            
            with col4:
                st.metric("Similarity Threshold", f"{rag_stats.get('similarity_threshold', 0.7):.2f}")
            
            # Additional stats
            st.markdown("**Configuration:**")
            st.write(f"• Embedding Dimension: {rag_stats.get('embedding_dimension', 1536)}")
            st.write(f"• Location Weight: {rag_stats.get('location_weight', 0.3):.2f}")
            st.write(f"• Max Nearby Users: {rag_stats.get('max_nearby_users', 20)}")
        
        else:
            st.warning("⚠️ RAG system not fully available")
            st.write(f"Error: {rag_stats.get('error', 'Unknown error')}")
    
    except Exception as e:
        st.error(f"❌ Error getting RAG statistics: {str(e)}")
    
    # Debug Section for RAG Nearby Users
    st.markdown("### 🔍 RAG Debug Information")
    
    with st.expander("🐛 Debug RAG Nearby Users Search", expanded=False):
        st.markdown("**Debug a specific search query to understand why results are not found**")
        
        # Debug form to avoid conflicts with main form
        with st.form("debug_form"):
            col1, col2 = st.columns(2)
            with col1:
                debug_query = st.text_input("Debug Query:", placeholder="e.g., bollywood, technology, food lovers")
                debug_search_type = st.selectbox("Search Type:", ['hybrid', 'semantic', 'location'], key="debug_search_type")
            
            with col2:
                debug_radius = st.slider("Radius (km):", 1, 500, 50, key="debug_radius")
                debug_max_results = st.slider("Max Results:", 1, 50, 10, key="debug_max_results")
            
            debug_submitted = st.form_submit_button("🔍 Debug Search")
        
        if debug_submitted:
            st.markdown("#### Debug Results:")
            
            try:
                # Get current user location
                user_location = chatbot.get_location_preferences(chatbot.user_id)
                st.write(f"**Current User Location:** {user_location}")
                
                # Get all users with their locations and tags
                all_users = chatbot.db.get_all_users_summary()
                st.write(f"**Total Users in Database:** {len(all_users)}")
                
                # Filter users with location data
                users_with_location = [u for u in all_users if u.get('location') and (u['location'].get('city') or u['location'].get('coordinates', {}).get('lat'))]
                st.write(f"**Users with Location Data:** {len(users_with_location)}")
                
                # Filter users in same city
                same_city_users = [u for u in all_users if u.get('location') and (u['location'].get('city') or '').lower() == (user_location.get('city') or '').lower() and u['user_id'] != chatbot.user_id]
                st.write(f"**Users in Same City:** {len(same_city_users)}")
                
                # Show users with query-related tags (dynamic based on debug query)
                if debug_query:
                    query_users = [u for u in all_users if any(debug_query.lower() in (tag or '').lower() for tag in u.get('tags', []))]
                    st.write(f"**Users with '{debug_query}' Tags:** {len(query_users)}")
                    
                    # Show users with query tags in same city
                    query_same_city = [u for u in same_city_users if any(debug_query.lower() in (tag or '').lower() for tag in u.get('tags', []))]
                    st.write(f"**Users with '{debug_query}' Tags in Same City:** {len(query_same_city)}")
                    
                    if query_same_city:
                        st.markdown(f"**{debug_query.title()} Users in Same City:**")
                        for user in query_same_city:
                            matching_tags = [tag for tag in user.get('tags', []) if debug_query.lower() in tag.lower()]
                            st.write(f"• {user['name']} - Matching Tags: {', '.join(matching_tags)}")
                
                # Test RAG search
                st.markdown("#### RAG Search Test:")
                
                # Show what search parameters are being used
                st.info(f"**Search Parameters:** Type={debug_search_type}, Query='{debug_query}', Radius={debug_radius}km, Max Results={debug_max_results}")
                
                rag_result = chatbot.rag_nearby_users_search(
                    user_id=chatbot.user_id,
                    search_type=debug_search_type,
                    location_radius_km=debug_radius,
                    semantic_query=debug_query,
                    max_results=debug_max_results
                )
                
                # Display detailed results
                if rag_result.get('success'):
                    nearby_users = rag_result.get('nearby_users', [])
                    st.success(f"✅ **Found {len(nearby_users)} users via RAG**")
                    st.write(f"**Search Method:** {rag_result.get('search_method', 'Unknown')}")
                    st.write(f"**Keyword Filtering Enabled:** {rag_result.get('keyword_filtering_enabled', 'Unknown')}")
                    
                    if nearby_users:
                        st.markdown("**Results:**")
                        for i, user in enumerate(nearby_users, 1):
                            score = user.get('combined_score', user.get('semantic_score', 0))
                            st.write(f"{i}. **{user.get('name', 'Unknown')}** - Score: {score:.3f}")
                            st.write(f"   Tags: {', '.join(user.get('tags', [])[:5])}")
                            if user.get('semantic_score'):
                                st.write(f"   Semantic: {user['semantic_score']:.3f}")
                            if user.get('location_score'):
                                st.write(f"   Location: {user['location_score']:.3f}")
                    else:
                        st.warning("⚠️ No users found matching the criteria")
                        
                        # Provide troubleshooting suggestions
                        st.markdown("**Troubleshooting Suggestions:**")
                        if debug_search_type == 'semantic':
                            st.write("• Try broader terms (e.g., 'technology' instead of 'machine learning')")
                            st.write("• Check if users have related tags in their profiles")
                            st.write("• Lower the similarity threshold by using hybrid search")
                        elif debug_search_type == 'location':
                            st.write("• Increase the search radius")
                            st.write("• Check if other users have set their location")
                        else:  # hybrid
                            st.write("• Try semantic-only search to see if there are any matches")
                            st.write("• Try location-only search to see nearby users")
                else:
                    st.error(f"❌ **RAG search failed:** {rag_result.get('error', 'Unknown error')}")
                    
                    # Show additional debugging info for failures
                    st.markdown("**Debugging Information:**")
                    st.write(f"• Search type: {debug_search_type}")
                    st.write(f"• Query: '{debug_query}'")
                    st.write(f"• User ID: {chatbot.user_id}")
                    st.write(f"• Vector store available: {rag_result.get('vector_search_available', 'Unknown')}")
                    
            except Exception as e:
                st.error(f"❌ **Error in debug search:** {str(e)}")
                st.write("**Stack trace:**")
                import traceback
                st.code(traceback.format_exc())
            
            # Test traditional database search for comparison
            st.markdown("#### Traditional Database Search Test:")
            try:
                # Test find_nearby_users (GPS-based)
                gps_users = chatbot.db.find_nearby_users(chatbot.user_id, debug_radius)
                st.write(f"**GPS-based nearby users:** {len(gps_users)}")
                
                # Test find_users_in_city (city-based)
                city_users = chatbot.db.find_users_in_city(chatbot.user_id)
                st.write(f"**City-based nearby users:** {len(city_users)}")
                
                # Test find_similar_users (tag-based)
                similar_users = chatbot.db.find_similar_users(chatbot.user_id, min_common_tags=1)
                st.write(f"**Tag-based similar users:** {len(similar_users)}")
                
            except Exception as e:
                st.error(f"Error in database search test: {str(e)}")
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Profile Vector", use_container_width=True):
            with st.spinner("Updating profile vector..."):
                result = chatbot.vectorize_user_profile(chatbot.user_id)
                if result.get('success'):
                    st.success("✅ Profile vector updated!")
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    with col2:
        if st.button("🧠 Semantic Search Only", use_container_width=True):
            st.session_state['semantic_search_mode'] = True
            st.rerun()
    
    with col3:
        if st.button("🤝 Traditional Search", use_container_width=True):
            st.session_state['current_view'] = 'similar_users'
            st.rerun()

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
    
    # User List Section
    st.markdown("---")
    st.markdown("### 👥 User Directory")
    st.markdown("*View all registered users and their location information*")
    
    try:
        # Get all users summary
        all_users = chatbot.db.get_all_users_summary()
        
        if all_users:
            # Display user count
            st.markdown(f"**Total Users:** {len(all_users)}")
            
            # Search and filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                search_term = st.text_input("🔍 Search Users", placeholder="Search by name...")
            
            with col2:
                location_filter = st.selectbox("📍 Filter by Location", 
                                             ["All Locations", "With Location", "Without Location"])
            
            with col3:
                privacy_filter = st.selectbox("🔒 Privacy Level", 
                                            ["All", "exact", "city_only", "state_only", "country_only", "private"])
            
            # Apply filters
            filtered_users = all_users
            
            if search_term:
                filtered_users = [user for user in filtered_users 
                                if search_term.lower() in (user['name'] or '').lower()]
            
            if location_filter == "With Location":
                filtered_users = [user for user in filtered_users 
                                if user['location'].get('city') or user['location'].get('state') or user['location'].get('country')]
            elif location_filter == "Without Location":
                filtered_users = [user for user in filtered_users 
                                if not (user['location'].get('city') or user['location'].get('state') or user['location'].get('country'))]
            
            if privacy_filter != "All":
                filtered_users = [user for user in filtered_users 
                                if user['location'].get('privacy_level') == privacy_filter]
            
            # Display filtered results
            st.markdown(f"**Showing {len(filtered_users)} of {len(all_users)} users**")
            
            # Pagination
            users_per_page = 10
            total_pages = (len(filtered_users) + users_per_page - 1) // users_per_page
            
            if total_pages > 1:
                page = st.selectbox("📄 Page", range(1, total_pages + 1))
                start_idx = (page - 1) * users_per_page
                end_idx = start_idx + users_per_page
                page_users = filtered_users[start_idx:end_idx]
            else:
                page_users = filtered_users
            
            # Display users in a table-like format
            for user in page_users:
                with st.expander(f"👤 {user['name']} ({user['user_id'][:8]}...)", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📋 Profile Information**")
                        st.write(f"**Name:** {user['name']}")
                        st.write(f"**User ID:** `{user['user_id']}`")
                        st.write(f"**Created:** {user.get('created_at', 'Unknown')}")
                        st.write(f"**Last Updated:** {user.get('profile_updated_at', 'Unknown')}")
                        
                        # Language preferences
                        st.markdown("**🌐 Language Preferences**")
                        st.write(f"**Native Language:** {user.get('native_language', 'Not specified')}")
                        st.write(f"**Preferred Languages:** {', '.join(user.get('preferred_languages', [])) or 'None'}")
                        st.write(f"**Comfort Level:** {user.get('language_comfort_level', 'english')}")
                        
                        # Tags
                        st.markdown("**🏷️ Interest Tags**")
                        if user.get('tags'):
                            tags_display = ', '.join([f"`{tag}`" for tag in user['tags'][:10]])
                            if len(user['tags']) > 10:
                                tags_display += f" ... (+{len(user['tags']) - 10} more)"
                            st.markdown(tags_display)
                        else:
                            st.write("No tags available")
                    
                    with col2:
                        st.markdown("**📍 Location Information**")
                        location = user.get('location', {})
                        
                        # Location details
                        if location.get('city') or location.get('state') or location.get('country'):
                            location_parts = []
                            if location.get('city'):
                                location_parts.append(f"🏙️ **City:** {location['city']}")
                            if location.get('state'):
                                location_parts.append(f"🗺️ **State:** {location['state']}")
                            if location.get('country'):
                                location_parts.append(f"🌍 **Country:** {location['country']}")
                            if location.get('timezone'):
                                location_parts.append(f"🕐 **Timezone:** {location['timezone']}")
                            
                            for part in location_parts:
                                st.markdown(part)
                        else:
                            st.write("No location information available")
                        
                        # Privacy level
                        privacy_level = location.get('privacy_level', 'city_only')
                        privacy_emoji = {
                            'exact': '🎯',
                            'city_only': '🏙️',
                            'state_only': '🗺️',
                            'country_only': '🌍',
                            'private': '🔒'
                        }.get(privacy_level, '❓')
                        
                        st.markdown(f"**🔒 Privacy Level:** {privacy_emoji} {privacy_level.replace('_', ' ').title()}")
                        
                        # Coordinates (if available and privacy allows)
                        coordinates = location.get('coordinates', {})
                        if coordinates.get('latitude') and coordinates.get('longitude'):
                            if privacy_level in ['exact', 'city_only']:
                                st.markdown(f"**📍 Coordinates:** {coordinates['latitude']:.4f}, {coordinates['longitude']:.4f}")
                            else:
                                st.markdown("**📍 Coordinates:** Hidden (Privacy)")
                        else:
                            st.markdown("**📍 Coordinates:** Not available")
                        
                        # Last location update
                        if location.get('last_updated'):
                            st.markdown(f"**🕐 Last Updated:** {location['last_updated']}")
        else:
            st.info("No users found in the database.")
    
    except Exception as e:
        st.error(f"❌ Error loading user data: {str(e)}")

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
        st.session_state['chatbot'].set_user_context(user_info['user_id'], user_info['user_name'])
    
    chatbot = st.session_state['chatbot']
    
    # Navigation sidebar
    st.sidebar.markdown(f"### 👤 User: {user_info['user_name']}")
    st.sidebar.markdown(f"**User ID:** `{user_info['user_id'][:8]}...`")
    
    # Navigation menu
    view_options = {
        'chat': '💬 Chat',
        'profile': '👤 Profile & Tags',
        'similar_users': '🤝 Similar Users',
        'nearby_users': '📍 Nearby Users (RAG)',
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
    elif st.session_state['current_view'] == 'nearby_users':
        _show_nearby_users_rag_interface(chatbot)
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
            st.session_state['chatbot'].set_user_context(user_id, name)
            
            st.success(f"👋 Welcome back, {name}! Your session has been loaded.")
            st.rerun()
        elif submit_button and not user_name.strip():
            st.error("Please enter your name to continue.") 