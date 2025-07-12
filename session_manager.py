import streamlit as st
import json
import uuid
from datetime import datetime, timedelta

class SessionManager:
    def __init__(self):
        self.session_key = "user_session"
        self.cookie_key = "chatbot_user_session"
    
    def save_user_session(self, user_id, user_name):
        """Save user session to persistent storage"""
        session_data = {
            'user_id': user_id,
            'user_name': user_name,
            'authenticated': True,
            'last_activity': datetime.now().isoformat()
        }
        
        # Save to Streamlit session state
        st.session_state['user_authenticated'] = True
        st.session_state['user_id'] = user_id
        st.session_state['user_name'] = user_name
        
        # Save to browser storage using st.query_params
        # This creates a URL parameter that persists across refreshes
        st.query_params["user_id"] = user_id
        st.query_params["user_name"] = user_name
        st.query_params["authenticated"] = "true"
        
        return session_data
    
    def load_user_session(self):
        """Load user session from persistent storage"""
        # First check URL parameters (most persistent)
        params = st.query_params
        
        if 'user_id' in params and 'user_name' in params and 'authenticated' in params:
            user_id = params['user_id']
            user_name = params['user_name']
            authenticated = params['authenticated'] == 'true'
            
            if authenticated:
                # Restore session state
                st.session_state['user_authenticated'] = True
                st.session_state['user_id'] = user_id
                st.session_state['user_name'] = user_name
                return {
                    'user_id': user_id,
                    'user_name': user_name,
                    'authenticated': True
                }
        
        # Fallback to session state
        if 'user_authenticated' in st.session_state and st.session_state['user_authenticated']:
            return {
                'user_id': st.session_state.get('user_id'),
                'user_name': st.session_state.get('user_name'),
                'authenticated': True
            }
        
        return None
    
    def clear_user_session(self):
        """Clear user session from all storage"""
        # Clear session state
        if 'user_authenticated' in st.session_state:
            del st.session_state['user_authenticated']
        if 'user_id' in st.session_state:
            del st.session_state['user_id']
        if 'user_name' in st.session_state:
            del st.session_state['user_name']
        if 'chatbot' in st.session_state:
            del st.session_state['chatbot']
        if 'current_view' in st.session_state:
            del st.session_state['current_view']
        
        # Clear URL parameters
        st.query_params.clear()
    
    def is_user_authenticated(self):
        """Check if user is authenticated"""
        session = self.load_user_session()
        return session is not None and session.get('authenticated', False)
    
    def get_user_info(self):
        """Get current user information"""
        session = self.load_user_session()
        if session and session.get('authenticated'):
            return {
                'user_id': session['user_id'],
                'user_name': session['user_name']
            }
        return None
    
    def update_last_activity(self):
        """Update last activity timestamp"""
        if self.is_user_authenticated():
            # Update URL parameters with new timestamp
            user_info = self.get_user_info()
            if user_info:
                st.query_params["user_id"] = user_info['user_id']
                st.query_params["user_name"] = user_info['user_name']
                st.query_params["authenticated"] = "true"
                st.query_params["last_activity"] = datetime.now().isoformat()

# Global session manager instance
session_manager = SessionManager() 