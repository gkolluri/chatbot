"""
Session Agent for Multi-Agent Chatbot System using LangGraph
==========================================================

This agent handles session management and persistence using LangGraph workflows, including:
- URL parameter-based session storage for cross-tab persistence
- Fallback to Streamlit session state for reliability
- Cross-browser tab synchronization
- Activity tracking and monitoring with timestamps
- Clean logout functionality with complete session clearing
- Language preference persistence across sessions
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid
from .base_agent import LangGraphBaseAgent, AgentState


class SessionAgent(LangGraphBaseAgent):
    """
    Agent responsible for session management and persistence using LangGraph.
    
    Manages user sessions, handles persistence across browser tabs,
    and provides session-related functionality with LangGraph workflows.
    """
    
    def __init__(self, db_interface=None):
        """Initialize the session agent."""
        super().__init__("SessionAgent", db_interface)
        self.active_sessions = {}
        self.session_timeout = timedelta(hours=24)  # 24-hour session timeout
        
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this agent provides."""
        return [
            "session_management",
            "cross_tab_synchronization",
            "activity_tracking",
            "session_persistence",
            "logout_handling",
            "session_cleanup",
            "url_parameter_management",
            "langgraph_workflow"
        ]
    
    def _get_agent_system_prompt(self) -> str:
        """Get the system prompt for session agent."""
        return """You are an AI agent specialized in session management and persistence.
        Your role is to:
        1. Manage user sessions and authentication
        2. Handle session persistence across browser tabs
        3. Track user activity and session state
        4. Provide session-related functionality
        5. Ensure secure session handling
        6. Manage session cleanup and logout
        
        Guidelines:
        - Maintain session security and privacy
        - Ensure reliable session persistence
        - Handle session timeouts gracefully
        - Provide clear session status information
        - Support cross-tab synchronization
        """
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process session-related requests using LangGraph workflow.
        
        Args:
            request: Request dictionary with parameters
            
        Returns:
            Response dictionary with results
        """
        request_type = request.get('type', '')
        
        if request_type == 'create_session':
            return self._create_user_session(request)
        elif request_type == 'validate_session':
            return self._validate_session(request)
        elif request_type == 'update_session':
            return self._update_session(request)
        elif request_type == 'get_session_info':
            return self._get_session_info(request)
        elif request_type == 'logout_session':
            return self._logout_session(request)
        elif request_type == 'cleanup_expired_sessions':
            return self._cleanup_expired_sessions(request)
        elif request_type == 'get_active_sessions':
            return self._get_active_sessions(request)
        else:
            return {
                'success': False,
                'error': f'Unknown request type: {request_type}',
                'available_types': ['create_session', 'validate_session', 'update_session', 'get_session_info', 'logout_session', 'cleanup_expired_sessions', 'get_active_sessions']
            }
    
    def _create_user_session(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new user session.
        
        Args:
            request: Request with user information
            
        Returns:
            Response with session data
        """
        user_name = request.get('user_name')
        language_preferences = request.get('language_preferences', {})
        
        if not user_name:
            return {
                'success': False,
                'error': 'Missing user_name'
            }
        
        # Generate unique user ID
        user_id = str(uuid.uuid4())
        
        # Create session data
        session_data = {
            'user_id': user_id,
            'user_name': user_name,
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'language_preferences': language_preferences,
            'is_active': True,
            'session_id': str(uuid.uuid4()),
            'url_parameters': {
                'user_id': user_id,
                'user_name': user_name,
                'authenticated': 'true',
                'last_activity': datetime.now().isoformat()
            }
        }
        
        # Store session
        self.active_sessions[user_id] = session_data
        
        # Save to database if available
        if self.db:
            try:
                self.db.create_user(user_name, user_id)
            except Exception as e:
                print(f"Error saving session to database: {e}")
        
        self.log_activity("Created user session", {
            'user_id': user_id,
            'user_name': user_name,
            'session_id': session_data['session_id']
        })
        
        return {
            'success': True,
            'session_data': session_data,
            'url_parameters': session_data['url_parameters'],
            'message': 'Session created successfully'
        }
    
    def _validate_session(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an existing session.
        
        Args:
            request: Request with session parameters
            
        Returns:
            Response with validation results
        """
        user_id = request.get('user_id')
        session_id = request.get('session_id')
        last_activity = request.get('last_activity')
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        # Check if session exists
        if user_id not in self.active_sessions:
            return {
                'success': False,
                'error': 'Session not found',
                'is_valid': False
            }
        
        session_data = self.active_sessions[user_id]
        
        # Check session timeout
        last_activity_time = datetime.fromisoformat(session_data['last_activity'])
        if datetime.now() - last_activity_time > self.session_timeout:
            # Session expired
            del self.active_sessions[user_id]
            return {
                'success': False,
                'error': 'Session expired',
                'is_valid': False
            }
        
        # Validate session ID if provided
        if session_id and session_id != session_data.get('session_id'):
            return {
                'success': False,
                'error': 'Invalid session ID',
                'is_valid': False
            }
        
        # Update last activity
        session_data['last_activity'] = datetime.now().isoformat()
        
        return {
            'success': True,
            'is_valid': True,
            'session_data': session_data,
            'url_parameters': session_data['url_parameters']
        }
    
    def _update_session(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update session data.
        
        Args:
            request: Request with session updates
            
        Returns:
            Response with updated session
        """
        user_id = request.get('user_id')
        updates = request.get('updates', {})
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        if user_id not in self.active_sessions:
            return {
                'success': False,
                'error': 'Session not found'
            }
        
        session_data = self.active_sessions[user_id]
        
        # Apply updates
        for key, value in updates.items():
            if key in session_data:
                session_data[key] = value
        
        # Update last activity
        session_data['last_activity'] = datetime.now().isoformat()
        
        # Update URL parameters
        session_data['url_parameters'] = {
            'user_id': session_data['user_id'],
            'user_name': session_data['user_name'],
            'authenticated': 'true',
            'last_activity': session_data['last_activity']
        }
        
        # Save to database if available
        if self.db and 'language_preferences' in updates:
            try:
                self.db.update_user_profile(user_id, {'language_preferences': updates['language_preferences']})
            except Exception as e:
                print(f"Error updating session in database: {e}")
        
        self.log_activity("Updated session", {
            'user_id': user_id,
            'updated_fields': list(updates.keys())
        })
        
        return {
            'success': True,
            'session_data': session_data,
            'url_parameters': session_data['url_parameters'],
            'message': 'Session updated successfully'
        }
    
    def _get_session_info(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get session information.
        
        Args:
            request: Request with user_id
            
        Returns:
            Response with session information
        """
        user_id = request.get('user_id')
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        if user_id not in self.active_sessions:
            return {
                'success': False,
                'error': 'Session not found'
            }
        
        session_data = self.active_sessions[user_id]
        
        # Calculate session duration
        created_at = datetime.fromisoformat(session_data['created_at'])
        session_duration = datetime.now() - created_at
        
        # Calculate time since last activity
        last_activity = datetime.fromisoformat(session_data['last_activity'])
        time_since_activity = datetime.now() - last_activity
        
        session_info = {
            'user_id': session_data['user_id'],
            'user_name': session_data['user_name'],
            'session_id': session_data['session_id'],
            'created_at': session_data['created_at'],
            'last_activity': session_data['last_activity'],
            'session_duration_hours': session_duration.total_seconds() / 3600,
            'time_since_activity_minutes': time_since_activity.total_seconds() / 60,
            'is_active': session_data['is_active'],
            'language_preferences': session_data.get('language_preferences', {}),
            'url_parameters': session_data['url_parameters']
        }
        
        return {
            'success': True,
            'session_info': session_info
        }
    
    def _logout_session(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Logout and clear session data.
        
        Args:
            request: Request with user_id
            
        Returns:
            Response with logout status
        """
        user_id = request.get('user_id')
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        if user_id not in self.active_sessions:
            return {
                'success': False,
                'error': 'Session not found'
            }
        
        # Get session data before clearing
        session_data = self.active_sessions[user_id]
        
        # Clear session
        del self.active_sessions[user_id]
        
        # Clear from database if available
        if self.db:
            try:
                # This would typically clear session data from database
                pass
            except Exception as e:
                print(f"Error clearing session from database: {e}")
        
        self.log_activity("User logged out", {
            'user_id': user_id,
            'user_name': session_data.get('user_name'),
            'session_duration': datetime.now() - datetime.fromisoformat(session_data['created_at'])
        })
        
        return {
            'success': True,
            'message': 'Logged out successfully',
            'cleared_session': {
                'user_id': user_id,
                'user_name': session_data.get('user_name'),
                'session_duration_hours': (datetime.now() - datetime.fromisoformat(session_data['created_at'])).total_seconds() / 3600
            }
        }
    
    def _cleanup_expired_sessions(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean up expired sessions.
        
        Args:
            request: Request parameters
            
        Returns:
            Response with cleanup results
        """
        current_time = datetime.now()
        expired_sessions = []
        
        # Find expired sessions
        for user_id, session_data in list(self.active_sessions.items()):
            last_activity = datetime.fromisoformat(session_data['last_activity'])
            if current_time - last_activity > self.session_timeout:
                expired_sessions.append({
                    'user_id': user_id,
                    'user_name': session_data.get('user_name'),
                    'last_activity': session_data['last_activity'],
                    'session_duration_hours': (current_time - datetime.fromisoformat(session_data['created_at'])).total_seconds() / 3600
                })
                del self.active_sessions[user_id]
        
        # Clean up from database if available
        if self.db and expired_sessions:
            try:
                # This would typically clean up expired sessions from database
                pass
            except Exception as e:
                print(f"Error cleaning up expired sessions from database: {e}")
        
        self.log_activity("Cleaned up expired sessions", {
            'expired_count': len(expired_sessions),
            'remaining_sessions': len(self.active_sessions)
        })
        
        return {
            'success': True,
            'expired_sessions': expired_sessions,
            'total_expired': len(expired_sessions),
            'remaining_sessions': len(self.active_sessions)
        }
    
    def _get_active_sessions(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get list of active sessions.
        
        Args:
            request: Request parameters
            
        Returns:
            Response with active sessions
        """
        active_sessions = []
        
        for user_id, session_data in self.active_sessions.items():
            last_activity = datetime.fromisoformat(session_data['last_activity'])
            session_duration = datetime.now() - datetime.fromisoformat(session_data['created_at'])
            
            active_sessions.append({
                'user_id': user_id,
                'user_name': session_data.get('user_name'),
                'session_id': session_data.get('session_id'),
                'created_at': session_data['created_at'],
                'last_activity': session_data['last_activity'],
                'session_duration_hours': session_duration.total_seconds() / 3600,
                'time_since_activity_minutes': (datetime.now() - last_activity).total_seconds() / 60,
                'is_active': session_data['is_active']
            })
        
        return {
            'success': True,
            'active_sessions': active_sessions,
            'total_active': len(active_sessions)
        }
    
    def restore_session_from_url_params(self, url_params: Dict[str, str]) -> Dict[str, Any]:
        """
        Restore session from URL parameters.
        
        Args:
            url_params: URL parameters from Streamlit
            
        Returns:
            Response with restored session data
        """
        user_id = url_params.get('user_id')
        user_name = url_params.get('user_name')
        authenticated = url_params.get('authenticated', 'false')
        last_activity = url_params.get('last_activity')
        
        if not user_id or not user_name or authenticated != 'true':
            return {
                'success': False,
                'error': 'Invalid URL parameters for session restoration'
            }
        
        # Check if session exists
        if user_id in self.active_sessions:
            session_data = self.active_sessions[user_id]
            
            # Validate session
            if last_activity:
                try:
                    last_activity_time = datetime.fromisoformat(last_activity)
                    if datetime.now() - last_activity_time > self.session_timeout:
                        # Session expired, remove it
                        del self.active_sessions[user_id]
                        return {
                            'success': False,
                            'error': 'Session expired'
                        }
                except ValueError:
                    pass
            
            # Update last activity
            session_data['last_activity'] = datetime.now().isoformat()
            
            return {
                'success': True,
                'session_data': session_data,
                'url_parameters': session_data['url_parameters'],
                'restored': True
            }
        else:
            # Session not found in memory, try to restore from database
            if self.db:
                try:
                    user_profile = self.db.get_user_profile(user_id)
                    if user_profile:
                        # Recreate session
                        session_data = {
                            'user_id': user_id,
                            'user_name': user_name,
                            'created_at': last_activity or datetime.now().isoformat(),
                            'last_activity': datetime.now().isoformat(),
                            'language_preferences': user_profile.get('language_preferences', {}),
                            'is_active': True,
                            'session_id': str(uuid.uuid4()),
                            'url_parameters': {
                                'user_id': user_id,
                                'user_name': user_name,
                                'authenticated': 'true',
                                'last_activity': datetime.now().isoformat()
                            }
                        }
                        
                        self.active_sessions[user_id] = session_data
                        
                        return {
                            'success': True,
                            'session_data': session_data,
                            'url_parameters': session_data['url_parameters'],
                            'restored': True
                        }
                except Exception as e:
                    print(f"Error restoring session from database: {e}")
            
            return {
                'success': False,
                'error': 'Session not found'
            }
    
    def get_session_url_parameters(self, user_id: str) -> Dict[str, str]:
        """
        Get URL parameters for session persistence.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of URL parameters
        """
        if user_id not in self.active_sessions:
            return {}
        
        session_data = self.active_sessions[user_id]
        return session_data.get('url_parameters', {})
    
    def is_session_active(self, user_id: str) -> bool:
        """
        Check if a session is active.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if session is active, False otherwise
        """
        if user_id not in self.active_sessions:
            return False
        
        session_data = self.active_sessions[user_id]
        last_activity = datetime.fromisoformat(session_data['last_activity'])
        
        return datetime.now() - last_activity <= self.session_timeout 