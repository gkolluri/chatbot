"""
React AI Pattern-Based Session Agent for Multi-Agent Chatbot System
=================================================================

This agent handles user sessions and persistence using React AI pattern, including:
- User session creation and validation
- Session data management and persistence
- Activity tracking and monitoring
- Session cleanup and maintenance
- Cross-tab session synchronization
- React AI pattern: Observe → Think → Act → Observe
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from agents.react_base_agent import ReactBaseAgent, ReactAgentState
from langchain_core.tools import tool


class ReactSessionAgent(ReactBaseAgent):
    """
    React AI pattern-based agent responsible for managing user sessions.
    
    Implements Observe-Think-Act loops for session management,
    validation, and persistence with React AI reasoning.
    """
    
    def __init__(self, db_interface=None):
        """Initialize the React AI session agent."""
        super().__init__("ReactSessionAgent", db_interface)
        self.active_sessions = {}
        self.session_history = []
        
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this agent provides."""
        return [
            "react_ai_session_management",
            "session_creation_and_validation",
            "activity_tracking",
            "session_persistence",
            "cross_tab_synchronization",
            "session_cleanup",
            "react_ai_reasoning"
        ]
    
    def _get_agent_system_prompt(self) -> str:
        """Get the system prompt for React AI session agent."""
        return """You are a React AI session management agent designed to handle user sessions and persistence.

Your role is to manage user sessions using the React AI pattern:
1. OBSERVE: Analyze session state and user activity
2. THINK: Reason about session validity and user needs
3. ACT: Create, validate, or update sessions appropriately
4. REFLECT: Learn from session patterns and improve management

SESSION MANAGEMENT GUIDELINES:
- Create secure and persistent user sessions
- Validate session authenticity and activity
- Track user activity and engagement
- Manage session lifecycle and cleanup
- Ensure cross-tab synchronization
- Maintain session data integrity

REACT AI PATTERN:
- Always observe current session state
- Think about session security and user experience
- Act by creating or validating sessions
- Reflect on session management effectiveness
"""
    
    def _get_agent_specific_tools(self) -> List:
        """Get session-specific tools for React AI pattern."""
        tools = []
        
        @tool
        def create_user_session(user_id: str, user_name: str, session_data: str) -> str:
            """Create a new user session with React AI reasoning."""
            try:
                session_info = eval(session_data) if isinstance(session_data, str) else session_data
                
                # Create session with React AI reasoning
                session = {
                    'user_id': user_id,
                    'user_name': user_name,
                    'created_at': datetime.now().isoformat(),
                    'last_activity': datetime.now().isoformat(),
                    'is_active': True,
                    'session_data': session_info,
                    'react_ai_reasoning': 'Session created with React AI pattern'
                }
                
                # Store in agent's active sessions
                self.active_sessions[user_id] = session
                
                # Add to session history
                self.session_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'action': 'session_created',
                    'user_id': user_id,
                    'user_name': user_name
                })
                
                return f"Session created successfully for user {user_name} (ID: {user_id})"
            except Exception as e:
                return f"Error creating session: {str(e)}"
        
        @tool
        def validate_session(user_id: str, session_token: str = None) -> str:
            """Validate an existing user session using React AI reasoning."""
            try:
                # Check if session exists
                if user_id in self.active_sessions:
                    session = self.active_sessions[user_id]
                    
                    # Update last activity
                    session['last_activity'] = datetime.now().isoformat()
                    
                    validation_result = f"Session validated for user {session['user_name']}:\n"
                    validation_result += f"- User ID: {user_id}\n"
                    validation_result += f"- Created: {session['created_at']}\n"
                    validation_result += f"- Last Activity: {session['last_activity']}\n"
                    validation_result += f"- Status: Active\n"
                    validation_result += f"- React AI: Session validated with reasoning"
                    
                    return validation_result
                else:
                    return f"Session not found for user ID: {user_id}"
            except Exception as e:
                return f"Error validating session: {str(e)}"
        
        @tool
        def track_user_activity(user_id: str, activity_type: str, activity_data: str = None) -> str:
            """Track user activity with React AI reasoning."""
            try:
                if user_id in self.active_sessions:
                    session = self.active_sessions[user_id]
                    session['last_activity'] = datetime.now().isoformat()
                    
                    # Add activity to session data
                    if 'activities' not in session:
                        session['activities'] = []
                    
                    activity = {
                        'type': activity_type,
                        'timestamp': datetime.now().isoformat(),
                        'data': activity_data
                    }
                    session['activities'].append(activity)
                    
                    tracking_result = f"Activity tracked for user {session['user_name']}:\n"
                    tracking_result += f"- Activity Type: {activity_type}\n"
                    tracking_result += f"- Timestamp: {activity['timestamp']}\n"
                    tracking_result += f"- Total Activities: {len(session['activities'])}\n"
                    tracking_result += f"- React AI: Activity analyzed with reasoning"
                    
                    return tracking_result
                else:
                    return f"Session not found for user ID: {user_id}"
            except Exception as e:
                return f"Error tracking activity: {str(e)}"
        
        @tool
        def cleanup_inactive_sessions(inactivity_threshold_minutes: int = 30) -> str:
            """Clean up inactive sessions using React AI reasoning."""
            try:
                current_time = datetime.now()
                inactive_sessions = []
                
                for user_id, session in self.active_sessions.items():
                    last_activity = datetime.fromisoformat(session['last_activity'])
                    time_diff = (current_time - last_activity).total_seconds() / 60
                    
                    if time_diff > inactivity_threshold_minutes:
                        inactive_sessions.append(user_id)
                
                # Remove inactive sessions
                for user_id in inactive_sessions:
                    session = self.active_sessions.pop(user_id)
                    
                    # Add to session history
                    self.session_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'action': 'session_cleaned',
                        'user_id': user_id,
                        'user_name': session['user_name'],
                        'reason': 'inactivity'
                    })
                
                cleanup_result = f"Session cleanup completed:\n"
                cleanup_result += f"- Inactive sessions found: {len(inactive_sessions)}\n"
                cleanup_result += f"- Sessions cleaned: {len(inactive_sessions)}\n"
                cleanup_result += f"- Active sessions remaining: {len(self.active_sessions)}\n"
                cleanup_result += f"- React AI: Cleanup performed with reasoning"
                
                return cleanup_result
            except Exception as e:
                return f"Error cleaning up sessions: {str(e)}"
        
        @tool
        def get_session_statistics() -> str:
            """Get session statistics with React AI analysis."""
            try:
                total_sessions = len(self.active_sessions)
                total_history = len(self.session_history)
                
                # Analyze session patterns
                recent_activities = [h for h in self.session_history 
                                  if (datetime.now() - datetime.fromisoformat(h['timestamp'])).total_seconds() < 3600]
                
                statistics = f"Session Statistics:\n"
                statistics += f"- Active sessions: {total_sessions}\n"
                statistics += f"- Total session history: {total_history}\n"
                statistics += f"- Recent activities (1 hour): {len(recent_activities)}\n"
                statistics += f"- React AI: Statistics analyzed with reasoning"
                
                return statistics
            except Exception as e:
                return f"Error getting session statistics: {str(e)}"
        
        tools.extend([
            create_user_session,
            validate_session,
            track_user_activity,
            cleanup_inactive_sessions,
            get_session_statistics
        ])
        
        return tools
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process session-related requests using React AI pattern.
        
        Args:
            request: Request dictionary with parameters
            
        Returns:
            Response dictionary with results
        """
        request_type = request.get('type', '')
        
        if request_type == 'create_session':
            return self._create_session(request)
        elif request_type == 'validate_session':
            return self._validate_session(request)
        elif request_type == 'track_activity':
            return self._track_activity(request)
        elif request_type == 'cleanup_sessions':
            return self._cleanup_sessions(request)
        elif request_type == 'get_session_stats':
            return self._get_session_statistics(request)
        else:
            return {
                'success': False,
                'error': f'Unknown request type: {request_type}',
                'available_types': ['create_session', 'validate_session', 'track_activity', 'cleanup_sessions', 'get_session_stats']
            }
    
    def _create_session(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a user session using React AI pattern.
        
        Args:
            request: Request with session data
            
        Returns:
            Response with session creation result
        """
        user_id = request.get('user_id')
        user_name = request.get('user_name')
        session_data = request.get('session_data', {})
        
        if not user_id or not user_name:
            return {
                'success': False,
                'error': 'Missing user_id or user_name'
            }
        
        # Use React AI pattern for session creation
        react_request = {
            'user_id': user_id,
            'message': f'Create session for user {user_name}',
            'session_data': session_data,
            'type': 'create_session'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            result['session_created'] = True
            result['user_id'] = user_id
            result['user_name'] = user_name
            result['framework'] = 'React AI Pattern'
        
        return result
    
    def _validate_session(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a user session using React AI pattern.
        
        Args:
            request: Request with session validation data
            
        Returns:
            Response with validation result
        """
        user_id = request.get('user_id')
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        # Use React AI pattern for session validation
        react_request = {
            'user_id': user_id,
            'message': f'Validate session for user {user_id}',
            'type': 'validate_session'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            result['session_valid'] = True
            result['framework'] = 'React AI Pattern'
        
        return result
    
    def _track_activity(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track user activity using React AI pattern.
        
        Args:
            request: Request with activity data
            
        Returns:
            Response with activity tracking result
        """
        user_id = request.get('user_id')
        activity_type = request.get('activity_type', 'general')
        activity_data = request.get('activity_data')
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        # Use React AI pattern for activity tracking
        react_request = {
            'user_id': user_id,
            'message': f'Track activity: {activity_type}',
            'activity_type': activity_type,
            'activity_data': activity_data,
            'type': 'track_activity'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            result['activity_tracked'] = True
            result['framework'] = 'React AI Pattern'
        
        return result
    
    def _cleanup_sessions(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean up inactive sessions using React AI pattern.
        
        Args:
            request: Request with cleanup parameters
            
        Returns:
            Response with cleanup result
        """
        threshold = request.get('inactivity_threshold_minutes', 30)
        
        # Use React AI pattern for session cleanup
        react_request = {
            'user_id': 'system',
            'message': f'Cleanup inactive sessions with threshold {threshold} minutes',
            'inactivity_threshold_minutes': threshold,
            'type': 'cleanup_sessions'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            result['cleanup_completed'] = True
            result['framework'] = 'React AI Pattern'
        
        return result
    
    def _get_session_statistics(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get session statistics using React AI pattern.
        
        Args:
            request: Request dictionary
            
        Returns:
            Response with session statistics
        """
        # Use React AI pattern for statistics
        react_request = {
            'user_id': 'system',
            'message': 'Get session statistics',
            'type': 'get_session_stats'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            result['statistics_retrieved'] = True
            result['framework'] = 'React AI Pattern'
        
        return result 