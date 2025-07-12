"""
Multi-Agent Chatbot System using LangGraph
=========================================

This module provides a comprehensive multi-agent chatbot system using LangGraph,
coordinating specialized agents for different functionalities while maintaining
the same interface as the original single-agent system.
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime

from agents import (
    ConversationAgent, TagAnalysisAgent, UserProfileAgent,
    GroupChatAgent, SessionAgent, LanguageAgent
)
from agents.base_agent import LangGraphAgentCoordinator
from db import DB as DatabaseInterface


class MultiAgentChatbot:
    """
    Multi-agent chatbot system using LangGraph for coordination.
    
    This class coordinates multiple specialized LangGraph-based agents to provide
    comprehensive chatbot functionality while maintaining the same interface as
    the original single-agent system.
    """
    
    def __init__(self, db_interface=None):
        """
        Initialize the multi-agent chatbot system.
        
        Args:
            db_interface: Database interface for data persistence
        """
        self.db = db_interface or DatabaseInterface()
        self.coordinator = LangGraphAgentCoordinator()
        
        # Initialize all agents
        self._initialize_agents()
        
        # System state
        self.system_status = {
            'initialized': True,
            'framework': 'LangGraph',
            'total_agents': 6,
            'initialized_at': datetime.now().isoformat()
        }
        
        print("Multi-Agent Chatbot System initialized with LangGraph framework")
        print(f"Total agents: {self.system_status['total_agents']}")
        print(f"Framework: {self.system_status['framework']}")
    
    def _initialize_agents(self):
        """Initialize all LangGraph-based agents."""
        # Create agents
        conversation_agent = ConversationAgent(self.db)
        tag_analysis_agent = TagAnalysisAgent(self.db)
        user_profile_agent = UserProfileAgent(self.db)
        group_chat_agent = GroupChatAgent(self.db)
        session_agent = SessionAgent(self.db)
        language_agent = LanguageAgent(self.db)
        
        # Register agents with coordinator
        self.coordinator.register_agent(conversation_agent)
        self.coordinator.register_agent(tag_analysis_agent)
        self.coordinator.register_agent(user_profile_agent)
        self.coordinator.register_agent(group_chat_agent)
        self.coordinator.register_agent(session_agent)
        self.coordinator.register_agent(language_agent)
        
        print("All LangGraph agents registered and ready")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            Dictionary with system status information
        """
        coordinator_status = self.coordinator.get_system_status()
        
        return {
            **self.system_status,
            'coordinator_status': coordinator_status,
            'framework_version': 'LangGraph v0.0.20',
            'architecture': 'Multi-Agent with LangGraph Coordination'
        }
    
    def process_conversation(self, user_id: str, user_name: str, message: str, 
                           language_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a conversation message using the conversation agent.
        
        Args:
            user_id: User identifier
            user_name: User's name
            message: User's message
            language_preferences: User's language preferences
            
        Returns:
            Response with bot message and metadata
        """
        request = {
            'type': 'send_message',
            'user_id': user_id,
            'user_name': user_name,
            'message': message,
            'language_preferences': language_preferences or {}
        }
        
        return self.coordinator.route_request(request)
    
    def analyze_conversation_for_tags(self, user_id: str, conversation_history: List[Dict[str, str]],
                                    language_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze conversation to infer user interests and generate tags.
        
        Args:
            user_id: User identifier
            conversation_history: List of conversation messages
            language_preferences: User's language preferences
            
        Returns:
            Response with inferred tags
        """
        request = {
            'type': 'analyze_conversation',
            'user_id': user_id,
            'conversation_history': conversation_history,
            'language_preferences': language_preferences or {}
        }
        
        return self.coordinator.route_request(request)
    
    def get_tag_suggestions(self, user_id: str, existing_tags: List[str],
                           language_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get tag suggestions in multiple categories.
        
        Args:
            user_id: User identifier
            existing_tags: User's current tags
            language_preferences: User's language preferences
            
        Returns:
            Response with categorized tag suggestions
        """
        request = {
            'type': 'get_tag_suggestions',
            'user_id': user_id,
            'existing_tags': existing_tags,
            'language_preferences': language_preferences or {}
        }
        
        return self.coordinator.route_request(request)
    
    def create_user_profile(self, user_id: str, user_name: str, tags: List[str],
                           language_preferences: Dict[str, Any] = None,
                           conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Create a comprehensive user profile.
        
        Args:
            user_id: User identifier
            user_name: User's name
            tags: User's tags
            language_preferences: User's language preferences
            conversation_history: User's conversation history
            
        Returns:
            Response with created profile
        """
        request = {
            'type': 'create_profile',
            'user_id': user_id,
            'user_name': user_name,
            'tags': tags,
            'language_preferences': language_preferences or {},
            'conversation_history': conversation_history or []
        }
        
        return self.coordinator.route_request(request)
    
    def find_similar_users(self, user_id: str, min_similarity: float = 0.3,
                          max_results: int = 10) -> Dict[str, Any]:
        """
        Find users with similar interests and profiles.
        
        Args:
            user_id: User identifier
            min_similarity: Minimum similarity score
            max_results: Maximum number of results
            
        Returns:
            Response with similar users
        """
        request = {
            'type': 'find_similar_users',
            'user_id': user_id,
            'min_similarity': min_similarity,
            'max_results': max_results
        }
        
        return self.coordinator.route_request(request)
    
    def create_group_chat(self, topic_name: str, user_id: str, user_name: str,
                         language_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a new group chat.
        
        Args:
            topic_name: Group topic name
            user_id: Creator's user ID
            user_name: Creator's name
            language_preferences: Language preferences
            
        Returns:
            Response with created group data
        """
        request = {
            'type': 'create_group',
            'topic_name': topic_name,
            'user_id': user_id,
            'user_name': user_name,
            'language_preferences': language_preferences or {}
        }
        
        return self.coordinator.route_request(request)
    
    def send_group_message(self, group_id: str, user_id: str, user_name: str,
                          message: str, language_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send a message to a group chat.
        
        Args:
            group_id: Group identifier
            user_id: User identifier
            user_name: User's name
            message: Message content
            language_preferences: Language preferences
            
        Returns:
            Response with message status and AI response
        """
        request = {
            'type': 'send_group_message',
            'group_id': group_id,
            'user_id': user_id,
            'user_name': user_name,
            'message': message,
            'language_preferences': language_preferences or {}
        }
        
        return self.coordinator.route_request(request)
    
    def get_group_messages(self, group_id: str, limit: int = 50) -> Dict[str, Any]:
        """
        Get messages from a group chat.
        
        Args:
            group_id: Group identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            Response with group messages
        """
        request = {
            'type': 'get_group_messages',
            'group_id': group_id,
            'limit': limit
        }
        
        return self.coordinator.route_request(request)
    
    def create_user_session(self, user_name: str, language_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a new user session.
        
        Args:
            user_name: User's name
            language_preferences: User's language preferences
            
        Returns:
            Response with session data
        """
        request = {
            'type': 'create_session',
            'user_name': user_name,
            'language_preferences': language_preferences or {}
        }
        
        return self.coordinator.route_request(request)
    
    def validate_session(self, user_id: str, session_id: str = None) -> Dict[str, Any]:
        """
        Validate an existing session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier (optional)
            
        Returns:
            Response with validation results
        """
        request = {
            'type': 'validate_session',
            'user_id': user_id,
            'session_id': session_id
        }
        
        return self.coordinator.route_request(request)
    
    def get_supported_languages(self) -> Dict[str, Any]:
        """
        Get list of supported languages.
        
        Returns:
            Response with supported languages
        """
        request = {
            'type': 'get_supported_languages'
        }
        
        return self.coordinator.route_request(request)
    
    def generate_personalized_greeting(self, user_name: str, language_preferences: Dict[str, Any],
                                     greeting_type: str = 'formal') -> Dict[str, Any]:
        """
        Generate a personalized greeting.
        
        Args:
            user_name: User's name
            language_preferences: User's language preferences
            greeting_type: Type of greeting (formal, casual, cultural, professional)
            
        Returns:
            Response with personalized greeting
        """
        request = {
            'type': 'generate_greeting',
            'user_name': user_name,
            'language_preferences': language_preferences,
            'greeting_type': greeting_type
        }
        
        return self.coordinator.route_request(request)
    
    def get_agent_capabilities(self, agent_name: str = None) -> Dict[str, Any]:
        """
        Get capabilities of agents.
        
        Args:
            agent_name: Specific agent name (optional)
            
        Returns:
            Response with agent capabilities
        """
        if agent_name:
            agent = self.coordinator.get_agent(agent_name)
            if agent:
                return {
                    'success': True,
                    'agent_name': agent_name,
                    'capabilities': agent.get_capabilities(),
                    'status': agent.get_status()
                }
            else:
                return {
                    'success': False,
                    'error': f'Agent {agent_name} not found'
                }
        else:
            # Get capabilities of all agents
            all_capabilities = {}
            for name, agent in self.coordinator.agents.items():
                all_capabilities[name] = agent.get_capabilities()
            
            return {
                'success': True,
                'all_capabilities': all_capabilities,
                'total_agents': len(all_capabilities)
            }
    
    def test_agent_functionality(self, agent_name: str, test_type: str = 'basic') -> Dict[str, Any]:
        """
        Test agent functionality.
        
        Args:
            agent_name: Name of the agent to test
            test_type: Type of test to perform
            
        Returns:
            Response with test results
        """
        agent = self.coordinator.get_agent(agent_name)
        if not agent:
            return {
                'success': False,
                'error': f'Agent {agent_name} not found'
            }
        
        # Create test request based on agent type
        test_requests = {
            'ConversationAgent': {
                'type': 'send_message',
                'user_id': 'test_user',
                'user_name': 'Test User',
                'message': 'Hello, this is a test message.',
                'language_preferences': {'native_language': 'english'}
            },
            'TagAnalysisAgent': {
                'type': 'get_tag_suggestions',
                'user_id': 'test_user',
                'existing_tags': ['technology', 'music'],
                'language_preferences': {'native_language': 'english'}
            },
            'UserProfileAgent': {
                'type': 'create_profile',
                'user_id': 'test_user',
                'user_name': 'Test User',
                'tags': ['technology', 'music'],
                'language_preferences': {'native_language': 'english'},
                'conversation_history': []
            },
            'GroupChatAgent': {
                'type': 'create_group',
                'topic_name': 'Test Group',
                'user_id': 'test_user',
                'user_name': 'Test User',
                'language_preferences': {'native_language': 'english'}
            },
            'SessionAgent': {
                'type': 'create_session',
                'user_name': 'Test User',
                'language_preferences': {'native_language': 'english'}
            },
            'LanguageAgent': {
                'type': 'get_supported_languages'
            }
        }
        
        test_request = test_requests.get(agent_name, {'type': 'test'})
        
        try:
            result = agent.process_request(test_request)
            return {
                'success': True,
                'agent_name': agent_name,
                'test_type': test_type,
                'test_result': result,
                'agent_status': agent.get_status()
            }
        except Exception as e:
            return {
                'success': False,
                'agent_name': agent_name,
                'error': f'Test failed: {str(e)}',
                'agent_status': agent.get_status()
            }
    
    def cleanup_system(self):
        """Cleanup system resources."""
        self.coordinator.cleanup_all()
        print("Multi-Agent Chatbot System cleaned up")
    
    def get_agent_status(self, agent_name: str = None) -> Dict[str, Any]:
        """
        Get status of specific agent or all agents.
        
        Args:
            agent_name: Specific agent name (optional)
            
        Returns:
            Response with agent status
        """
        if agent_name:
            agent = self.coordinator.get_agent(agent_name)
            if agent:
                return {
                    'success': True,
                    'agent_name': agent_name,
                    'status': agent.get_status()
                }
            else:
                return {
                    'success': False,
                    'error': f'Agent {agent_name} not found'
                }
        else:
            return self.coordinator.get_system_status()
    
    def get_framework_info(self) -> Dict[str, Any]:
        """
        Get framework information.
        
        Returns:
            Dictionary with framework details
        """
        return {
            'version': 'LangGraph v0.0.20',
            'architecture': 'Multi-Agent with LangGraph Coordination',
            'features': [
                'Conversation Management',
                'Tag Analysis',
                'User Profiling',
                'Group Chat',
                'Session Management',
                'Language Support'
            ]
        }
    
    # Interface methods to match original chatbot
    def process_user_message(self, message: str) -> str:
        """
        Process a user message and return bot response.
        
        Args:
            message: User's message
            
        Returns:
            Bot's response
        """
        if not hasattr(self, 'user_id') or not hasattr(self, 'user_name'):
            return "Error: User not initialized"
        
        result = self.process_conversation(
            user_id=self.user_id,
            user_name=self.user_name,
            message=message,
            language_preferences=self.get_language_preferences()
        )
        
        if result.get('success'):
            return result.get('response', 'No response generated')
        else:
            return f"Error: {result.get('error', 'Unknown error')}"
    
    def get_conversation(self) -> List[tuple]:
        """
        Get conversation history.
        
        Returns:
            List of (role, message) tuples
        """
        if not hasattr(self, 'user_id'):
            return []
        
        # Get conversation from database
        if self.db and hasattr(self.db, 'get_user_conversation'):
            return self.db.get_user_conversation(self.user_id)
        return []
    
    def get_conversation_turns(self) -> int:
        """
        Get number of conversation turns.
        
        Returns:
            Number of turns
        """
        conversation = self.get_conversation()
        return len(conversation)
    
    def get_last_question(self) -> Optional[str]:
        """
        Get the last follow-up question.
        
        Returns:
            Last question or None
        """
        if not hasattr(self, 'user_id'):
            return None
        
        # Get from database
        if self.db and hasattr(self.db, 'get_last_question'):
            return self.db.get_last_question(self.user_id)
        return None
    
    def get_accepted_questions(self) -> List[str]:
        """
        Get list of accepted questions.
        
        Returns:
            List of accepted questions
        """
        if not hasattr(self, 'user_id'):
            return []
        
        if self.db and hasattr(self.db, 'get_accepted_questions'):
            return self.db.get_accepted_questions(self.user_id)
        return []
    
    def get_rejected_questions(self) -> List[str]:
        """
        Get list of rejected questions.
        
        Returns:
            List of rejected questions
        """
        if not hasattr(self, 'user_id'):
            return []
        
        if self.db and hasattr(self.db, 'get_rejected_questions'):
            return self.db.get_rejected_questions(self.user_id)
        return []
    
    def get_question_stats(self) -> Dict[str, int]:
        """
        Get question statistics.
        
        Returns:
            Dictionary with question stats
        """
        accepted = len(self.get_accepted_questions())
        rejected = len(self.get_rejected_questions())
        
        return {
            'accepted_count': accepted,
            'rejected_count': rejected,
            'total_questions': accepted + rejected
        }
    
    def get_user_tags(self) -> List[str]:
        """
        Get user's tags.
        
        Returns:
            List of user tags
        """
        if not hasattr(self, 'user_id'):
            return []
        
        if self.db and hasattr(self.db, 'get_user_tags'):
            return self.db.get_user_tags(self.user_id)
        return []
    
    def add_manual_tag(self, tag: str) -> bool:
        """
        Add a manual tag for the user.
        
        Args:
            tag: Tag to add
            
        Returns:
            Success status
        """
        if not hasattr(self, 'user_id'):
            return False
        
        if self.db and hasattr(self.db, 'add_user_tag'):
            try:
                self.db.add_user_tag(self.user_id, tag, 'manual')
                return True
            except Exception as e:
                print(f"Error adding tag: {e}")
                return False
        return False
    
    def remove_tag(self, tag: str) -> bool:
        """
        Remove a tag for the user.
        
        Args:
            tag: Tag to remove
            
        Returns:
            Success status
        """
        if not hasattr(self, 'user_id'):
            return False
        
        if self.db and hasattr(self.db, 'remove_user_tag'):
            try:
                self.db.remove_user_tag(self.user_id, tag)
                return True
            except Exception as e:
                print(f"Error removing tag: {e}")
                return False
        return False
    
    def update_language_preferences(self, native_language: str = None, 
                                  preferred_languages: List[str] = None,
                                  language_comfort_level: str = 'english') -> bool:
        """
        Update user's language preferences.
        
        Args:
            native_language: Native language
            preferred_languages: List of preferred languages
            language_comfort_level: Comfort level
            
        Returns:
            Success status
        """
        if not hasattr(self, 'user_id'):
            return False
        
        if self.db and hasattr(self.db, 'update_language_preferences'):
            try:
                self.db.update_language_preferences(
                    self.user_id,
                    native_language,
                    preferred_languages or [],
                    language_comfort_level
                )
                return True
            except Exception as e:
                print(f"Error updating language preferences: {e}")
                return False
        return False
    
    def get_language_preferences(self) -> Dict[str, Any]:
        """
        Get user's language preferences.
        
        Returns:
            Dictionary with language preferences
        """
        if not hasattr(self, 'user_id'):
            return {
                'native_language': None,
                'preferred_languages': [],
                'language_comfort_level': 'english'
            }
        
        if self.db and hasattr(self.db, 'get_language_preferences'):
            return self.db.get_language_preferences(self.user_id)
        
        return {
            'native_language': None,
            'preferred_languages': [],
            'language_comfort_level': 'english'
        }
    
    def process_request(self, request_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a request using the multi-agent system.
        
        Args:
            request_type: Type of request
            params: Request parameters
            
        Returns:
            Response dictionary
        """
        if not hasattr(self, 'user_id'):
            return {'success': False, 'error': 'User not initialized'}
        
        # Add user context to request
        request = {
            'type': request_type,
            'user_id': self.user_id,
            'user_name': getattr(self, 'user_name', 'Unknown'),
            **(params or {})
        }
        
        return self.coordinator.route_request(request)
    
    def check_status(self):
        """
        Check the status of the chatbot system.
        
        Returns:
            Dictionary with status information
        """
        try:
            # Check system status
            system_status = self.get_system_status()
            
            # Check agent status
            agent_status = self.get_agent_status()
            
            # Check database connection
            db_status = self.db.check_connection() if hasattr(self.db, 'check_connection') else {
                'success': True,
                'message': 'Database connection not available'
            }
            
            return {
                'success': True,
                'message': 'System is operational',
                'system_status': system_status,
                'agent_status': agent_status,
                'db_status': db_status,
                'total_agents': system_status.get('total_agents', 0),
                'framework': system_status.get('framework', 'Unknown')
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Status check failed: {str(e)}'
            } 