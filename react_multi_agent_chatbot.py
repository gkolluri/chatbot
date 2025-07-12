"""
React AI Pattern-Based Multi-Agent Chatbot System
================================================

This module provides a multi-agent chatbot system using React AI pattern
with Observe-Think-Act loops, dynamic tool calling, and reasoning capabilities.
The system coordinates multiple specialized React AI agents for different
functionalities while maintaining the same interface as the original chatbot.
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from agents.react_base_agent import ReactAgentCoordinator, ReactBaseAgent
from agents.react_conversation_agent import ReactConversationAgent
from agents.react_tag_analysis_agent import ReactTagAnalysisAgent
from agents.react_session_agent import ReactSessionAgent
from agents.react_user_profile_agent import ReactUserProfileAgent
from agents.react_language_agent import ReactLanguageAgent
from agents.react_group_chat_agent import ReactGroupChatAgent
from db import DB as DatabaseInterface


class ReactMultiAgentChatbot:
    """
    Multi-agent chatbot system using React AI pattern.
    
    Coordinates multiple React AI agents for different functionalities:
    - Conversation handling with reasoning
    - Tag analysis with dynamic inference
    - User profiling with adaptive learning
    - Group chat with AI participation
    - Session management with persistence
    - Language preferences with cultural context
    """
    
    def __init__(self):
        """Initialize the React AI multi-agent chatbot system."""
        self.db = DatabaseInterface()
        self.coordinator = ReactAgentCoordinator()
        self.agents = {}
        
        # Initialize React AI agents
        self._initialize_agents()
        
        print("React AI Multi-Agent Chatbot System initialized")
        print(f"Total agents: {len(self.agents)}")
        print("Framework: React AI Pattern")
    
    def _initialize_agents(self):
        """Initialize all React AI agents."""
        # Create React AI agents
        self.agents['ConversationAgent'] = ReactConversationAgent(self.db)
        self.agents['TagAnalysisAgent'] = ReactTagAnalysisAgent(self.db)
        self.agents['SessionAgent'] = ReactSessionAgent(self.db)
        self.agents['UserProfileAgent'] = ReactUserProfileAgent(self.db)
        self.agents['LanguageAgent'] = ReactLanguageAgent(self.db)
        self.agents['GroupChatAgent'] = ReactGroupChatAgent(self.db)
        
        # Register agents with coordinator
        for agent in self.agents.values():
            self.coordinator.register_agent(agent)
        
        print("All React AI agents registered and ready")
    
    def create_user_session(self, user_name: str, native_language: str = None, 
                           preferred_languages: List[str] = None, 
                           language_comfort_level: str = 'english') -> Dict[str, Any]:
        """
        Create a new user session using React AI pattern.
        
        Args:
            user_name: User's name
            native_language: User's native language
            preferred_languages: List of preferred languages
            language_comfort_level: Language comfort level
            
        Returns:
            Session creation result
        """
        try:
            user_id = str(uuid.uuid4())
            
            # Create user in database
            if self.db:
                self.db.create_user(user_name, user_id)
                # Update language preferences
                self.db.update_language_preferences(
                    user_id, 
                    native_language=native_language,
                    preferred_languages=preferred_languages or [],
                    language_comfort_level=language_comfort_level
                )
            
            # Use React AI pattern for session creation
            react_request = {
                'type': 'create_session',
                'user_id': user_id,
                'user_name': user_name,
                'native_language': native_language,
                'preferred_languages': preferred_languages or [],
                'language_comfort_level': language_comfort_level
            }
            
            result = self.coordinator.route_request(react_request)
            
            if result.get('success'):
                result['user_id'] = user_id
                result['session_created'] = True
                result['framework'] = 'React AI Pattern'
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error creating user session: {str(e)}'
            }
    
    def send_message(self, user_id: str, message: str, 
                    language_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send a message using React AI pattern.
        
        Args:
            user_id: User identifier
            message: User message
            language_preferences: User's language preferences
            
        Returns:
            Response with bot message and metadata
        """
        try:
            # Use React AI pattern for message processing
            react_request = {
                'type': 'send_message',
                'user_id': user_id,
                'message': message,
                'language_preferences': language_preferences or {}
            }
            
            result = self.coordinator.route_request(react_request)
            
            if result.get('success'):
                # Add React AI specific metadata
                result['framework'] = 'React AI Pattern'
                result['reasoning_steps'] = len(result.get('reasoning_chain', []))
                result['processing_timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error sending message: {str(e)}'
            }
    
    def analyze_conversation_for_tags(self, user_id: str, 
                                    conversation_text: str = None,
                                    language_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze conversation for tag inference using React AI pattern.
        
        Args:
            user_id: User identifier
            conversation_text: Conversation text to analyze
            language_preferences: User's language preferences
            
        Returns:
            Analysis results with tag suggestions
        """
        try:
            # Get conversation text if not provided
            if not conversation_text and self.db:
                conversations = self.db.get_user_conversations(user_id)
                conversation_text = " ".join([msg[1] if isinstance(msg, tuple) else msg.get('content', '') 
                                           for msg in conversations[-10:]])  # Last 10 messages
            
            # Use React AI pattern for tag analysis
            react_request = {
                'type': 'analyze_conversation',
                'user_id': user_id,
                'conversation_text': conversation_text or '',
                'language_preferences': language_preferences or {}
            }
            
            result = self.coordinator.route_request(react_request)
            
            if result.get('success'):
                result['framework'] = 'React AI Pattern'
                result['analysis_method'] = 'React AI Observe-Think-Act'
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error analyzing conversation: {str(e)}'
            }
    
    def get_tag_suggestions(self, user_id: str, analysis_result: str = None,
                           existing_tags: List[str] = None) -> Dict[str, Any]:
        """
        Get tag suggestions using React AI pattern.
        
        Args:
            user_id: User identifier
            analysis_result: Previous analysis result
            existing_tags: User's existing tags
            
        Returns:
            Tag suggestions with reasoning
        """
        try:
            # Get existing tags if not provided
            if not existing_tags and self.db:
                existing_tags = [tag['tag'] for tag in self.db.get_user_tags(user_id)]
            
            # Use React AI pattern for tag suggestions
            react_request = {
                'type': 'get_tag_suggestions',
                'user_id': user_id,
                'analysis_result': analysis_result or '',
                'existing_tags': existing_tags or []
            }
            
            result = self.coordinator.route_request(react_request)
            
            if result.get('success'):
                result['framework'] = 'React AI Pattern'
                result['suggestion_method'] = 'React AI Reasoning'
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error getting tag suggestions: {str(e)}'
            }
    
    def create_user_profile(self, user_id: str, tags: List[str] = None,
                          language_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create user profile using React AI pattern.
        
        Args:
            user_id: User identifier
            tags: User's tags
            language_preferences: User's language preferences
            
        Returns:
            Profile creation result
        """
        try:
            # Use React AI pattern for profile creation
            react_request = {
                'type': 'create_profile',
                'user_id': user_id,
                'tags': tags or [],
                'language_preferences': language_preferences or {}
            }
            
            result = self.coordinator.route_request(react_request)
            
            if result.get('success'):
                # Save tags to database
                if tags and self.db:
                    for tag in tags:
                        self.db.add_user_tag(user_id, tag, 'manual')
                
                result['framework'] = 'React AI Pattern'
                result['profile_created'] = True
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error creating user profile: {str(e)}'
            }
    
    def find_similar_users(self, user_id: str, limit: int = 5) -> Dict[str, Any]:
        """
        Find similar users using React AI pattern.
        
        Args:
            user_id: User identifier
            limit: Maximum number of similar users to return
            
        Returns:
            Similar users with reasoning
        """
        try:
            # Get user tags
            user_tags = []
            if self.db:
                user_tags = [tag['tag'] for tag in self.db.get_user_tags(user_id)]
            
            # Use React AI pattern for user matching
            react_request = {
                'type': 'find_similar_users',
                'user_id': user_id,
                'user_tags': user_tags,
                'limit': limit
            }
            
            result = self.coordinator.route_request(react_request)
            
            if result.get('success'):
                result['framework'] = 'React AI Pattern'
                result['matching_method'] = 'React AI Reasoning'
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error finding similar users: {str(e)}'
            }
    
    def create_group_chat(self, topic_name: str, user_ids: List[str],
                         created_by: str) -> Dict[str, Any]:
        """
        Create group chat using React AI pattern.
        
        Args:
            topic_name: Group chat topic
            user_ids: List of user IDs
            created_by: Creator user ID
            
        Returns:
            Group chat creation result
        """
        try:
            group_id = str(uuid.uuid4())
            
            # Save to database
            if self.db:
                self.db.create_group_chat(group_id, topic_name, user_ids, created_by)
            
            # Use React AI pattern for group creation
            react_request = {
                'type': 'create_group_chat',
                'group_id': group_id,
                'topic_name': topic_name,
                'user_ids': user_ids,
                'created_by': created_by
            }
            
            result = self.coordinator.route_request(react_request)
            
            if result.get('success'):
                result['group_id'] = group_id
                result['framework'] = 'React AI Pattern'
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error creating group chat: {str(e)}'
            }
    
    def send_group_message(self, group_id: str, user_id: str, message: str) -> Dict[str, Any]:
        """
        Send group message using React AI pattern.
        
        Args:
            group_id: Group identifier
            user_id: User identifier
            message: Message content
            
        Returns:
            Message sending result
        """
        try:
            # Save message to database
            if self.db:
                self.db.add_group_message(group_id, user_id, message)
            
            # Use React AI pattern for group message
            react_request = {
                'type': 'send_group_message',
                'group_id': group_id,
                'user_id': user_id,
                'message': message
            }
            
            result = self.coordinator.route_request(react_request)
            
            if result.get('success'):
                result['framework'] = 'React AI Pattern'
                result['message_sent'] = True
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error sending group message: {str(e)}'
            }
    
    def get_group_messages(self, group_id: str) -> Dict[str, Any]:
        """
        Get group messages using React AI pattern.
        
        Args:
            group_id: Group identifier
            
        Returns:
            Group messages with context
        """
        try:
            # Get messages from database
            messages = []
            if self.db:
                messages = self.db.get_group_messages(group_id)
            
            # Use React AI pattern for message retrieval
            react_request = {
                'type': 'get_group_messages',
                'group_id': group_id,
                'messages': messages
            }
            
            result = self.coordinator.route_request(react_request)
            
            if result.get('success'):
                result['messages'] = messages
                result['framework'] = 'React AI Pattern'
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error getting group messages: {str(e)}'
            }
    
    def get_supported_languages(self) -> Dict[str, Any]:
        """
        Get supported languages using React AI pattern.
        
        Returns:
            Supported languages with cultural context
        """
        try:
            # Use React AI pattern for language information
            react_request = {
                'type': 'get_supported_languages',
                'include_cultural_context': True
            }
            
            result = self.coordinator.route_request(react_request)
            
            if result.get('success'):
                result['framework'] = 'React AI Pattern'
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error getting supported languages: {str(e)}'
            }
    
    def generate_greeting(self, user_name: str, 
                         language_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate greeting using React AI pattern.
        
        Args:
            user_name: User's name
            language_preferences: User's language preferences
            
        Returns:
            Greeting with cultural context
        """
        try:
            # Use React AI pattern for greeting generation
            react_request = {
                'type': 'generate_greeting',
                'user_name': user_name,
                'language_preferences': language_preferences or {}
            }
            
            result = self.coordinator.route_request(react_request)
            
            if result.get('success'):
                result['framework'] = 'React AI Pattern'
                result['greeting_method'] = 'React AI Cultural Context'
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error generating greeting: {str(e)}'
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status with React AI pattern information.
        
        Returns:
            System status with agent information
        """
        try:
            # Get coordinator status
            coordinator_status = self.coordinator.get_system_status()
            
            # Get agent details
            agent_details = {}
            for name, agent in self.agents.items():
                agent_details[name] = {
                    'status': agent.get_status(),
                    'capabilities': agent.get_capabilities(),
                    'reasoning_history_count': len(getattr(agent, 'reasoning_history', [])),
                    'tool_usage_count': len(getattr(agent, 'tool_usage_history', []))
                }
            
            return {
                'success': True,
                'framework': 'React AI Pattern',
                'total_agents': len(self.agents),
                'coordinator_status': coordinator_status,
                'agent_details': agent_details,
                'system_timestamp': datetime.now().isoformat(),
                'react_ai_features': [
                    'Observe-Think-Act loops',
                    'Dynamic tool calling',
                    'Reasoning and reflection',
                    'Cultural context integration',
                    'Adaptive learning'
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error getting system status: {str(e)}'
            }
    
    def check_status(self) -> Dict[str, Any]:
        """
        Check system status for health monitoring.
        
        Returns:
            Status check result
        """
        try:
            # Check database connection
            db_status = self.db.check_connection() if self.db else False
            
            # Check agent status
            agent_status = all(agent.is_active for agent in self.agents.values())
            
            # Check coordinator status
            coordinator_status = len(self.coordinator.agents) > 0
            
            return {
                'success': True,
                'database_connected': db_status,
                'agents_active': agent_status,
                'coordinator_ready': coordinator_status,
                'framework': 'React AI Pattern',
                'total_agents': len(self.agents),
                'active_agents': len([a for a in self.agents.values() if a.is_active])
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error checking status: {str(e)}'
            }
    
    def cleanup(self):
        """Clean up all agents and resources."""
        try:
            # Clean up all agents
            for agent in self.agents.values():
                agent.cleanup()
            
            # Clean up coordinator
            self.coordinator.cleanup_all()
            
            print("React AI Multi-Agent Chatbot System cleaned up")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def get_agent_graph_data(self) -> Dict[str, Any]:
        """
        Get agent graph data for visualization.
        
        Returns:
            Agent graph data with React AI pattern information
        """
        try:
            nodes = []
            edges = []
            
            # Add START node
            nodes.append({
                'id': 'START',
                'label': 'START',
                'type': 'start',
                'color': '#2E8B57',
                'size': 20,
                'icon': '▶️'
            })
            
            # Add agent nodes
            for name, agent in self.agents.items():
                status = agent.get_status()
                nodes.append({
                    'id': name,
                    'label': name,
                    'type': 'agent',
                    'color': '#4169E1' if status['is_active'] else '#DC143C',
                    'size': 25,
                    'icon': '🤖',
                    'capabilities': agent.get_capabilities(),
                    'framework': 'React AI Pattern',
                    'tools_count': status.get('tools_count', 0),
                    'reasoning_history': status.get('reasoning_history_count', 0)
                })
                
                # Add edge from START to agent
                edges.append({
                    'from': 'START',
                    'to': name,
                    'type': 'workflow',
                    'style': 'solid'
                })
            
            # Add END node
            nodes.append({
                'id': 'END',
                'label': 'END',
                'type': 'end',
                'color': '#FF4500',
                'size': 20,
                'icon': '⏹️'
            })
            
            # Add edges from agents to END
            for name in self.agents.keys():
                edges.append({
                    'from': name,
                    'to': 'END',
                    'type': 'workflow',
                    'style': 'solid'
                })
            
            return {
                'success': True,
                'nodes': nodes,
                'edges': edges,
                'framework': 'React AI Pattern',
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'pattern': 'Observe-Think-Act Loop'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error getting agent graph data: {str(e)}'
            } 

    def get_language_preferences(self) -> Dict[str, Any]:
        """
        Get user's language preferences.
        
        Returns:
            Language preferences dictionary
        """
        try:
            if self.db and hasattr(self, 'user_id'):
                return self.db.get_language_preferences(self.user_id)
            return {
                'native_language': None,
                'preferred_languages': [],
                'language_comfort_level': 'english'
            }
        except Exception as e:
            return {
                'native_language': None,
                'preferred_languages': [],
                'language_comfort_level': 'english'
            }

    def get_conversation_turns(self) -> int:
        """
        Get conversation turn count.
        
        Returns:
            Number of conversation turns
        """
        try:
            if self.db and hasattr(self, 'user_id'):
                conversations = self.db.get_user_conversations(self.user_id)
                return len(conversations)
            return 0
        except Exception as e:
            return 0

    def get_question_stats(self) -> Dict[str, int]:
        """
        Get question statistics.
        
        Returns:
            Question statistics dictionary
        """
        try:
            if self.db and hasattr(self, 'user_id'):
                return self.db.get_question_stats(self.user_id)
            return {
                'accepted_count': 0,
                'rejected_count': 0,
                'total_questions': 0
            }
        except Exception as e:
            return {
                'accepted_count': 0,
                'rejected_count': 0,
                'total_questions': 0
            }

    def get_conversation(self) -> List[tuple]:
        """
        Get conversation history.
        
        Returns:
            List of conversation tuples (role, message)
        """
        try:
            if self.db and hasattr(self, 'user_id'):
                return self.db.get_user_conversations(self.user_id)
            return []
        except Exception as e:
            return []

    def get_last_question(self) -> Optional[str]:
        """
        Get the last follow-up question.
        
        Returns:
            Last question or None
        """
        try:
            if self.db and hasattr(self, 'user_id'):
                return self.db.get_last_question(self.user_id)
            return None
        except Exception as e:
            return None

    def process_user_message(self, message: str) -> Dict[str, Any]:
        """
        Process a user message.
        
        Args:
            message: User message
            
        Returns:
            Processing result
        """
        try:
            if not hasattr(self, 'user_id'):
                return {'success': False, 'error': 'User not set'}
            
            # Use React AI pattern for message processing
            react_request = {
                'type': 'send_message',
                'user_id': self.user_id,
                'message': message,
                'language_preferences': self.get_language_preferences()
            }
            
            result = self.coordinator.route_request(react_request)
            
            if result.get('success'):
                # Add React AI specific metadata
                result['framework'] = 'React AI Pattern'
                result['reasoning_steps'] = len(result.get('reasoning_chain', []))
                result['processing_timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error processing message: {str(e)}'
            }

    def get_accepted_questions(self) -> List[str]:
        """
        Get accepted questions.
        
        Returns:
            List of accepted questions
        """
        try:
            if self.db and hasattr(self, 'user_id'):
                return self.db.get_accepted_questions(self.user_id)
            return []
        except Exception as e:
            return []

    def get_rejected_questions(self) -> List[str]:
        """
        Get rejected questions.
        
        Returns:
            List of rejected questions
        """
        try:
            if self.db and hasattr(self, 'user_id'):
                return self.db.get_rejected_questions(self.user_id)
            return []
        except Exception as e:
            return []

    def get_user_tags(self) -> List[str]:
        """
        Get user tags.
        
        Returns:
            List of user tags
        """
        try:
            if self.db and hasattr(self, 'user_id'):
                tags = self.db.get_user_tags(self.user_id)
                return [tag['tag'] for tag in tags]
            return []
        except Exception as e:
            return []

    def add_user_tag(self, tag: str) -> bool:
        """
        Add a tag to user profile.
        
        Args:
            tag: Tag to add
            
        Returns:
            Success status
        """
        try:
            if not hasattr(self, 'user_id'):
                return False
            
            # Use React AI pattern for tag addition
            react_request = {
                'type': 'create_profile',
                'user_id': self.user_id,
                'tags': [tag],
                'language_preferences': self.get_language_preferences()
            }
            
            result = self.coordinator.route_request(react_request)
            return result.get('success', False)
            
        except Exception as e:
            return False

    def update_language_preferences(self, native_language: str = None,
                                  preferred_languages: List[str] = None,
                                  language_comfort_level: str = 'english') -> bool:
        """
        Update language preferences.
        
        Args:
            native_language: Native language
            preferred_languages: List of preferred languages
            language_comfort_level: Language comfort level
            
        Returns:
            Success status
        """
        try:
            if not hasattr(self, 'user_id') or not self.db:
                return False
            
            self.db.update_language_preferences(
                self.user_id,
                native_language,
                preferred_languages or [],
                language_comfort_level
            )
            return True
            
        except Exception as e:
            return False

    def find_similar_users(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar users.
        
        Args:
            limit: Maximum number of users to return
            
        Returns:
            List of similar users
        """
        try:
            if not hasattr(self, 'user_id'):
                return []
            
            # Use React AI pattern for user matching
            react_request = {
                'type': 'find_similar_users',
                'user_id': self.user_id,
                'limit': limit
            }
            
            result = self.coordinator.route_request(react_request)
            
            if result.get('success'):
                return result.get('similar_users', [])
            return []
            
        except Exception as e:
            return []

    def get_user_group_chats(self) -> List[Dict[str, Any]]:
        """
        Get user's group chats.
        
        Returns:
            List of group chats
        """
        try:
            if self.db and hasattr(self, 'user_id'):
                return self.db.get_user_group_chats(self.user_id)
            return []
        except Exception as e:
            return []

    def create_group_chat(self, topic_name: str, user_ids: List[str] = None) -> Dict[str, Any]:
        """
        Create a group chat.
        
        Args:
            topic_name: Topic name
            user_ids: List of user IDs
            
        Returns:
            Group chat creation result
        """
        try:
            if not hasattr(self, 'user_id'):
                return {'success': False, 'error': 'User not set'}
            
            # Use React AI pattern for group chat creation
            react_request = {
                'type': 'create_group_chat',
                'topic_name': topic_name,
                'user_ids': user_ids or [self.user_id],
                'created_by': self.user_id
            }
            
            result = self.coordinator.route_request(react_request)
            
            if result.get('success'):
                result['framework'] = 'React AI Pattern'
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error creating group chat: {str(e)}'
            }

    def get_group_chat_info(self, group_id: str) -> Optional[Dict[str, Any]]:
        """
        Get group chat information.
        
        Args:
            group_id: Group chat ID
            
        Returns:
            Group chat info or None
        """
        try:
            if self.db:
                return self.db.get_group_chat_info(group_id)
            return None
        except Exception as e:
            return None

    def get_group_messages(self, group_id: str) -> List[Dict[str, Any]]:
        """
        Get group chat messages.
        
        Args:
            group_id: Group chat ID
            
        Returns:
            List of messages
        """
        try:
            if self.db:
                return self.db.get_group_messages(group_id)
            return []
        except Exception as e:
            return []

    def send_group_message(self, group_id: str, message: str) -> Dict[str, Any]:
        """
        Send a message to a group chat.
        
        Args:
            group_id: Group chat ID
            message: Message to send
            
        Returns:
            Message sending result
        """
        try:
            if not hasattr(self, 'user_id'):
                return {'success': False, 'error': 'User not set'}
            
            # Use React AI pattern for group message
            react_request = {
                'type': 'send_group_message',
                'group_id': group_id,
                'user_id': self.user_id,
                'message': message
            }
            
            result = self.coordinator.route_request(react_request)
            
            if result.get('success'):
                result['framework'] = 'React AI Pattern'
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error sending group message: {str(e)}'
            } 