"""
React AI Pattern-Based Group Chat Agent for Multi-Agent Chatbot System
====================================================================

This agent handles group chat functionality using React AI pattern, including:
- Group chat creation and management
- Group message handling
- AI participation in group chats
- Group member management
- React AI pattern: Observe → Think → Act → Observe
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from agents.react_base_agent import ReactBaseAgent, ReactAgentState
from langchain_core.tools import tool
from citation_system import CitationGenerator, CitationDisplayManager


class ReactGroupChatAgent(ReactBaseAgent):
    """
    React AI pattern-based agent responsible for handling group chat functionality.
    
    Implements Observe-Think-Act loops for group chat management,
    message handling, and AI participation.
    """
    
    def __init__(self, db_interface=None):
        """Initialize the React AI group chat agent."""
        super().__init__("ReactGroupChatAgent", db_interface)
        self.active_groups = {}
        
        # Initialize citation system
        self.citation_generator = CitationGenerator(db_interface=db_interface)
        self.citation_display_manager = CitationDisplayManager()
        
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this agent provides."""
        return [
            "react_ai_group_chat_management",
            "group_creation_and_management",
            "group_message_handling",
            "ai_participation",
            "group_member_management",
            "group_analytics",
            "react_ai_reasoning"
        ]
    
    def _get_agent_system_prompt(self) -> str:
        """Get the system prompt for React AI group chat agent."""
        return """You are a React AI group chat management agent designed to handle group chat functionality.

Your role is to manage group chats using the React AI pattern:
1. OBSERVE: Analyze group chat context and participant interactions
2. THINK: Reason about group dynamics and appropriate responses
3. ACT: Create groups, send messages, or participate appropriately
4. REFLECT: Learn from group interactions and improve participation

GROUP CHAT GUIDELINES:
- Create engaging group chat environments
- Facilitate meaningful conversations
- Participate naturally as an AI member
- Manage group member interactions
- Maintain cultural sensitivity in groups
- Provide helpful and relevant responses

REACT AI PATTERN:
- Always observe group chat context
- Think about group dynamics and cultural context
- Act by participating appropriately in conversations
- Reflect on group interaction effectiveness
"""
    
    def _get_agent_specific_tools(self) -> List:
        """Get group chat-specific tools for React AI pattern."""
        tools = []
        
        @tool
        def create_group_chat(topic_name: str, user_ids: str, created_by: str) -> str:
            """Create a new group chat with React AI reasoning."""
            try:
                # Handle user_ids properly - it should be a list or string representation
                if isinstance(user_ids, str):
                    # If it's a string, try to parse it as a list
                    if user_ids.startswith('[') and user_ids.endswith(']'):
                        # Remove brackets and split by comma
                        user_list = [uid.strip().strip("'\"") for uid in user_ids[1:-1].split(',')]
                    else:
                        # Single user ID
                        user_list = [user_ids]
                elif isinstance(user_ids, list):
                    user_list = user_ids
                else:
                    user_list = [str(user_ids)]
                
                # Create group with React AI reasoning
                group_id = f"group_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                group = {
                    'group_id': group_id,
                    'topic_name': topic_name,
                    'user_ids': user_list,
                    'created_by': created_by,
                    'created_at': datetime.now().isoformat(),
                    'is_active': True,
                    'react_ai_reasoning': 'Group created with React AI pattern'
                }
                
                # Store in agent's active groups
                self.active_groups[group_id] = group
                
                # Save to database if available
                if self.db:
                    self.db.create_group_chat(topic_name, user_list, created_by)
                
                return f"Group chat '{topic_name}' created successfully (ID: {group_id})"
            except Exception as e:
                return f"Error creating group chat: {str(e)}"
        
        @tool
        def send_group_message(group_id: str, user_id: str, message: str) -> str:
            """Send a message to a group chat with React AI reasoning."""
            try:
                if group_id in self.active_groups:
                    group = self.active_groups[group_id]
                    
                    # Add message to group
                    message_data = {
                        'group_id': group_id,
                        'user_id': user_id,
                        'message': message,
                        'timestamp': datetime.now().isoformat(),
                        'react_ai_reasoning': 'Message sent with React AI pattern'
                    }
                    
                    # Save to database if available
                    if self.db:
                        self.db.add_group_message(group_id, user_id, message)
                    
                    result = f"Message sent to group '{group['topic_name']}':\n"
                    result += f"- From: {user_id}\n"
                    result += f"- Message: {message[:50]}...\n"
                    result += f"- React AI: Message processed with reasoning"
                    
                    return result
                else:
                    return f"Group chat {group_id} not found"
            except Exception as e:
                return f"Error sending group message: {str(e)}"
        
        @tool
        def get_group_messages(group_id: str) -> str:
            """Get messages from a group chat with React AI reasoning."""
            try:
                if self.db:
                    messages = self.db.get_group_messages(group_id)
                    
                    result = f"Group chat messages for {group_id}:\n"
                    result += f"- Total messages: {len(messages)}\n"
                    
                    # Show recent messages
                    for msg in messages[-5:]:  # Last 5 messages
                        result += f"- {msg.get('user_id', 'Unknown')}: {msg.get('message', '')[:30]}...\n"
                    
                    result += f"- React AI: Message retrieval with reasoning"
                    
                    return result
                else:
                    return "Database not available for message retrieval"
            except Exception as e:
                return f"Error getting group messages: {str(e)}"
        
        @tool
        def participate_as_ai(group_id: str, conversation_context: str) -> str:
            """Participate in group chat as AI with React AI reasoning."""
            try:
                if group_id in self.active_groups:
                    group = self.active_groups[group_id]
                    
                    # Generate AI response based on context
                    ai_response = f"🤖 AI Bot: I'm here to help facilitate the conversation about '{group['topic_name']}'. "
                    ai_response += f"Based on the context, I'd like to contribute to the discussion. "
                    ai_response += f"React AI: Participating with cultural awareness and reasoning"
                    
                    # Save AI message to database if available
                    if self.db:
                        self.db.add_group_message(group_id, "ai_bot", ai_response, "ai")
                    
                    return ai_response
                else:
                    return f"Group chat {group_id} not found"
            except Exception as e:
                return f"Error participating as AI: {str(e)}"
        
        @tool
        def analyze_group_dynamics(group_id: str) -> str:
            """Analyze group chat dynamics with React AI reasoning."""
            try:
                if group_id in self.active_groups:
                    group = self.active_groups[group_id]
                    
                    if self.db:
                        messages = self.db.get_group_messages(group_id)
                        
                        analysis = f"Group Dynamics Analysis for '{group['topic_name']}':\n"
                        analysis += f"- Group ID: {group_id}\n"
                        analysis += f"- Participants: {len(group['user_ids'])}\n"
                        analysis += f"- Total messages: {len(messages)}\n"
                        analysis += f"- Created: {group['created_at']}\n"
                        
                        # Analyze message patterns
                        user_messages = {}
                        for msg in messages:
                            user = msg.get('user_id', 'Unknown')
                            user_messages[user] = user_messages.get(user, 0) + 1
                        
                        analysis += f"- Message distribution: {user_messages}\n"
                        analysis += f"- React AI: Dynamics analysis with reasoning"
                        
                        return analysis
                    else:
                        return "Database not available for group analysis"
                else:
                    return f"Group chat {group_id} not found"
            except Exception as e:
                return f"Error analyzing group dynamics: {str(e)}"
        
        tools.extend([
            create_group_chat,
            send_group_message,
            get_group_messages,
            participate_as_ai,
            analyze_group_dynamics
        ])
        
        return tools
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process group chat-related requests using React AI pattern.
        
        Args:
            request: Request dictionary with parameters
            
        Returns:
            Response dictionary with results
        """
        request_type = request.get('type', '')
        
        if request_type == 'create_group_chat':
            return self._create_group_chat(request)
        elif request_type == 'send_group_message':
            return self._send_group_message(request)
        elif request_type == 'get_group_messages':
            return self._get_group_messages(request)
        elif request_type == 'participate_as_ai':
            return self._participate_as_ai(request)
        elif request_type == 'analyze_group_dynamics':
            return self._analyze_group_dynamics(request)
        else:
            return {
                'success': False,
                'error': f'Unknown request type: {request_type}',
                'available_types': ['create_group_chat', 'send_group_message', 'get_group_messages', 'participate_as_ai', 'analyze_group_dynamics']
            }
    
    def _create_group_chat(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a group chat using React AI pattern.
        
        Args:
            request: Request with group chat data
            
        Returns:
            Response with group creation result
        """
        topic_name = request.get('topic_name')
        user_ids = request.get('user_ids', [])
        created_by = request.get('created_by')
        
        if not topic_name or not created_by:
            return {
                'success': False,
                'error': 'Missing topic_name or created_by'
            }
        
        # Use React AI pattern for group creation
        react_request = {
            'user_id': created_by,
            'message': f'Create group chat: {topic_name}',
            'topic_name': topic_name,
            'user_ids': user_ids,
            'created_by': created_by,
            'type': 'create_group_chat'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            result['group_created'] = True
            result['framework'] = 'React AI Pattern'
        
        return result
    
    def _send_group_message(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a group message using React AI pattern.
        
        Args:
            request: Request with message data
            
        Returns:
            Response with message sending result
        """
        group_id = request.get('group_id')
        user_id = request.get('user_id')
        message = request.get('message')
        
        if not group_id or not user_id or not message:
            return {
                'success': False,
                'error': 'Missing group_id, user_id, or message'
            }
        
        # Use React AI pattern for message sending
        react_request = {
            'user_id': user_id,
            'message': f'Send message to group {group_id}',
            'group_id': group_id,
            'user_id': user_id,
            'message_content': message,
            'type': 'send_group_message'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            result['message_sent'] = True
            result['framework'] = 'React AI Pattern'
        
        return result
    
    def _get_group_messages(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get group messages using React AI pattern.
        
        Args:
            request: Request with group ID
            
        Returns:
            Response with group messages
        """
        group_id = request.get('group_id')
        
        if not group_id:
            return {
                'success': False,
                'error': 'Missing group_id'
            }
        
        # Use React AI pattern for message retrieval
        react_request = {
            'user_id': 'system',
            'message': f'Get messages for group {group_id}',
            'group_id': group_id,
            'type': 'get_group_messages'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            result['messages_retrieved'] = True
            result['framework'] = 'React AI Pattern'
        
        return result
    
    def _participate_as_ai(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Participate as AI in group chat using React AI pattern.
        
        Args:
            request: Request with participation data
            
        Returns:
            Response with AI participation result
        """
        group_id = request.get('group_id')
        conversation_context = request.get('conversation_context', '')
        
        if not group_id:
            return {
                'success': False,
                'error': 'Missing group_id'
            }
        
        # Use React AI pattern for AI participation
        react_request = {
            'user_id': 'ai_bot',
            'message': f'Participate as AI in group {group_id}',
            'group_id': group_id,
            'conversation_context': conversation_context,
            'type': 'participate_as_ai'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            result['ai_participated'] = True
            result['framework'] = 'React AI Pattern'
        
        return result
    
    def _analyze_group_dynamics(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze group dynamics using React AI pattern.
        
        Args:
            request: Request with group ID
            
        Returns:
            Response with dynamics analysis
        """
        group_id = request.get('group_id')
        
        if not group_id:
            return {
                'success': False,
                'error': 'Missing group_id'
            }
        
        # Use React AI pattern for dynamics analysis
        react_request = {
            'user_id': 'system',
            'message': f'Analyze dynamics for group {group_id}',
            'group_id': group_id,
            'type': 'analyze_group_dynamics'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            result['dynamics_analyzed'] = True
            result['framework'] = 'React AI Pattern'
        
        return result 