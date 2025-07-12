"""
Group Chat Agent for Multi-Agent Chatbot System using LangGraph
==============================================================

This agent handles group chat functionality using LangGraph workflows, including:
- Multi-user group chat creation with topic selection
- AI bot participation in group conversations
- Topic-based chat organization with proper naming
- Persistent group message history with timestamps
- Real-time conversation context for AI responses
- Participant management with proper user identification
- Cultural context awareness in group interactions
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
from .base_agent import LangGraphBaseAgent, AgentState


class GroupChatAgent(LangGraphBaseAgent):
    """
    Agent responsible for group chat functionality using LangGraph.
    
    Manages group conversations, AI participation, and provides
    group chat-related functionality with LangGraph workflows.
    """
    
    def __init__(self, db_interface=None):
        """Initialize the group chat agent."""
        super().__init__("GroupChatAgent", db_interface)
        self.active_groups = {}
        self.group_messages = {}
        
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this agent provides."""
        return [
            "group_chat_management",
            "ai_participation",
            "topic_organization",
            "message_history",
            "participant_management",
            "cultural_context_awareness",
            "real_time_context",
            "langgraph_workflow"
        ]
    
    def _get_agent_system_prompt(self) -> str:
        """Get the system prompt for group chat agent."""
        return """You are an AI agent specialized in group chat management and participation.
        Your role is to:
        1. Manage group chat conversations and topics
        2. Participate naturally in group discussions
        3. Provide context-aware responses in group settings
        4. Facilitate meaningful conversations among participants
        5. Consider cultural context in group interactions
        6. Maintain conversation flow and engagement
        
        Guidelines:
        - Be conversational and engaging in group settings
        - Consider the group topic and context
        - Respect diverse perspectives and backgrounds
        - Facilitate connections between participants
        - Provide helpful insights and suggestions
        - Maintain appropriate group dynamics
        """
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process group chat requests using LangGraph workflow.
        
        Args:
            request: Request dictionary with parameters
            
        Returns:
            Response dictionary with results
        """
        request_type = request.get('type', '')
        
        if request_type == 'create_group':
            return self._create_group_chat(request)
        elif request_type == 'join_group':
            return self._join_group_chat(request)
        elif request_type == 'send_group_message':
            return self._send_group_message(request)
        elif request_type == 'get_group_messages':
            return self._get_group_messages(request)
        elif request_type == 'get_active_groups':
            return self._get_active_groups(request)
        elif request_type == 'leave_group':
            return self._leave_group_chat(request)
        elif request_type == 'get_group_info':
            return self._get_group_info(request)
        elif request_type == 'get_user_groups':
            return self._get_user_groups(request)
        elif request_type == 'suggest_group_topics':
            return self._suggest_group_topics(request)
        else:
            return {
                'success': False,
                'error': f'Unknown request type: {request_type}',
                'available_types': ['create_group', 'join_group', 'send_group_message', 'get_group_messages', 'get_active_groups', 'leave_group', 'get_group_info', 'get_user_groups', 'suggest_group_topics']
            }
    
    def _create_group_chat(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new group chat.
        
        Args:
            request: Request with group creation parameters
            
        Returns:
            Response with created group data
        """
        topic_name = request.get('topic_name', '')
        created_by = request.get('user_id')
        user_name = request.get('user_name', '')
        language_preferences = request.get('language_preferences', {})
        
        if not topic_name or not created_by:
            return {
                'success': False,
                'error': 'Missing topic_name or user_id'
            }
        
        # Generate unique group ID
        group_id = str(uuid.uuid4())
        
        # Create group data
        group_data = {
            'group_id': group_id,
            'topic_name': topic_name,
            'created_by': created_by,
            'created_by_name': user_name,
            'created_at': datetime.now().isoformat(),
            'participants': [created_by],
            'participant_names': [user_name],
            'is_active': True,
            'language_preferences': language_preferences,
            'message_count': 0
        }
        
        # Store group
        self.active_groups[group_id] = group_data
        self.group_messages[group_id] = []
        
        # Save to database if available
        if self.db:
            try:
                self.db.create_group_chat(topic_name, [created_by], created_by)
            except Exception as e:
                print(f"Error saving group to database: {e}")
        
        # Add welcome message
        welcome_message = self._generate_welcome_message(topic_name, user_name, language_preferences)
        self._add_group_message(group_id, 'ai_bot', welcome_message, 'ai')
        
        self.log_activity("Created group chat", {
            'group_id': group_id,
            'topic_name': topic_name,
            'created_by': created_by,
            'user_name': user_name
        })
        
        return {
            'success': True,
            'group_data': group_data,
            'welcome_message': welcome_message,
            'message': 'Group chat created successfully'
        }
    
    def _generate_welcome_message(self, topic_name: str, user_name: str, 
                                language_preferences: Dict[str, Any]) -> str:
        """
        Generate a welcome message for the group.
        
        Args:
            topic_name: Group topic name
            user_name: Creator's name
            language_preferences: Language preferences
            
        Returns:
            Welcome message
        """
        prompt = f"""Generate a welcoming message for a new group chat about "{topic_name}".
        
        Context:
        - Group topic: {topic_name}
        - Created by: {user_name}
        - Language preferences: {language_preferences}
        
        Guidelines:
        - Be welcoming and inclusive
        - Encourage participation and discussion
        - Mention the group topic
        - Keep it conversational and engaging
        - Consider cultural context if relevant
        
        Generate a natural welcome message:"""
        
        # Use LangGraph workflow for message generation
        state = self._request_to_state({
            'user_id': 'group_welcome',
            'message': prompt,
            'language_preferences': language_preferences
        })
        
        result = self.workflow.invoke(state)
        welcome_message = result.get('response', f'Welcome to the {topic_name} group! Let\'s start a great conversation.')
        
        return welcome_message
    
    def _join_group_chat(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Join an existing group chat.
        
        Args:
            request: Request with join parameters
            
        Returns:
            Response with join status
        """
        group_id = request.get('group_id')
        user_id = request.get('user_id')
        user_name = request.get('user_name', '')
        
        if not group_id or not user_id:
            return {
                'success': False,
                'error': 'Missing group_id or user_id'
            }
        
        if group_id not in self.active_groups:
            return {
                'success': False,
                'error': 'Group not found'
            }
        
        group_data = self.active_groups[group_id]
        
        # Check if user is already a participant
        if user_id in group_data['participants']:
            return {
                'success': False,
                'error': 'User is already a participant'
            }
        
        # Add user to group
        group_data['participants'].append(user_id)
        group_data['participant_names'].append(user_name)
        
        # Generate join message
        join_message = self._generate_join_message(user_name, group_data['topic_name'])
        self._add_group_message(group_id, 'ai_bot', join_message, 'ai')
        
        # Save to database if available
        if self.db:
            try:
                # Update group participants in database
                pass
            except Exception as e:
                print(f"Error updating group in database: {e}")
        
        self.log_activity("User joined group", {
            'group_id': group_id,
            'user_id': user_id,
            'user_name': user_name,
            'topic_name': group_data['topic_name']
        })
        
        return {
            'success': True,
            'group_data': group_data,
            'join_message': join_message,
            'message': 'Successfully joined group'
        }
    
    def _generate_join_message(self, user_name: str, topic_name: str) -> str:
        """
        Generate a join message for a new participant.
        
        Args:
            user_name: Name of the joining user
            topic_name: Group topic name
            
        Returns:
            Join message
        """
        prompt = f"""Generate a message welcoming {user_name} to the "{topic_name}" group chat.
        
        Guidelines:
        - Be welcoming and friendly
        - Encourage them to participate
        - Keep it brief and natural
        - Make them feel included
        
        Generate a welcoming message:"""
        
        # Use LangGraph workflow for message generation
        state = self._request_to_state({
            'user_id': 'group_join',
            'message': prompt
        })
        
        result = self.workflow.invoke(state)
        join_message = result.get('response', f'Welcome {user_name}! Great to have you join our {topic_name} discussion.')
        
        return join_message
    
    def _send_group_message(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a message to a group chat.
        
        Args:
            request: Request with message parameters
            
        Returns:
            Response with message status
        """
        group_id = request.get('group_id')
        user_id = request.get('user_id')
        user_name = request.get('user_name', '')
        message = request.get('message', '')
        language_preferences = request.get('language_preferences', {})
        
        if not group_id or not user_id or not message:
            return {
                'success': False,
                'error': 'Missing group_id, user_id, or message'
            }
        
        if group_id not in self.active_groups:
            return {
                'success': False,
                'error': 'Group not found'
            }
        
        group_data = self.active_groups[group_id]
        
        # Check if user is a participant
        if user_id not in group_data['participants']:
            return {
                'success': False,
                'error': 'User is not a participant in this group'
            }
        
        # Add user message
        message_id = self._add_group_message(group_id, user_id, message, 'user')
        
        # Check if AI should respond
        ai_response = self._generate_ai_response(group_id, message, user_name, language_preferences)
        if ai_response:
            self._add_group_message(group_id, 'ai_bot', ai_response, 'ai')
        
        # Save to database if available
        if self.db:
            try:
                self.db.add_group_message(group_id, user_id, message, 'user')
                if ai_response:
                    self.db.add_group_message(group_id, 'ai_bot', ai_response, 'ai')
            except Exception as e:
                print(f"Error saving group message to database: {e}")
        
        self.log_activity("Sent group message", {
            'group_id': group_id,
            'user_id': user_id,
            'user_name': user_name,
            'message_length': len(message),
            'has_ai_response': bool(ai_response)
        })
        
        return {
            'success': True,
            'message_id': message_id,
            'ai_response': ai_response,
            'group_data': group_data
        }
    
    def _generate_ai_response(self, group_id: str, user_message: str, user_name: str,
                            language_preferences: Dict[str, Any]) -> str:
        """
        Generate an AI response in the group chat.
        
        Args:
            group_id: Group identifier
            user_message: User's message
            user_name: User's name
            language_preferences: Language preferences
            
        Returns:
            AI response or empty string if no response needed
        """
        # Get recent group context
        recent_messages = self.group_messages.get(group_id, [])[-10:]  # Last 10 messages
        group_data = self.active_groups.get(group_id, {})
        topic_name = group_data.get('topic_name', '')
        
        # Build context for AI response
        context = f"Group topic: {topic_name}\n"
        context += f"User: {user_name}\n"
        context += "Recent conversation:\n"
        
        for msg in recent_messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                context += f"User: {content}\n"
            else:
                context += f"AI: {content}\n"
        
        prompt = f"""You are participating in a group chat about "{topic_name}".
        
        Context:
        {context}
        
        Guidelines:
        - Respond naturally and conversationally
        - Consider the group topic and context
        - Be helpful and engaging
        - Encourage discussion and participation
        - Keep responses appropriate for group setting
        - Consider cultural context if relevant
        
        Generate a natural response to the user's message, or indicate if no response is needed."""
        
        # Use LangGraph workflow for AI response generation
        state = self._request_to_state({
            'user_id': 'group_ai',
            'message': prompt,
            'language_preferences': language_preferences
        })
        
        result = self.workflow.invoke(state)
        response = result.get('response', '')
        
        # Only respond if the AI generated a meaningful response
        if response and len(response.strip()) > 10:
            return response
        
        return ""
    
    def _add_group_message(self, group_id: str, sender_id: str, content: str, 
                          message_type: str) -> str:
        """
        Add a message to the group chat.
        
        Args:
            group_id: Group identifier
            sender_id: Sender ID (user_id or 'ai_bot')
            content: Message content
            message_type: Message type ('user' or 'ai')
            
        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())
        
        message_data = {
            'message_id': message_id,
            'sender_id': sender_id,
            'content': content,
            'message_type': message_type,
            'timestamp': datetime.now().isoformat(),
            'role': 'user' if message_type == 'user' else 'ai'
        }
        
        if group_id not in self.group_messages:
            self.group_messages[group_id] = []
        
        self.group_messages[group_id].append(message_data)
        
        # Update group message count
        if group_id in self.active_groups:
            self.active_groups[group_id]['message_count'] += 1
        
        return message_id
    
    def _get_group_messages(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get messages from a group chat.
        
        Args:
            request: Request with group_id
            
        Returns:
            Response with group messages
        """
        group_id = request.get('group_id')
        limit = request.get('limit', 50)
        
        if not group_id:
            return {
                'success': False,
                'error': 'Missing group_id'
            }
        
        if group_id not in self.active_groups:
            return {
                'success': False,
                'error': 'Group not found'
            }
        
        messages = self.group_messages.get(group_id, [])
        
        # Apply limit
        if limit and len(messages) > limit:
            messages = messages[-limit:]
        
        return {
            'success': True,
            'messages': messages,
            'total_messages': len(self.group_messages.get(group_id, [])),
            'group_data': self.active_groups[group_id]
        }
    
    def _get_active_groups(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get list of active groups.
        
        Args:
            request: Request parameters
            
        Returns:
            Response with active groups
        """
        user_id = request.get('user_id')
        
        if user_id:
            # Get groups where user is a participant
            user_groups = []
            for group_id, group_data in self.active_groups.items():
                if user_id in group_data['participants']:
                    user_groups.append({
                        'group_id': group_id,
                        'topic_name': group_data['topic_name'],
                        'created_by': group_data['created_by'],
                        'created_by_name': group_data['created_by_name'],
                        'created_at': group_data['created_at'],
                        'participant_count': len(group_data['participants']),
                        'message_count': group_data['message_count'],
                        'is_active': group_data['is_active']
                    })
            
            return {
                'success': True,
                'user_groups': user_groups,
                'total_user_groups': len(user_groups)
            }
        else:
            # Get all active groups
            all_groups = []
            for group_id, group_data in self.active_groups.items():
                all_groups.append({
                    'group_id': group_id,
                    'topic_name': group_data['topic_name'],
                    'created_by': group_data['created_by'],
                    'created_by_name': group_data['created_by_name'],
                    'created_at': group_data['created_at'],
                    'participant_count': len(group_data['participants']),
                    'message_count': group_data['message_count'],
                    'is_active': group_data['is_active']
                })
            
            return {
                'success': True,
                'all_groups': all_groups,
                'total_groups': len(all_groups)
            }
    
    def _leave_group_chat(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Leave a group chat.
        
        Args:
            request: Request with leave parameters
            
        Returns:
            Response with leave status
        """
        group_id = request.get('group_id')
        user_id = request.get('user_id')
        user_name = request.get('user_name', '')
        
        if not group_id or not user_id:
            return {
                'success': False,
                'error': 'Missing group_id or user_id'
            }
        
        if group_id not in self.active_groups:
            return {
                'success': False,
                'error': 'Group not found'
            }
        
        group_data = self.active_groups[group_id]
        
        # Check if user is a participant
        if user_id not in group_data['participants']:
            return {
                'success': False,
                'error': 'User is not a participant in this group'
            }
        
        # Remove user from group
        participant_index = group_data['participants'].index(user_id)
        group_data['participants'].pop(participant_index)
        group_data['participant_names'].pop(participant_index)
        
        # Generate leave message
        leave_message = self._generate_leave_message(user_name)
        self._add_group_message(group_id, 'ai_bot', leave_message, 'ai')
        
        # Save to database if available
        if self.db:
            try:
                # Update group participants in database
                pass
            except Exception as e:
                print(f"Error updating group in database: {e}")
        
        self.log_activity("User left group", {
            'group_id': group_id,
            'user_id': user_id,
            'user_name': user_name,
            'topic_name': group_data['topic_name']
        })
        
        return {
            'success': True,
            'group_data': group_data,
            'leave_message': leave_message,
            'message': 'Successfully left group'
        }
    
    def _generate_leave_message(self, user_name: str) -> str:
        """
        Generate a leave message for a departing participant.
        
        Args:
            user_name: Name of the leaving user
            
        Returns:
            Leave message
        """
        prompt = f"""Generate a message acknowledging that {user_name} has left the group chat.
        
        Guidelines:
        - Be polite and respectful
        - Keep it brief and natural
        - Thank them for participating
        - Encourage remaining participants to continue
        
        Generate a leave acknowledgment message:"""
        
        # Use LangGraph workflow for message generation
        state = self._request_to_state({
            'user_id': 'group_leave',
            'message': prompt
        })
        
        result = self.workflow.invoke(state)
        leave_message = result.get('response', f'Thanks for participating, {user_name}! The conversation continues.')
        
        return leave_message
    
    def _get_group_info(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed information about a group.
        
        Args:
            request: Request with group_id
            
        Returns:
            Response with group information
        """
        group_id = request.get('group_id')
        
        if not group_id:
            return {
                'success': False,
                'error': 'Missing group_id'
            }
        
        if group_id not in self.active_groups:
            return {
                'success': False,
                'error': 'Group not found'
            }
        
        group_data = self.active_groups[group_id]
        messages = self.group_messages.get(group_id, [])
        
        # Calculate group statistics
        user_messages = [msg for msg in messages if msg.get('message_type') == 'user']
        ai_messages = [msg for msg in messages if msg.get('message_type') == 'ai']
        
        group_info = {
            'group_id': group_id,
            'topic_name': group_data['topic_name'],
            'created_by': group_data['created_by'],
            'created_by_name': group_data['created_by_name'],
            'created_at': group_data['created_at'],
            'participants': group_data['participants'],
            'participant_names': group_data['participant_names'],
            'participant_count': len(group_data['participants']),
            'total_messages': len(messages),
            'user_messages': len(user_messages),
            'ai_messages': len(ai_messages),
            'is_active': group_data['is_active'],
            'language_preferences': group_data.get('language_preferences', {})
        }
        
        return {
            'success': True,
            'group_info': group_info
        }
    
    def _get_user_groups(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get groups for a specific user.
        
        Args:
            request: Request with user_id
            
        Returns:
            Response with user's groups
        """
        user_id = request.get('user_id')
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        # Get groups where user is a participant
        user_groups = []
        for group_id, group_data in self.active_groups.items():
            if user_id in group_data['participants']:
                # Get participant names
                participant_names = []
                for uid in group_data['participants']:
                    if uid == "ai_bot":
                        participant_names.append("AI Assistant")
                    else:
                        # Try to get user name from database or use default
                        if self.db:
                            try:
                                user_profile = self.db.get_user_profile(uid)
                                if user_profile:
                                    participant_names.append(user_profile.get('name', 'Unknown User'))
                                else:
                                    participant_names.append('Unknown User')
                            except:
                                participant_names.append('Unknown User')
                        else:
                            participant_names.append('Unknown User')
                
                # Parse created_at string to datetime for formatting
                try:
                    from datetime import datetime
                    created_at = datetime.fromisoformat(group_data['created_at'])
                except:
                    created_at = group_data['created_at']
                
                user_groups.append({
                    'group_id': group_id,
                    'topic_name': group_data['topic_name'],
                    'participants': participant_names,
                    'created_at': created_at,
                    'is_active': group_data['is_active']
                })
        
        return {
            'success': True,
            'user_groups': user_groups
        }
    
    def _suggest_group_topics(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest group topics based on user tags.
        
        Args:
            request: Request with user_tags
            
        Returns:
            Response with suggested topics
        """
        user_tags = request.get('user_tags', [])
        
        if not user_tags:
            # Default suggestions if no tags
            default_topics = [
                "General Discussion",
                "Technology Talk",
                "Cultural Exchange",
                "Language Learning",
                "Travel Stories",
                "Food & Cuisine",
                "Music & Entertainment",
                "Sports & Fitness",
                "Business & Career",
                "Education & Learning"
            ]
            return {
                'success': True,
                'suggested_topics': default_topics
            }
        
        # Generate topic suggestions based on tags
        prompt = f"""Based on these user interests: {', '.join(user_tags)}
        
        Generate 5-8 engaging group chat topic suggestions that would appeal to people with these interests.
        
        Guidelines:
        - Make topics broad enough for group discussion
        - Include both specific and general topics
        - Consider cultural and contemporary relevance
        - Make topics engaging and discussion-worthy
        - Keep names concise and clear
        
        Generate topic suggestions:"""
        
        # Use LangGraph workflow for topic generation
        state = self._request_to_state({
            'user_id': 'topic_suggestion',
            'message': prompt
        })
        
        result = self.workflow.invoke(state)
        response = result.get('response', '')
        
        # Parse response into individual topics
        topics = []
        if response:
            # Split by lines and clean up
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('-') and not line.startswith('•'):
                    # Remove numbering if present
                    if line[0].isdigit() and line[1] in ['.', ')', ':']:
                        line = line[2:].strip()
                    topics.append(line)
        
        # Fallback if parsing fails
        if not topics:
            topics = [
                f"{tag.title()} Discussion" for tag in user_tags[:5]
            ] + [
                "General Discussion",
                "Cultural Exchange",
                "Technology Talk"
            ]
        
        return {
            'success': True,
            'suggested_topics': topics[:8]  # Limit to 8 topics
        } 