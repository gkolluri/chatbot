"""
Conversation Agent for Multi-Agent Chatbot System using LangGraph
===============================================================

This agent handles individual chat conversations using LangGraph workflows, including:
- OpenAI GPT-3.5-turbo powered conversations
- Language-aware response generation with cultural context
- Follow-up question generation every 3 turns with learning
- Conversation history tracking with turn counting
- Question acceptance/rejection learning for better suggestions
- Real-time conversation analysis for tag inference
- Native language phrase integration when appropriate
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from .base_agent import LangGraphBaseAgent, AgentState


class ConversationAgent(LangGraphBaseAgent):
    """
    Agent responsible for handling individual chat conversations using LangGraph.
    
    Maintains conversation state, generates responses, and manages
    follow-up questions with learning capabilities.
    """
    
    def __init__(self, db_interface=None):
        """Initialize the conversation agent."""
        super().__init__("ConversationAgent", db_interface)
        self.conversation_turns = 0
        self.last_question = None
        self.accepted_questions = []
        self.rejected_questions = []
        
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this agent provides."""
        return [
            "individual_chat_conversations",
            "language_aware_responses",
            "follow_up_questions",
            "conversation_history_tracking",
            "question_learning",
            "cultural_context_integration",
            "langgraph_workflow"
        ]
    
    def _get_agent_system_prompt(self) -> str:
        """Get the system prompt for conversation agent."""
        return """You are a helpful AI assistant designed to connect Indian users and NRIs through shared interests. 
        Your primary focus is helping users find common ground and shared interests, with subtle Indian cultural context as an underlying layer.
        
        Guidelines:
        - Be conversational and engaging
        - Focus on interests and connections
        - Include subtle cultural context when appropriate
        - Respect diverse traditions and customs
        - Keep responses helpful and informative
        - Generate follow-up questions every 3 turns to learn about user interests
        """
    
    def _build_messages(self, state: AgentState) -> List:
        """
        Build messages for the LLM with conversation-specific context.
        
        Args:
            state: Current agent state
            
        Returns:
            List of messages for the LLM
        """
        # Get base messages from parent
        messages = super()._build_messages(state)
        
        # Add conversation-specific context
        if state.get("language_preferences"):
            lang_context = self._build_language_context(state["language_preferences"])
            if lang_context:
                # Update the system message with language context
                system_msg = messages[0].content + "\n\n" + lang_context
                messages[0] = type(messages[0])(content=system_msg)
        
        return messages
    
    def _build_language_context(self, language_preferences: Dict[str, Any]) -> str:
        """
        Build language-specific context for the conversation.
        
        Args:
            language_preferences: User's language preferences
            
        Returns:
            Language context string
        """
        native_lang = language_preferences.get('native_language')
        comfort_level = language_preferences.get('language_comfort_level', 'english')
        
        if not native_lang or native_lang == 'english':
            return ""
        
        context = f"\nLanguage Context:"
        context += f"\n- User's native language: {native_lang.title()}"
        context += f"\n- Language comfort level: {comfort_level}"
        
        if comfort_level == 'mixed':
            context += f"\n- User is comfortable with both English and {native_lang.title()}. You can occasionally use {native_lang.title()} phrases when appropriate."
        elif comfort_level == 'native':
            context += f"\n- User prefers {native_lang.title()} but you should respond in English with subtle cultural awareness."
        else:
            context += f"\n- User's native language is {native_lang.title()} but prefers English conversations."
        
        return context
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process conversation-related requests using LangGraph workflow.
        
        Args:
            request: Request dictionary with parameters
            
        Returns:
            Response dictionary with results
        """
        request_type = request.get('type', '')
        
        if request_type == 'send_message':
            return self._handle_user_message(request)
        elif request_type == 'get_conversation':
            return self._get_conversation_history(request)
        elif request_type == 'get_turns':
            return self._get_conversation_turns(request)
        elif request_type == 'get_question_stats':
            return self._get_question_statistics(request)
        elif request_type == 'get_last_question':
            return self._get_last_question(request)
        elif request_type == 'process_followup':
            return self._process_followup_response(request)
        else:
            return {
                'success': False,
                'error': f'Unknown request type: {request_type}',
                'available_types': ['send_message', 'get_conversation', 'get_turns', 'get_question_stats', 'get_last_question', 'process_followup']
            }
    
    def _handle_user_message(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a user message and generate a response.
        
        Args:
            request: Request with user message and context
            
        Returns:
            Response with bot message and metadata
        """
        user_id = request.get('user_id')
        message = request.get('message', '')
        language_preferences = request.get('language_preferences', {})
        
        if not user_id or not message:
            return {
                'success': False,
                'error': 'Missing user_id or message'
            }
        
        # Check if this is a follow-up question response
        if message.lower() in ['yes', 'no'] and self.last_question:
            return self._process_followup_response({
                'user_id': user_id,
                'response': message.lower(),
                'question': self.last_question
            })
        
        # Save user message
        self._save_message(user_id, 'user', message)
        self.conversation_turns += 1
        
        # Generate bot response using LangGraph workflow
        state = self._request_to_state({
            'user_id': user_id,
            'message': message,
            'language_preferences': language_preferences,
            'conversation_history': self._get_conversation_context(user_id)
        })
        
        result = self.workflow.invoke(state)
        bot_response = result.get('response', 'I apologize, but I am having trouble responding right now.')
        
        # Save bot response
        self._save_message(user_id, 'bot', bot_response)
        
        # Check if we should generate a follow-up question
        follow_up_question = None
        if self.conversation_turns % 3 == 0:
            follow_up_question = self._generate_followup_question(user_id, language_preferences)
            self.last_question = follow_up_question
        
        # Log activity
        self.log_activity("Processed user message", {
            'user_id': user_id,
            'message_length': len(message),
            'response_length': len(bot_response),
            'turns': self.conversation_turns,
            'has_followup': follow_up_question is not None
        })
        
        return {
            'success': True,
            'bot_response': bot_response,
            'conversation_turns': self.conversation_turns,
            'follow_up_question': follow_up_question,
            'message_id': f"msg_{datetime.now().timestamp()}"
        }
    
    def _generate_followup_question(self, user_id: str, language_preferences: Dict[str, Any]) -> str:
        """
        Generate a follow-up question based on conversation context.
        
        Args:
            user_id: User identifier
            language_preferences: User's language preferences
            
        Returns:
            Generated follow-up question
        """
        conversation = self._get_conversation_context(user_id)
        
        # Build prompt for follow-up question
        prompt = """Based on the conversation context, generate a natural follow-up question that:
        1. Relates to the user's interests
        2. Helps understand their preferences better
        3. Could help them connect with others
        4. Is conversational and engaging
        
        Return only the question, no explanations."""
        
        # Add conversation context
        if conversation:
            prompt += "\n\nConversation context:\n"
            for role, msg in conversation[-10:]:
                prompt += f"{role}: {msg}\n"
        
        # Use LangGraph workflow for question generation
        state = self._request_to_state({
            'user_id': user_id,
            'message': prompt,
            'language_preferences': language_preferences
        })
        
        result = self.workflow.invoke(state)
        response = result.get('response', '')
        
        return response if response else "What interests you most about this topic?"
    
    def _process_followup_response(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user's response to a follow-up question.
        
        Args:
            request: Request with user response and question
            
        Returns:
            Response with acknowledgment and learning
        """
        user_id = request.get('user_id')
        response = request.get('response', '').lower()
        question = request.get('question', self.last_question)
        
        if not question:
            return {
                'success': False,
                'error': 'No follow-up question to process'
            }
        
        # Learn from the response
        if response == 'yes':
            self.accepted_questions.append(question)
        elif response == 'no':
            self.rejected_questions.append(question)
        
        # Clear the last question
        self.last_question = None
        
        # Generate acknowledgment
        if response == 'yes':
            acknowledgment = "Great! I'll keep that in mind to help you connect with people who share similar interests."
        else:
            acknowledgment = "Thanks for letting me know. I'll adjust my suggestions to better match your interests."
        
        # Save the interaction
        self._save_message(user_id, 'user', f"Response to follow-up: {response}")
        self._save_message(user_id, 'bot', acknowledgment)
        
        self.log_activity("Processed follow-up response", {
            'user_id': user_id,
            'response': response,
            'question': question
        })
        
        return {
            'success': True,
            'acknowledgment': acknowledgment,
            'question_learned': True
        }
    
    def _get_conversation_context(self, user_id: str) -> List[Tuple[str, str]]:
        """
        Get conversation context for the user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of (role, message) tuples
        """
        if not self.db:
            return []
        
        try:
            conversations = self.db.get_user_conversations(user_id, limit=20)
            return conversations  # Already returns list of (role, message) tuples
        except Exception as e:
            print(f"Error getting conversation context: {e}")
            return []
    
    def _save_message(self, user_id: str, role: str, message: str):
        """
        Save a message to the database.
        
        Args:
            user_id: User identifier
            role: Message role (user/bot)
            message: Message content
        """
        if not self.db:
            return
        
        try:
            self.db.add_conversation(user_id, role, message, self.conversation_turns)
        except Exception as e:
            print(f"Error saving message: {e}")
    
    def _get_conversation_history(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get conversation history for a user."""
        user_id = request.get('user_id')
        if not user_id:
            return {'success': False, 'error': 'Missing user_id'}
        
        conversation = self._get_conversation_context(user_id)
        return {
            'success': True,
            'conversation': conversation
        }
    
    def _get_conversation_turns(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get current conversation turn count."""
        return {
            'success': True,
            'turns': self.conversation_turns
        }
    
    def _get_question_statistics(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about follow-up questions."""
        return {
            'success': True,
            'accepted_count': len(self.accepted_questions),
            'rejected_count': len(self.rejected_questions),
            'total_questions': len(self.accepted_questions) + len(self.rejected_questions),
            'accepted_questions': self.accepted_questions,
            'rejected_questions': self.rejected_questions
        }
    
    def _get_last_question(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get the last follow-up question if any."""
        return {
            'success': True,
            'last_question': self.last_question
        } 