"""
React AI Pattern-Based Conversation Agent for Multi-Agent Chatbot System
======================================================================

This agent handles individual chat conversations using React AI pattern, including:
- OpenAI GPT-3.5-turbo powered conversations with reasoning
- Language-aware response generation with cultural context
- Follow-up question generation with learning and reflection
- Conversation history tracking with turn counting
- Question acceptance/rejection learning for better suggestions
- Real-time conversation analysis for tag inference
- Native language phrase integration when appropriate
- React AI pattern: Observe → Think → Act → Observe
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from agents.react_base_agent import ReactBaseAgent, ReactAgentState
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage


class ReactConversationAgent(ReactBaseAgent):
    """
    React AI pattern-based agent responsible for handling individual chat conversations.
    
    Implements Observe-Think-Act loops for dynamic conversation management,
    reasoning about user needs, and generating appropriate responses.
    """
    
    def __init__(self, db_interface=None):
        """Initialize the React AI conversation agent."""
        super().__init__("ReactConversationAgent", db_interface)
        self.conversation_turns = 0
        self.last_question = None
        self.accepted_questions = []
        self.rejected_questions = []
        self.conversation_patterns = []
        
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this agent provides."""
        return [
            "react_ai_conversations",
            "dynamic_response_generation",
            "reasoning_based_responses",
            "follow_up_questions_with_learning",
            "conversation_pattern_analysis",
            "cultural_context_integration",
            "language_aware_responses",
            "adaptive_conversation_flow"
        ]
    
    def _get_agent_system_prompt(self) -> str:
        """Get the system prompt for React AI conversation agent."""
        return """You are a React AI conversation agent designed to connect Indian users and NRIs through shared interests.

Your role is to engage in meaningful conversations using the React AI pattern:
1. OBSERVE: Understand the user's message, context, and conversation history
2. THINK: Reason about what the user is asking, their interests, and cultural context
3. ACT: Generate an appropriate response that helps build connections

CONVERSATION GUIDELINES:
- Be conversational and engaging
- Focus on interests and connections
- Include subtle cultural context when appropriate
- Respect diverse traditions and customs
- Keep responses helpful and informative
- Generate follow-up questions every 3 turns to learn about user interests
- Use reasoning to understand user needs and provide better responses

CULTURAL CONTEXT:
- Be aware of Indian cultural nuances
- Respect language preferences
- Include cultural references when appropriate
- Maintain professional and inclusive approach

REACT AI PATTERN:
- Always observe the current conversation state
- Think about what the user is really asking
- Act by generating the most helpful response
- Reflect on your response quality
"""
    
    def _get_agent_specific_tools(self) -> List:
        """Get conversation-specific tools for React AI pattern."""
        tools = []
        
        @tool
        def analyze_conversation_context(user_id: str, message: str) -> str:
            """Analyze the conversation context to understand user needs and interests."""
            try:
                # Get conversation history
                history = self.db.get_user_conversations(user_id) if self.db else []
                
                analysis = f"Conversation Analysis for user {user_id}:\n"
                analysis += f"Current message: {message}\n"
                analysis += f"Conversation turns: {len(history)}\n"
                
                # Analyze patterns
                if history:
                    recent_messages = history[-5:]  # Last 5 messages
                    analysis += f"Recent conversation pattern: {recent_messages}\n"
                    
                    # Look for interest indicators
                    interest_keywords = ['like', 'enjoy', 'love', 'interested', 'passionate', 'hobby']
                    found_interests = []
                    for msg in recent_messages:
                        if isinstance(msg, tuple):
                            content = msg[1]
                        else:
                            content = msg.get('content', '')
                        
                        for keyword in interest_keywords:
                            if keyword.lower() in content.lower():
                                found_interests.append(keyword)
                    
                    if found_interests:
                        analysis += f"Detected interests: {list(set(found_interests))}\n"
                
                return analysis
            except Exception as e:
                return f"Error analyzing conversation context: {str(e)}"
        
        @tool
        def generate_cultural_response(message: str, language_preferences: str) -> str:
            """Generate a response with appropriate cultural context."""
            try:
                # Parse language preferences
                prefs = eval(language_preferences) if isinstance(language_preferences, str) else language_preferences
                native_lang = prefs.get('native_language', 'english')
                comfort_level = prefs.get('language_comfort_level', 'english')
                
                cultural_context = f"User's native language: {native_lang}, Comfort level: {comfort_level}\n"
                
                # Generate culturally aware response
                if native_lang != 'english' and comfort_level in ['mixed', 'native']:
                    cultural_context += f"Consider including subtle {native_lang.title()} cultural references when appropriate.\n"
                
                cultural_context += "Focus on connecting through shared interests while being culturally sensitive."
                
                return cultural_context
            except Exception as e:
                return f"Error generating cultural response: {str(e)}"
        
        @tool
        def generate_followup_question(user_id: str, conversation_context: str) -> str:
            """Generate a follow-up question based on conversation context."""
            try:
                # Get user tags to understand interests
                user_tags = self.db.get_user_tags(user_id) if self.db else []
                
                question_context = f"User tags: {user_tags}\n"
                question_context += f"Conversation context: {conversation_context}\n"
                question_context += "Generate an engaging follow-up question that helps discover shared interests."
                
                # Use LLM to generate question
                messages = [
                    SystemMessage(content="You are a helpful assistant. Generate an engaging follow-up question."),
                    HumanMessage(content=question_context)
                ]
                
                response = self.llm.invoke(messages)
                return response.content
            except Exception as e:
                return f"Error generating follow-up question: {str(e)}"
        
        @tool
        def learn_from_response(user_id: str, question: str, response: str) -> str:
            """Learn from user's response to improve future questions."""
            try:
                if response.lower() in ['yes', 'y', 'true', '1']:
                    self.accepted_questions.append(question)
                    learning = f"Question accepted: {question}\n"
                    learning += f"Total accepted questions: {len(self.accepted_questions)}"
                elif response.lower() in ['no', 'n', 'false', '0']:
                    self.rejected_questions.append(question)
                    learning = f"Question rejected: {question}\n"
                    learning += f"Total rejected questions: {len(self.rejected_questions)}"
                else:
                    learning = f"Neutral response to question: {question}\n"
                    learning += f"Response: {response}"
                
                return learning
            except Exception as e:
                return f"Error learning from response: {str(e)}"
        
        @tool
        def search_current_information(query: str) -> str:
            """Search for current information using web search."""
            try:
                # Check if query needs current information
                current_keywords = ['current', 'latest', 'recent', 'now', 'today', '2024', '2025', 'news', 'trending']
                needs_web_search = any(keyword in query.lower() for keyword in current_keywords)
                
                if needs_web_search:
                    search_info = f"Searching for current information about: {query}\n"
                    try:
                        web_results = self._search_web_reactive(query, "User is asking about current information")
                        search_info += f"Web search results: {web_results[:500]}...\n"
                    except Exception as e:
                        search_info += f"Web search unavailable: {str(e)}\n"
                    
                    return search_info
                else:
                    return f"No current information search needed for: {query}"
            except Exception as e:
                return f"Error searching current information: {str(e)}"
        
        @tool
        def reflect_on_conversation_quality(user_id: str, response: str) -> str:
            """Reflect on the quality of the generated response."""
            try:
                reflection = f"Reflecting on response quality for user {user_id}:\n"
                reflection += f"Response: {response}\n"
                
                # Analyze response characteristics
                response_length = len(response)
                has_question = '?' in response
                has_cultural_elements = any(word in response.lower() for word in ['indian', 'culture', 'tradition'])
                
                reflection += f"Response length: {response_length} characters\n"
                reflection += f"Contains question: {has_question}\n"
                reflection += f"Cultural elements: {has_cultural_elements}\n"
                
                # Suggest improvements
                if response_length < 50:
                    reflection += "Suggestion: Consider providing more detailed responses.\n"
                if not has_question and self.conversation_turns % 3 == 0:
                    reflection += "Suggestion: Consider adding a follow-up question.\n"
                if not has_cultural_elements:
                    reflection += "Suggestion: Consider adding subtle cultural context.\n"
                
                return reflection
            except Exception as e:
                return f"Error reflecting on conversation quality: {str(e)}"
        
        tools.extend([
            analyze_conversation_context,
            generate_cultural_response,
            generate_followup_question,
            learn_from_response,
            search_current_information,
            reflect_on_conversation_quality
        ])
        
        return tools
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process conversation-related requests using React AI pattern.
        
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
        Handle a user message using React AI pattern.
        
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
        
        # Use React AI pattern to generate response
        react_request = {
            'user_id': user_id,
            'message': message,
            'language_preferences': language_preferences,
            'conversation_history': self._get_conversation_context(user_id),
            'type': 'send_message'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            bot_response = result.get('response', 'I apologize, but I am having trouble responding right now.')
            
            # Save bot response
            self._save_message(user_id, 'bot', bot_response)
            
            # Check if we should generate a follow-up question
            follow_up_question = None
            if self.conversation_turns % 3 == 0:
                follow_up_question = self._generate_followup_question(user_id, language_preferences)
                self.last_question = follow_up_question
            
            # Add React AI metadata
            result['conversation_turns'] = self.conversation_turns
            result['follow_up_question'] = follow_up_question
            result['reasoning_chain'] = result.get('reasoning_chain', [])
            
            return result
        else:
            return result
    
    def _generate_followup_question(self, user_id: str, language_preferences: Dict[str, Any]) -> str:
        """
        Generate a follow-up question using React AI pattern.
        
        Args:
            user_id: User identifier
            language_preferences: User's language preferences
            
        Returns:
            Generated follow-up question
        """
        try:
            # Use React AI pattern to generate question
            react_request = {
                'user_id': user_id,
                'message': 'Generate a follow-up question',
                'language_preferences': language_preferences,
                'conversation_history': self._get_conversation_context(user_id),
                'type': 'generate_followup'
            }
            
            result = self.react_loop(react_request)
            
            if result.get('success'):
                return result.get('response', 'What interests you most?')
            else:
                return 'What interests you most?'
                
        except Exception as e:
            print(f"Error generating follow-up question: {e}")
            return 'What interests you most?'
    
    def _process_followup_response(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a follow-up question response using React AI pattern.
        
        Args:
            request: Request with response data
            
        Returns:
            Response with learning results
        """
        user_id = request.get('user_id')
        response = request.get('response', '')
        question = request.get('question', '')
        
        if not user_id or not response or not question:
            return {
                'success': False,
                'error': 'Missing user_id, response, or question'
            }
        
        # Use React AI pattern to learn from response
        react_request = {
            'user_id': user_id,
            'message': f'Learn from response: {response} to question: {question}',
            'type': 'learn_from_response'
        }
        
        result = self.react_loop(react_request)
        
        # Update learning statistics
        if response.lower() in ['yes', 'y', 'true', '1']:
            self.accepted_questions.append(question)
        elif response.lower() in ['no', 'n', 'false', '0']:
            self.rejected_questions.append(question)
        
        result['accepted_questions'] = len(self.accepted_questions)
        result['rejected_questions'] = len(self.rejected_questions)
        
        return result
    
    def _get_conversation_context(self, user_id: str) -> List[Tuple[str, str]]:
        """
        Get conversation context for React AI pattern.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of conversation tuples (role, message)
        """
        try:
            if self.db:
                conversations = self.db.get_user_conversations(user_id)
                return conversations
            else:
                return []
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
        try:
            if self.db:
                self.db.add_conversation(user_id, role, message, self.conversation_turns)
        except Exception as e:
            print(f"Error saving message: {e}")
    
    def _get_conversation_history(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get conversation history.
        
        Args:
            request: Request with user_id
            
        Returns:
            Response with conversation history
        """
        user_id = request.get('user_id')
        if not user_id:
            return {'success': False, 'error': 'Missing user_id'}
        
        try:
            conversations = self._get_conversation_context(user_id)
            return {
                'success': True,
                'conversations': conversations,
                'total_turns': len(conversations)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _get_conversation_turns(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get conversation turn count.
        
        Args:
            request: Request with user_id
            
        Returns:
            Response with turn count
        """
        return {
            'success': True,
            'turns': self.conversation_turns
        }
    
    def _get_question_statistics(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get question acceptance statistics.
        
        Args:
            request: Request dictionary
            
        Returns:
            Response with statistics
        """
        total_questions = len(self.accepted_questions) + len(self.rejected_questions)
        acceptance_rate = len(self.accepted_questions) / total_questions if total_questions > 0 else 0
        
        return {
            'success': True,
            'accepted_questions': len(self.accepted_questions),
            'rejected_questions': len(self.rejected_questions),
            'total_questions': total_questions,
            'acceptance_rate': acceptance_rate
        }
    
    def _get_last_question(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the last follow-up question.
        
        Args:
            request: Request dictionary
            
        Returns:
            Response with last question
        """
        return {
            'success': True,
            'last_question': self.last_question
        } 