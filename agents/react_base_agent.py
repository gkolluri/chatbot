"""
React AI Pattern-Based Agent System for Multi-Agent Chatbot
==========================================================

This module provides the foundation for all React AI pattern-based agents in the
multi-agent chatbot system. Each agent uses the ReAct (Reasoning + Acting) loop
with Observe-Think-Act cycles for dynamic reasoning and tool usage.
"""

import os
import json
from typing import Dict, Any, List, Optional, TypedDict, Tuple
from datetime import datetime
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool, BaseTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class ReactAgentState(TypedDict):
    """State definition for React AI pattern agents"""
    user_id: str
    user_name: str
    message: str
    response: str
    conversation_history: List[Dict[str, str]]
    language_preferences: Dict[str, Any]
    tags: List[str]
    session_data: Dict[str, Any]
    agent_name: str
    timestamp: str
    metadata: Dict[str, Any]
    # React AI specific fields
    observations: List[str]
    thoughts: List[str]
    actions: List[Dict[str, Any]]
    tools_available: List[str]
    reasoning_chain: List[Dict[str, Any]]


class ReactBaseAgent:
    """
    Base class for all React AI pattern-based agents.
    
    Implements the ReAct loop: Observe → Think → Act → Observe
    with dynamic tool calling and reasoning capabilities.
    """
    
    def __init__(self, agent_name: str, db_interface=None):
        """
        Initialize the React AI pattern-based agent.
        
        Args:
            agent_name: Name of the agent for identification
            db_interface: Database interface for data persistence
        """
        self.agent_name = agent_name
        self.db = db_interface
        self.api_key = os.getenv('OPENAI_API_KEY')
        # Use latest GPT-4o model with web search capabilities
        self.llm = ChatOpenAI(
            api_key=self.api_key, 
            model="gpt-4o-mini",
            temperature=0.7
        ) if self.api_key else None
        
        # Web search enabled LLM for current information
        self.web_search_llm = ChatOpenAI(
            api_key=self.api_key,
            model="gpt-4o-mini",
            temperature=0.7
        ) if self.api_key else None
        
        # Agent state
        self.is_active = True
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # React AI specific state
        self.max_iterations = 5
        self.reasoning_history = []
        self.tool_usage_history = []
        
        # Initialize tools
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent_executor()
        
        # Web search tool for current information
        self.web_search_tool = {"type": "web_search_preview"}
        
    def _create_tools(self) -> List[BaseTool]:
        """
        Create tools available to this agent.
        
        Returns:
            List of available tools
        """
        tools = []
        
        @tool
        def observe_current_state(user_id: str) -> str:
            """Observe the current state of the system for a user."""
            try:
                # Get user data
                user_data = self.db.get_user_profile(user_id) if self.db else {}
                conversation_history = self.db.get_user_conversations(user_id) if self.db else []
                user_tags = self.db.get_user_tags(user_id) if self.db else []
                
                observation = f"Current state for user {user_id}:\n"
                observation += f"- User data: {user_data}\n"
                observation += f"- Conversation turns: {len(conversation_history)}\n"
                observation += f"- User tags: {user_tags}\n"
                observation += f"- Agent: {self.agent_name}\n"
                
                return observation
            except Exception as e:
                return f"Error observing state: {str(e)}"
        
        @tool
        def think_about_next_action(context: str, available_actions: str) -> str:
            """Think about what action to take next based on current context."""
            try:
                reasoning = f"Based on the context: {context}\n"
                reasoning += f"Available actions: {available_actions}\n"
                reasoning += f"I need to determine the best course of action for {self.agent_name}.\n"
                
                # Add agent-specific reasoning
                if self.agent_name == "ConversationAgent":
                    reasoning += "As a conversation agent, I should focus on generating helpful responses and maintaining engaging dialogue."
                elif self.agent_name == "TagAnalysisAgent":
                    reasoning += "As a tag analysis agent, I should analyze conversation content for interest patterns and suggest relevant tags."
                elif self.agent_name == "UserProfileAgent":
                    reasoning += "As a user profile agent, I should manage user preferences and profile data effectively."
                
                return reasoning
            except Exception as e:
                return f"Error in reasoning: {str(e)}"
        
        @tool
        def execute_action(action_type: str, action_data: str) -> str:
            """Execute a specific action based on the agent's capabilities."""
            try:
                action_result = f"Executing {action_type} with data: {action_data}\n"
                
                if action_type == "generate_response":
                    # Generate response using LLM
                    messages = [SystemMessage(content=self._get_agent_system_prompt())]
                    messages.append(HumanMessage(content=action_data))
                    response = self.llm.invoke(messages)
                    action_result += f"Generated response: {response.content}"
                    
                elif action_type == "analyze_conversation":
                    # Analyze conversation for patterns
                    action_result += f"Analyzed conversation for patterns and interests"
                    
                elif action_type == "update_profile":
                    # Update user profile
                    action_result += f"Updated user profile with new information"
                    
                elif action_type == "suggest_tags":
                    # Suggest tags based on analysis
                    action_result += f"Generated tag suggestions based on conversation analysis"
                    
                return action_result
            except Exception as e:
                return f"Error executing action: {str(e)}"
        
        @tool
        def reflect_on_actions(action_history: str) -> str:
            """Reflect on previous actions and their outcomes."""
            try:
                reflection = f"Reflecting on action history: {action_history}\n"
                reflection += f"Agent: {self.agent_name}\n"
                reflection += "Considering what worked well and what could be improved.\n"
                
                # Add to reasoning history
                self.reasoning_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'reflection': reflection,
                    'agent': self.agent_name
                })
                
                return reflection
            except Exception as e:
                return f"Error in reflection: {str(e)}"
        
        @tool
        def get_location_context(user_id: str) -> str:
            """Get location context for a user."""
            try:
                if not self.db:
                    return "No database connection available"
                
                location_prefs = self.db.get_location_preferences(user_id)
                if not location_prefs:
                    return "No location information available"
                
                context = f"Location context for user {user_id}:\n"
                context += f"- City: {location_prefs.get('city', 'Not set')}\n"
                context += f"- State: {location_prefs.get('state', 'Not set')}\n"
                context += f"- Country: {location_prefs.get('country', 'Not set')}\n"
                context += f"- Privacy Level: {location_prefs.get('privacy_level', 'private')}\n"
                
                return context
            except Exception as e:
                return f"Error getting location context: {str(e)}"
        
        @tool
        def get_cultural_context(user_id: str) -> str:
            """Get cultural context based on user's location."""
            try:
                if not self.db:
                    return "No database connection available"
                
                location_prefs = self.db.get_location_preferences(user_id)
                if not location_prefs:
                    return "No location information for cultural context"
                
                state = location_prefs.get('state', '').lower()
                city = location_prefs.get('city', '').lower()
                
                # Indian cultural context mapping
                cultural_context = f"Cultural context for user {user_id}:\n"
                
                if state:
                    # Add state-specific cultural context
                    state_cultures = {
                        'maharashtra': 'Marathi culture, Bollywood, Vada Pav, Ganesh Chaturthi',
                        'kerala': 'Malayalam culture, Kathakali, Backwaters, Onam',
                        'tamil nadu': 'Tamil culture, Bharatanatyam, Filter coffee, Pongal',
                        'karnataka': 'Kannada culture, Mysore Palace, Dosa, Dasara',
                        'gujarat': 'Gujarati culture, Garba, Dhokla, Navratri',
                        'rajasthan': 'Rajasthani culture, Folk music, Dal Baati, Teej',
                        'punjab': 'Punjabi culture, Bhangra, Butter chicken, Baisakhi',
                        'west bengal': 'Bengali culture, Durga Puja, Fish curry, Kali Puja'
                    }
                    
                    cultural_context += f"- State Culture: {state_cultures.get(state, 'Regional Indian culture')}\n"
                
                if city:
                    # Add city-specific context
                    city_cultures = {
                        'mumbai': 'Financial capital, Street food, Local trains, Bollywood',
                        'delhi': 'Capital city, Mughal architecture, Chaat, Political hub',
                        'bangalore': 'IT hub, Pub culture, Pleasant weather, Cosmopolitan',
                        'chennai': 'Auto industry, Classical music, Marina beach, Conservative',
                        'hyderabad': 'IT city, Biryani, Nizami culture, Charminar',
                        'pune': 'Educational hub, Pleasant climate, IT sector, Cultural city'
                    }
                    
                    cultural_context += f"- City Culture: {city_cultures.get(city, 'Local Indian culture')}\n"
                
                return cultural_context
            except Exception as e:
                return f"Error getting cultural context: {str(e)}"
        
        @tool
        def calculate_distance(user_id: str, other_user_id: str) -> str:
            """Calculate distance between two users."""
            try:
                if not self.db:
                    return "No database connection available"
                
                user1_location = self.db.get_location_preferences(user_id)
                user2_location = self.db.get_location_preferences(other_user_id)
                
                if not user1_location or not user2_location:
                    return "Location information not available for one or both users"
                
                coords1 = user1_location.get('coordinates')
                coords2 = user2_location.get('coordinates')
                
                if not coords1 or not coords2:
                    return "GPS coordinates not available for distance calculation"
                
                # Use database method for distance calculation
                distance = self.db._calculate_distance(
                    coords1.get('lat'), coords1.get('lng'),
                    coords2.get('lat'), coords2.get('lng')
                )
                
                return f"Distance between users: {distance:.2f} km"
            except Exception as e:
                return f"Error calculating distance: {str(e)}"
        
        tools.extend([observe_current_state, think_about_next_action, execute_action, reflect_on_actions, 
                     get_location_context, get_cultural_context, calculate_distance])
        
        # Add agent-specific tools
        agent_tools = self._get_agent_specific_tools()
        tools.extend(agent_tools)
        
        return tools
    
    def _search_web_reactive(self, query: str, context: str = None) -> str:
        """
        Search the web for current information using React AI pattern with web search.
        
        Args:
            query: Search query
            context: Additional context for the search
            
        Returns:
            Search results and analysis
        """
        if not self.web_search_llm:
            return "Web search not available - OpenAI API key required"
        
        try:
            # Use web search with React AI pattern
            llm_with_search = self.web_search_llm.bind_tools([self.web_search_tool])
            
            search_prompt = f"""
            Use the React AI pattern to search for current information:
            
            OBSERVE: Query: {query}
            {f"Context: {context}" if context else ""}
            
            THINK: What current information do I need to find about this topic?
            - Recent developments and trends
            - Up-to-date facts and statistics
            - Cultural context and regional relevance
            - Reliable sources and verification
            
            ACT: Search the web for comprehensive, current information
            
            OBSERVE: Analyze the search results and provide:
            1. Current, verified information
            2. Recent trends and developments
            3. Cultural context if applicable
            4. Source reliability assessment
            
            Provide a comprehensive response based on the latest information found.
            """
            
            response = llm_with_search.invoke(search_prompt)
            return response.content
            
        except Exception as e:
            return f"Web search error: {str(e)}"
    
    def _get_trending_topics(self, category: str = None) -> str:
        """
        Get current trending topics using web search.
        
        Args:
            category: Optional category to focus on
            
        Returns:
            Current trending topics
        """
        if category:
            query = f"trending topics {category} 2024 2025 current popular"
        else:
            query = "trending topics 2024 2025 current popular news"
        
        return self._search_web_reactive(query, f"Focus on current trends and popular topics")
    
    def _get_current_events(self, topic: str) -> str:
        """
        Get current events and news about a topic.
        
        Args:
            topic: Topic to get current events for
            
        Returns:
            Current events and news
        """
        query = f"current events news {topic} 2024 2025 latest updates"
        return self._search_web_reactive(query, f"Focus on recent news and events about {topic}")
    
    def _get_agent_specific_tools(self) -> List[BaseTool]:
        """
        Get agent-specific tools. Override in subclasses.
        
        Returns:
            List of agent-specific tools
        """
        return []
    
    def _create_agent_executor(self) -> AgentExecutor:
        """
        Create the agent executor with React AI pattern.
        
        Returns:
            AgentExecutor with React AI capabilities
        """
        if not self.llm:
            return None
        
        # Create the prompt template with React AI pattern
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_react_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        
        # Create the executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=self.max_iterations,
            return_intermediate_steps=True
        )
        
        return agent_executor
    
    def _get_react_system_prompt(self) -> str:
        """
        Get the React AI system prompt.
        
        Returns:
            System prompt for React AI pattern
        """
        base_prompt = f"""You are the {self.agent_name} in a multi-agent chatbot system.

You follow the React AI pattern: Observe → Think → Act → Observe

AVAILABLE TOOLS:
{self._get_tools_description()}

REACT AI PATTERN:
1. OBSERVE: Use observe_current_state to understand the current situation
2. THINK: Use think_about_next_action to reason about what to do next
3. ACT: Use execute_action to perform the chosen action
4. REFLECT: Use reflect_on_actions to learn from your actions

GUIDELINES:
- Always start by observing the current state
- Think step-by-step about what action to take
- Use available tools to accomplish your goals
- Reflect on your actions to improve future performance
- Be helpful, accurate, and culturally aware
- Focus on connecting users through shared interests

AGENT SPECIFIC ROLE: {self._get_agent_system_prompt()}
"""
        return base_prompt
    
    def _get_tools_description(self) -> str:
        """Get description of available tools."""
        tool_descriptions = []
        for tool in self.tools:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(tool_descriptions)
    
    def _get_agent_system_prompt(self) -> str:
        """
        Get the system prompt specific to this agent.
        
        Returns:
            System prompt string
        """
        return "You are a helpful AI assistant. Respond naturally and helpfully."
    
    def react_loop(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the React AI loop: Observe → Think → Act → Observe
        
        Args:
            request: Request dictionary with parameters
            
        Returns:
            Response dictionary with results
        """
        try:
            self.update_activity()
            
            # Initialize state
            state = self._request_to_state(request)
            
            # Build the input for the agent
            agent_input = self._build_agent_input(state)
            
            # Execute the React AI loop
            result = self.agent_executor.invoke({
                "input": agent_input,
                "chat_history": self._get_chat_history(state)
            })
            
            # Process the result
            response = self._process_react_result(result, state)
            
            # Log activity
            self.log_activity("Completed React AI loop", {
                'user_id': state.get('user_id'),
                'iterations': len(result.get('intermediate_steps', [])),
                'agent': self.agent_name
            })
            
            return response
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error in {self.agent_name} React AI loop: {str(e)}'
            }
    
    def _build_agent_input(self, state: ReactAgentState) -> str:
        """
        Build input for the React AI agent.
        
        Args:
            state: Current agent state
            
        Returns:
            Formatted input string
        """
        input_text = f"User message: {state.get('message', '')}\n"
        input_text += f"User ID: {state.get('user_id', '')}\n"
        input_text += f"Agent: {self.agent_name}\n"
        
        if state.get('language_preferences'):
            input_text += f"Language preferences: {state['language_preferences']}\n"
        
        return input_text
    
    def _get_chat_history(self, state: ReactAgentState) -> List:
        """
        Get chat history for the agent.
        
        Args:
            state: Current agent state
            
        Returns:
            List of chat messages
        """
        history = []
        if state.get("conversation_history"):
            for msg in state["conversation_history"][-5:]:  # Last 5 messages
                if isinstance(msg, tuple):
                    role, content = msg
                else:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                
                if role == "user":
                    history.append(HumanMessage(content=content))
                else:
                    history.append(AIMessage(content=content))
        
        return history
    
    def _process_react_result(self, result: Dict[str, Any], state: ReactAgentState) -> Dict[str, Any]:
        """
        Process the result from React AI loop.
        
        Args:
            result: Result from agent executor
            state: Original state
            
        Returns:
            Processed response
        """
        # Extract the final response
        response_text = result.get('output', 'I apologize, but I am having trouble responding right now.')
        
        # Extract reasoning chain from intermediate steps
        reasoning_chain = []
        for step in result.get('intermediate_steps', []):
            action, observation = step
            reasoning_chain.append({
                'action': action.tool,
                'action_input': action.tool_input,
                'observation': observation
            })
        
        # Update state with React AI specific data
        state['observations'] = [step['observation'] for step in reasoning_chain]
        state['actions'] = [{'tool': step['action'], 'input': step['action_input']} for step in reasoning_chain]
        state['reasoning_chain'] = reasoning_chain
        
        return {
            'success': True,
            'response': response_text,
            'reasoning_chain': reasoning_chain,
            'agent_name': self.agent_name,
            'iterations': len(reasoning_chain),
            'metadata': {
                'agent_name': self.agent_name,
                'processing_time': datetime.now().isoformat(),
                'framework': 'React AI Pattern',
                'max_iterations': self.max_iterations
            }
        }
    
    def _request_to_state(self, request: Dict[str, Any]) -> ReactAgentState:
        """
        Convert request to React AI state.
        
        Args:
            request: Request dictionary
            
        Returns:
            ReactAgentState
        """
        return ReactAgentState(
            user_id=request.get('user_id', ''),
            user_name=request.get('user_name', ''),
            message=request.get('message', ''),
            response='',
            conversation_history=request.get('conversation_history', []),
            language_preferences=request.get('language_preferences', {}),
            tags=request.get('tags', []),
            session_data=request.get('session_data', {}),
            agent_name=self.agent_name,
            timestamp=datetime.now().isoformat(),
            metadata={},
            observations=[],
            thoughts=[],
            actions=[],
            tools_available=[tool.name for tool in self.tools],
            reasoning_chain=[]
        )
    
    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity = datetime.now()
        
    def get_status(self) -> Dict[str, Any]:
        """Get agent status information."""
        return {
            'agent_name': self.agent_name,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'has_llm': self.llm is not None,
            'framework': 'React AI Pattern',
            'tools_count': len(self.tools),
            'max_iterations': self.max_iterations,
            'reasoning_history_count': len(self.reasoning_history)
        }
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request using React AI pattern.
        
        Args:
            request: Request dictionary with parameters
            
        Returns:
            Response dictionary with results
        """
        return self.react_loop(request)
    
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this agent provides."""
        return [
            "react_ai_pattern",
            "dynamic_tool_calling",
            "reasoning_and_reflection",
            "observe_think_act_loop",
            "adaptive_behavior"
        ]
    
    def validate_request(self, request: Dict[str, Any]) -> bool:
        """
        Validate the incoming request.
        
        Args:
            request: Request to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['user_id', 'message']
        return all(field in request for field in required_fields)
    
    def log_activity(self, activity: str, details: Dict[str, Any] = None):
        """
        Log agent activity.
        
        Args:
            activity: Activity description
            details: Additional details
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent': self.agent_name,
            'activity': activity,
            'details': details or {}
        }
        
        print(f"[{self.agent_name}] {activity}")
        if details:
            print(f"Details: {details}")
    
    def cleanup(self):
        """Clean up agent resources."""
        self.is_active = False
        self.reasoning_history.clear()
        self.tool_usage_history.clear()
        print(f"[{self.agent_name}] Agent deactivated")


class ReactAgentCoordinator:
    """
    Coordinator for React AI pattern-based agents.
    
    Manages agent registration, request routing, and system-wide
    React AI pattern coordination.
    """
    
    def __init__(self):
        """Initialize the React AI agent coordinator."""
        self.agents = {}
        self.request_history = []
        self.system_reasoning = []
        
    def register_agent(self, agent: ReactBaseAgent):
        """
        Register a React AI agent.
        
        Args:
            agent: React AI agent to register
        """
        self.agents[agent.agent_name] = agent
        print(f"Registered React AI agent: {agent.agent_name}")
    
    def get_agent(self, agent_name: str) -> Optional[ReactBaseAgent]:
        """
        Get a registered agent by name.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent if found, None otherwise
        """
        return self.agents.get(agent_name)
    
    def route_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a request to the appropriate React AI agent.
        
        Args:
            request: Request to route
            
        Returns:
            Response from the agent
        """
        request_type = request.get('type', '')
        
        # Map request types to agents
        routing_map = {
            'send_message': 'ReactConversationAgent',
            'analyze_conversation': 'ReactTagAnalysisAgent',
            'get_tag_suggestions': 'ReactTagAnalysisAgent',
            'suggest_location_tags': 'ReactTagAnalysisAgent',
            'create_profile': 'ReactUserProfileAgent',
            'find_similar_users': 'ReactUserProfileAgent',
            'find_similar_users_with_location': 'ReactUserProfileAgent',
            'update_location': 'ReactUserProfileAgent',
            'find_nearby_users': 'ReactUserProfileAgent',
            'find_users_in_city': 'ReactUserProfileAgent',
            'rag_nearby_users': 'ReactRAGNearbyUsersAgent',
            'semantic_search_users': 'ReactRAGNearbyUsersAgent',
            'hybrid_user_search': 'ReactRAGNearbyUsersAgent',
            'vectorize_user_profile': 'ReactRAGNearbyUsersAgent',
            'create_group_chat': 'ReactGroupChatAgent',
            'send_group_message': 'ReactGroupChatAgent',
            'get_group_messages': 'ReactGroupChatAgent',
            'create_session': 'ReactSessionAgent',
            'validate_session': 'ReactSessionAgent',
            'get_supported_languages': 'ReactLanguageAgent',
            'generate_greeting': 'ReactLanguageAgent'
        }
        
        target_agent = routing_map.get(request_type)
        
        if not target_agent:
            return {
                'success': False,
                'error': f'Unknown request type: {request_type}',
                'available_types': list(routing_map.keys())
            }
        
        agent = self.get_agent(target_agent)
        if not agent:
            return {
                'success': False,
                'error': f'Agent not found: {target_agent}'
            }
        
        # Add system reasoning before routing
        system_reasoning = self._generate_system_reasoning(request, target_agent)
        self.system_reasoning.append(system_reasoning)
        
        # Route to agent
        try:
            response = agent.process_request(request)
            
            # Log request
            self.request_history.append({
                'timestamp': datetime.now().isoformat(),
                'request_type': request_type,
                'target_agent': target_agent,
                'success': response.get('success', False)
            })
            
            return response
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error in {target_agent}: {str(e)}'
            }
    
    def _generate_system_reasoning(self, request: Dict[str, Any], target_agent: str) -> Dict[str, Any]:
        """
        Generate system-level reasoning about request routing.
        
        Args:
            request: The request being routed
            target_agent: Target agent name
            
        Returns:
            Reasoning information
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'request_type': request.get('type', ''),
            'target_agent': target_agent,
            'reasoning': f"Routing {request.get('type', '')} request to {target_agent} based on request type mapping",
            'available_agents': list(self.agents.keys())
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information."""
        agent_statuses = {}
        for name, agent in self.agents.items():
            agent_statuses[name] = agent.get_status()
        
        return {
            'total_agents': len(self.agents),
            'agent_statuses': agent_statuses,
            'framework': 'React AI Pattern',
            'request_history_count': len(self.request_history),
            'system_reasoning_count': len(self.system_reasoning),
            'active_agents': [name for name, agent in self.agents.items() if agent.is_active]
        }
    
    def cleanup_all(self):
        """Clean up all agents."""
        for agent in self.agents.values():
            agent.cleanup()
        self.agents.clear()
        self.request_history.clear()
        self.system_reasoning.clear()
        print("All React AI agents cleaned up") 