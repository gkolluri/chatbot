"""
LangGraph-Based Agent System for Multi-Agent Chatbot
===================================================

This module provides the foundation for all LangGraph-based agents in the
multi-agent chatbot system. Each agent uses LangGraph's state management
and routing capabilities for better coordination and scalability.
"""

import os
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool


class AgentState(TypedDict):
    """State definition for LangGraph agents"""
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


class LangGraphBaseAgent:
    """
    Base class for all LangGraph-based agents.
    
    Provides common functionality like OpenAI integration, state management,
    and basic agent lifecycle management using LangGraph.
    """
    
    def __init__(self, agent_name: str, db_interface=None):
        """
        Initialize the LangGraph-based agent.
        
        Args:
            agent_name: Name of the agent for identification
            db_interface: Database interface for data persistence
        """
        self.agent_name = agent_name
        self.db = db_interface
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.llm = ChatOpenAI(api_key=self.api_key, model="gpt-3.5-turbo") if self.api_key else None
        
        # Agent state
        self.is_active = True
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()
        
    def _create_workflow(self) -> StateGraph:
        """
        Create the LangGraph workflow for this agent.
        
        Returns:
            StateGraph workflow
        """
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("process_request", self._process_request_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("update_state", self._update_state_node)
        
        # Define edges
        workflow.set_entry_point("process_request")
        workflow.add_edge("process_request", "generate_response")
        workflow.add_edge("generate_response", "update_state")
        workflow.add_edge("update_state", END)
        
        # Compile the workflow
        return workflow.compile()
    
    def _process_request_node(self, state: AgentState) -> AgentState:
        """
        Process the incoming request and prepare state.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        self.update_activity()
        
        # Add timestamp
        state["timestamp"] = datetime.now().isoformat()
        state["agent_name"] = self.agent_name
        
        # Log activity
        self.log_activity("Processing request", {
            'user_id': state.get('user_id'),
            'message_length': len(state.get('message', '')),
            'agent': self.agent_name
        })
        
        return state
    
    def _generate_response_node(self, state: AgentState) -> AgentState:
        """
        Generate response using the LLM.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with response
        """
        if not self.llm:
            state["response"] = "I'm having trouble responding right now. Please check your API configuration."
            return state
        
        try:
            # Build messages for the LLM
            messages = self._build_messages(state)
            
            # Generate response
            response = self.llm.invoke(messages)
            
            state["response"] = response.content
            
        except Exception as e:
            print(f"Error in {self.agent_name} LLM call: {e}")
            state["response"] = "I'm having trouble responding right now. Please try again."
        
        return state
    
    def _update_state_node(self, state: AgentState) -> AgentState:
        """
        Update the state with final processing.
        
        Args:
            state: Current agent state
            
        Returns:
            Final state
        """
        # Update last activity
        self.last_activity = datetime.now()
        
        # Add metadata
        state["metadata"] = {
            "agent_name": self.agent_name,
            "processing_time": datetime.now().isoformat(),
            "success": True
        }
        
        return state
    
    def _build_messages(self, state: AgentState) -> List:
        """
        Build messages for the LLM based on agent type.
        
        Args:
            state: Current agent state
            
        Returns:
            List of messages for the LLM
        """
        # Base system message
        system_message = f"You are the {self.agent_name} agent. "
        system_message += self._get_agent_system_prompt()
        
        messages = [SystemMessage(content=system_message)]
        
        # Add conversation history if available
        if state.get("conversation_history"):
            for msg in state["conversation_history"][-5:]:  # Last 5 messages
                # Handle both tuple format (role, message) and dict format (role, content)
                if isinstance(msg, tuple):
                    role, content = msg
                else:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                
                if role == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))
        
        # Add current message
        if state.get("message"):
            messages.append(HumanMessage(content=state["message"]))
        
        return messages
    
    def _get_agent_system_prompt(self) -> str:
        """
        Get the system prompt specific to this agent.
        
        Returns:
            System prompt string
        """
        return "You are a helpful AI assistant. Respond naturally and helpfully."
    
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
            'framework': 'LangGraph'
        }
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request using LangGraph workflow.
        
        Args:
            request: Request dictionary with parameters
            
        Returns:
            Response dictionary with results
        """
        # Convert request to state
        state = self._request_to_state(request)
        
        # Run the workflow
        try:
            result = self.workflow.invoke(state)
            return self._state_to_response(result)
        except Exception as e:
            return {
                'success': False,
                'error': f'Error in {self.agent_name}: {str(e)}'
            }
    
    def _request_to_state(self, request: Dict[str, Any]) -> AgentState:
        """
        Convert request dictionary to AgentState.
        
        Args:
            request: Request dictionary
            
        Returns:
            AgentState object
        """
        return AgentState(
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
            metadata={}
        )
    
    def _state_to_response(self, state: AgentState) -> Dict[str, Any]:
        """
        Convert AgentState to response dictionary.
        
        Args:
            state: AgentState object
            
        Returns:
            Response dictionary
        """
        return {
            'success': state.get('metadata', {}).get('success', True),
            'response': state.get('response', ''),
            'agent_name': self.agent_name,
            'timestamp': state.get('timestamp'),
            'metadata': state.get('metadata', {})
        }
    
    def get_capabilities(self) -> List[str]:
        """
        Get list of capabilities this agent provides.
        
        Returns:
            List of capability strings
        """
        return [
            "langgraph_workflow",
            "state_management",
            "llm_integration",
            "error_handling"
        ]
    
    def validate_request(self, request: Dict[str, Any]) -> bool:
        """
        Validate incoming request format.
        
        Args:
            request: Request to validate
            
        Returns:
            True if valid, False otherwise
        """
        return isinstance(request, dict) and 'type' in request
    
    def log_activity(self, activity: str, details: Dict[str, Any] = None):
        """
        Log agent activity for monitoring.
        
        Args:
            activity: Activity description
            details: Additional details
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent': self.agent_name,
            'activity': activity,
            'details': details or {},
            'framework': 'LangGraph'
        }
        print(f"[{self.agent_name}] {activity}")
        
        # Could be extended to write to database or file
        if self.db and hasattr(self.db, 'log_agent_activity'):
            self.db.log_agent_activity(log_entry)
    
    def cleanup(self):
        """Cleanup resources when agent is deactivated."""
        self.is_active = False
        self.log_activity("Agent deactivated")


class LangGraphAgentCoordinator:
    """
    Coordinates multiple LangGraph agents and manages their interactions.
    
    This class acts as the central orchestrator for the multi-agent system,
    routing requests to appropriate agents and managing agent lifecycle.
    """
    
    def __init__(self):
        """Initialize the agent coordinator."""
        self.agents = {}
        self.request_history = []
        
    def register_agent(self, agent: LangGraphBaseAgent):
        """
        Register an agent with the coordinator.
        
        Args:
            agent: Agent instance to register
        """
        self.agents[agent.agent_name] = agent
        print(f"Registered LangGraph agent: {agent.agent_name}")
        
    def get_agent(self, agent_name: str) -> Optional[LangGraphBaseAgent]:
        """
        Get a specific agent by name.
        
        Args:
            agent_name: Name of the agent to retrieve
            
        Returns:
            Agent instance or None if not found
        """
        return self.agents.get(agent_name)
    
    def route_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a request to the appropriate agent.
        
        Args:
            request: Request to route
            
        Returns:
            Response from the appropriate agent
        """
        request_type = request.get('type', '')
        target_agent = request.get('agent', '')
        
        # Route based on request type or target agent
        if target_agent and target_agent in self.agents:
            agent = self.agents[target_agent]
            if agent.validate_request(request):
                return agent.process_request(request)
        
        # Fallback routing based on request type
        agent_mapping = {
            'send_message': 'ConversationAgent',
            'analyze_conversation': 'TagAnalysisAgent',
            'get_tag_suggestions': 'TagAnalysisAgent',
            'create_profile': 'UserProfileAgent',
            'find_similar_users': 'UserProfileAgent',
            'get_similar_users': 'UserProfileAgent',
            'create_group': 'GroupChatAgent',
            'create_group_chat': 'GroupChatAgent',
            'send_group_message': 'GroupChatAgent',
            'get_group_messages': 'GroupChatAgent',
            'get_group_chat': 'GroupChatAgent',
            'get_user_groups': 'GroupChatAgent',
            'suggest_group_topics': 'GroupChatAgent',
            'get_user_profile': 'UserProfileAgent',
            'create_session': 'SessionAgent',
            'validate_session': 'SessionAgent',
            'get_supported_languages': 'LanguageAgent',
            'generate_greeting': 'LanguageAgent',
            'conversation': 'ConversationAgent',
            'tag_analysis': 'TagAnalysisAgent',
            'user_profile': 'UserProfileAgent',
            'group_chat': 'GroupChatAgent',
            'session': 'SessionAgent',
            'language': 'LanguageAgent'
        }
        
        target_agent_name = agent_mapping.get(request_type)
        if target_agent_name and target_agent_name in self.agents:
            agent = self.agents[target_agent_name]
            if agent.validate_request(request):
                return agent.process_request(request)
        
        # Default response if no agent can handle the request
        return {
            'success': False,
            'error': f'No agent found to handle request type: {request_type}',
            'available_agents': list(self.agents.keys())
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get status of all agents in the system.
        
        Returns:
            Dictionary with status of all agents
        """
        status = {
            'total_agents': len(self.agents),
            'active_agents': sum(1 for agent in self.agents.values() if agent.is_active),
            'framework': 'LangGraph',
            'agents': {}
        }
        
        for name, agent in self.agents.items():
            status['agents'][name] = agent.get_status()
            
        return status
    
    def cleanup_all(self):
        """Cleanup all registered agents."""
        for agent in self.agents.values():
            agent.cleanup()
        print("All LangGraph agents cleaned up") 