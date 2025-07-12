"""
Multi-Agent System for AI Chatbot Platform using LangGraph
=========================================================

This package contains specialized LangGraph-based agents that work together to provide
comprehensive chatbot functionality while maintaining the same UI and features
as the original single-agent system.

Agents:
- ConversationAgent: Handles individual chat conversations
- TagAnalysisAgent: Manages tag inference and suggestions
- UserProfileAgent: Handles user profiling and recommendations
- GroupChatAgent: Manages group chat functionality
- SessionAgent: Handles session management and persistence
- LanguageAgent: Manages language preferences and cultural context
"""

from .conversation_agent import ConversationAgent
from .tag_analysis_agent import TagAnalysisAgent
from .user_profile_agent import UserProfileAgent
from .group_chat_agent import GroupChatAgent
from .session_agent import SessionAgent
from .language_agent import LanguageAgent

__all__ = [
    'ConversationAgent',
    'TagAnalysisAgent', 
    'UserProfileAgent',
    'GroupChatAgent',
    'SessionAgent',
    'LanguageAgent'
] 