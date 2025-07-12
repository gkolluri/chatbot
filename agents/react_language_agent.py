"""
React AI Pattern-Based Language Agent for Multi-Agent Chatbot System
==================================================================

This agent handles language preferences and cultural context using React AI pattern, including:
- Language preference management
- Cultural context integration
- Greeting generation
- Language support information
- React AI pattern: Observe → Think → Act → Observe
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from agents.react_base_agent import ReactBaseAgent, ReactAgentState
from langchain_core.tools import tool


class ReactLanguageAgent(ReactBaseAgent):
    """
    React AI pattern-based agent responsible for handling language preferences and cultural context.
    
    Implements Observe-Think-Act loops for language management,
    cultural context integration, and greeting generation.
    """
    
    def __init__(self, db_interface=None):
        """Initialize the React AI language agent."""
        super().__init__("ReactLanguageAgent", db_interface)
        self.supported_languages = [
            "English", "Hindi", "Bengali", "Telugu", "Marathi", "Tamil", 
            "Gujarati", "Kannada", "Odia", "Punjabi", "Assamese", "Urdu", 
            "Malayalam", "Konkani", "Manipuri", "Nepali", "Bodo", "Santhali", 
            "Dogri", "Kashmiri"
        ]
        
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this agent provides."""
        return [
            "react_ai_language_management",
            "language_preference_handling",
            "cultural_context_integration",
            "greeting_generation",
            "language_support_information",
            "cultural_sensitivity",
            "react_ai_reasoning"
        ]
    
    def _get_agent_system_prompt(self) -> str:
        """Get the system prompt for React AI language agent."""
        return """You are a React AI language management agent designed to handle language preferences and cultural context.

Your role is to manage language preferences using the React AI pattern:
1. OBSERVE: Analyze user language preferences and cultural background
2. THINK: Reason about appropriate language and cultural context
3. ACT: Generate culturally appropriate responses and greetings
4. REFLECT: Learn from language preferences and improve cultural sensitivity

LANGUAGE MANAGEMENT GUIDELINES:
- Support 21 major Indian languages
- Provide culturally appropriate greetings
- Integrate cultural context naturally
- Respect language comfort levels
- Maintain professional and inclusive approach
- Balance traditional and contemporary elements

REACT AI PATTERN:
- Always observe user language preferences
- Think about cultural context and sensitivity
- Act by generating appropriate language responses
- Reflect on cultural integration effectiveness
"""
    
    def _get_agent_specific_tools(self) -> List:
        """Get language-specific tools for React AI pattern."""
        tools = []
        
        @tool
        def get_supported_languages() -> str:
            """Get list of supported languages with React AI reasoning."""
            try:
                result = f"Supported Languages ({len(self.supported_languages)}):\n"
                for i, lang in enumerate(self.supported_languages, 1):
                    result += f"{i}. {lang}\n"
                result += f"\nReact AI: Language support analysis with reasoning"
                return result
            except Exception as e:
                return f"Error getting supported languages: {str(e)}"
        
        @tool
        def generate_cultural_greeting(user_name: str, language_preferences: str) -> str:
            """Generate a culturally appropriate greeting with React AI reasoning."""
            try:
                prefs = eval(language_preferences) if isinstance(language_preferences, str) else language_preferences
                native_lang = prefs.get('native_language', 'English')
                comfort_level = prefs.get('language_comfort_level', 'english')
                
                greeting = f"Namaste {user_name}! 🙏\n"
                
                if native_lang != 'English' and comfort_level in ['mixed', 'native']:
                    if native_lang == 'Hindi':
                        greeting += "Aap kaise hain? (How are you?)\n"
                    elif native_lang == 'Bengali':
                        greeting += "Apni kemon achen? (How are you?)\n"
                    elif native_lang == 'Telugu':
                        greeting += "Meeru ela unnaru? (How are you?)\n"
                    elif native_lang == 'Tamil':
                        greeting += "Neenga eppadi irukkeenga? (How are you?)\n"
                    elif native_lang == 'Gujarati':
                        greeting += "Tamne kem cho? (How are you?)\n"
                    elif native_lang == 'Marathi':
                        greeting += "Tumhi kase aahat? (How are you?)\n"
                    else:
                        greeting += f"Welcome in {native_lang}!\n"
                
                greeting += f"Welcome to our Indian connection platform!\n"
                greeting += f"We're here to help you connect through shared interests.\n"
                greeting += f"- React AI: Cultural greeting with reasoning"
                
                return greeting
            except Exception as e:
                return f"Error generating greeting: {str(e)}"
        
        @tool
        def analyze_language_preferences(user_id: str) -> str:
            """Analyze user language preferences with React AI reasoning."""
            try:
                if self.db:
                    prefs = self.db.get_language_preferences(user_id)
                    
                    analysis = f"Language Preferences Analysis for user {user_id}:\n"
                    analysis += f"- Native Language: {prefs.get('native_language', 'Not specified')}\n"
                    analysis += f"- Preferred Languages: {', '.join(prefs.get('preferred_languages', []))}\n"
                    analysis += f"- Comfort Level: {prefs.get('language_comfort_level', 'english')}\n"
                    
                    # Cultural context analysis
                    if prefs.get('native_language') and prefs.get('native_language') != 'English':
                        analysis += f"- Cultural Context: User has Indian language background\n"
                        analysis += f"- Integration Level: {'High' if prefs.get('language_comfort_level') in ['mixed', 'native'] else 'Moderate'}\n"
                    else:
                        analysis += f"- Cultural Context: English-focused user\n"
                        analysis += f"- Integration Level: Basic cultural awareness\n"
                    
                    analysis += f"- React AI: Language preference analysis with reasoning"
                    
                    return analysis
                else:
                    return "Database not available for language preference analysis"
            except Exception as e:
                return f"Error analyzing language preferences: {str(e)}"
        
        @tool
        def suggest_language_improvements(user_id: str) -> str:
            """Suggest language improvements with React AI reasoning."""
            try:
                if self.db:
                    prefs = self.db.get_language_preferences(user_id)
                    
                    suggestions = f"Language Improvement Suggestions for user {user_id}:\n"
                    
                    if not prefs.get('native_language'):
                        suggestions += "- Consider adding your native language for better cultural integration\n"
                    
                    if len(prefs.get('preferred_languages', [])) < 2:
                        suggestions += "- Add more preferred languages for broader communication\n"
                    
                    if prefs.get('language_comfort_level') == 'english':
                        suggestions += "- Consider trying mixed language mode for cultural context\n"
                    
                    suggestions += f"- React AI: Improvement suggestions with reasoning"
                    
                    return suggestions
                else:
                    return "Database not available for language suggestions"
            except Exception as e:
                return f"Error suggesting improvements: {str(e)}"
        
        tools.extend([
            get_supported_languages,
            generate_cultural_greeting,
            analyze_language_preferences,
            suggest_language_improvements
        ])
        
        return tools
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process language-related requests using React AI pattern.
        
        Args:
            request: Request dictionary with parameters
            
        Returns:
            Response dictionary with results
        """
        request_type = request.get('type', '')
        
        if request_type == 'get_supported_languages':
            return self._get_supported_languages(request)
        elif request_type == 'generate_greeting':
            return self._generate_greeting(request)
        elif request_type == 'analyze_preferences':
            return self._analyze_preferences(request)
        elif request_type == 'suggest_improvements':
            return self._suggest_improvements(request)
        else:
            return {
                'success': False,
                'error': f'Unknown request type: {request_type}',
                'available_types': ['get_supported_languages', 'generate_greeting', 'analyze_preferences', 'suggest_improvements']
            }
    
    def _get_supported_languages(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get supported languages using React AI pattern.
        
        Args:
            request: Request dictionary
            
        Returns:
            Response with supported languages
        """
        # Use React AI pattern for language information
        react_request = {
            'user_id': 'system',
            'message': 'Get supported languages information',
            'type': 'get_supported_languages'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            result['languages_retrieved'] = True
            result['supported_languages'] = self.supported_languages
            result['framework'] = 'React AI Pattern'
        
        return result
    
    def _generate_greeting(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate greeting using React AI pattern.
        
        Args:
            request: Request with greeting parameters
            
        Returns:
            Response with greeting
        """
        user_name = request.get('user_name', 'User')
        language_preferences = request.get('language_preferences', {})
        
        # Use React AI pattern for greeting generation
        react_request = {
            'user_id': 'system',
            'message': f'Generate greeting for {user_name}',
            'user_name': user_name,
            'language_preferences': language_preferences,
            'type': 'generate_greeting'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            result['greeting_generated'] = True
            result['framework'] = 'React AI Pattern'
        
        return result
    
    def _analyze_preferences(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze language preferences using React AI pattern.
        
        Args:
            request: Request with user ID
            
        Returns:
            Response with preference analysis
        """
        user_id = request.get('user_id')
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        # Use React AI pattern for preference analysis
        react_request = {
            'user_id': user_id,
            'message': f'Analyze language preferences for user {user_id}',
            'type': 'analyze_preferences'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            result['preferences_analyzed'] = True
            result['framework'] = 'React AI Pattern'
        
        return result
    
    def _suggest_improvements(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest language improvements using React AI pattern.
        
        Args:
            request: Request with user ID
            
        Returns:
            Response with improvement suggestions
        """
        user_id = request.get('user_id')
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        # Use React AI pattern for improvement suggestions
        react_request = {
            'user_id': user_id,
            'message': f'Suggest language improvements for user {user_id}',
            'type': 'suggest_improvements'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            result['improvements_suggested'] = True
            result['framework'] = 'React AI Pattern'
        
        return result 