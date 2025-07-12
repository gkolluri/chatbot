"""
Language Agent for Multi-Agent Chatbot System using LangGraph
===========================================================

This agent handles language preferences and cultural context using LangGraph workflows, including:
- Native language selection (21 Indian languages)
- Preferred languages multi-selection
- Language comfort level settings (English Only, Mixed, Native Preferred)
- Subtle cultural greeting personalization
- Language-aware AI responses that feel natural
- Bilingual interface elements where appropriate
- Regional language support with proper transliteration
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from .base_agent import LangGraphBaseAgent, AgentState


class LanguageAgent(LangGraphBaseAgent):
    """
    Agent responsible for language preferences and cultural context using LangGraph.
    
    Manages user language settings, provides cultural context,
    and ensures appropriate language-aware interactions.
    """
    
    def __init__(self, db_interface=None):
        """Initialize the language agent."""
        super().__init__("LanguageAgent", db_interface)
        self.supported_languages = self._initialize_supported_languages()
        self.cultural_greetings = self._initialize_cultural_greetings()
        
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this agent provides."""
        return [
            "language_preference_management",
            "cultural_context_analysis",
            "greeting_personalization",
            "bilingual_support",
            "regional_language_handling",
            "language_aware_responses",
            "cultural_sensitivity",
            "langgraph_workflow"
        ]
    
    def _get_agent_system_prompt(self) -> str:
        """Get the system prompt for language agent."""
        return """You are an AI agent specialized in language preferences and cultural context.
        Your role is to:
        1. Manage user language preferences and settings
        2. Provide cultural context for interactions
        3. Generate appropriate greetings and responses
        4. Ensure language-aware communication
        5. Handle regional language variations
        6. Maintain cultural sensitivity
        
        Guidelines:
        - Respect diverse language preferences
        - Provide subtle cultural context
        - Ensure natural language flow
        - Consider regional variations
        - Maintain professional approach
        """
    
    def _initialize_supported_languages(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize supported Indian languages with metadata.
        
        Returns:
            Dictionary of language codes to language metadata
        """
        return {
            'hindi': {
                'name': 'Hindi',
                'native_name': 'हिंदी',
                'region': 'North India',
                'script': 'Devanagari',
                'greeting': 'नमस्ते',
                'cultural_context': 'Most widely spoken Indian language'
            },
            'english': {
                'name': 'English',
                'native_name': 'English',
                'region': 'International',
                'script': 'Latin',
                'greeting': 'Hello',
                'cultural_context': 'Primary language for business and education'
            },
            'bengali': {
                'name': 'Bengali',
                'native_name': 'বাংলা',
                'region': 'West Bengal, Bangladesh',
                'script': 'Bengali',
                'greeting': 'নমস্কার',
                'cultural_context': 'Rich literary and cultural tradition'
            },
            'telugu': {
                'name': 'Telugu',
                'native_name': 'తెలుగు',
                'region': 'Andhra Pradesh, Telangana',
                'script': 'Telugu',
                'greeting': 'నమస్కారం',
                'cultural_context': 'Classical language with ancient roots'
            },
            'marathi': {
                'name': 'Marathi',
                'native_name': 'मराठी',
                'region': 'Maharashtra',
                'script': 'Devanagari',
                'greeting': 'नमस्कार',
                'cultural_context': 'Rich cultural and literary heritage'
            },
            'tamil': {
                'name': 'Tamil',
                'native_name': 'தமிழ்',
                'region': 'Tamil Nadu',
                'script': 'Tamil',
                'greeting': 'வணக்கம்',
                'cultural_context': 'One of the oldest living languages'
            },
            'gujarati': {
                'name': 'Gujarati',
                'native_name': 'ગુજરાતી',
                'region': 'Gujarat',
                'script': 'Gujarati',
                'greeting': 'નમસ્તે',
                'cultural_context': 'Business and trade language'
            },
            'kannada': {
                'name': 'Kannada',
                'native_name': 'ಕನ್ನಡ',
                'region': 'Karnataka',
                'script': 'Kannada',
                'greeting': 'ನಮಸ್ಕಾರ',
                'cultural_context': 'Classical language with rich literature'
            },
            'odia': {
                'name': 'Odia',
                'native_name': 'ଓଡ଼ିଆ',
                'region': 'Odisha',
                'script': 'Odia',
                'greeting': 'ନମସ୍କାର',
                'cultural_context': 'Ancient language with cultural significance'
            },
            'punjabi': {
                'name': 'Punjabi',
                'native_name': 'ਪੰਜਾਬੀ',
                'region': 'Punjab',
                'script': 'Gurmukhi',
                'greeting': 'ਸਤ ਸ੍ਰੀ ਅਕਾਲ',
                'cultural_context': 'Language of the Sikh community'
            },
            'assamese': {
                'name': 'Assamese',
                'native_name': 'অসমীয়া',
                'region': 'Assam',
                'script': 'Assamese',
                'greeting': 'নমস্কাৰ',
                'cultural_context': 'Northeastern cultural heritage'
            },
            'sanskrit': {
                'name': 'Sanskrit',
                'native_name': 'संस्कृतम्',
                'region': 'Classical',
                'script': 'Devanagari',
                'greeting': 'नमः',
                'cultural_context': 'Ancient classical language'
            },
            'urdu': {
                'name': 'Urdu',
                'native_name': 'اردو',
                'region': 'North India, Pakistan',
                'script': 'Perso-Arabic',
                'greeting': 'السلام علیکم',
                'cultural_context': 'Poetic and literary language'
            },
            'malayalam': {
                'name': 'Malayalam',
                'native_name': 'മലയാളം',
                'region': 'Kerala',
                'script': 'Malayalam',
                'greeting': 'നമസ്കാരം',
                'cultural_context': 'Dravidian language with unique script'
            },
            'konkani': {
                'name': 'Konkani',
                'native_name': 'कोंकणी',
                'region': 'Goa, Konkan',
                'script': 'Devanagari',
                'greeting': 'नमस्कार',
                'cultural_context': 'Coastal cultural language'
            },
            'manipuri': {
                'name': 'Manipuri',
                'native_name': 'মৈতৈলোন্',
                'region': 'Manipur',
                'script': 'Meitei Mayek',
                'greeting': 'ꯍꯥꯌ',
                'cultural_context': 'Northeastern cultural language'
            },
            'nepali': {
                'name': 'Nepali',
                'native_name': 'नेपाली',
                'region': 'Nepal, Sikkim',
                'script': 'Devanagari',
                'greeting': 'नमस्ते',
                'cultural_context': 'Himalayan cultural language'
            },
            'bodo': {
                'name': 'Bodo',
                'native_name': 'बड़ो',
                'region': 'Assam',
                'script': 'Devanagari',
                'greeting': 'नमस्कार',
                'cultural_context': 'Northeastern tribal language'
            },
            'santhali': {
                'name': 'Santhali',
                'native_name': 'ᱥᱟᱱᱛᱟᱲᱤ',
                'region': 'Jharkhand, West Bengal',
                'script': 'Ol Chiki',
                'greeting': 'ᱡᱷᱚᱞᱟᱠ',
                'cultural_context': 'Tribal cultural language'
            },
            'dogri': {
                'name': 'Dogri',
                'native_name': 'डोगरी',
                'region': 'Jammu and Kashmir',
                'script': 'Devanagari',
                'greeting': 'नमस्कार',
                'cultural_context': 'Himalayan cultural language'
            },
            'kashmiri': {
                'name': 'Kashmiri',
                'native_name': 'कॉशुर',
                'region': 'Jammu and Kashmir',
                'script': 'Devanagari',
                'greeting': 'आशीर्वाद',
                'cultural_context': 'Kashmiri cultural heritage'
            }
        }
    
    def _initialize_cultural_greetings(self) -> Dict[str, List[str]]:
        """
        Initialize cultural greetings for different contexts.
        
        Returns:
            Dictionary of greeting types to greeting lists
        """
        return {
            'formal': [
                'Namaste! Welcome to our community.',
                'Greetings! We\'re glad to have you here.',
                'Welcome! Let\'s connect through shared interests.',
                'Hello! Ready to explore and connect?',
                'Namaskar! Join us in meaningful conversations.'
            ],
            'casual': [
                'Hey there! What\'s on your mind?',
                'Hi! Ready to chat and connect?',
                'Hello! What interests you today?',
                'Hey! Let\'s talk about what matters to you.',
                'Hi there! What would you like to explore?'
            ],
            'cultural': [
                'Namaste! Let\'s discover common ground together.',
                'Greetings! May we find shared interests.',
                'Welcome! Let\'s bridge cultures through conversation.',
                'Namaskar! Ready to connect across boundaries?',
                'Hello! Let\'s celebrate our diverse interests.'
            ],
            'professional': [
                'Welcome! How can I assist you today?',
                'Greetings! I\'m here to help you connect.',
                'Hello! Let\'s explore your interests together.',
                'Welcome! Ready to discover meaningful connections?',
                'Greetings! How may I help you today?'
            ]
        }
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process language-related requests using LangGraph workflow.
        
        Args:
            request: Request dictionary with parameters
            
        Returns:
            Response dictionary with results
        """
        request_type = request.get('type', '')
        
        if request_type == 'get_supported_languages':
            return self._get_supported_languages(request)
        elif request_type == 'validate_language_preferences':
            return self._validate_language_preferences(request)
        elif request_type == 'generate_greeting':
            return self._generate_personalized_greeting(request)
        elif request_type == 'get_cultural_context':
            return self._get_cultural_context(request)
        elif request_type == 'translate_phrase':
            return self._translate_phrase(request)
        elif request_type == 'get_language_stats':
            return self._get_language_statistics(request)
        else:
            return {
                'success': False,
                'error': f'Unknown request type: {request_type}',
                'available_types': ['get_supported_languages', 'validate_language_preferences', 'generate_greeting', 'get_cultural_context', 'translate_phrase', 'get_language_stats']
            }
    
    def _get_supported_languages(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get list of supported languages with metadata.
        
        Args:
            request: Request parameters
            
        Returns:
            Response with supported languages
        """
        return {
            'success': True,
            'supported_languages': self.supported_languages,
            'total_languages': len(self.supported_languages),
            'comfort_levels': [
                'english_only',
                'mixed_language',
                'native_preferred'
            ]
        }
    
    def _validate_language_preferences(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate user language preferences.
        
        Args:
            request: Request with language preferences
            
        Returns:
            Response with validation results
        """
        native_language = request.get('native_language')
        preferred_languages = request.get('preferred_languages', [])
        comfort_level = request.get('language_comfort_level', 'english')
        
        validation_errors = []
        
        # Validate native language
        if native_language and native_language not in self.supported_languages:
            validation_errors.append(f'Native language "{native_language}" is not supported')
        
        # Validate preferred languages
        for lang in preferred_languages:
            if lang not in self.supported_languages:
                validation_errors.append(f'Preferred language "{lang}" is not supported')
        
        # Validate comfort level
        valid_comfort_levels = ['english_only', 'mixed_language', 'native_preferred']
        if comfort_level not in valid_comfort_levels:
            validation_errors.append(f'Comfort level "{comfort_level}" is not valid')
        
        is_valid = len(validation_errors) == 0
        
        return {
            'success': True,
            'is_valid': is_valid,
            'errors': validation_errors,
            'validated_preferences': {
                'native_language': native_language,
                'preferred_languages': preferred_languages,
                'comfort_level': comfort_level
            }
        }
    
    def _generate_personalized_greeting(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a personalized greeting based on language preferences.
        
        Args:
            request: Request with user context
            
        Returns:
            Response with personalized greeting
        """
        user_name = request.get('user_name', '')
        language_preferences = request.get('language_preferences', {})
        greeting_type = request.get('greeting_type', 'formal')
        time_of_day = request.get('time_of_day', 'general')
        
        # Build greeting prompt
        prompt = f"""Generate a personalized greeting for a user with the following context:
        
        User Name: {user_name}
        Language Preferences: {language_preferences}
        Greeting Type: {greeting_type}
        Time of Day: {time_of_day}
        
        Guidelines:
        - Be warm and welcoming
        - Consider cultural context if relevant
        - Keep it natural and conversational
        - Focus on connection and shared interests
        - Be respectful of language preferences
        
        Generate a greeting that feels personal and appropriate."""
        
        # Use LangGraph workflow for greeting generation
        state = self._request_to_state({
            'user_id': 'greeting',
            'message': prompt,
            'language_preferences': language_preferences
        })
        
        result = self.workflow.invoke(state)
        greeting = result.get('response', 'Hello! Welcome to our community.')
        
        # Add cultural context if relevant
        cultural_context = self._get_cultural_context_for_greeting(language_preferences)
        
        return {
            'success': True,
            'greeting': greeting,
            'cultural_context': cultural_context,
            'greeting_type': greeting_type,
            'personalized': True
        }
    
    def _get_cultural_context_for_greeting(self, language_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get cultural context for greeting generation.
        
        Args:
            language_preferences: User's language preferences
            
        Returns:
            Cultural context dictionary
        """
        native_lang = language_preferences.get('native_language')
        comfort_level = language_preferences.get('language_comfort_level', 'english')
        
        context = {
            'has_cultural_context': False,
            'native_language_info': None,
            'greeting_style': 'standard'
        }
        
        if native_lang and native_lang in self.supported_languages:
            lang_info = self.supported_languages[native_lang]
            context['has_cultural_context'] = True
            context['native_language_info'] = lang_info
            
            if comfort_level == 'native_preferred':
                context['greeting_style'] = 'cultural'
            elif comfort_level == 'mixed_language':
                context['greeting_style'] = 'bilingual'
            else:
                context['greeting_style'] = 'english_with_context'
        
        return context
    
    def _get_cultural_context(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get cultural context for a user.
        
        Args:
            request: Request with user context
            
        Returns:
            Response with cultural context
        """
        language_preferences = request.get('language_preferences', {})
        user_tags = request.get('tags', [])
        
        native_lang = language_preferences.get('native_language')
        comfort_level = language_preferences.get('language_comfort_level', 'english')
        
        cultural_context = {
            'native_language': native_lang,
            'language_info': self.supported_languages.get(native_lang, {}) if native_lang else {},
            'comfort_level': comfort_level,
            'cultural_tags': [],
            'regional_context': None,
            'greeting_preferences': self._get_greeting_preferences(comfort_level)
        }
        
        # Identify cultural tags
        cultural_keywords = [
            'indian', 'hindi', 'bengali', 'telugu', 'marathi', 'tamil', 'gujarati',
            'kannada', 'odia', 'punjabi', 'assamese', 'sanskrit', 'urdu', 'malayalam',
            'classical', 'traditional', 'folk', 'heritage', 'culture', 'regional'
        ]
        
        for tag in user_tags:
            if any(keyword in tag.lower() for keyword in cultural_keywords):
                cultural_context['cultural_tags'].append(tag)
        
        # Add regional context
        if native_lang and native_lang in self.supported_languages:
            lang_info = self.supported_languages[native_lang]
            cultural_context['regional_context'] = {
                'region': lang_info.get('region'),
                'script': lang_info.get('script'),
                'cultural_significance': lang_info.get('cultural_context')
            }
        
        return {
            'success': True,
            'cultural_context': cultural_context
        }
    
    def _get_greeting_preferences(self, comfort_level: str) -> Dict[str, Any]:
        """
        Get greeting preferences based on comfort level.
        
        Args:
            comfort_level: User's language comfort level
            
        Returns:
            Greeting preferences dictionary
        """
        if comfort_level == 'english_only':
            return {
                'primary_style': 'english',
                'cultural_elements': 'minimal',
                'bilingual_support': False
            }
        elif comfort_level == 'mixed_language':
            return {
                'primary_style': 'english',
                'cultural_elements': 'moderate',
                'bilingual_support': True
            }
        else:  # native_preferred
            return {
                'primary_style': 'cultural',
                'cultural_elements': 'prominent',
                'bilingual_support': True
            }
    
    def _translate_phrase(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate a phrase to/from different languages.
        
        Args:
            request: Request with translation parameters
            
        Returns:
            Response with translation
        """
        phrase = request.get('phrase', '')
        source_language = request.get('source_language', 'english')
        target_language = request.get('target_language', 'hindi')
        
        if not phrase:
            return {
                'success': False,
                'error': 'No phrase provided for translation'
            }
        
        # Build translation prompt
        prompt = f"""Translate the following phrase from {source_language} to {target_language}:
        
        Phrase: "{phrase}"
        Source Language: {source_language}
        Target Language: {target_language}
        
        Provide the translation in a natural, culturally appropriate way.
        If the target language uses a different script, provide both transliteration and native script.
        
        Translation:"""
        
        # Use LangGraph workflow for translation
        state = self._request_to_state({
            'user_id': 'translation',
            'message': prompt,
            'language_preferences': {
                'source_language': source_language,
                'target_language': target_language
            }
        })
        
        result = self.workflow.invoke(state)
        translation = result.get('response', '')
        
        return {
            'success': True,
            'original_phrase': phrase,
            'translation': translation,
            'source_language': source_language,
            'target_language': target_language,
            'translation_timestamp': datetime.now().isoformat()
        }
    
    def _get_language_statistics(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get language usage statistics.
        
        Args:
            request: Request parameters
            
        Returns:
            Response with language statistics
        """
        # This would typically come from database analytics
        # For now, return sample statistics
        stats = {
            'total_languages_supported': len(self.supported_languages),
            'most_common_native_languages': [
                'hindi', 'english', 'bengali', 'telugu', 'marathi'
            ],
            'comfort_level_distribution': {
                'english_only': 0.4,
                'mixed_language': 0.4,
                'native_preferred': 0.2
            },
            'cultural_greetings_used': len(self.cultural_greetings),
            'translation_requests': 0  # Would be tracked in real usage
        }
        
        return {
            'success': True,
            'statistics': stats,
            'last_updated': datetime.now().isoformat()
        } 