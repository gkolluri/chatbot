"""
React AI Pattern-Based Tag Analysis Agent for Multi-Agent Chatbot System
======================================================================

This agent analyzes conversations for interest inference using React AI pattern, including:
- AI-powered tag inference from conversation analysis
- Dynamic tag suggestions with multiple categories
- Intelligent reasoning about user interests
- Cultural context integration in tag analysis
- Automatic tag addition with user control
- Tag validation and cleaning with proper formatting
- Language-aware tag suggestions with cultural context
- React AI pattern: Observe → Think → Act → Observe
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re
from agents.react_base_agent import ReactBaseAgent, ReactAgentState
from langchain_core.tools import tool


class ReactTagAnalysisAgent(ReactBaseAgent):
    """
    React AI pattern-based agent responsible for analyzing conversations and generating tag suggestions.
    
    Implements Observe-Think-Act loops for dynamic tag analysis, reasoning about user interests,
    and generating appropriate tag suggestions with cultural context.
    """
    
    def __init__(self, db_interface=None):
        """Initialize the React AI tag analysis agent."""
        super().__init__("ReactTagAnalysisAgent", db_interface)
        self.tag_categories = {
            'technology': ['programming', 'ai', 'startup', 'software', 'tech', 'digital'],
            'entertainment': ['music', 'movies', 'gaming', 'streaming', 'art', 'media'],
            'sports': ['cricket', 'football', 'yoga', 'gym', 'fitness', 'sports'],
            'food': ['cooking', 'indian_food', 'street_food', 'restaurants', 'cuisine'],
            'travel': ['travel', 'photography', 'hiking', 'adventure', 'tourism'],
            'arts': ['art', 'classical_music', 'folk_art', 'dance', 'literature'],
            'business': ['entrepreneurship', 'finance', 'career', 'marketing', 'business'],
            'education': ['online_courses', 'languages', 'learning', 'academic', 'education'],
            'health': ['meditation', 'ayurveda', 'wellness', 'health', 'fitness'],
            'lifestyle': ['fashion', 'self_improvement', 'lifestyle', 'personal_growth'],
            'social': ['volunteering', 'networking', 'community', 'social_causes'],
            'creative': ['photography', 'writing', 'crafts', 'design', 'creative'],
            'regional': ['regional_cinema', 'classical_dance', 'folk_music', 'regional_culture'],
            'contemporary': ['sustainability', 'social_media', 'trends', 'modern_life']
        }
        self.analysis_history = []
        
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this agent provides."""
        return [
            "react_ai_tag_analysis",
            "dynamic_tag_inference",
            "reasoning_based_tag_suggestions",
            "cultural_context_tag_analysis",
            "multi_category_tag_suggestions",
            "intelligent_tag_validation",
            "language_aware_tag_generation",
            "adaptive_tag_learning"
        ]
    
    def _get_agent_system_prompt(self) -> str:
        """Get the system prompt for React AI tag analysis agent."""
        return """You are a React AI tag analysis agent designed to analyze conversations and suggest relevant interest tags.

Your role is to understand user interests through conversation analysis using the React AI pattern:
1. OBSERVE: Analyze conversation content, context, and patterns
2. THINK: Reason about user interests, cultural context, and tag relevance
3. ACT: Generate appropriate tag suggestions and validate them

TAG ANALYSIS GUIDELINES:
- Analyze conversation content for interest indicators
- Consider cultural context and language preferences
- Generate tags that reflect genuine interests
- Provide multiple categories of suggestions
- Validate and clean tag formats
- Learn from user feedback to improve suggestions

CATEGORIES TO CONSIDER:
- Technology & Digital (programming, AI, startup, etc.)
- Entertainment & Media (music, movies, gaming, etc.)
- Sports & Fitness (cricket, yoga, gym, etc.)
- Food & Cuisine (Indian food, cooking, street food, etc.)
- Travel & Adventure (travel, photography, hiking, etc.)
- Arts & Culture (art, classical music, folk art, etc.)
- Business & Career (entrepreneurship, finance, etc.)
- Education & Learning (online courses, languages, etc.)
- Health & Wellness (meditation, ayurveda, etc.)
- Lifestyle & Personal (fashion, self-improvement, etc.)
- Social & Community (volunteering, networking, etc.)
- Creative & Hobbies (photography, writing, crafts, etc.)
- Regional & Cultural (regional cinema, classical dance, etc.)
- Contemporary (sustainability, social media, etc.)

CULTURAL CONTEXT:
- Be aware of Indian cultural interests and traditions
- Consider regional language and cultural preferences
- Include both traditional and contemporary interests
- Respect diverse cultural backgrounds

REACT AI PATTERN:
- Always observe the conversation content and context
- Think about what interests the user is expressing
- Act by generating relevant and appropriate tags
- Reflect on tag quality and user feedback
"""
    
    def _get_agent_specific_tools(self) -> List:
        """Get tag analysis-specific tools for React AI pattern."""
        tools = []
        
        @tool
        def analyze_conversation_for_interests(conversation_text: str) -> str:
            """Analyze conversation text to identify potential interests and topics."""
            try:
                analysis = f"Conversation Analysis:\n"
                analysis += f"Text length: {len(conversation_text)} characters\n"
                
                # Look for interest indicators
                interest_indicators = {
                    'technology': ['programming', 'coding', 'ai', 'startup', 'tech', 'software', 'app', 'website'],
                    'entertainment': ['movie', 'film', 'music', 'song', 'game', 'gaming', 'streaming', 'netflix'],
                    'sports': ['cricket', 'football', 'yoga', 'gym', 'fitness', 'sports', 'exercise', 'workout'],
                    'food': ['cooking', 'food', 'restaurant', 'cuisine', 'recipe', 'chef', 'dining'],
                    'travel': ['travel', 'trip', 'vacation', 'photography', 'hiking', 'adventure', 'tourism'],
                    'arts': ['art', 'painting', 'music', 'dance', 'literature', 'poetry', 'creative'],
                    'business': ['business', 'entrepreneur', 'finance', 'career', 'marketing', 'startup'],
                    'education': ['learning', 'course', 'study', 'education', 'language', 'academic'],
                    'health': ['health', 'wellness', 'meditation', 'ayurveda', 'fitness', 'wellbeing'],
                    'lifestyle': ['lifestyle', 'fashion', 'personal', 'growth', 'improvement', 'life'],
                    'social': ['community', 'volunteer', 'social', 'network', 'help', 'charity'],
                    'creative': ['creative', 'art', 'writing', 'photography', 'craft', 'design'],
                    'regional': ['regional', 'culture', 'tradition', 'classical', 'folk', 'heritage'],
                    'contemporary': ['modern', 'trend', 'sustainability', 'social_media', 'digital']
                }
                
                found_interests = {}
                text_lower = conversation_text.lower()
                
                for category, keywords in interest_indicators.items():
                    category_matches = []
                    for keyword in keywords:
                        if keyword in text_lower:
                            category_matches.append(keyword)
                    if category_matches:
                        found_interests[category] = category_matches
                
                analysis += f"Found interest categories: {list(found_interests.keys())}\n"
                for category, matches in found_interests.items():
                    analysis += f"- {category}: {matches}\n"
                
                return analysis
            except Exception as e:
                return f"Error analyzing conversation for interests: {str(e)}"
        
        @tool
        def generate_tag_suggestions(analysis_result: str, user_tags: str) -> str:
            """Generate tag suggestions based on analysis and existing user tags."""
            try:
                # Parse existing tags
                existing_tags = eval(user_tags) if isinstance(user_tags, str) else user_tags
                
                suggestions = f"Tag Suggestions based on analysis:\n"
                suggestions += f"Analysis: {analysis_result}\n"
                suggestions += f"Existing tags: {existing_tags}\n"
                
                # Generate suggestions for each category found
                analysis_lines = analysis_result.split('\n')
                categories_found = []
                
                for line in analysis_lines:
                    if line.startswith('- ') and ':' in line:
                        category = line.split(':')[0].replace('- ', '')
                        categories_found.append(category)
                
                suggestions += f"Categories to suggest tags for: {categories_found}\n"
                
                # Generate specific tag suggestions
                tag_suggestions = []
                for category in categories_found:
                    if category in self.tag_categories:
                        category_tags = self.tag_categories[category]
                        # Filter out existing tags
                        new_tags = [tag for tag in category_tags if tag not in existing_tags]
                        if new_tags:
                            tag_suggestions.extend(new_tags[:3])  # Top 3 from each category
                
                suggestions += f"Suggested tags: {tag_suggestions}\n"
                
                return suggestions
            except Exception as e:
                return f"Error generating tag suggestions: {str(e)}"
        
        @tool
        def validate_and_clean_tags(tag_list: str) -> str:
            """Validate and clean tag formats."""
            try:
                tags = eval(tag_list) if isinstance(tag_list, str) else tag_list
                
                cleaned_tags = []
                validation_results = []
                
                for tag in tags:
                    # Clean the tag
                    cleaned_tag = tag.lower().strip()
                    cleaned_tag = re.sub(r'[^a-z0-9_]', '_', cleaned_tag)
                    cleaned_tag = re.sub(r'_+', '_', cleaned_tag)
                    cleaned_tag = cleaned_tag.strip('_')
                    
                    # Validate
                    if len(cleaned_tag) >= 2 and len(cleaned_tag) <= 20:
                        cleaned_tags.append(cleaned_tag)
                        validation_results.append(f"✓ {tag} → {cleaned_tag}")
                    else:
                        validation_results.append(f"✗ {tag} (invalid length)")
                
                result = f"Tag Validation Results:\n"
                result += "\n".join(validation_results)
                result += f"\n\nCleaned tags: {cleaned_tags}"
                
                return result
            except Exception as e:
                return f"Error validating tags: {str(e)}"
        
        @tool
        def analyze_cultural_context(language_preferences: str, conversation_text: str) -> str:
            """Analyze cultural context for tag suggestions."""
            try:
                prefs = eval(language_preferences) if isinstance(language_preferences, str) else language_preferences
                native_lang = prefs.get('native_language', 'english')
                
                analysis = f"Cultural Context Analysis:\n"
                analysis += f"Native language: {native_lang}\n"
                
                # Look for cultural indicators
                cultural_indicators = {
                    'hindi': ['hindi', 'bollywood', 'classical', 'traditional'],
                    'bengali': ['bengali', 'rabindra', 'kolkata', 'bengal'],
                    'telugu': ['telugu', 'tollywood', 'andhra', 'telangana'],
                    'marathi': ['marathi', 'mumbai', 'maharashtra', 'marathi_culture'],
                    'tamil': ['tamil', 'kollywood', 'chennai', 'tamil_nadu'],
                    'gujarati': ['gujarati', 'gujarat', 'gujarati_culture'],
                    'kannada': ['kannada', 'sandalwood', 'karnataka', 'kannada_culture'],
                    'odia': ['odia', 'odisha', 'oriya', 'odia_culture'],
                    'punjabi': ['punjabi', 'punjab', 'bhangra', 'punjabi_culture'],
                    'assamese': ['assamese', 'assam', 'assamese_culture'],
                    'urdu': ['urdu', 'ghazal', 'urdu_literature', 'urdu_culture'],
                    'malayalam': ['malayalam', 'mollywood', 'kerala', 'malayalam_culture']
                }
                
                text_lower = conversation_text.lower()
                found_cultural_elements = []
                
                if native_lang in cultural_indicators:
                    for indicator in cultural_indicators[native_lang]:
                        if indicator in text_lower:
                            found_cultural_elements.append(indicator)
                
                analysis += f"Cultural elements found: {found_cultural_elements}\n"
                
                if found_cultural_elements:
                    analysis += "Consider adding cultural context tags to suggestions.\n"
                else:
                    analysis += "No specific cultural elements detected.\n"
                
                return analysis
            except Exception as e:
                return f"Error analyzing cultural context: {str(e)}"
        
        @tool
        def learn_from_tag_feedback(tag_suggestions: str, user_feedback: str) -> str:
            """Learn from user feedback on tag suggestions."""
            try:
                suggestions = eval(tag_suggestions) if isinstance(tag_suggestions, str) else tag_suggestions
                feedback = eval(user_feedback) if isinstance(user_feedback, str) else user_feedback
                
                learning = f"Learning from tag feedback:\n"
                learning += f"Suggested tags: {suggestions}\n"
                learning += f"User feedback: {feedback}\n"
                
                # Analyze feedback patterns
                accepted_tags = feedback.get('accepted', [])
                rejected_tags = feedback.get('rejected', [])
                
                learning += f"Accepted tags: {accepted_tags}\n"
                learning += f"Rejected tags: {rejected_tags}\n"
                
                # Update analysis history
                self.analysis_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'suggestions': suggestions,
                    'feedback': feedback,
                    'accepted': accepted_tags,
                    'rejected': rejected_tags
                })
                
                learning += f"Analysis history entries: {len(self.analysis_history)}\n"
                
                return learning
            except Exception as e:
                return f"Error learning from feedback: {str(e)}"
        
        tools.extend([
            analyze_conversation_for_interests,
            generate_tag_suggestions,
            validate_and_clean_tags,
            analyze_cultural_context,
            learn_from_tag_feedback
        ])
        
        return tools
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process tag analysis-related requests using React AI pattern.
        
        Args:
            request: Request dictionary with parameters
            
        Returns:
            Response dictionary with results
        """
        request_type = request.get('type', '')
        
        if request_type == 'analyze_conversation':
            return self._analyze_conversation_for_tags(request)
        elif request_type == 'get_tag_suggestions':
            return self._get_tag_suggestions(request)
        elif request_type == 'validate_tags':
            return self._validate_tags(request)
        elif request_type == 'learn_from_feedback':
            return self._learn_from_tag_feedback(request)
        elif request_type == 'get_analysis_history':
            return self._get_analysis_history(request)
        else:
            return {
                'success': False,
                'error': f'Unknown request type: {request_type}',
                'available_types': ['analyze_conversation', 'get_tag_suggestions', 'validate_tags', 'learn_from_feedback', 'get_analysis_history']
            }
    
    def _analyze_conversation_for_tags(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze conversation for tag inference using React AI pattern.
        
        Args:
            request: Request with conversation data
            
        Returns:
            Response with analysis results
        """
        user_id = request.get('user_id')
        conversation_text = request.get('conversation_text', '')
        language_preferences = request.get('language_preferences', {})
        
        if not user_id or not conversation_text:
            return {
                'success': False,
                'error': 'Missing user_id or conversation_text'
            }
        
        # Use React AI pattern to analyze conversation
        react_request = {
            'user_id': user_id,
            'message': f'Analyze this conversation for interests: {conversation_text}',
            'language_preferences': language_preferences,
            'type': 'analyze_conversation'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            # Extract analysis from reasoning chain
            analysis_summary = self._extract_analysis_from_reasoning(result.get('reasoning_chain', []))
            
            result['analysis_summary'] = analysis_summary
            result['conversation_length'] = len(conversation_text)
            result['analysis_timestamp'] = datetime.now().isoformat()
            
            return result
        else:
            return result
    
    def _get_tag_suggestions(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get tag suggestions using React AI pattern.
        
        Args:
            request: Request with analysis data
            
        Returns:
            Response with tag suggestions
        """
        user_id = request.get('user_id')
        analysis_result = request.get('analysis_result', '')
        existing_tags = request.get('existing_tags', [])
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        # Use React AI pattern to generate suggestions
        react_request = {
            'user_id': user_id,
            'message': f'Generate tag suggestions based on analysis: {analysis_result}',
            'existing_tags': existing_tags,
            'type': 'get_tag_suggestions'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            # Extract suggestions from reasoning chain
            suggestions = self._extract_suggestions_from_reasoning(result.get('reasoning_chain', []))
            
            result['suggestions'] = suggestions
            result['suggestion_count'] = len(suggestions)
            result['categories_covered'] = list(set([tag.split('_')[0] for tag in suggestions if '_' in tag]))
            
            return result
        else:
            return result
    
    def _validate_tags(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tags using React AI pattern.
        
        Args:
            request: Request with tags to validate
            
        Returns:
            Response with validation results
        """
        tags = request.get('tags', [])
        
        if not tags:
            return {
                'success': False,
                'error': 'No tags provided for validation'
            }
        
        # Use React AI pattern to validate tags
        react_request = {
            'user_id': 'system',
            'message': f'Validate and clean these tags: {tags}',
            'type': 'validate_tags'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            # Extract validation results from reasoning chain
            validation_results = self._extract_validation_from_reasoning(result.get('reasoning_chain', []))
            
            result['validation_results'] = validation_results
            result['valid_tags'] = validation_results.get('cleaned_tags', [])
            result['invalid_tags'] = validation_results.get('invalid_tags', [])
            
            return result
        else:
            return result
    
    def _learn_from_tag_feedback(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Learn from tag feedback using React AI pattern.
        
        Args:
            request: Request with feedback data
            
        Returns:
            Response with learning results
        """
        user_id = request.get('user_id')
        suggestions = request.get('suggestions', [])
        feedback = request.get('feedback', {})
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        # Use React AI pattern to learn from feedback
        react_request = {
            'user_id': user_id,
            'message': f'Learn from feedback on suggestions: {suggestions}',
            'feedback': feedback,
            'type': 'learn_from_feedback'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            result['analysis_history_count'] = len(self.analysis_history)
            result['learning_timestamp'] = datetime.now().isoformat()
            
            return result
        else:
            return result
    
    def _get_analysis_history(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get analysis history.
        
        Args:
            request: Request dictionary
            
        Returns:
            Response with analysis history
        """
        return {
            'success': True,
            'analysis_history': self.analysis_history,
            'total_entries': len(self.analysis_history)
        }
    
    def _extract_analysis_from_reasoning(self, reasoning_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract analysis results from reasoning chain.
        
        Args:
            reasoning_chain: List of reasoning steps
            
        Returns:
            Analysis summary
        """
        analysis = {
            'categories_found': [],
            'interest_indicators': [],
            'cultural_elements': []
        }
        
        for step in reasoning_chain:
            observation = step.get('observation', '')
            if 'categories to suggest tags for:' in observation.lower():
                # Extract categories
                lines = observation.split('\n')
                for line in lines:
                    if 'categories to suggest tags for:' in line.lower():
                        categories = line.split(':')[1].strip()
                        analysis['categories_found'] = eval(categories) if categories else []
                        break
        
        return analysis
    
    def _extract_suggestions_from_reasoning(self, reasoning_chain: List[Dict[str, Any]]) -> List[str]:
        """
        Extract tag suggestions from reasoning chain.
        
        Args:
            reasoning_chain: List of reasoning steps
            
        Returns:
            List of suggested tags
        """
        suggestions = []
        
        for step in reasoning_chain:
            observation = step.get('observation', '')
            if 'suggested tags:' in observation.lower():
                # Extract suggestions
                lines = observation.split('\n')
                for line in lines:
                    if 'suggested tags:' in line.lower():
                        tags_str = line.split(':')[1].strip()
                        suggestions = eval(tags_str) if tags_str else []
                        break
        
        return suggestions
    
    def _extract_validation_from_reasoning(self, reasoning_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract validation results from reasoning chain.
        
        Args:
            reasoning_chain: List of reasoning steps
            
        Returns:
            Validation results
        """
        validation = {
            'cleaned_tags': [],
            'invalid_tags': [],
            'validation_messages': []
        }
        
        for step in reasoning_chain:
            observation = step.get('observation', '')
            if 'cleaned tags:' in observation.lower():
                # Extract cleaned tags
                lines = observation.split('\n')
                for line in lines:
                    if 'cleaned tags:' in line.lower():
                        tags_str = line.split(':')[1].strip()
                        validation['cleaned_tags'] = eval(tags_str) if tags_str else []
                        break
        
        return validation 