"""
Tag Analysis Agent for Multi-Agent Chatbot System using LangGraph
===============================================================

This agent handles tag analysis and suggestions using LangGraph workflows, including:
- AI-powered tag inference from conversations every 5 turns
- Dynamic tag suggestions with multiple categories (AI, category, synonym, related)
- Automatic tag addition with user control and feedback
- Tag validation and cleaning with proper formatting
- Language-aware tag suggestions with cultural context
- Automatic tag analysis with duplicate handling
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import re
from .base_agent import LangGraphBaseAgent, AgentState


class TagAnalysisAgent(LangGraphBaseAgent):
    """
    Agent responsible for tag analysis and suggestions using LangGraph.
    
    Analyzes conversations to infer user interests, generates tag suggestions,
    and manages tag-related operations with AI-powered intelligence.
    """
    
    def __init__(self, db_interface=None):
        """Initialize the tag analysis agent."""
        super().__init__("TagAnalysisAgent", db_interface)
        self.tag_categories = self._initialize_tag_categories()
        self.analyzed_conversations = set()
        
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this agent provides."""
        return [
            "ai_powered_tag_inference",
            "dynamic_tag_suggestions",
            "multi_category_suggestions",
            "tag_validation_and_cleaning",
            "language_aware_suggestions",
            "duplicate_handling",
            "conversation_analysis",
            "langgraph_workflow"
        ]
    
    def _get_agent_system_prompt(self) -> str:
        """Get the system prompt for tag analysis agent."""
        return """You are an AI agent specialized in analyzing conversations and identifying user interests.
        Your role is to:
        1. Analyze conversation content to identify user interests and topics
        2. Generate relevant tags that represent user interests
        3. Provide tag suggestions in multiple categories
        4. Consider cultural context and language preferences
        5. Focus on interests that could help users connect with others
        
        Guidelines:
        - Generate tags that represent genuine interests
        - Consider both explicit and implicit interests
        - Include cultural and regional interests when relevant
        - Keep tags concise and meaningful
        - Avoid overly generic tags
        - Focus on topics that facilitate connections
        """
    
    def _initialize_tag_categories(self) -> Dict[str, List[str]]:
        """
        Initialize tag categories with diverse interests.
        
        Returns:
            Dictionary of category names to tag lists
        """
        return {
            "Technology & Digital": [
                "programming", "artificial intelligence", "startup", "digital marketing",
                "web development", "mobile apps", "data science", "cybersecurity",
                "blockchain", "cloud computing", "machine learning", "software engineering"
            ],
            "Entertainment & Media": [
                "music", "movies", "gaming", "streaming", "podcasts", "books",
                "comedy", "drama", "action", "romance", "documentaries", "anime"
            ],
            "Sports & Fitness": [
                "cricket", "football", "yoga", "gym", "running", "swimming",
                "tennis", "badminton", "meditation", "fitness", "sports", "workout"
            ],
            "Food & Cuisine": [
                "indian food", "cooking", "street food", "restaurants", "baking",
                "vegetarian", "non-vegetarian", "desserts", "tea", "coffee", "spices"
            ],
            "Travel & Adventure": [
                "travel", "photography", "hiking", "backpacking", "mountains",
                "beaches", "cities", "villages", "road trips", "international travel"
            ],
            "Arts & Culture": [
                "art", "classical music", "folk art", "dance", "theater",
                "literature", "poetry", "crafts", "painting", "sculpture"
            ],
            "Business & Career": [
                "entrepreneurship", "finance", "marketing", "consulting",
                "freelancing", "remote work", "leadership", "innovation"
            ],
            "Education & Learning": [
                "online courses", "languages", "certifications", "workshops",
                "skill development", "academic research", "teaching"
            ],
            "Health & Wellness": [
                "meditation", "ayurveda", "mental health", "nutrition",
                "wellness", "alternative medicine", "holistic health"
            ],
            "Lifestyle & Personal": [
                "fashion", "self-improvement", "minimalism", "sustainability",
                "personal development", "lifestyle", "productivity"
            ],
            "Social & Community": [
                "volunteering", "networking", "community service", "social work",
                "mentoring", "collaboration", "team building"
            ],
            "Creative & Hobbies": [
                "photography", "writing", "crafts", "gardening", "collecting",
                "DIY projects", "creative arts", "handmade items"
            ],
            "Regional & Cultural": [
                "regional cinema", "classical dance", "folk music", "traditional arts",
                "regional languages", "cultural festivals", "heritage"
            ],
            "Contemporary": [
                "sustainability", "social media", "influencer culture", "trends",
                "modern lifestyle", "digital nomad", "conscious living"
            ]
        }
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process tag analysis requests using LangGraph workflow.
        
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
        elif request_type == 'validate_tag':
            return self._validate_tag(request)
        elif request_type == 'get_categories':
            return self._get_tag_categories(request)
        elif request_type == 'clean_tags':
            return self._clean_tags(request)
        elif request_type == 'get_analyzed_conversations':
            return self._get_analyzed_conversations(request)
        else:
            return {
                'success': False,
                'error': f'Unknown request type: {request_type}',
                'available_types': ['analyze_conversation', 'get_tag_suggestions', 'validate_tag', 'get_categories', 'clean_tags', 'get_analyzed_conversations']
            }
    
    def _analyze_conversation_for_tags(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze conversation to infer user interests and generate tags.
        
        Args:
            request: Request with conversation data
            
        Returns:
            Response with inferred tags
        """
        user_id = request.get('user_id')
        conversation_history = request.get('conversation_history', [])
        language_preferences = request.get('language_preferences', {})
        
        if not user_id or not conversation_history:
            return {
                'success': False,
                'error': 'Missing user_id or conversation_history'
            }
        
        # Check if we've already analyzed this conversation
        conversation_key = f"{user_id}_{len(conversation_history)}"
        if conversation_key in self.analyzed_conversations:
            return {
                'success': True,
                'message': 'Conversation already analyzed',
                'tags': []
            }
        
        # Build analysis prompt
        analysis_prompt = self._build_analysis_prompt(conversation_history, language_preferences)
        
        # Use LangGraph workflow for analysis
        state = self._request_to_state({
            'user_id': user_id,
            'message': analysis_prompt,
            'language_preferences': language_preferences,
            'conversation_history': conversation_history
        })
        
        result = self.workflow.invoke(state)
        analysis_response = result.get('response', '')
        
        # Extract tags from response
        inferred_tags = self._extract_tags_from_response(analysis_response)
        
        # Clean and validate tags
        cleaned_tags = self._clean_and_validate_tags(inferred_tags)
        
        # Mark conversation as analyzed
        self.analyzed_conversations.add(conversation_key)
        
        # Log activity
        self.log_activity("Analyzed conversation for tags", {
            'user_id': user_id,
            'conversation_length': len(conversation_history),
            'inferred_tags': len(cleaned_tags),
            'tags': cleaned_tags
        })
        
        return {
            'success': True,
            'inferred_tags': cleaned_tags,
            'analysis_response': analysis_response,
            'conversation_key': conversation_key
        }
    
    def _build_analysis_prompt(self, conversation_history: List[Dict[str, str]], 
                              language_preferences: Dict[str, Any]) -> str:
        """
        Build prompt for conversation analysis.
        
        Args:
            conversation_history: List of conversation messages
            language_preferences: User's language preferences
            
        Returns:
            Analysis prompt string
        """
        prompt = """Analyze the following conversation and identify the user's interests and topics they've discussed.
        
        Focus on:
        1. Explicit interests mentioned by the user
        2. Implicit interests based on what they talk about
        3. Cultural and regional interests if relevant
        4. Topics that could help them connect with others
        
        Return a list of relevant tags (single words or short phrases) separated by commas.
        Keep tags concise and meaningful. Focus on genuine interests, not generic topics.
        
        Conversation:
        """
        
        # Add conversation context
        for msg in conversation_history[-10:]:  # Last 10 messages
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            prompt += f"{role}: {content}\n"
        
        # Add language context if relevant
        native_lang = language_preferences.get('native_language')
        if native_lang and native_lang != 'english':
            prompt += f"\nNote: User's native language is {native_lang.title()}. Consider cultural context."
        
        prompt += "\n\nTags (comma-separated):"
        
        return prompt
    
    def _extract_tags_from_response(self, response: str) -> List[str]:
        """
        Extract tags from AI response.
        
        Args:
            response: AI response containing tags
            
        Returns:
            List of extracted tags
        """
        # Clean the response
        response = response.strip()
        
        # Remove common prefixes
        prefixes = ['tags:', 'interests:', 'topics:', 'suggestions:']
        for prefix in prefixes:
            if response.lower().startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Split by commas and clean
        tags = [tag.strip().lower() for tag in response.split(',')]
        
        # Remove empty tags and duplicates
        tags = [tag for tag in tags if tag and len(tag) > 1]
        tags = list(set(tags))
        
        return tags
    
    def _clean_and_validate_tags(self, tags: List[str]) -> List[str]:
        """
        Clean and validate a list of tags.
        
        Args:
            tags: List of raw tags
            
        Returns:
            List of cleaned and validated tags
        """
        cleaned_tags = []
        
        for tag in tags:
            # Clean the tag
            cleaned_tag = self._clean_single_tag(tag)
            
            # Validate the tag
            if self._validate_single_tag(cleaned_tag):
                cleaned_tags.append(cleaned_tag)
        
        # Remove duplicates
        cleaned_tags = list(set(cleaned_tags))
        
        return cleaned_tags
    
    def _clean_single_tag(self, tag: str) -> str:
        """
        Clean a single tag.
        
        Args:
            tag: Raw tag string
            
        Returns:
            Cleaned tag string
        """
        # Remove special characters except hyphens and spaces
        tag = re.sub(r'[^\w\s-]', '', tag)
        
        # Convert to lowercase
        tag = tag.lower()
        
        # Remove extra whitespace
        tag = ' '.join(tag.split())
        
        # Replace spaces with hyphens for consistency
        tag = tag.replace(' ', '-')
        
        return tag
    
    def _validate_single_tag(self, tag: str) -> bool:
        """
        Validate a single tag.
        
        Args:
            tag: Tag to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not tag or len(tag) < 2:
            return False
        
        if len(tag) > 50:
            return False
        
        # Check for common invalid patterns
        invalid_patterns = [
            r'^\d+$',  # Only numbers
            r'^[a-z]{1,2}$',  # Very short words
            r'^(the|and|or|but|in|on|at|to|for|of|with|by)$',  # Common words
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, tag):
                return False
        
        return True
    
    def _get_tag_suggestions(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get tag suggestions in multiple categories.
        
        Args:
            request: Request with context for suggestions
            
        Returns:
            Response with categorized tag suggestions
        """
        user_id = request.get('user_id')
        existing_tags = request.get('existing_tags', [])
        language_preferences = request.get('language_preferences', {})
        
        # Generate AI-based suggestions
        ai_suggestions = self._generate_ai_suggestions(user_id, existing_tags, language_preferences)
        
        # Get category-based suggestions
        category_suggestions = self._get_category_suggestions(existing_tags)
        
        # Get synonym suggestions
        synonym_suggestions = self._get_synonym_suggestions(existing_tags)
        
        # Get related concept suggestions
        related_suggestions = self._get_related_suggestions(existing_tags)
        
        return {
            'success': True,
            'suggestions': {
                'ai_generated': ai_suggestions,
                'category_based': category_suggestions,
                'synonyms': synonym_suggestions,
                'related_concepts': related_suggestions
            },
            'total_suggestions': len(ai_suggestions) + len(category_suggestions) + len(synonym_suggestions) + len(related_suggestions)
        }
    
    def _generate_ai_suggestions(self, user_id: str, existing_tags: List[str], 
                                language_preferences: Dict[str, Any]) -> List[str]:
        """
        Generate AI-based tag suggestions.
        
        Args:
            user_id: User identifier
            existing_tags: User's current tags
            language_preferences: User's language preferences
            
        Returns:
            List of AI-generated tag suggestions
        """
        prompt = f"""Based on the user's existing tags: {', '.join(existing_tags)}
        
        Generate 5-10 additional tag suggestions that:
        1. Complement their existing interests
        2. Could help them connect with others
        3. Represent related or adjacent interests
        4. Consider cultural context if relevant
        
        Return only the tags, comma-separated."""
        
        # Use LangGraph workflow for AI suggestions
        state = self._request_to_state({
            'user_id': user_id,
            'message': prompt,
            'language_preferences': language_preferences
        })
        
        result = self.workflow.invoke(state)
        response = result.get('response', '')
        
        suggestions = self._extract_tags_from_response(response)
        return suggestions[:10]  # Limit to 10 suggestions
    
    def _get_category_suggestions(self, existing_tags: List[str]) -> List[str]:
        """
        Get category-based tag suggestions.
        
        Args:
            existing_tags: User's current tags
            
        Returns:
            List of category-based suggestions
        """
        suggestions = []
        
        # Find categories that match existing tags
        matching_categories = []
        for category, tags in self.tag_categories.items():
            for tag in existing_tags:
                if tag in tags:
                    matching_categories.append(category)
                    break
        
        # Get suggestions from matching categories
        for category in matching_categories:
            category_tags = self.tag_categories[category]
            for tag in category_tags:
                if tag not in existing_tags and tag not in suggestions:
                    suggestions.append(tag)
                    if len(suggestions) >= 5:
                        break
            if len(suggestions) >= 5:
                break
        
        return suggestions[:5]
    
    def _get_synonym_suggestions(self, existing_tags: List[str]) -> List[str]:
        """
        Get synonym-based tag suggestions.
        
        Args:
            existing_tags: User's current tags
            
        Returns:
            List of synonym suggestions
        """
        # Simple synonym mapping
        synonym_map = {
            'programming': ['coding', 'software development', 'tech'],
            'music': ['songs', 'melody', 'rhythm'],
            'cooking': ['food', 'cuisine', 'recipes'],
            'travel': ['tourism', 'exploration', 'adventure'],
            'fitness': ['exercise', 'workout', 'health'],
            'reading': ['books', 'literature', 'novels'],
            'photography': ['photos', 'camera', 'images'],
            'gaming': ['video games', 'esports', 'play'],
            'art': ['painting', 'drawing', 'creative'],
            'sports': ['athletics', 'games', 'physical activity']
        }
        
        suggestions = []
        for tag in existing_tags:
            if tag in synonym_map:
                synonyms = synonym_map[tag]
                for synonym in synonyms:
                    if synonym not in existing_tags and synonym not in suggestions:
                        suggestions.append(synonym)
                        if len(suggestions) >= 3:
                            break
            if len(suggestions) >= 3:
                break
        
        return suggestions[:3]
    
    def _get_related_suggestions(self, existing_tags: List[str]) -> List[str]:
        """
        Get related concept suggestions.
        
        Args:
            existing_tags: User's current tags
            
        Returns:
            List of related concept suggestions
        """
        # Simple related concepts mapping
        related_map = {
            'programming': ['artificial intelligence', 'data science', 'web development'],
            'music': ['instruments', 'concerts', 'classical music'],
            'cooking': ['baking', 'restaurants', 'food photography'],
            'travel': ['photography', 'culture', 'languages'],
            'fitness': ['nutrition', 'meditation', 'wellness'],
            'reading': ['writing', 'poetry', 'book clubs'],
            'photography': ['travel', 'art', 'social media'],
            'gaming': ['streaming', 'technology', 'community'],
            'art': ['museums', 'galleries', 'creative workshops'],
            'sports': ['team building', 'coaching', 'fitness']
        }
        
        suggestions = []
        for tag in existing_tags:
            if tag in related_map:
                related = related_map[tag]
                for concept in related:
                    if concept not in existing_tags and concept not in suggestions:
                        suggestions.append(concept)
                        if len(suggestions) >= 3:
                            break
            if len(suggestions) >= 3:
                break
        
        return suggestions[:3]
    
    def _validate_tag(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single tag."""
        tag = request.get('tag', '')
        is_valid = self._validate_single_tag(tag)
        cleaned_tag = self._clean_single_tag(tag) if tag else ''
        
        return {
            'success': True,
            'is_valid': is_valid,
            'cleaned_tag': cleaned_tag,
            'original_tag': tag
        }
    
    def _get_tag_categories(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get all available tag categories."""
        return {
            'success': True,
            'categories': self.tag_categories
        }
    
    def _clean_tags(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Clean a list of tags."""
        tags = request.get('tags', [])
        cleaned_tags = self._clean_and_validate_tags(tags)
        
        return {
            'success': True,
            'original_tags': tags,
            'cleaned_tags': cleaned_tags
        }
    
    def _get_analyzed_conversations(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get list of analyzed conversations."""
        return {
            'success': True,
            'analyzed_conversations': list(self.analyzed_conversations),
            'count': len(self.analyzed_conversations)
        } 