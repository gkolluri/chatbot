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
        self.tag_categories = self._initialize_tag_categories()
        self.analysis_history = []
        self.analyzed_conversations = set()  # Track analyzed conversations
        self.synonym_map = self._initialize_synonym_map()
        self.related_concepts_map = self._initialize_related_concepts_map()
    
    def _initialize_tag_categories(self) -> Dict[str, List[str]]:
        """
        Initialize comprehensive tag categories matching original TagAnalysisAgent.
        
        Returns:
            Dictionary of category names to tag lists
        """
        return {
            "Technology & Digital": [
                "programming", "artificial intelligence", "startup", "digital marketing",
                "web development", "mobile apps", "data science", "cybersecurity",
                "blockchain", "cloud computing", "machine learning", "software engineering",
                "ai", "tech", "software", "coding", "algorithms", "database", "api",
                "frontend", "backend", "devops", "automation", "robotics"
            ],
            "Entertainment & Media": [
                "music", "movies", "gaming", "streaming", "podcasts", "books",
                "comedy", "drama", "action", "romance", "documentaries", "anime",
                "bollywood", "tollywood", "kollywood", "regional cinema", "art house",
                "netflix", "amazon prime", "hotstar", "youtube", "instagram", "tiktok"
            ],
            "Sports & Fitness": [
                "cricket", "football", "yoga", "gym", "running", "swimming",
                "tennis", "badminton", "meditation", "fitness", "sports", "workout",
                "ipl", "bcci", "hockey", "kabaddi", "athletics", "marathon",
                "cycling", "weightlifting", "pilates", "crossfit", "martial arts"
            ],
            "Food & Cuisine": [
                "indian food", "cooking", "street food", "restaurants", "baking",
                "vegetarian", "non-vegetarian", "desserts", "tea", "coffee", "spices",
                "north indian", "south indian", "biryani", "curry", "roti", "dal",
                "recipes", "chef", "food photography", "food blogging", "wine", "beer"
            ],
            "Travel & Adventure": [
                "travel", "photography", "hiking", "backpacking", "mountains",
                "beaches", "cities", "villages", "road trips", "international travel",
                "india travel", "himalayas", "goa", "kerala", "rajasthan", "varanasi",
                "adventure sports", "trekking", "camping", "wildlife", "heritage sites"
            ],
            "Arts & Culture": [
                "art", "classical music", "folk art", "dance", "theater",
                "literature", "poetry", "crafts", "painting", "sculpture",
                "carnatic", "hindustani", "ghazal", "qawwali", "classical dance",
                "folk music", "regional culture", "museums", "galleries", "exhibitions"
            ],
            "Business & Career": [
                "entrepreneurship", "finance", "marketing", "consulting",
                "freelancing", "remote work", "leadership", "innovation",
                "startup india", "make in india", "msme", "digital payments",
                "investment", "stock market", "cryptocurrency", "business development"
            ],
            "Education & Learning": [
                "online courses", "languages", "learning", "academic", "education",
                "skill development", "professional development", "certifications",
                "research", "science", "mathematics", "physics", "chemistry", "biology",
                "history", "geography", "philosophy", "psychology"
            ],
            "Health & Wellness": [
                "meditation", "ayurveda", "wellness", "health", "fitness",
                "yoga science", "mental health", "nutrition", "diet", "healthcare",
                "alternative medicine", "mindfulness", "spirituality", "self-care"
            ],
            "Lifestyle & Personal": [
                "fashion", "self-improvement", "lifestyle", "personal growth",
                "relationships", "parenting", "home decor", "gardening",
                "sustainability", "minimalism", "productivity", "time management"
            ],
            "Social & Community": [
                "volunteering", "networking", "community", "social causes",
                "ngo", "charity", "social work", "activism", "environment",
                "women empowerment", "education for all", "rural development"
            ],
            "Creative & Hobbies": [
                "photography", "writing", "crafts", "design", "creative",
                "blogging", "content creation", "video editing", "graphic design",
                "music production", "singing", "dancing", "acting", "drawing"
            ],
            "Regional & Cultural": [
                "regional cinema", "classical dance", "folk music", "regional culture",
                "hindi literature", "bengali literature", "marathi literature",
                "tamil literature", "regional languages", "festivals", "traditions"
            ],
            "Contemporary & Trends": [
                "sustainability", "social media", "trends", "modern life",
                "digital india", "smart cities", "electric vehicles", "renewable energy",
                "climate change", "artificial intelligence", "future technology"
            ]
        }
    
    def _initialize_synonym_map(self) -> Dict[str, List[str]]:
        """Initialize synonym mappings for tag suggestions."""
        return {
            'programming': ['coding', 'software development', 'tech', 'development'],
            'music': ['songs', 'melody', 'rhythm', 'audio', 'sound'],
            'cooking': ['food', 'cuisine', 'recipes', 'culinary', 'chef'],
            'travel': ['tourism', 'exploration', 'adventure', 'journey', 'vacation'],
            'fitness': ['exercise', 'workout', 'health', 'gym', 'training'],
            'reading': ['books', 'literature', 'novels', 'study', 'learning'],
            'photography': ['photos', 'camera', 'images', 'visual', 'capture'],
            'gaming': ['video games', 'esports', 'play', 'entertainment'],
            'art': ['painting', 'drawing', 'creative', 'visual art', 'design'],
            'sports': ['athletics', 'games', 'physical activity', 'competition'],
            'business': ['entrepreneurship', 'startup', 'commerce', 'trade'],
            'education': ['learning', 'study', 'academic', 'knowledge', 'school'],
            'technology': ['tech', 'innovation', 'digital', 'computer', 'software'],
            'health': ['wellness', 'medical', 'fitness', 'healthcare', 'wellbeing']
        }
    
    def _initialize_related_concepts_map(self) -> Dict[str, List[str]]:
        """Initialize related concepts mappings for tag suggestions."""
        return {
            'programming': ['artificial intelligence', 'data science', 'web development', 'software engineering'],
            'music': ['instruments', 'concerts', 'classical music', 'audio production'],
            'cooking': ['baking', 'restaurants', 'food photography', 'nutrition'],
            'travel': ['photography', 'culture', 'languages', 'adventure'],
            'fitness': ['nutrition', 'meditation', 'wellness', 'sports'],
            'reading': ['writing', 'poetry', 'book clubs', 'literature'],
            'photography': ['travel', 'art', 'social media', 'visual storytelling'],
            'gaming': ['streaming', 'technology', 'community', 'esports'],
            'art': ['museums', 'galleries', 'creative workshops', 'design'],
            'sports': ['team building', 'coaching', 'fitness', 'competition'],
            'business': ['finance', 'marketing', 'leadership', 'innovation'],
            'education': ['research', 'teaching', 'professional development', 'skills'],
            'technology': ['innovation', 'startups', 'digital transformation', 'automation'],
            'health': ['nutrition', 'exercise', 'mental health', 'preventive care']
        }
        
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
        def generate_tag_suggestions(analysis_result: str, user_tags: List[str]) -> str:
            """Generate tag suggestions based on analysis and existing user tags."""
            try:
                # Parse existing tags
                existing_tags = user_tags
                
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
        def validate_and_clean_tags(tag_list: List[str]) -> str:
            """Validate and clean tag formats."""
            try:
                tags = tag_list
                
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
        def analyze_cultural_context(language_preferences: Dict[str, Any], conversation_text: str) -> str:
            """Analyze cultural context for tag suggestions."""
            try:
                prefs = language_preferences
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
        def search_trending_tags(category: str = None) -> str:
            """Search for trending tags and topics using web search."""
            try:
                search_info = f"Searching for trending tags"
                if category:
                    search_info += f" in {category}"
                search_info += ":\n"
                
                try:
                    if category:
                        query = f"trending topics {category} 2024 2025 popular interests"
                    else:
                        query = "trending topics 2024 2025 popular interests hobbies"
                    
                    web_results = self._search_web_reactive(query, "Finding trending topics and interests")
                    search_info += f"Trending information: {web_results[:500]}...\n"
                except Exception as e:
                    search_info += f"Web search unavailable: {str(e)}\n"
                
                return search_info
            except Exception as e:
                return f"Error searching trending tags: {str(e)}"
        
        @tool
        def learn_from_tag_feedback(tag_suggestions: List[str], user_feedback: Dict[str, Any]) -> str:
            """Learn from user feedback on tag suggestions."""
            try:
                suggestions = tag_suggestions
                feedback = user_feedback
                
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
            search_trending_tags,
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
        elif request_type == 'suggest_location_tags':
            return self._suggest_location_tags(request)
        elif request_type == 'validate_tag':
            return self._validate_tag(request)
        elif request_type == 'validate_tags':
            return self._validate_tags(request)
        elif request_type == 'get_categories':
            return self._get_tag_categories(request)
        elif request_type == 'clean_tags':
            return self._clean_tags(request)
        elif request_type == 'get_analyzed_conversations':
            return self._get_analyzed_conversations(request)
        elif request_type == 'learn_from_feedback':
            return self._learn_from_tag_feedback(request)
        elif request_type == 'get_analysis_history':
            return self._get_analysis_history(request)
        else:
            return {
                'success': False,
                'error': f'Unknown request type: {request_type}',
                'available_types': ['analyze_conversation', 'get_tag_suggestions', 'suggest_location_tags', 'validate_tag', 'validate_tags', 'get_categories', 'clean_tags', 'get_analyzed_conversations', 'learn_from_feedback', 'get_analysis_history']
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
        
        # Check if we've already analyzed this conversation
        conversation_key = f"{user_id}_{len(conversation_text)}"
        if conversation_key in self.analyzed_conversations:
            return {
                'success': True,
                'message': 'Conversation already analyzed',
                'tags': [],
                'conversation_key': conversation_key
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
            
            # Mark conversation as analyzed
            self.analyzed_conversations.add(conversation_key)
            
            result['analysis_summary'] = analysis_summary
            result['conversation_length'] = len(conversation_text)
            result['analysis_timestamp'] = datetime.now().isoformat()
            result['conversation_key'] = conversation_key
            
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
            # Get multi-category suggestions like original agent
            ai_suggestions = self._generate_ai_suggestions(user_id, existing_tags, {})
            category_suggestions = self._get_category_suggestions(existing_tags)
            synonym_suggestions = self._get_synonym_suggestions(existing_tags)
            related_suggestions = self._get_related_suggestions(existing_tags)
            
            result['suggestions'] = {
                'ai_generated': ai_suggestions,
                'category_based': category_suggestions,
                'synonyms': synonym_suggestions,
                'related_concepts': related_suggestions
            }
            result['total_suggestions'] = len(ai_suggestions) + len(category_suggestions) + len(synonym_suggestions) + len(related_suggestions)
            
            return result
        else:
            return result
    
    def _suggest_location_tags(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get location-aware tag suggestions using React AI pattern.
        
        Args:
            request: Request with user ID and limit
            
        Returns:
            Response with location-aware tag suggestions
        """
        user_id = request.get('user_id')
        limit = request.get('limit', 10)
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        try:
            # Get user's location preferences
            location_prefs = self.db.get_location_preferences(user_id) if self.db else {}
            
            # Use React AI pattern for location-aware suggestions
            react_request = {
                'user_id': user_id,
                'message': f'Generate location-aware tag suggestions for user {user_id}',
                'location_preferences': location_prefs,
                'limit': limit,
                'type': 'suggest_location_tags'
            }
            
            result = self.react_loop(react_request)
            
            if result.get('success'):
                # Generate location-specific suggestions
                location_suggestions = self._get_location_based_suggestions(location_prefs, limit)
                
                result['suggestions'] = location_suggestions
                result['location_context'] = location_prefs
                result['suggestion_method'] = 'Location-aware analysis'
                result['framework'] = 'React AI Pattern'
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error generating location-aware suggestions: {str(e)}',
                'suggestions': []
            }
    
    def _get_location_based_suggestions(self, location_prefs: Dict[str, Any], limit: int) -> List[str]:
        """
        Generate location-based tag suggestions.
        
        Args:
            location_prefs: User's location preferences
            limit: Maximum number of suggestions
            
        Returns:
            List of location-based tag suggestions
        """
        suggestions = []
        
        if not location_prefs:
            return suggestions
        
        state = location_prefs.get('state', '').lower()
        city = location_prefs.get('city', '').lower()
        country = location_prefs.get('country', '').lower()
        
        # Indian state-specific suggestions
        state_suggestions = {
            'maharashtra': ['bollywood', 'marathi culture', 'vada pav', 'ganesh chaturthi', 'mumbai street food', 'pune tech'],
            'kerala': ['kathakali', 'backwaters', 'ayurveda', 'coconut cuisine', 'monsoon', 'spices'],
            'tamil nadu': ['bharatanatyam', 'filter coffee', 'carnatic music', 'temple architecture', 'tamil literature'],
            'karnataka': ['carnatic music', 'mysore palace', 'dosa varieties', 'it hub', 'western ghats'],
            'gujarat': ['garba', 'dhokla', 'navratri', 'business culture', 'textiles', 'gujarati cuisine'],
            'rajasthan': ['folk music', 'dal baati', 'desert culture', 'royal heritage', 'handicrafts'],
            'punjab': ['bhangra', 'butter chicken', 'sikh culture', 'farming', 'punjabi music'],
            'west bengal': ['durga puja', 'fish curry', 'bengali literature', 'cultural festivals', 'adda'],
            'delhi': ['mughal architecture', 'chaat', 'political hub', 'delhi metro', 'historical monuments'],
            'uttar pradesh': ['classical music', 'lucknowi cuisine', 'ganga aarti', 'taj mahal', 'hindi literature'],
            'madhya pradesh': ['khajuraho', 'tribal culture', 'wildlife sanctuaries', 'madhya pradesh tourism'],
            'bihar': ['bihar cuisine', 'ancient history', 'maithili culture', 'buddhist heritage'],
            'jharkhand': ['tribal art', 'jharkhand culture', 'mining heritage', 'folk traditions'],
            'odisha': ['jagannath temple', 'odissi dance', 'puri beach', 'odia cuisine', 'tribal culture'],
            'chhattisgarh': ['tribal culture', 'chhattisgarhi cuisine', 'folk art', 'forest heritage'],
            'assam': ['bihu', 'assamese culture', 'tea gardens', 'silk weaving', 'northeast cuisine'],
            'manipur': ['manipuri dance', 'martial arts', 'northeast culture', 'handloom'],
            'meghalaya': ['living root bridges', 'khasi culture', 'northeast music', 'hill stations'],
            'tripura': ['tripuri culture', 'bamboo crafts', 'northeast traditions', 'tribal festivals'],
            'mizoram': ['mizo culture', 'bamboo dance', 'northeast cuisine', 'hill culture'],
            'nagaland': ['naga culture', 'hornbill festival', 'tribal traditions', 'northeast music'],
            'arunachal pradesh': ['tribal culture', 'buddhist monasteries', 'northeast adventure', 'mountain culture'],
            'sikkim': ['buddhist culture', 'himalayan cuisine', 'mountain trekking', 'organic farming'],
            'himachal pradesh': ['hill stations', 'himachali cuisine', 'mountain culture', 'adventure sports'],
            'uttarakhand': ['himalayan culture', 'yoga and meditation', 'pilgrimage sites', 'mountain adventure'],
            'jammu and kashmir': ['kashmiri cuisine', 'valley culture', 'handicrafts', 'mountain beauty'],
            'ladakh': ['buddhist culture', 'high altitude', 'adventure tourism', 'mountain monasteries'],
            'andhra pradesh': ['telugu culture', 'spicy cuisine', 'classical dance', 'hyderabadi biryani'],
            'telangana': ['telangana culture', 'hyderabadi cuisine', 'it hub', 'nizami heritage'],
            'goa': ['beach culture', 'goan cuisine', 'portuguese heritage', 'coastal life', 'feni']
        }
        
        # City-specific suggestions
        city_suggestions = {
            'mumbai': ['bollywood', 'street food', 'local trains', 'financial capital', 'marine drive'],
            'delhi': ['historical monuments', 'chaat', 'metro culture', 'political center', 'connaught place'],
            'bangalore': ['it hub', 'pub culture', 'pleasant weather', 'startup ecosystem', 'garden city'],
            'hyderabad': ['biryani', 'it city', 'nizami culture', 'charminar', 'tech hub'],
            'chennai': ['filter coffee', 'carnatic music', 'marina beach', 'auto industry', 'tamil culture'],
            'kolkata': ['fish curry', 'cultural festivals', 'trams', 'literary culture', 'adda'],
            'pune': ['educational hub', 'it sector', 'pleasant climate', 'cultural city', 'pune nightlife'],
            'ahmedabad': ['gujarati cuisine', 'textile industry', 'sabarmati ashram', 'business hub'],
            'jaipur': ['pink city', 'rajasthani cuisine', 'royal heritage', 'handicrafts', 'desert culture'],
            'lucknow': ['lucknowi cuisine', 'nawabi culture', 'tehzeeb', 'chikankari', 'awadhi food'],
            'kanpur': ['leather industry', 'uttar pradesh culture', 'industrial city', 'ganges'],
            'nagpur': ['orange city', 'central india', 'tiger reserves', 'maharashtrian culture'],
            'indore': ['street food', 'madhya pradesh culture', 'commercial hub', 'indori cuisine'],
            'bhopal': ['lakes', 'madhya pradesh capital', 'begum culture', 'tribal heritage'],
            'visakhapatnam': ['beaches', 'port city', 'andhra cuisine', 'coastal culture', 'vizag'],
            'vadodara': ['cultural city', 'gujarati culture', 'industrial hub', 'baroda'],
            'coimbatore': ['textile hub', 'tamil nadu', 'industrial city', 'western ghats'],
            'agra': ['taj mahal', 'mughal heritage', 'petha', 'historical monuments', 'uttar pradesh'],
            'nashik': ['wine country', 'religious tourism', 'maharashtra', 'grapes', 'kumbh mela'],
            'faridabad': ['industrial city', 'ncr', 'haryana culture', 'manufacturing hub'],
            'meerut': ['uttar pradesh', 'sports goods', 'historical significance', 'industrial city'],
            'rajkot': ['gujarat', 'business hub', 'kathiawadi cuisine', 'industrial center'],
            'kalyan-dombivli': ['mumbai suburbs', 'maharashtra', 'residential city', 'local trains'],
            'vasai-virar': ['mumbai suburbs', 'coastal culture', 'maharashtra', 'growing city'],
            'varanasi': ['spiritual city', 'ganga aarti', 'classical music', 'silk weaving', 'ancient culture'],
            'srinagar': ['kashmir valley', 'dal lake', 'kashmiri cuisine', 'houseboats', 'mountain beauty'],
            'aurangabad': ['ajanta ellora', 'historical monuments', 'maharashtra', 'tourism hub'],
            'dhanbad': ['coal mining', 'jharkhand', 'industrial city', 'mining heritage'],
            'amritsar': ['golden temple', 'punjabi culture', 'sikh heritage', 'punjabi cuisine'],
            'allahabad': ['sangam city', 'kumbh mela', 'uttar pradesh', 'religious significance'],
            'gwalior': ['madhya pradesh', 'historical fort', 'classical music', 'royal heritage'],
            'jabalpur': ['madhya pradesh', 'marble rocks', 'narmada river', 'central india'],
            'coimbatore': ['textile city', 'tamil nadu', 'industrial hub', 'western ghats']
        }
        
        # Add state-based suggestions
        if state in state_suggestions:
            suggestions.extend(state_suggestions[state][:limit//2])
        
        # Add city-based suggestions
        if city in city_suggestions:
            suggestions.extend(city_suggestions[city][:limit//2])
        
        # Add country-based suggestions
        if country == 'india':
            suggestions.extend(['indian culture', 'festivals', 'regional cuisine', 'classical arts', 'heritage sites'])
        
        # Remove duplicates and limit results
        suggestions = list(set(suggestions))[:limit]
        
        return suggestions
    
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
        
        # Use React AI pattern for AI suggestions
        react_request = {
            'user_id': user_id,
            'message': prompt,
            'type': 'ai_suggestions'
        }
        
        result = self.react_loop(react_request)
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
        suggestions = []
        for tag in existing_tags:
            if tag in self.synonym_map:
                synonyms = self.synonym_map[tag]
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
        suggestions = []
        for tag in existing_tags:
            if tag in self.related_concepts_map:
                related = self.related_concepts_map[tag]
                for concept in related:
                    if concept not in existing_tags and concept not in suggestions:
                        suggestions.append(concept)
                        if len(suggestions) >= 3:
                            break
            if len(suggestions) >= 3:
                break
        
        return suggestions[:3]
    
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