"""
User Profile Agent for Multi-Agent Chatbot System using LangGraph
===============================================================

This agent handles user profiling and recommendations using LangGraph workflows, including:
- Automatic conversation analysis for interest discovery
- Interest-based user profiling with tag weighting
- Similar user discovery algorithm with similarity scoring
- Tag-based matching with cultural context consideration
- User recommendation system with proper filtering
- Real-time profile updates based on interactions
- Cultural and linguistic context consideration
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import Counter
from .base_agent import LangGraphBaseAgent, AgentState


class UserProfileAgent(LangGraphBaseAgent):
    """
    Agent responsible for user profiling and recommendations using LangGraph.
    
    Analyzes user behavior, manages profiles, and provides intelligent
    recommendations for user connections and content.
    """
    
    def __init__(self, db_interface=None):
        """Initialize the user profile agent."""
        super().__init__("UserProfileAgent", db_interface)
        self.profile_cache = {}
        self.similarity_cache = {}
        
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this agent provides."""
        return [
            "user_profiling",
            "interest_analysis",
            "similar_user_discovery",
            "recommendation_engine",
            "profile_management",
            "cultural_context_analysis",
            "real_time_updates",
            "langgraph_workflow"
        ]
    
    def _get_agent_system_prompt(self) -> str:
        """Get the system prompt for user profile agent."""
        return """You are an AI agent specialized in user profiling and recommendations.
        Your role is to:
        1. Analyze user behavior and interests
        2. Create comprehensive user profiles
        3. Find similar users for connections
        4. Generate personalized recommendations
        5. Consider cultural and linguistic context
        6. Focus on interests that facilitate connections
        
        Guidelines:
        - Focus on genuine interests and preferences
        - Consider cultural context and language preferences
        - Prioritize connections based on shared interests
        - Provide diverse and relevant recommendations
        - Respect user privacy and preferences
        """
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user profile requests using LangGraph workflow.
        
        Args:
            request: Request dictionary with parameters
            
        Returns:
            Response dictionary with results
        """
        request_type = request.get('type', '')
        
        if request_type == 'create_profile':
            return self._create_user_profile(request)
        elif request_type == 'update_profile':
            return self._update_user_profile(request)
        elif request_type == 'get_profile':
            return self._get_user_profile(request)
        elif request_type == 'find_similar_users':
            return self._find_similar_users(request)
        elif request_type == 'get_similar_users':
            return self._get_similar_users(request)
        elif request_type == 'get_recommendations':
            return self._get_recommendations(request)
        elif request_type == 'analyze_interests':
            return self._analyze_user_interests(request)
        elif request_type == 'get_profile_stats':
            return self._get_profile_statistics(request)
        else:
            return {
                'success': False,
                'error': f'Unknown request type: {request_type}',
                'available_types': ['create_profile', 'update_profile', 'get_profile', 'find_similar_users', 'get_similar_users', 'get_recommendations', 'analyze_interests', 'get_profile_stats']
            }
    
    def _get_similar_users(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get similar users - wrapper for find_similar_users.
        
        Args:
            request: Request with parameters
            
        Returns:
            Response with similar users
        """
        # Convert min_common_tags to min_similarity if present
        min_common_tags = request.get('min_common_tags', 1)
        min_similarity = min_common_tags * 0.1  # Convert to similarity score
        
        # Create request for find_similar_users
        find_request = {
            'user_id': request.get('user_id'),
            'min_similarity': min_similarity,
            'max_results': request.get('max_results', 10)
        }
        
        # Call the existing find_similar_users method
        result = self._find_similar_users(find_request)
        
        # Convert the response format to match UI expectations
        if result.get('success'):
            similar_users = result.get('similar_users', [])
            # Ensure each user has the expected fields
            for user in similar_users:
                if 'common_tags' not in user:
                    user['common_tags'] = user.get('shared_tags', [])
                if 'similarity_score' not in user:
                    user['similarity_score'] = user.get('similarity', 0)
            
            return {
                'success': True,
                'similar_users': similar_users
            }
        else:
            return result
    
    def _create_user_profile(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a comprehensive user profile.
        
        Args:
            request: Request with user data
            
        Returns:
            Response with created profile
        """
        user_id = request.get('user_id')
        user_name = request.get('user_name')
        language_preferences = request.get('language_preferences', {})
        tags = request.get('tags', [])
        conversation_history = request.get('conversation_history', [])
        
        if not user_id or not user_name:
            return {
                'success': False,
                'error': 'Missing user_id or user_name'
            }
        
        # Analyze user interests from conversation
        interest_analysis = self._analyze_user_interests_from_conversation(
            conversation_history, language_preferences
        )
        
        # Build comprehensive profile
        profile = {
            'user_id': user_id,
            'user_name': user_name,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'language_preferences': language_preferences,
            'tags': tags,
            'interest_analysis': interest_analysis,
            'profile_completeness': self._calculate_profile_completeness(tags, interest_analysis),
            'cultural_context': self._extract_cultural_context(language_preferences, tags),
            'activity_level': 'new'  # Will be updated based on usage
        }
        
        # Save to database
        if self.db:
            try:
                self.db.update_user_profile(user_id, profile)
            except Exception as e:
                print(f"Error saving profile: {e}")
        
        # Cache the profile
        self.profile_cache[user_id] = profile
        
        self.log_activity("Created user profile", {
            'user_id': user_id,
            'user_name': user_name,
            'tags_count': len(tags),
            'profile_completeness': profile['profile_completeness']
        })
        
        return {
            'success': True,
            'profile': profile,
            'message': 'Profile created successfully'
        }
    
    def _analyze_user_interests_from_conversation(self, conversation_history: List[Dict[str, str]], 
                                                language_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze user interests from conversation history.
        
        Args:
            conversation_history: List of conversation messages
            language_preferences: User's language preferences
            
        Returns:
            Dictionary with interest analysis
        """
        if not conversation_history:
            return {
                'primary_interests': [],
                'secondary_interests': [],
                'cultural_interests': [],
                'conversation_topics': [],
                'interest_confidence': 0.0
            }
        
        # Build analysis prompt
        prompt = f"""Analyze the following conversation to identify the user's interests and preferences.
        
        Focus on:
        1. Primary interests (most mentioned/discussed)
        2. Secondary interests (mentioned occasionally)
        3. Cultural interests and preferences
        4. Topics they seem passionate about
        5. Areas they want to learn more about
        
        Consider the user's language preferences: {language_preferences}
        
        Conversation:
        """
        
        # Add conversation context
        for msg in conversation_history[-20:]:  # Last 20 messages
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            prompt += f"{role}: {content}\n"
        
        prompt += "\n\nProvide analysis in this format:\nPrimary interests: [list]\nSecondary interests: [list]\nCultural interests: [list]\nTopics: [list]\nConfidence: [0-1]"
        
        # Use LangGraph workflow for analysis
        state = self._request_to_state({
            'user_id': 'analysis',
            'message': prompt,
            'language_preferences': language_preferences
        })
        
        result = self.workflow.invoke(state)
        analysis_response = result.get('response', '')
        
        # Parse the analysis response
        return self._parse_interest_analysis(analysis_response)
    
    def _parse_interest_analysis(self, analysis_response: str) -> Dict[str, Any]:
        """
        Parse the interest analysis response.
        
        Args:
            analysis_response: Raw analysis response
            
        Returns:
            Parsed interest analysis
        """
        try:
            lines = analysis_response.split('\n')
            primary_interests = []
            secondary_interests = []
            cultural_interests = []
            topics = []
            confidence = 0.5  # Default confidence
            
            for line in lines:
                line = line.strip()
                if line.startswith('Primary interests:'):
                    primary_interests = [item.strip() for item in line.replace('Primary interests:', '').strip('[]').split(',') if item.strip()]
                elif line.startswith('Secondary interests:'):
                    secondary_interests = [item.strip() for item in line.replace('Secondary interests:', '').strip('[]').split(',') if item.strip()]
                elif line.startswith('Cultural interests:'):
                    cultural_interests = [item.strip() for item in line.replace('Cultural interests:', '').strip('[]').split(',') if item.strip()]
                elif line.startswith('Topics:'):
                    topics = [item.strip() for item in line.replace('Topics:', '').strip('[]').split(',') if item.strip()]
                elif line.startswith('Confidence:'):
                    try:
                        confidence = float(line.replace('Confidence:', '').strip())
                    except:
                        pass
            
            return {
                'primary_interests': primary_interests,
                'secondary_interests': secondary_interests,
                'cultural_interests': cultural_interests,
                'conversation_topics': topics,
                'interest_confidence': confidence
            }
            
        except Exception as e:
            print(f"Error parsing interest analysis: {e}")
            return {
                'primary_interests': [],
                'secondary_interests': [],
                'cultural_interests': [],
                'conversation_topics': [],
                'interest_confidence': 0.0
            }
    
    def _calculate_profile_completeness(self, tags: List[str], interest_analysis: Dict[str, Any]) -> float:
        """
        Calculate profile completeness score.
        
        Args:
            tags: User's tags
            interest_analysis: Interest analysis results
            
        Returns:
            Completeness score (0-1)
        """
        score = 0.0
        
        # Tags contribute 40%
        if tags:
            score += min(len(tags) / 10.0, 0.4)
        
        # Interest analysis contributes 40%
        if interest_analysis.get('primary_interests'):
            score += min(len(interest_analysis['primary_interests']) / 5.0, 0.4)
        
        # Cultural context contributes 20%
        if interest_analysis.get('cultural_interests'):
            score += min(len(interest_analysis['cultural_interests']) / 3.0, 0.2)
        
        return min(score, 1.0)
    
    def _extract_cultural_context(self, language_preferences: Dict[str, Any], tags: List[str]) -> Dict[str, Any]:
        """
        Extract cultural context from language preferences and tags.
        
        Args:
            language_preferences: User's language preferences
            tags: User's tags
            
        Returns:
            Cultural context dictionary
        """
        context = {
            'native_language': language_preferences.get('native_language'),
            'preferred_languages': language_preferences.get('preferred_languages', []),
            'comfort_level': language_preferences.get('language_comfort_level', 'english'),
            'cultural_tags': [],
            'regional_interests': []
        }
        
        # Identify cultural tags
        cultural_keywords = [
            'indian', 'hindi', 'bengali', 'telugu', 'marathi', 'tamil', 'gujarati',
            'kannada', 'odia', 'punjabi', 'assamese', 'sanskrit', 'urdu', 'malayalam',
            'classical', 'traditional', 'folk', 'heritage', 'culture', 'regional'
        ]
        
        for tag in tags:
            if any(keyword in tag.lower() for keyword in cultural_keywords):
                context['cultural_tags'].append(tag)
        
        return context
    
    def _update_user_profile(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing user profile.
        
        Args:
            request: Request with updated profile data
            
        Returns:
            Response with updated profile
        """
        user_id = request.get('user_id')
        updates = request.get('updates', {})
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        # Get current profile
        current_profile = self._get_user_profile({'user_id': user_id})
        if not current_profile.get('success'):
            return current_profile
        
        profile = current_profile['profile']
        
        # Apply updates
        for key, value in updates.items():
            if key in profile:
                profile[key] = value
        
        # Update timestamp
        profile['last_updated'] = datetime.now().isoformat()
        
        # Recalculate completeness
        profile['profile_completeness'] = self._calculate_profile_completeness(
            profile.get('tags', []), profile.get('interest_analysis', {})
        )
        
        # Save to database
        if self.db:
            try:
                self.db.update_user_profile(user_id, profile)
            except Exception as e:
                print(f"Error updating profile: {e}")
        
        # Update cache
        self.profile_cache[user_id] = profile
        
        self.log_activity("Updated user profile", {
            'user_id': user_id,
            'updated_fields': list(updates.keys()),
            'new_completeness': profile['profile_completeness']
        })
        
        return {
            'success': True,
            'profile': profile,
            'message': 'Profile updated successfully'
        }
    
    def _get_user_profile(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get user profile from cache or database.
        
        Args:
            request: Request with user_id
            
        Returns:
            Response with user profile
        """
        user_id = request.get('user_id')
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        # Check cache first
        if user_id in self.profile_cache:
            return {
                'success': True,
                'profile': self.profile_cache[user_id],
                'source': 'cache'
            }
        
        # Get from database
        if self.db:
            try:
                profile = self.db.get_user_profile(user_id)
                if profile:
                    # Cache the profile
                    self.profile_cache[user_id] = profile
                    return {
                        'success': True,
                        'profile': profile,
                        'source': 'database'
                    }
            except Exception as e:
                print(f"Error getting profile: {e}")
        
        return {
            'success': False,
            'error': 'Profile not found'
        }
    
    def _find_similar_users(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find users with similar interests and profiles.
        
        Args:
            request: Request with user_id and criteria
            
        Returns:
            Response with similar users
        """
        user_id = request.get('user_id')
        min_similarity = request.get('min_similarity', 0.3)
        max_results = request.get('max_results', 10)
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        # Get user's profile
        profile_response = self._get_user_profile({'user_id': user_id})
        if not profile_response.get('success'):
            return profile_response
        
        user_profile = profile_response['profile']
        user_tags = set(user_profile.get('tags', []))
        
        # Find similar users
        similar_users = []
        
        if self.db:
            try:
                all_users = self.db.get_all_users()
                
                for other_user in all_users:
                    if other_user['user_id'] == user_id:
                        continue
                    
                    # Get other user's profile
                    other_profile_response = self._get_user_profile({'user_id': other_user['user_id']})
                    if not other_profile_response.get('success'):
                        continue
                    
                    other_profile = other_profile_response['profile']
                    other_tags = set(other_profile.get('tags', []))
                    
                    # Calculate similarity
                    similarity_score = self._calculate_user_similarity(
                        user_profile, other_profile, user_tags, other_tags
                    )
                    
                    if similarity_score >= min_similarity:
                        similar_users.append({
                            'user_id': other_user['user_id'],
                            'name': other_user['name'],
                            'similarity_score': similarity_score,
                            'common_tags': list(user_tags.intersection(other_tags)),
                            'profile_completeness': other_profile.get('profile_completeness', 0.0)
                        })
                
                # Sort by similarity score
                similar_users.sort(key=lambda x: x['similarity_score'], reverse=True)
                similar_users = similar_users[:max_results]
                
            except Exception as e:
                print(f"Error finding similar users: {e}")
        
        return {
            'success': True,
            'similar_users': similar_users,
            'total_found': len(similar_users),
            'user_profile_completeness': user_profile.get('profile_completeness', 0.0)
        }
    
    def _calculate_user_similarity(self, user_profile: Dict[str, Any], other_profile: Dict[str, Any],
                                 user_tags: set, other_tags: set) -> float:
        """
        Calculate similarity between two users.
        
        Args:
            user_profile: Current user's profile
            other_profile: Other user's profile
            user_tags: Current user's tags
            other_tags: Other user's tags
            
        Returns:
            Similarity score (0-1)
        """
        score = 0.0
        
        # Tag similarity (40% weight)
        if user_tags and other_tags:
            common_tags = user_tags.intersection(other_tags)
            tag_similarity = len(common_tags) / max(len(user_tags), len(other_tags))
            score += tag_similarity * 0.4
        
        # Interest similarity (30% weight)
        user_interests = user_profile.get('interest_analysis', {})
        other_interests = other_profile.get('interest_analysis', {})
        
        user_primary = set(user_interests.get('primary_interests', []))
        other_primary = set(other_interests.get('primary_interests', []))
        
        if user_primary and other_primary:
            common_interests = user_primary.intersection(other_primary)
            interest_similarity = len(common_interests) / max(len(user_primary), len(other_primary))
            score += interest_similarity * 0.3
        
        # Cultural similarity (20% weight)
        user_cultural = user_profile.get('cultural_context', {})
        other_cultural = other_profile.get('cultural_context', {})
        
        if (user_cultural.get('native_language') and 
            other_cultural.get('native_language') and
            user_cultural['native_language'] == other_cultural['native_language']):
            score += 0.2
        
        # Language comfort similarity (10% weight)
        if (user_cultural.get('comfort_level') and 
            other_cultural.get('comfort_level') and
            user_cultural['comfort_level'] == other_cultural['comfort_level']):
            score += 0.1
        
        return min(score, 1.0)
    
    def _get_recommendations(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get personalized recommendations for a user.
        
        Args:
            request: Request with user_id and recommendation type
            
        Returns:
            Response with recommendations
        """
        user_id = request.get('user_id')
        recommendation_type = request.get('type', 'general')
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        # Get user profile
        profile_response = self._get_user_profile({'user_id': user_id})
        if not profile_response.get('success'):
            return profile_response
        
        user_profile = profile_response['profile']
        
        # Generate recommendations based on type
        if recommendation_type == 'similar_users':
            return self._find_similar_users({'user_id': user_id, 'max_results': 5})
        elif recommendation_type == 'tags':
            return self._get_tag_recommendations(user_profile)
        elif recommendation_type == 'activities':
            return self._get_activity_recommendations(user_profile)
        else:
            return self._get_general_recommendations(user_profile)
    
    def _get_tag_recommendations(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Get tag recommendations for user."""
        current_tags = user_profile.get('tags', [])
        interests = user_profile.get('interest_analysis', {})
        
        # Generate tag suggestions based on interests
        suggested_tags = []
        
        for interest in interests.get('primary_interests', []):
            if interest not in current_tags:
                suggested_tags.append(interest)
        
        return {
            'success': True,
            'recommendations': {
                'type': 'tags',
                'suggested_tags': suggested_tags[:5],
                'reason': 'Based on your conversation interests'
            }
        }
    
    def _get_activity_recommendations(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Get activity recommendations for user."""
        tags = user_profile.get('tags', [])
        cultural_context = user_profile.get('cultural_context', {})
        
        activities = []
        
        # Suggest activities based on tags
        if 'music' in tags or 'classical' in tags:
            activities.append('Join a music appreciation group')
        if 'cooking' in tags or 'food' in tags:
            activities.append('Participate in cooking workshops')
        if 'travel' in tags:
            activities.append('Join travel discussion groups')
        if 'technology' in tags or 'programming' in tags:
            activities.append('Join tech meetups')
        
        # Cultural activities
        native_lang = cultural_context.get('native_language')
        if native_lang and native_lang != 'english':
            activities.append(f'Join {native_lang.title()} language groups')
        
        return {
            'success': True,
            'recommendations': {
                'type': 'activities',
                'suggested_activities': activities[:5],
                'reason': 'Based on your interests and cultural background'
            }
        }
    
    def _get_general_recommendations(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Get general recommendations for user."""
        profile_completeness = user_profile.get('profile_completeness', 0.0)
        tags = user_profile.get('tags', [])
        
        recommendations = []
        
        if profile_completeness < 0.5:
            recommendations.append('Add more tags to your profile to find better connections')
        
        if len(tags) < 3:
            recommendations.append('Explore more interests to connect with diverse people')
        
        if not user_profile.get('cultural_context', {}).get('cultural_tags'):
            recommendations.append('Consider adding cultural interests to your profile')
        
        return {
            'success': True,
            'recommendations': {
                'type': 'general',
                'suggestions': recommendations,
                'profile_completeness': profile_completeness
            }
        }
    
    def _analyze_user_interests(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user interests from various sources."""
        user_id = request.get('user_id')
        conversation_history = request.get('conversation_history', [])
        language_preferences = request.get('language_preferences', {})
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        # Analyze interests from conversation
        interest_analysis = self._analyze_user_interests_from_conversation(
            conversation_history, language_preferences
        )
        
        return {
            'success': True,
            'interest_analysis': interest_analysis,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _get_profile_statistics(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get profile statistics and metrics."""
        user_id = request.get('user_id')
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        # Get user profile
        profile_response = self._get_user_profile({'user_id': user_id})
        if not profile_response.get('success'):
            return profile_response
        
        profile = profile_response['profile']
        
        stats = {
            'profile_completeness': profile.get('profile_completeness', 0.0),
            'total_tags': len(profile.get('tags', [])),
            'primary_interests': len(profile.get('interest_analysis', {}).get('primary_interests', [])),
            'secondary_interests': len(profile.get('interest_analysis', {}).get('secondary_interests', [])),
            'cultural_tags': len(profile.get('cultural_context', {}).get('cultural_tags', [])),
            'days_since_created': (datetime.now() - datetime.fromisoformat(profile.get('created_at', datetime.now().isoformat()))).days,
            'last_updated_days': (datetime.now() - datetime.fromisoformat(profile.get('last_updated', datetime.now().isoformat()))).days
        }
        
        return {
            'success': True,
            'statistics': stats,
            'profile_summary': {
                'user_name': profile.get('user_name'),
                'activity_level': profile.get('activity_level', 'new'),
                'native_language': profile.get('cultural_context', {}).get('native_language')
            }
        } 