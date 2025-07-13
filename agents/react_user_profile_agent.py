"""
React AI Pattern-Based User Profile Agent for Multi-Agent Chatbot System
======================================================================

This agent handles user profiles and preferences using React AI pattern, including:
- User profile creation and management
- Tag management and suggestions
- User similarity matching
- Profile data validation
- React AI pattern: Observe → Think → Act → Observe
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from agents.react_base_agent import ReactBaseAgent, ReactAgentState
from langchain_core.tools import tool


class ReactUserProfileAgent(ReactBaseAgent):
    """
    React AI pattern-based agent responsible for managing user profiles.
    
    Implements Observe-Think-Act loops for profile management,
    tag handling, and user similarity matching.
    """
    
    def __init__(self, db_interface=None):
        """Initialize the React AI user profile agent."""
        super().__init__("ReactUserProfileAgent", db_interface)
        self.profile_cache = {}
        
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this agent provides."""
        return [
            "react_ai_profile_management",
            "user_profile_creation",
            "tag_management",
            "user_similarity_matching",
            "profile_data_validation",
            "preference_management",
            "react_ai_reasoning"
        ]
    
    def _get_agent_system_prompt(self) -> str:
        """Get the system prompt for React AI user profile agent."""
        return """You are a React AI user profile management agent designed to handle user profiles and preferences.

Your role is to manage user profiles using the React AI pattern:
1. OBSERVE: Analyze user profile data and preferences
2. THINK: Reason about user needs and profile requirements
3. ACT: Create, update, or manage profiles appropriately
4. REFLECT: Learn from profile patterns and improve management

PROFILE MANAGEMENT GUIDELINES:
- Create comprehensive user profiles
- Manage user tags and preferences
- Find similar users based on interests
- Validate profile data integrity
- Maintain user privacy and security
- Provide personalized recommendations

REACT AI PATTERN:
- Always observe current profile state
- Think about user needs and preferences
- Act by creating or updating profiles
- Reflect on profile management effectiveness
"""
    
    def _get_agent_specific_tools(self) -> List:
        """Get profile-specific tools for React AI pattern."""
        tools = []
        
        @tool
        def create_user_profile(user_id: str, profile_data: str) -> str:
            """Create a new user profile with React AI reasoning."""
            try:
                profile_info = eval(profile_data) if isinstance(profile_data, str) else profile_data
                
                # Create profile with React AI reasoning
                profile = {
                    'user_id': user_id,
                    'created_at': datetime.now().isoformat(),
                    'profile_data': profile_info,
                    'react_ai_reasoning': 'Profile created with React AI pattern'
                }
                
                # Store in agent's cache
                self.profile_cache[user_id] = profile
                
                return f"Profile created successfully for user {user_id}"
            except Exception as e:
                return f"Error creating profile: {str(e)}"
        
        @tool
        def find_similar_users(user_id: str, min_common_tags: int = 2) -> str:
            """Find similar users using React AI reasoning."""
            try:
                if self.db:
                    similar_users = self.db.find_similar_users(user_id, min_common_tags)
                    
                    result = f"Similar users found for user {user_id}:\n"
                    result += f"- Total similar users: {len(similar_users)}\n"
                    result += f"- Minimum common tags: {min_common_tags}\n"
                    
                    for user in similar_users[:5]:  # Show top 5
                        result += f"- {user['name']}: {user['similarity_score']} common tags\n"
                    
                    result += f"- React AI: Similarity analysis with reasoning"
                    
                    return result
                else:
                    return "Database not available for similarity search"
            except Exception as e:
                return f"Error finding similar users: {str(e)}"
        
        @tool
        def manage_user_tags(user_id: str, action: str, tags: str) -> str:
            """Manage user tags using React AI reasoning."""
            try:
                tag_list = eval(tags) if isinstance(tags, str) else tags
                
                if action == "add":
                    for tag in tag_list:
                        if self.db:
                            self.db.add_user_tag(user_id, tag, "manual")
                    
                    result = f"Added {len(tag_list)} tags to user {user_id}"
                elif action == "remove":
                    for tag in tag_list:
                        if self.db:
                            self.db.remove_user_tag(user_id, tag)
                    
                    result = f"Removed {len(tag_list)} tags from user {user_id}"
                else:
                    result = f"Unknown action: {action}"
                
                result += f"\n- React AI: Tag management with reasoning"
                return result
            except Exception as e:
                return f"Error managing tags: {str(e)}"
        
        @tool
        def get_user_profile(user_id: str) -> str:
            """Get user profile with React AI analysis."""
            try:
                if self.db:
                    profile = self.db.get_user_profile(user_id)
                    user_tags = self.db.get_user_tags(user_id)
                    
                    result = f"Profile for user {user_id}:\n"
                    if profile:
                        result += f"- Name: {profile.get('name', 'Unknown')}\n"
                        result += f"- Created: {profile.get('created_at', 'Unknown')}\n"
                        result += f"- Tags: {len(user_tags)} total\n"
                        result += f"- React AI: Profile analysis with reasoning"
                    else:
                        result += "- Profile not found\n"
                    
                    return result
                else:
                    return "Database not available for profile retrieval"
            except Exception as e:
                return f"Error getting profile: {str(e)}"
        
        tools.extend([
            create_user_profile,
            find_similar_users,
            manage_user_tags,
            get_user_profile
        ])
        
        return tools
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process profile-related requests using React AI pattern.
        
        Args:
            request: Request dictionary with parameters
            
        Returns:
            Response dictionary with results
        """
        request_type = request.get('type', '')
        
        if request_type == 'create_profile':
            return self._create_profile(request)
        elif request_type == 'find_similar_users':
            return self._find_similar_users(request)
        elif request_type == 'find_similar_users_with_location':
            return self._find_similar_users_with_location(request)
        elif request_type == 'update_location':
            return self._update_location(request)
        elif request_type == 'find_nearby_users':
            return self._find_nearby_users(request)
        elif request_type == 'find_users_in_city':
            return self._find_users_in_city(request)
        elif request_type == 'manage_tags':
            return self._manage_tags(request)
        elif request_type == 'get_profile':
            return self._get_profile(request)
        else:
            return {
                'success': False,
                'error': f'Unknown request type: {request_type}',
                'available_types': ['create_profile', 'find_similar_users', 'find_similar_users_with_location', 
                                  'update_location', 'find_nearby_users', 'find_users_in_city', 'manage_tags', 'get_profile']
            }
    
    def _create_profile(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a user profile using React AI pattern.
        
        Args:
            request: Request with profile data
            
        Returns:
            Response with profile creation result
        """
        user_id = request.get('user_id')
        profile_data = request.get('profile_data', {})
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        # Use React AI pattern for profile creation
        react_request = {
            'user_id': user_id,
            'message': f'Create profile for user {user_id}',
            'profile_data': profile_data,
            'type': 'create_profile'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            result['profile_created'] = True
            result['user_id'] = user_id
            result['framework'] = 'React AI Pattern'
        
        return result
    
    def _find_similar_users(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find similar users using React AI pattern.
        
        Args:
            request: Request with search parameters
            
        Returns:
            Response with similar users
        """
        user_id = request.get('user_id')
        min_common_tags = request.get('min_common_tags', 2)
        limit = request.get('limit', 5)
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        try:
            # Use React AI pattern for similarity search
            react_request = {
                'user_id': user_id,
                'message': f'Find similar users for {user_id}',
                'min_common_tags': min_common_tags,
                'type': 'find_similar_users'
            }
            
            result = self.react_loop(react_request)
            
            if result.get('success'):
                # Get actual similar users from database
                similar_users = []
                if self.db:
                    db_similar_users = self.db.find_similar_users(user_id, min_common_tags, include_location=True)
                    
                    # Convert to expected format
                    for user in db_similar_users[:limit]:
                        similar_users.append({
                            'user_id': user['user_id'],
                            'name': user['name'],
                            'similarity_score': user['total_score'],
                            'common_tags': user['common_tags'],
                            'location_info': user.get('location_info', {}),
                            'matching_method': 'tag_based_similarity'
                        })
                
                result['similar_users'] = similar_users
                result['total_found'] = len(similar_users)
                result['similarity_search_completed'] = True
                result['framework'] = 'React AI Pattern'
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error finding similar users: {str(e)}',
                'similar_users': []
            }
    
    def _manage_tags(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage user tags using React AI pattern.
        
        Args:
            request: Request with tag management data
            
        Returns:
            Response with tag management result
        """
        user_id = request.get('user_id')
        action = request.get('action', 'add')
        tags = request.get('tags', [])
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        # Use React AI pattern for tag management
        react_request = {
            'user_id': user_id,
            'message': f'{action} tags for user {user_id}',
            'action': action,
            'tags': tags,
            'type': 'manage_tags'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            result['tags_managed'] = True
            result['action'] = action
            result['framework'] = 'React AI Pattern'
        
        return result
    
    def _get_profile(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get user profile using React AI pattern.
        
        Args:
            request: Request with user ID
            
        Returns:
            Response with profile data
        """
        user_id = request.get('user_id')
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        # Use React AI pattern for profile retrieval
        react_request = {
            'user_id': user_id,
            'message': f'Get profile for user {user_id}',
            'type': 'get_profile'
        }
        
        result = self.react_loop(react_request)
        
        if result.get('success'):
            result['profile_retrieved'] = True
            result['framework'] = 'React AI Pattern'
        
        return result 

    def _find_similar_users_with_location(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find similar users with location filtering using React AI pattern.
        
        Args:
            request: Request with search parameters and location filter
            
        Returns:
            Response with similar users considering location
        """
        user_id = request.get('user_id')
        search_type = request.get('search_type', 'interests')
        location_filter = request.get('location_filter', {})
        limit = request.get('limit', 5)
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        try:
            # Use React AI pattern for location-aware similarity search
            react_request = {
                'user_id': user_id,
                'message': f'Find similar users with location for {user_id}',
                'search_type': search_type,
                'location_filter': location_filter,
                'limit': limit,
                'type': 'find_similar_users_with_location'
            }
            
            result = self.react_loop(react_request)
            
            if result.get('success'):
                # Get actual similar users from database
                if search_type == 'location':
                    if location_filter.get('type') == 'nearby':
                        similar_users = self.db.find_nearby_users(
                            user_id, location_filter.get('radius_km', 50), limit
                        ) if self.db else []
                    elif location_filter.get('type') == 'city':
                        similar_users = self.db.find_users_in_city(
                            user_id, location_filter.get('city', ''),
                            location_filter.get('state'), limit
                        ) if self.db else []
                    else:
                        similar_users = []
                else:
                    # For interests or both, use enhanced similarity matching
                    similar_users = self.db.find_similar_users(user_id, limit) if self.db else []
                
                result['similar_users'] = similar_users
                result['search_type'] = search_type
                result['location_filter_applied'] = bool(location_filter)
                result['framework'] = 'React AI Pattern'
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error finding similar users with location: {str(e)}',
                'similar_users': []
            }

    def _update_location(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user location using React AI pattern.
        
        Args:
            request: Request with location data
            
        Returns:
            Response with location update result
        """
        user_id = request.get('user_id')
        city = request.get('city')
        state = request.get('state')
        country = request.get('country')
        timezone = request.get('timezone')
        coordinates = request.get('coordinates')
        privacy_level = request.get('privacy_level', 'city_only')
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        try:
            # Use React AI pattern for location update
            react_request = {
                'user_id': user_id,
                'message': f'Update location for user {user_id}',
                'city': city,
                'state': state,
                'country': country,
                'timezone': timezone,
                'coordinates': coordinates,
                'privacy_level': privacy_level,
                'type': 'update_location'
            }
            
            result = self.react_loop(react_request)
            
            if result.get('success'):
                # Update location in database
                if self.db:
                    self.db.update_location_preferences(
                        user_id, city, state, country, timezone, coordinates, privacy_level
                    )
                
                result['location_updated'] = True
                result['privacy_level'] = privacy_level
                result['framework'] = 'React AI Pattern'
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error updating location: {str(e)}'
            }

    def _find_nearby_users(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find nearby users using React AI pattern.
        
        Args:
            request: Request with search parameters
            
        Returns:
            Response with nearby users
        """
        user_id = request.get('user_id')
        radius_km = request.get('radius_km', 50)
        limit = request.get('limit', 10)
        
        if not user_id:
            return {
                'success': False,
                'error': 'Missing user_id'
            }
        
        try:
            # Use React AI pattern for nearby user search
            react_request = {
                'user_id': user_id,
                'message': f'Find nearby users for {user_id}',
                'radius_km': radius_km,
                'limit': limit,
                'type': 'find_nearby_users'
            }
            
            result = self.react_loop(react_request)
            
            if result.get('success'):
                # Get nearby users from database
                nearby_users = self.db.find_nearby_users(user_id, radius_km, limit) if self.db else []
                
                result['nearby_users'] = nearby_users
                result['search_radius_km'] = radius_km
                result['search_method'] = 'GPS-based proximity'
                result['framework'] = 'React AI Pattern'
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error finding nearby users: {str(e)}',
                'nearby_users': []
            }

    def _find_users_in_city(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find users in a specific city using React AI pattern.
        
        Args:
            request: Request with city search parameters
            
        Returns:
            Response with users in the city
        """
        user_id = request.get('user_id')
        city = request.get('city')
        state = request.get('state')
        limit = request.get('limit', 10)
        
        if not user_id or not city:
            return {
                'success': False,
                'error': 'Missing user_id or city'
            }
        
        try:
            # Use React AI pattern for city-based user search
            react_request = {
                'user_id': user_id,
                'message': f'Find users in {city} for {user_id}',
                'city': city,
                'state': state,
                'limit': limit,
                'type': 'find_users_in_city'
            }
            
            result = self.react_loop(react_request)
            
            if result.get('success'):
                # Get users in city from database
                city_users = self.db.find_users_in_city(user_id, city, state, limit) if self.db else []
                
                result['city_users'] = city_users
                result['search_city'] = city
                result['search_state'] = state
                result['search_method'] = 'City-based search'
                result['framework'] = 'React AI Pattern'
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error finding users in city: {str(e)}',
                'city_users': []
            } 