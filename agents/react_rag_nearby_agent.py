"""
React AI Pattern-Based RAG Nearby Users Agent for Multi-Agent Chatbot System
==========================================================================

This agent handles nearby user discovery using RAG (Retrieval-Augmented Generation) with vector search, including:
- User profile vectorization and embedding storage
- Semantic similarity search for user profiles and interests
- Location-based filtering combined with semantic search
- Intelligent user recommendations based on profile similarity
- React AI pattern: Observe → Think → Act → Observe
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from agents.react_base_agent import ReactBaseAgent, ReactAgentState
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
# Removed FAISS - using MongoDB as vector store
import math


class ReactRAGNearbyUsersAgent(ReactBaseAgent):
    """
    React AI pattern-based agent for finding nearby users using RAG and vector search.
    
    Implements Observe-Think-Act loops for intelligent user discovery,
    combining location-based filtering with semantic similarity search
    for enhanced user matching and recommendations.
    """
    
    def __init__(self, db_interface=None):
        """Initialize the React AI RAG nearby users agent."""
        super().__init__("ReactRAGNearbyUsersAgent", db_interface)
        
        # Initialize embeddings (MongoDB is the vector store)
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv('OPENAI_API_KEY')
        ) if os.getenv('OPENAI_API_KEY') else None
        
        # Cache for quick access
        self.user_embeddings_cache = {}
        self.profile_update_timestamps = {}
        
        # RAG configuration
        self.embedding_dimension = 1536  # OpenAI embedding dimension
        self.similarity_threshold = 0.7
        self.max_nearby_users = 20
        self.location_weight = 0.3  # Weight for location vs semantic similarity
        
        # Log initialization
        self.log_activity("MongoDB vector store initialized")
        
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this agent provides."""
        return [
            "rag_nearby_user_discovery",
            "semantic_user_matching",
            "vector_search_profiles",
            "location_semantic_fusion",
            "intelligent_recommendations",
            "profile_vectorization",
            "similarity_scoring",
            "react_ai_reasoning"
        ]
    
    def _is_vector_store_available(self) -> bool:
        """Check if MongoDB vector store is available."""
        return self.db is not None and self.embeddings is not None
    
    def _get_agent_specific_tools(self) -> List:
        """Get agent-specific tools for RAG nearby users functionality."""
        tools = []
        
        @tool
        def vectorize_user_profile(user_id: str) -> str:
            """Vectorize a user's profile for semantic search and store in MongoDB."""
            try:
                if not self._is_vector_store_available():
                    return "Database or embeddings not available"
                
                # Get user profile and tags
                user_profile = self.db.get_user_profile(user_id)
                user_tags = self.db.get_user_tags(user_id)
                location_prefs = self.db.get_location_preferences(user_id)
                
                if not user_profile:
                    return f"User profile not found for {user_id}"
                
                # Create profile text for embedding
                profile_text = self._create_profile_text(user_profile, user_tags, location_prefs)
                
                # Generate embedding
                embedding = self.embeddings.embed_query(profile_text)
                
                # Store in MongoDB vector store
                metadata = {
                    'user_id': user_id,
                    'name': user_profile.get('name', ''),
                    'tags': user_tags,
                    'city': location_prefs.get('city', ''),
                    'state': location_prefs.get('state', ''),
                    'country': location_prefs.get('country', ''),
                    'coordinates': location_prefs.get('coordinates', {}),
                    'privacy_level': location_prefs.get('privacy_level', 'city_only')
                }
                
                result = self.db.store_user_embedding(user_id, embedding, profile_text, metadata)
                
                # Update cache
                self.user_embeddings_cache[user_id] = {
                    'embedding': embedding,
                    'profile_text': profile_text,
                    'timestamp': datetime.now(),
                    'location': location_prefs,
                    'metadata': metadata
                }
                
                if result:
                    return f"Successfully vectorized and stored profile for user {user_id}"
                else:
                    return f"Error storing embedding for user {user_id}"
                
            except Exception as e:
                return f"Error vectorizing user profile: {str(e)}"
        
        @tool
        def semantic_search_nearby_users(user_id: str, query: str, max_results: int = 10) -> str:
            """Search for nearby users using semantic similarity with MongoDB."""
            try:
                if not self._is_vector_store_available():
                    return "Vector store or embeddings not available"
                
                # Generate query embedding
                query_embedding = self.embeddings.embed_query(query)
                
                # Get user's location for filtering
                user_location = self.db.get_location_preferences(user_id) if self.db else {}
                
                # Perform semantic search using MongoDB
                results = self.db.semantic_search_users(
                    query_embedding, 
                    user_id=user_id, 
                    max_results=max_results * 3,  # Get more for filtering
                    min_similarity=self.similarity_threshold * 0.8  # Lower threshold for broader search
                )
                
                # Filter and format results
                filtered_results = []
                for result in results:
                    # Apply location filtering
                    if self._is_location_relevant(user_location, result['metadata']):
                        # Apply keyword relevance filter
                        if self._is_keyword_relevant(query, result):
                            filtered_results.append({
                                'user_id': result['user_id'],
                                'name': result['metadata'].get('name', ''),
                                'semantic_score': float(result['similarity']),
                                'tags': result['metadata'].get('tags', []),
                                'location': {
                                    'city': result['metadata'].get('city', ''),
                                    'state': result['metadata'].get('state', ''),
                                    'country': result['metadata'].get('country', ''),
                                    'privacy_level': result['metadata'].get('privacy_level', 'city_only')
                                }
                            })
                
                # Apply diversity scoring to reduce keyword dominance
                final_results = self._diversify_semantic_results(filtered_results, max_results)
                
                return json.dumps({
                    'success': True,
                    'results': final_results,
                    'total_found': len(final_results),
                    'search_method': 'mongodb_semantic_similarity_with_keyword_filter'
                })
                
            except Exception as e:
                return f"Error in semantic search: {str(e)}"
        
        @tool
        def hybrid_location_semantic_search(user_id: str, location_radius_km: float = 50, 
                                          semantic_query: str = "", max_results: int = 10) -> str:
            """Perform hybrid search combining location proximity and semantic similarity."""
            try:
                if not self.db:
                    return "Database not available"
                
                # Get location-based nearby users with fallback strategy
                nearby_users = []
                
                # First try GPS-based search
                gps_users = self.db.find_nearby_users(user_id, location_radius_km)
                if gps_users:
                    nearby_users = gps_users
                else:
                    # Fallback to city-based search if no GPS coordinates
                    city_users = self.db.find_users_in_city(user_id)
                    # Convert city users to nearby users format
                    for user in city_users:
                        nearby_users.append({
                            'user_id': user['user_id'],
                            'name': user['name'],
                            'distance_km': 0,  # Same city, no distance
                            'city': user.get('city', ''),
                            'state': user.get('state', ''),
                            'privacy_level': user.get('privacy_level', 'city_only')
                        })
                
                # Use MongoDB's hybrid geospatial-semantic search if available
                if semantic_query and self._is_vector_store_available():
                    # Generate query embedding
                    query_embedding = self.embeddings.embed_query(semantic_query)
                    
                    # Use MongoDB's optimized hybrid search
                    hybrid_results = self.db.hybrid_geospatial_semantic_search(
                        query_embedding=query_embedding,
                        user_id=user_id,
                        max_distance_km=location_radius_km,
                        max_results=max_results,
                        min_similarity=self.similarity_threshold * 0.8,
                        location_weight=self.location_weight
                    )
                    
                    # Format results to match expected structure with keyword filtering
                    formatted_results = []
                    for result in hybrid_results:
                        # Apply keyword relevance filter for hybrid search (only if semantic query provided)
                        if not semantic_query or self._is_keyword_relevant(semantic_query, result):
                            user_profile = result.get('user_profile', {})
                            formatted_results.append({
                                'user_id': result['user_id'],
                                'name': user_profile.get('name', ''),
                                'semantic_score': result['semantic_similarity'],
                                'location_score': result['location_score'],
                                'combined_score': result['combined_score'],
                                'city': user_profile.get('location', {}).get('city', ''),
                                'state': user_profile.get('location', {}).get('state', ''),
                                'privacy_level': user_profile.get('location', {}).get('privacy_level', 'city_only'),
                                'tags': self.db.get_user_tags(result['user_id']) if self.db else []
                            })
                    
                    return json.dumps({
                        'success': True,
                        'results': formatted_results,
                        'total_found': len(formatted_results),
                        'search_method': 'mongodb_hybrid_geospatial_semantic_with_keyword_filter',
                        'location_radius_km': location_radius_km,
                        'used_gps_search': True,
                        'used_city_fallback': False,
                        'used_mongodb_optimization': True,
                        'keyword_filtering_applied': True
                    })
                
                # Fallback to separate location + semantic search
                semantic_results = []
                if semantic_query and self._is_vector_store_available():
                    # Generate query embedding
                    query_embedding = self.embeddings.embed_query(semantic_query)
                    
                    # Get semantic search results
                    semantic_search_results = self.db.semantic_search_users(
                        query_embedding, 
                        user_id=user_id, 
                        max_results=max_results * 2,
                        min_similarity=self.similarity_threshold * 0.8
                    )
                    
                    for result in semantic_search_results:
                        # Apply keyword relevance filter for semantic results in hybrid search (only if semantic query provided)
                        if not semantic_query or self._is_keyword_relevant(semantic_query, result):
                            semantic_results.append({
                                'user_id': result['user_id'],
                                'semantic_score': float(result['similarity']),
                                'metadata': result['metadata']
                            })
                
                # Combine and rank results
                combined_results = self._combine_location_semantic_results(
                    nearby_users, semantic_results, max_results
                )
                
                return json.dumps({
                    'success': True,
                    'results': combined_results,
                    'total_found': len(combined_results),
                    'search_method': 'hybrid_location_semantic_with_fallback_and_keyword_filter',
                    'location_radius_km': location_radius_km,
                    'used_gps_search': len(gps_users) > 0,
                    'used_city_fallback': len(gps_users) == 0 and len(city_users) > 0,
                    'keyword_filtering_applied': bool(semantic_query)
                })
                
            except Exception as e:
                error_msg = str(e)
                self.log_activity(f"Hybrid search error: {error_msg}")
                
                # Try fallback to semantic search only
                try:
                    if semantic_query and self._is_vector_store_available():
                        query_embedding = self.embeddings.embed_query(semantic_query)
                        semantic_results = self.db.semantic_search_users(
                            query_embedding, 
                            user_id=user_id, 
                            max_results=max_results,
                            min_similarity=self.similarity_threshold * 0.8
                        )
                        
                        formatted_results = []
                        for result in semantic_results:
                            # Apply keyword relevance filter for fallback semantic search (only if semantic query provided)
                            if not semantic_query or self._is_keyword_relevant(semantic_query, result):
                                metadata = result['metadata']
                                formatted_results.append({
                                    'user_id': result['user_id'],
                                    'name': metadata.get('name', ''),
                                    'semantic_score': result['similarity'],
                                    'location_score': 0.0,
                                    'combined_score': result['similarity'],
                                    'city': metadata.get('city', ''),
                                    'state': metadata.get('state', ''),
                                    'privacy_level': metadata.get('privacy_level', 'city_only'),
                                    'tags': metadata.get('tags', [])
                                })
                        
                        return json.dumps({
                            'success': True,
                            'results': formatted_results,
                            'total_found': len(formatted_results),
                            'search_method': 'semantic_fallback_with_keyword_filter',
                            'location_radius_km': location_radius_km,
                            'used_gps_search': False,
                            'used_city_fallback': False,
                            'error_handled': True,
                            'original_error': error_msg,
                            'keyword_filtering_applied': bool(semantic_query)
                        })
                    
                    return json.dumps({
                        'success': False,
                        'results': [],
                        'total_found': 0,
                        'error': f"Hybrid search failed: {error_msg}",
                        'search_method': 'failed'
                    })
                    
                except Exception as fallback_error:
                    return f"Error in hybrid search and fallback: {error_msg}. Fallback error: {str(fallback_error)}"
        
        @tool
        def get_user_similarity_score(user_id1: str, user_id2: str) -> str:
            """Calculate similarity score between two users using embeddings."""
            try:
                if not self.embeddings:
                    return "Embeddings not available"
                
                # Get or create embeddings for both users
                embedding1 = self._get_user_embedding(user_id1)
                embedding2 = self._get_user_embedding(user_id2)
                
                if embedding1 is None or embedding2 is None:
                    return "Could not generate embeddings for one or both users"
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(embedding1, embedding2)
                
                return json.dumps({
                    'user_id1': user_id1,
                    'user_id2': user_id2,
                    'similarity_score': float(similarity),
                    'similarity_level': self._get_similarity_level(similarity)
                })
                
            except Exception as e:
                return f"Error calculating similarity: {str(e)}"
        
        @tool
        def update_user_vector_profile(user_id: str) -> str:
            """Update user's vector profile when their data changes."""
            try:
                # Remove old embedding from cache
                if user_id in self.user_embeddings_cache:
                    del self.user_embeddings_cache[user_id]
                
                # Vectorize updated profile
                result = vectorize_user_profile(user_id)
                
                # Update timestamp
                self.profile_update_timestamps[user_id] = datetime.now()
                
                return f"Updated vector profile for user {user_id}: {result}"
                
            except Exception as e:
                return f"Error updating user vector profile: {str(e)}"
        
        tools.extend([
            vectorize_user_profile,
            semantic_search_nearby_users,
            hybrid_location_semantic_search,
            get_user_similarity_score,
            update_user_vector_profile
        ])
        
        return tools
    
    def _create_profile_text(self, user_profile: Dict, user_tags: List[str], 
                           location_prefs: Dict) -> str:
        """Create balanced text representation of user profile for embedding with keyword diversity."""
        profile_parts = []
        
        # User basic info
        if user_profile.get('name'):
            profile_parts.append(f"User: {user_profile['name']}")
        
        # Process tags with diversity and deduplication
        if user_tags:
            # Deduplicate and normalize tags
            processed_tags = self._process_tags_for_diversity(user_tags)
            
            # Group tags by category for balanced representation
            categorized_tags = self._categorize_tags(processed_tags)
            
            # Create balanced interests text
            interests_parts = []
            for category, tags in categorized_tags.items():
                if tags:
                    # Limit tags per category to prevent dominance
                    limited_tags = tags[:3]  # Max 3 tags per category
                    interests_parts.append(f"{category}: {', '.join(limited_tags)}")
            
            if interests_parts:
                profile_parts.append(f"Interests: {' | '.join(interests_parts)}")
        
        # Location information (reduced weight)
        location_parts = []
        if location_prefs.get('city'):
            location_parts.append(location_prefs['city'])
        if location_prefs.get('state'):
            location_parts.append(location_prefs['state'])
        if location_prefs.get('country'):
            location_parts.append(location_prefs['country'])
        
        if location_parts:
            profile_parts.append(f"Location: {', '.join(location_parts)}")
        
        # Language preferences (reduced weight)
        language_parts = []
        if user_profile.get('native_language'):
            language_parts.append(user_profile['native_language'])
        if user_profile.get('preferred_languages'):
            language_parts.extend(user_profile['preferred_languages'][:2])  # Max 2 languages
        
        if language_parts:
            # Deduplicate languages
            unique_languages = list(set(language_parts))
            profile_parts.append(f"Languages: {', '.join(unique_languages)}")
        
        return " | ".join(profile_parts)
    
    def _process_tags_for_diversity(self, tags: List[str]) -> List[str]:
        """Process tags to reduce redundancy and improve diversity."""
        if not tags:
            return []
        
        # Normalize tags
        normalized_tags = []
        seen_roots = set()
        
        for tag in tags:
            tag_lower = tag.lower().strip()
            
            # Skip very short tags
            if len(tag_lower) < 2:
                continue
            
            # Extract root words to avoid redundancy
            root_words = self._extract_root_words(tag_lower)
            
            # Check if we've seen similar root words
            if not any(root in seen_roots for root in root_words):
                normalized_tags.append(tag)
                seen_roots.update(root_words)
        
        return normalized_tags
    
    def _extract_root_words(self, tag: str) -> List[str]:
        """Extract root words from a tag to detect similarity."""
        # Simple root word extraction
        words = tag.split()
        root_words = []
        
        for word in words:
            # Remove common suffixes
            if word.endswith('ing'):
                root_words.append(word[:-3])
            elif word.endswith('ed'):
                root_words.append(word[:-2])
            elif word.endswith('er'):
                root_words.append(word[:-2])
            elif word.endswith('s') and len(word) > 3:
                root_words.append(word[:-1])
            else:
                root_words.append(word)
        
        return root_words
    
    def _categorize_tags(self, tags: List[str]) -> Dict[str, List[str]]:
        """Categorize tags to ensure balanced representation."""
        categories = {
            'Food & Cuisine': [],
            'Technology': [],
            'Entertainment': [],
            'Location': [],
            'Lifestyle': [],
            'Professional': [],
            'Other': []
        }
        
        # Define category keywords
        category_keywords = {
            'Food & Cuisine': ['food', 'cuisine', 'restaurant', 'cooking', 'dining', 'indian', 'chinese', 'italian', 'mexican', 'asian', 'spicy', 'sweet', 'recipe'],
            'Technology': ['ai', 'technology', 'digital', 'virtual', 'assistant', 'computer', 'software', 'programming', 'tech', 'data', 'analytics', 'machine', 'learning'],
            'Entertainment': ['music', 'bollywood', 'streaming', 'movies', 'tv', 'series', 'entertainment', 'gaming', 'sports', 'dance', 'art', 'culture'],
            'Location': ['frisco', 'texas', 'california', 'india', 'city', 'state', 'country', 'local', 'neighborhood'],
            'Lifestyle': ['fitness', 'health', 'travel', 'outdoor', 'indoor', 'hobby', 'family', 'friends', 'social'],
            'Professional': ['work', 'career', 'business', 'professional', 'job', 'office', 'management', 'leadership']
        }
        
        # Categorize each tag
        for tag in tags:
            tag_lower = tag.lower()
            categorized = False
            
            for category, keywords in category_keywords.items():
                if any(keyword in tag_lower for keyword in keywords):
                    categories[category].append(tag)
                    categorized = True
                    break
            
            if not categorized:
                categories['Other'].append(tag)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _diversify_semantic_results(self, results: List[Dict], max_results: int) -> List[Dict]:
        """Diversify semantic search results to reduce keyword dominance."""
        if not results:
            return results
        
        # Sort by similarity score first
        sorted_results = sorted(results, key=lambda x: x.get('semantic_score', 0), reverse=True)
        
        # Apply diversity scoring
        diversified_results = []
        seen_tag_categories = set()
        
        for result in sorted_results:
            # Calculate diversity score
            user_tags = result.get('tags', [])
            tag_categories = self._get_tag_categories(user_tags)
            
            # Penalize results with categories we've already seen
            diversity_penalty = sum(1 for cat in tag_categories if cat in seen_tag_categories)
            diversity_bonus = len(tag_categories) - diversity_penalty
            
            # Adjust semantic score with diversity
            original_score = result.get('semantic_score', 0)
            diversity_factor = max(0.1, 1.0 - (diversity_penalty * 0.2))  # Reduce by 20% per repeated category
            
            result['diversity_adjusted_score'] = original_score * diversity_factor + (diversity_bonus * 0.05)
            result['diversity_factor'] = diversity_factor
            result['tag_categories'] = tag_categories
            
            diversified_results.append(result)
            seen_tag_categories.update(tag_categories)
        
        # Re-sort by diversity-adjusted score
        diversified_results.sort(key=lambda x: x.get('diversity_adjusted_score', 0), reverse=True)
        
        return diversified_results[:max_results]
    
    def _get_tag_categories(self, tags: List[str]) -> List[str]:
        """Get categories for a list of tags."""
        categorized = self._categorize_tags(tags)
        return list(categorized.keys())
    
    def _is_location_relevant(self, user_location: Dict, other_metadata: Dict) -> bool:
        """Check if another user's location is relevant for search."""
        # If no location preferences set, include all users
        if not user_location:
            return True
        
        # Check privacy level
        privacy_level = other_metadata.get('privacy_level', 'private')
        if privacy_level == 'private':
            return False
        
        # If user has location preferences, check relevance
        user_city = (user_location.get('city') or '').lower()
        user_state = (user_location.get('state') or '').lower()
        user_country = (user_location.get('country') or '').lower()
        
        other_city = (other_metadata.get('city') or '').lower()
        other_state = (other_metadata.get('state') or '').lower()
        other_country = (other_metadata.get('country') or '').lower()
        
        # Same city is highly relevant
        if user_city and other_city and user_city == other_city:
            return True
        
        # Same state is relevant
        if user_state and other_state and user_state == other_state:
            return True
        
        # Same country is somewhat relevant
        if user_country and other_country and user_country == other_country:
            return True
        
        # If no location match but privacy allows, include
        return privacy_level in ['city_only', 'state_only', 'country_only', 'exact']

    def _is_keyword_relevant(self, query: str, result: Dict) -> bool:
        """Check if the query terms are actually relevant to the user's profile."""
        try:
            # Get user's profile text and tags
            profile_text = result.get('profile_text', '') or ''
            profile_text = profile_text.lower()
            
            user_tags = result.get('metadata', {}).get('tags', []) or []
            user_tags_text = ' '.join([str(tag) for tag in user_tags if tag is not None]).lower()
            
            # Normalize query
            query_lower = (query or '').lower().strip()
            
            if not query_lower:
                return True  # Empty query matches everything
            
            # Define related terms for common queries
            query_expansions = {
                'bollywood': ['bollywood', 'hindi cinema', 'hindi film', 'hindi movie', 'mumbai film', 'indian cinema', 'film industry', 'movie', 'cinema', 'entertainment'],
                'technology': ['technology', 'tech', 'programming', 'software', 'computer', 'coding', 'ai', 'artificial intelligence', 'machine learning', 'developer', 'engineering'],
                'food': ['food', 'cooking', 'cuisine', 'restaurant', 'chef', 'recipe', 'dining', 'culinary', 'meal', 'dish'],
                'music': ['music', 'song', 'singer', 'artist', 'instrument', 'concert', 'album', 'melody', 'rhythm', 'classical'],
                'travel': ['travel', 'trip', 'vacation', 'tourism', 'destination', 'journey', 'adventure', 'explore', 'wanderlust'],
                'sports': ['sports', 'game', 'fitness', 'exercise', 'athlete', 'competition', 'team', 'cricket', 'football', 'basketball'],
                'art': ['art', 'artist', 'painting', 'drawing', 'creative', 'design', 'sculpture', 'gallery', 'exhibition'],
                'business': ['business', 'entrepreneur', 'startup', 'company', 'finance', 'marketing', 'management', 'corporate', 'professional'],
                'education': ['education', 'learning', 'study', 'school', 'university', 'course', 'teaching', 'knowledge', 'academic'],
                'health': ['health', 'fitness', 'wellness', 'medical', 'doctor', 'exercise', 'nutrition', 'yoga', 'meditation']
            }
            
            # Get relevant terms for the query
            relevant_terms = []
            
            # Direct query match
            relevant_terms.append(query_lower)
            
            # Query expansion
            for key, expansions in query_expansions.items():
                if key in query_lower or query_lower in expansions:
                    relevant_terms.extend(expansions)
                    break
            
            # If no expansion found, use query words
            if len(relevant_terms) == 1:
                relevant_terms.extend(query_lower.split())
            
            # Check if any relevant terms appear in profile or tags
            combined_text = f"{profile_text} {user_tags_text}"
            
            for term in relevant_terms:
                if term and term in combined_text:
                    return True
            
            # Special case: if semantic score is very high (>0.9), allow it through
            if result.get('similarity', 0) > 0.9:
                return True
            
            return False
            
        except Exception as e:
            # If there's an error, default to allowing the result
            return True
    
    def _get_user_embedding(self, user_id: str) -> Optional[List[float]]:
        """Get or create embedding for a user."""
        try:
            # Check cache first
            if user_id in self.user_embeddings_cache:
                return self.user_embeddings_cache[user_id]['embedding']
            
            # Create new embedding
            if not self.db or not self.embeddings:
                return None
            
            user_profile = self.db.get_user_profile(user_id)
            user_tags = self.db.get_user_tags(user_id)
            location_prefs = self.db.get_location_preferences(user_id)
            
            if not user_profile:
                return None
            
            profile_text = self._create_profile_text(user_profile, user_tags, location_prefs)
            embedding = self.embeddings.embed_query(profile_text)
            
            # Cache the embedding
            self.user_embeddings_cache[user_id] = {
                'embedding': embedding,
                'profile_text': profile_text,
                'timestamp': datetime.now(),
                'location': location_prefs
            }
            
            return embedding
            
        except Exception as e:
            self.log_activity(f"Error getting user embedding: {str(e)}")
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            self.log_activity(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    def _get_similarity_level(self, similarity_score: float) -> str:
        """Get human-readable similarity level."""
        if similarity_score >= 0.9:
            return "Very High"
        elif similarity_score >= 0.8:
            return "High"
        elif similarity_score >= 0.7:
            return "Medium"
        elif similarity_score >= 0.6:
            return "Low"
        else:
            return "Very Low"
    
    def _combine_location_semantic_results(self, nearby_users: List[Dict], 
                                         semantic_results: List[Dict], 
                                         max_results: int) -> List[Dict]:
        """Combine location-based and semantic search results."""
        combined = {}
        
        # Add location-based results
        for user in nearby_users:
            user_id = user['user_id']
            # Get user tags from database
            user_tags = self.db.get_user_tags(user_id) if self.db else []
            
            combined[user_id] = {
                'user_id': user_id,
                'name': user['name'],
                'location_score': 1.0 / (1.0 + user.get('distance_km', 0) / 10),  # Closer = higher score
                'semantic_score': 0.0,
                'combined_score': 0.0,
                'distance_km': user.get('distance_km', 0),
                'city': user.get('city', ''),
                'state': user.get('state', ''),
                'privacy_level': user.get('privacy_level', 'city_only'),
                'tags': user_tags  # Use actual tags from database
            }
        
        # Add semantic results
        for result in semantic_results:
            user_id = result['user_id']
            if user_id in combined:
                combined[user_id]['semantic_score'] = result['semantic_score']
            else:
                metadata = result['metadata']
                combined[user_id] = {
                    'user_id': user_id,
                    'name': metadata.get('name', ''),
                    'location_score': 0.0,
                    'semantic_score': result['semantic_score'],
                    'combined_score': 0.0,
                    'distance_km': None,
                    'city': metadata.get('city', ''),
                    'state': metadata.get('state', ''),
                    'privacy_level': metadata.get('privacy_level', 'city_only'),
                    'tags': metadata.get('tags', [])
                }
        
        # Calculate combined scores
        for user_id, user_data in combined.items():
            location_score = user_data['location_score']
            semantic_score = user_data['semantic_score']
            
            # Weighted combination
            combined_score = (self.location_weight * location_score + 
                            (1 - self.location_weight) * semantic_score)
            
            user_data['combined_score'] = combined_score
        
        # Sort by combined score and return top results
        sorted_results = sorted(combined.values(), key=lambda x: x['combined_score'], reverse=True)
        return sorted_results[:max_results]
    
    def _get_agent_system_prompt(self) -> str:
        """Get the system prompt specific to this agent."""
        return """You are the ReactRAGNearbyUsersAgent, specialized in finding nearby users using advanced RAG (Retrieval-Augmented Generation) techniques.

Your capabilities include:
- Vectorizing user profiles for semantic search
- Performing semantic similarity searches across user profiles
- Combining location-based and semantic search results
- Calculating user similarity scores using embeddings
- Providing intelligent user recommendations

You use React AI pattern to:
1. OBSERVE: Analyze user profiles and location data
2. THINK: Reason about the best search strategy (location vs semantic vs hybrid)
3. ACT: Execute appropriate search tools and algorithms
4. REFLECT: Evaluate search results and improve recommendations

Focus on finding users who are both geographically nearby and semantically similar in interests, creating meaningful connections through intelligent matching."""
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override base process_request to handle RAG-specific requests directly.
        
        Args:
            request: Request dictionary with parameters
            
        Returns:
            Response dictionary with results
        """
        request_type = request.get('type', '')
        
        # Handle RAG-specific requests directly
        if request_type == 'rag_nearby_users':
            return self.process_rag_nearby_request(request)
        
        # For other request types, fall back to React AI loop
        return super().process_request(request)
    
    def process_rag_nearby_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a RAG-based nearby users request.
        
        Args:
            request: Request dictionary with search parameters
            
        Returns:
            Response dictionary with nearby users found using RAG
        """
        try:
            self.update_activity()
            
            # Extract request parameters
            user_id = request.get('user_id')
            search_type = request.get('search_type', 'hybrid')  # 'location', 'semantic', 'hybrid'
            location_radius_km = request.get('location_radius_km', 50)
            semantic_query = request.get('semantic_query', '')
            max_results = request.get('max_results', 10)
            
            if not user_id:
                return {
                    'success': False,
                    'error': 'Missing user_id parameter'
                }
            
            # Ensure user profile is vectorized
            self._ensure_user_vectorized(user_id)
            
            # Get the tools directly
            tools = self._get_agent_specific_tools()
            
            # Find the appropriate tool based on search type
            target_tool = None
            if search_type == 'semantic':
                # Use semantic search tool with keyword filtering
                for tool in tools:
                    if tool.name == 'semantic_search_nearby_users':
                        target_tool = tool
                        break
            else:
                # Use hybrid search tool for hybrid and location searches
                for tool in tools:
                    if tool.name == 'hybrid_location_semantic_search':
                        target_tool = tool
                        break
            
            if not target_tool:
                return {
                    'success': False,
                    'error': f'Search tool not found for type: {search_type}',
                    'nearby_users': []
                }
            
            # Call the appropriate tool
            if search_type == 'semantic':
                tool_result = target_tool.func(user_id, semantic_query, max_results)
            else:
                tool_result = target_tool.func(user_id, location_radius_km, semantic_query, max_results)
            
            # Parse the tool result (it's a JSON string)
            if isinstance(tool_result, str):
                import json
                try:
                    parsed_result = json.loads(tool_result)
                    if parsed_result.get('success'):
                        # Extract results and map to expected format
                        results = parsed_result.get('results', [])
                        
                        # Ensure each result has required tags field
                        for result in results:
                            if 'tags' not in result or not result['tags']:
                                # Get tags from database as fallback
                                user_tags = self.db.get_user_tags(result['user_id']) if self.db else []
                                result['tags'] = user_tags
                        
                        return {
                            'success': True,
                            'nearby_users': results,  # Map 'results' to 'nearby_users'
                            'total_found': parsed_result.get('total_found', len(results)),
                            'search_method': parsed_result.get('search_method', search_type),
                            'framework': 'React AI + RAG',
                            'rag_enabled': True,
                            'vector_search_available': self._is_vector_store_available(),
                            'embeddings_available': self.embeddings is not None,
                            'location_radius_km': location_radius_km,
                            'semantic_query': semantic_query,
                            'used_gps_search': parsed_result.get('used_gps_search', False),
                            'used_city_fallback': parsed_result.get('used_city_fallback', False),
                            'keyword_filtering_enabled': search_type in ['semantic', 'hybrid']
                        }
                    else:
                        return {
                            'success': False,
                            'error': f'Tool search failed: {tool_result}',
                            'nearby_users': []
                        }
                except json.JSONDecodeError:
                    return {
                        'success': False,
                        'error': f'Failed to parse tool result: {tool_result}',
                        'nearby_users': []
                    }
            else:
                return {
                    'success': False,
                    'error': f'Unexpected tool result format: {type(tool_result)}',
                    'nearby_users': []
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error processing RAG nearby request: {str(e)}',
                'nearby_users': []
            }
    
    def _ensure_user_vectorized(self, user_id: str):
        """Ensure user profile is vectorized and up-to-date in MongoDB."""
        try:
            # Check if user needs vectorization
            needs_vectorization = False
            
            # Check cache first
            if user_id not in self.user_embeddings_cache:
                needs_vectorization = True
            elif self._is_profile_outdated(user_id):
                needs_vectorization = True
            else:
                # Check if embedding exists in MongoDB
                stored_embedding = self.db.get_user_embedding(user_id) if self.db else None
                if not stored_embedding:
                    needs_vectorization = True
            
            if needs_vectorization and self._is_vector_store_available():
                # Vectorize user profile
                user_profile = self.db.get_user_profile(user_id)
                user_tags = self.db.get_user_tags(user_id)
                location_prefs = self.db.get_location_preferences(user_id)
                
                if user_profile:
                    profile_text = self._create_profile_text(user_profile, user_tags, location_prefs)
                    embedding = self.embeddings.embed_query(profile_text)
                    
                    # Store in MongoDB
                    metadata = {
                        'user_id': user_id,
                        'name': user_profile.get('name', ''),
                        'tags': user_tags,
                        'city': location_prefs.get('city', ''),
                        'state': location_prefs.get('state', ''),
                        'country': location_prefs.get('country', ''),
                        'coordinates': location_prefs.get('coordinates', {}),
                        'privacy_level': location_prefs.get('privacy_level', 'city_only')
                    }
                    
                    result = self.db.store_user_embedding(user_id, embedding, profile_text, metadata)
                    
                    # Update cache
                    self.user_embeddings_cache[user_id] = {
                        'embedding': embedding,
                        'profile_text': profile_text,
                        'timestamp': datetime.now(),
                        'location': location_prefs,
                        'metadata': metadata
                    }
                    
                    if result:
                        self.log_activity(f"Vectorized profile for user {user_id}")
                    else:
                        self.log_activity(f"Failed to store embedding for user {user_id}")
                        
        except Exception as e:
            self.log_activity(f"Error ensuring user vectorization: {str(e)}")
    
    def _is_profile_outdated(self, user_id: str) -> bool:
        """Check if user's profile vectorization is outdated."""
        if user_id not in self.user_embeddings_cache:
            return True
        
        # Check if profile was updated recently (within last hour)
        cached_time = self.user_embeddings_cache[user_id]['timestamp']
        time_diff = datetime.now() - cached_time
        
        return time_diff.total_seconds() > 3600  # 1 hour
    
    def get_rag_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics for MongoDB vector store."""
        try:
            # Get MongoDB embedding statistics
            total_embeddings = 0
            if self.db:
                # Count embeddings in MongoDB
                total_embeddings = self.db.embeddings_collection.count_documents({})
            
            return {
                'rag_enabled': self.embeddings is not None,
                'vector_store_type': 'MongoDB',
                'vector_store_available': self._is_vector_store_available(),
                'embeddings_available': self.embeddings is not None,
                'cached_embeddings': len(self.user_embeddings_cache),
                'total_stored_embeddings': total_embeddings,
                'similarity_threshold': self.similarity_threshold,
                'location_weight': self.location_weight,
                'embedding_dimension': self.embedding_dimension,
                'max_nearby_users': self.max_nearby_users,
                'geospatial_indexing': True,
                'hybrid_search_available': True,
                'framework': 'React AI + RAG + MongoDB',
                'last_activity': self.last_activity.isoformat()
            }
        except Exception as e:
            return {
                'rag_enabled': False,
                'error': str(e),
                'vector_store_type': 'MongoDB (Error)',
                'framework': 'React AI + RAG + MongoDB',
                'last_activity': self.last_activity.isoformat()
            } 