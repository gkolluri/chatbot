"""
Citation System for AI Responses in Group Chats
=============================================

This module provides citation generation and display functionality for AI responses
in group conversations, similar to ChatGPT's source links but with a more subtle
and integrated approach.
"""

import os
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from openai import OpenAI
from dataclasses import dataclass, asdict
from logging_utils import citation_logger, debug, info, error


@dataclass
class Citation:
    """Represents a citation for an AI response"""
    id: str
    type: str  # 'user_profile', 'conversation_context', 'group_topic', 'language_preference', 'cultural_context', 'web_search'
    source: str
    content: str
    relevance_score: float
    timestamp: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert citation to dictionary for database storage"""
        return {
            'id': self.id,
            'type': self.type,
            'source': self.source,
            'content': self.content,
            'relevance_score': self.relevance_score,
            'timestamp': self.timestamp,
            'metadata': self.metadata or {}
        }


class CitationGenerator:
    """Enhanced citation generator with comprehensive logging"""
    
    def __init__(self, db_interface=None):
        self.db = db_interface
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        info("CitationGenerator initialized", db_available=db_interface is not None, openai_available=self.client is not None)
    
    def generate_citations_for_response(self, response_text: str, group_context: Dict[str, Any], 
                                      user_message: str, user_id: str, participants: List[str]) -> List[Citation]:
        """Generate comprehensive citations for AI response with enhanced logging"""
        start_time = time.time()
        
        info(f"Generating citations for response", user_id=user_id, response_length=len(response_text))
        
        try:
            citations = []
            
            # Generate different types of citations
            citation_types = []
            
            # User profile citations
            profile_citations = self._generate_user_profile_citations(response_text, user_id)
            citations.extend(profile_citations)
            if profile_citations:
                citation_types.append("user_profile")
            
            # Conversation context citations
            context_citations = self._generate_conversation_context_citations(response_text, group_context, user_message)
            citations.extend(context_citations)
            if context_citations:
                citation_types.append("conversation_context")
            
            # Group topic citations
            topic_citations = self._generate_group_topic_citations(response_text, group_context)
            citations.extend(topic_citations)
            if topic_citations:
                citation_types.append("group_topic")
            
            # Language preference citations
            language_citations = self._generate_language_preference_citations(response_text, user_id)
            citations.extend(language_citations)
            if language_citations:
                citation_types.append("language_preference")
            
            # Cultural context citations
            cultural_citations = self._generate_cultural_context_citations(response_text, user_id)
            citations.extend(cultural_citations)
            if cultural_citations:
                citation_types.append("cultural_context")
            
            # RAG context citations (topic-focused)
            rag_citations = self._generate_rag_context_citations(response_text, group_context, user_id, participants)
            citations.extend(rag_citations)
            if rag_citations:
                citation_types.append("rag_context")
            
            # Web search citations
            web_citations = self._generate_web_search_citations(response_text, user_message)
            citations.extend(web_citations)
            if web_citations:
                citation_types.append("web_search")
            
            processing_time = time.time() - start_time
            
            citation_logger.citations_generated(
                citation_count=len(citations),
                citation_types=citation_types,
                response_preview=response_text,
                user_id=user_id
            )
            
            info(f"Generated {len(citations)} citations in {processing_time:.3f}s", 
                 user_id=user_id, citation_count=len(citations), citation_types=citation_types, 
                 processing_time=processing_time)
            
            return citations
            
        except Exception as e:
            error(f"Error generating citations", error=e, user_id=user_id)
            return []
    
    def _generate_user_profile_citations(self, response_text: str, user_id: str) -> List[Citation]:
        """Generate citations based on user profile with enhanced relevance detection"""
        citations = []
        
        try:
            if not self.db:
                return citations
            
            user_profile = self.db.get_user_profile(user_id)
            if not user_profile:
                return citations
            
            user_tags = self.db.get_user_tags(user_id)
            
            # Check if response mentions user interests (exact match)
            for tag in user_tags:
                if tag.lower() in response_text.lower():
                    citation = Citation(
                        id=f"user_profile_{tag}",
                        type="user_profile",
                        source="User Profile",
                        content=f"Interest in {tag}",
                        relevance_score=0.8,
                        timestamp=datetime.now().isoformat(),
                        metadata={
                            "user_id": user_id,
                            "tag": tag,
                            "profile_name": user_profile.get('name', 'Unknown'),
                            "match_type": "exact"
                        }
                    )
                    citations.append(citation)
                    
                    citation_logger.topic_citation(
                        citation_type="user_profile",
                        topic_name=tag,
                        content=f"Interest in {tag}",
                        relevance_score=0.8,
                        user_id=user_id
                    )
            
            # Enhanced relevance detection for food-related responses
            if self._is_food_related_response(response_text):
                for tag in user_tags:
                    if self._is_food_related_tag(tag) and not any(c.metadata.get('tag') == tag for c in citations):
                        citation = Citation(
                            id=f"user_profile_{tag}_contextual",
                            type="user_profile",
                            source="User Profile",
                            content=f"Food preference: {tag}",
                            relevance_score=0.9,  # Higher relevance for contextual food matches
                            timestamp=datetime.now().isoformat(),
                            metadata={
                                "user_id": user_id,
                                "tag": tag,
                                "profile_name": user_profile.get('name', 'Unknown'),
                                "match_type": "contextual_food"
                            }
                        )
                        citations.append(citation)
                        
                        citation_logger.topic_citation(
                            citation_type="user_profile",
                            topic_name=tag,
                            content=f"Food preference: {tag}",
                            relevance_score=0.9,
                            user_id=user_id
                        )
            
            # Enhanced relevance detection for restaurant-related responses
            if self._is_restaurant_related_response(response_text):
                for tag in user_tags:
                    if self._is_restaurant_related_tag(tag) and not any(c.metadata.get('tag') == tag for c in citations):
                        citation = Citation(
                            id=f"user_profile_{tag}_restaurant",
                            type="user_profile",
                            source="User Profile",
                            content=f"Restaurant preference: {tag}",
                            relevance_score=0.85,
                            timestamp=datetime.now().isoformat(),
                            metadata={
                                "user_id": user_id,
                                "tag": tag,
                                "profile_name": user_profile.get('name', 'Unknown'),
                                "match_type": "contextual_restaurant"
                            }
                        )
                        citations.append(citation)
                        
                        citation_logger.topic_citation(
                            citation_type="user_profile",
                            topic_name=tag,
                            content=f"Restaurant preference: {tag}",
                            relevance_score=0.85,
                            user_id=user_id
                        )
            
            debug(f"Generated {len(citations)} user profile citations", 
                  user_id=user_id, user_tags=user_tags, citations_count=len(citations))
            
        except Exception as e:
            error(f"Error generating user profile citations", error=e, user_id=user_id)
        
        return citations
    
    def _is_food_related_response(self, response_text: str) -> bool:
        """Check if response is related to food/dining"""
        food_keywords = [
            'restaurant', 'dining', 'food', 'cuisine', 'meal', 'lunch', 'dinner', 
            'breakfast', 'brunch', 'eat', 'eating', 'dish', 'menu', 'chef', 
            'cooking', 'recipe', 'cafe', 'bistro', 'bar', 'grill', 'kitchen',
            'taste', 'flavor', 'spicy', 'sweet', 'savory', 'delicious'
        ]
        response_lower = response_text.lower()
        return any(keyword in response_lower for keyword in food_keywords)
    
    def _is_food_related_tag(self, tag: str) -> bool:
        """Check if tag is related to food preferences"""
        food_related_tags = [
            'south indian food', 'north indian food', 'indian food', 'chinese food',
            'italian food', 'mexican food', 'thai food', 'japanese food', 'korean food',
            'american food', 'fast food', 'street food', 'vegetarian', 'vegan',
            'spicy food', 'dessert', 'cuisine', 'cooking', 'recipe', 'chef',
            'bakery', 'cafe', 'dining', 'lunch', 'dinner', 'breakfast', 'brunch'
        ]
        return tag.lower() in food_related_tags
    
    def _is_restaurant_related_response(self, response_text: str) -> bool:
        """Check if response is specifically about restaurants/places to eat"""
        restaurant_keywords = [
            'restaurant', 'place to eat', 'dining', 'bistro', 'cafe', 'bar',
            'grill', 'eatery', 'food court', 'buffet', 'takeout', 'delivery',
            'reservation', 'menu', 'recommend', 'suggestion', 'location'
        ]
        response_lower = response_text.lower()
        return any(keyword in response_lower for keyword in restaurant_keywords)
    
    def _is_restaurant_related_tag(self, tag: str) -> bool:
        """Check if tag is related to restaurant preferences"""
        restaurant_related_tags = [
            'restaurant', 'dining', 'south indian food', 'north indian food', 
            'indian food', 'chinese food', 'italian food', 'mexican food', 
            'thai food', 'japanese food', 'korean food', 'american food',
            'fast food', 'street food', 'cuisine', 'cafe', 'bakery',
            'lunch', 'dinner', 'breakfast', 'brunch'
        ]
        return tag.lower() in restaurant_related_tags
    
    def _generate_conversation_context_citations(self, response_text: str, group_context: Dict[str, Any], 
                                               user_message: str) -> List[Citation]:
        """Generate citations based on conversation context with logging"""
        citations = []
        
        try:
            recent_messages = group_context.get('recent_messages', [])
            
            # Check if response references recent conversation
            for msg in recent_messages[-3:]:  # Last 3 messages
                if msg['message_type'] == 'user' and msg['message'] in response_text:
                    citation = Citation(
                        id=f"conversation_context_{msg.get('timestamp', 'unknown')}",
                        type="conversation_context",
                        source="Recent Conversation",
                        content=f"Reference to previous message: \"{msg['message'][:50]}...\"",
                        relevance_score=0.7,
                        timestamp=datetime.now().isoformat(),
                        metadata={
                            "original_message": msg['message'],
                            "message_timestamp": msg.get('timestamp'),
                            "user_id": msg.get('user_id')
                        }
                    )
                    citations.append(citation)
            
            debug(f"Generated {len(citations)} conversation context citations", 
                  recent_messages_count=len(recent_messages), citations_count=len(citations))
            
        except Exception as e:
            error(f"Error generating conversation context citations", error=e)
        
        return citations
    
    def _generate_group_topic_citations(self, response_text: str, group_context: Dict[str, Any]) -> List[Citation]:
        """Generate citations based on group topic with logging"""
        citations = []
        
        try:
            topic_name = group_context.get('topic_name', '')
            if topic_name and topic_name.lower() in response_text.lower():
                citation = Citation(
                    id=f"group_topic_{topic_name.replace(' ', '_')}",
                    type="group_topic",
                    source="Group Topic",
                    content=f"Group discussion topic: {topic_name}",
                    relevance_score=0.9,
                    timestamp=datetime.now().isoformat(),
                    metadata={
                        "topic_name": topic_name,
                        "group_context": group_context
                    }
                )
                citations.append(citation)
                
                debug(f"Generated group topic citation", topic_name=topic_name)
            
        except Exception as e:
            error(f"Error generating group topic citations", error=e)
        
        return citations
    
    def _generate_language_preference_citations(self, response_text: str, user_id: str) -> List[Citation]:
        """Generate citations based on language preferences with logging"""
        citations = []
        
        try:
            if not self.db:
                return citations
            
            user_profile = self.db.get_user_profile(user_id)
            if not user_profile:
                return citations
            
            native_language = user_profile.get('native_language')
            if native_language and native_language.lower() in response_text.lower():
                citation = Citation(
                    id=f"language_preference_{native_language}",
                    type="language_preference",
                    source="Language Preferences",
                    content=f"Native language: {native_language}",
                    relevance_score=0.6,
                    timestamp=datetime.now().isoformat(),
                    metadata={
                        "user_id": user_id,
                        "native_language": native_language
                    }
                )
                citations.append(citation)
                
                debug(f"Generated language preference citation", user_id=user_id, native_language=native_language)
            
        except Exception as e:
            error(f"Error generating language preference citations", error=e, user_id=user_id)
        
        return citations
    
    def _generate_cultural_context_citations(self, response_text: str, user_id: str) -> List[Citation]:
        """Generate citations based on cultural context with logging"""
        citations = []
        
        try:
            if not self.db:
                return citations
            
            user_profile = self.db.get_user_profile(user_id)
            if not user_profile:
                return citations
            
            # Check for cultural elements in response
            cultural_keywords = ['indian', 'culture', 'tradition', 'festival', 'cuisine', 'language']
            for keyword in cultural_keywords:
                if keyword.lower() in response_text.lower():
                    citation = Citation(
                        id=f"cultural_context_{keyword}",
                        type="cultural_context",
                        source="Cultural Context",
                        content=f"Cultural reference: {keyword}",
                        relevance_score=0.7,
                        timestamp=datetime.now().isoformat(),
                        metadata={
                            "user_id": user_id,
                            "cultural_keyword": keyword
                        }
                    )
                    citations.append(citation)
                    
                    debug(f"Generated cultural context citation", user_id=user_id, cultural_keyword=keyword)
            
        except Exception as e:
            error(f"Error generating cultural context citations", error=e, user_id=user_id)
        
        return citations
    
    def _generate_rag_context_citations(self, response_text: str, group_context: Dict[str, Any], user_id: str, participants: List[str]) -> List[Citation]:
        """Generate citations based on RAG context (user tags, location, mutual interests)"""
        citations = []
        
        if not self.db:
            return citations
        
        try:
            # Get RAG context from group context
            rag_context = group_context.get('rag_context', '')
            topic_name = group_context.get('topic_name', '')
            
            # Generate citations for ALL shared interests (not just topic-related)
            shared_interests = self._get_shared_interests(participants, topic_name)
            for interest, users in shared_interests.items():
                if interest.lower() in response_text.lower():
                    # Check if this is topic-related for appropriate labeling
                    is_topic_related = self._is_tag_related_to_topic(interest, topic_name)
                    
                    citations.append(Citation(
                        id=f"shared_interest_{interest}",
                        type="shared_interest",
                        source="RAG Context",
                        content=f"Shared interest in {interest} among {', '.join(users)}",
                        relevance_score=0.9 if is_topic_related else 0.7,  # Higher score for topic-related interests
                        timestamp=datetime.now().isoformat(),
                        metadata={
                            "interest": interest,
                            "participants": users,
                            "topic_name": topic_name,
                            "is_topic_related": is_topic_related
                        }
                    ))
            
            # Generate citations for topic-relevant location context
            topic_location_citations = self._generate_topic_relevant_location_citations(response_text, participants, topic_name)
            citations.extend(topic_location_citations)
            
            # Generate citations for topic-related user interests
            topic_interest_citations = self._generate_topic_interest_citations(response_text, participants, topic_name)
            citations.extend(topic_interest_citations)
            
        except Exception as e:
            print(f"Error generating RAG context citations: {e}")
        
        return citations
    
    def _get_topic_related_shared_interests(self, participants: List[str], topic_name: str) -> Dict[str, List[str]]:
        """Get shared interests that are directly related to the group topic"""
        shared_interests = {}
        
        try:
            all_user_tags = {}
            
            # Collect all user tags
            for user_id in participants:
                if user_id != "ai_bot":
                    user_tags = self.db.get_user_tags(user_id)
                    all_user_tags[user_id] = user_tags
            
            # Find shared tags that are topic-related
            for user_id, tags in all_user_tags.items():
                for tag in tags:
                    if tag in shared_interests:
                        shared_interests[tag].append(user_id)
                    else:
                        shared_interests[tag] = [user_id]
            
            # Filter to only topic-related tags shared by multiple users
            topic_related_shared = {}
            for tag, users in shared_interests.items():
                if len(users) > 1 and self._is_tag_related_to_topic(tag, topic_name):
                    topic_related_shared[tag] = users
            
            return topic_related_shared
            
        except Exception as e:
            print(f"Error getting topic-related shared interests: {e}")
            return {}
    
    def _is_tag_related_to_topic(self, tag: str, topic_name: str) -> bool:
        """Check if a tag is related to the group topic"""
        tag_lower = tag.lower()
        topic_lower = topic_name.lower()
        
        # Direct match
        if tag_lower in topic_lower or topic_lower in tag_lower:
            return True
        
        # Topic-tag mappings
        topic_mappings = {
            'technology': ['tech', 'programming', 'coding', 'software', 'computer', 'ai', 'machine learning', 'python', 'javascript', 'web development'],
            'food': ['cooking', 'recipe', 'restaurant', 'cuisine', 'chef', 'indian food', 'south indian food', 'north indian food', 'chinese food', 'italian food', 'mexican food', 'thai food', 'japanese food', 'korean food', 'american food', 'fast food', 'street food', 'vegetarian', 'vegan', 'spicy food', 'dessert', 'bakery', 'cafe', 'dining', 'lunch', 'dinner', 'breakfast', 'brunch'],
            'sports': ['fitness', 'exercise', 'gym', 'athletics', 'game', 'football', 'basketball', 'soccer', 'tennis', 'swimming', 'running', 'yoga', 'cycling'],
            'music': ['song', 'artist', 'concert', 'band', 'melody', 'guitar', 'piano', 'singing', 'classical', 'rock', 'pop', 'jazz', 'blues'],
            'travel': ['trip', 'vacation', 'destination', 'tourism', 'explore', 'adventure', 'backpacking', 'cruise', 'flight', 'hotel'],
            'business': ['entrepreneur', 'startup', 'company', 'work', 'career', 'management', 'finance', 'marketing', 'sales'],
            'education': ['learning', 'study', 'course', 'school', 'university', 'college', 'teaching', 'research', 'academic'],
            'health': ['fitness', 'wellness', 'medical', 'exercise', 'nutrition', 'doctor', 'medicine', 'healthy', 'diet']
        }
        
        # Check if tag matches any topic mapping
        for topic_key, related_tags in topic_mappings.items():
            if topic_key in topic_lower:
                if any(related_tag in tag_lower for related_tag in related_tags):
                    return True
        
        # Check if any topic mapping words are in the tag
        for topic_key, related_tags in topic_mappings.items():
            if topic_key in topic_lower:
                if tag_lower in related_tags:
                    return True
        
        return False
    
    def _get_shared_interests(self, participants: List[str], topic_name: str) -> Dict[str, List[str]]:
        """Get shared interests among participants"""
        shared_interests = {}
        
        try:
            all_user_tags = {}
            
            # Collect all user tags
            for user_id in participants:
                if user_id != "ai_bot":
                    user_tags = self.db.get_user_tags(user_id)
                    all_user_tags[user_id] = user_tags
            
            # Find shared tags
            for user_id, tags in all_user_tags.items():
                for tag in tags:
                    if tag in shared_interests:
                        shared_interests[tag].append(user_id)
                    else:
                        shared_interests[tag] = [user_id]
            
            # Filter to only tags shared by multiple users
            return {tag: users for tag, users in shared_interests.items() if len(users) > 1}
            
        except Exception as e:
            print(f"Error getting shared interests: {e}")
            return {}
    
    def _generate_topic_relevant_location_citations(self, response_text: str, participants: List[str], topic_name: str) -> List[Citation]:
        """Generate citations based on location context only if relevant to the topic"""
        citations = []
        
        try:
            locations = []
            cities = set()
            
            for user_id in participants:
                if user_id != "ai_bot":
                    location = self.db.get_location_preferences(user_id)
                    if location and location.get('city'):
                        cities.add(location['city'])
                        locations.append({
                            'user_id': user_id,
                            'city': location['city'],
                            'state': location.get('state', ''),
                            'country': location.get('country', '')
                        })
            
            # Check if location is relevant to the topic
            topic_lower = topic_name.lower()
            
            # Define location-relevant topics
            location_relevant_topics = {
                'travel': ['travel', 'tourism', 'vacation', 'destination', 'explore'],
                'business': ['business', 'startup', 'entrepreneur', 'company', 'work'],
                'food': ['food', 'cooking', 'restaurant', 'cuisine', 'dining'],
                'sports': ['sports', 'fitness', 'exercise', 'gym', 'athletics'],
                'technology': ['technology', 'tech', 'startup', 'innovation', 'silicon valley'],
                'education': ['education', 'university', 'college', 'school', 'learning'],
                'culture': ['culture', 'arts', 'museum', 'theater', 'festival']
            }
            
            # Check if topic is location-relevant
            is_location_relevant = False
            for topic_key, keywords in location_relevant_topics.items():
                if any(keyword in topic_lower for keyword in keywords):
                    is_location_relevant = True
                    break
            
            # Also check for city-specific topics
            city_specific_topics = ['san francisco', 'new york', 'los angeles', 'chicago', 'boston', 'seattle']
            if any(city in topic_lower for city in city_specific_topics):
                is_location_relevant = True
            
            if not is_location_relevant:
                return citations
            
            # Check if response mentions any of the cities
            for location_info in locations:
                city = location_info['city']
                if city.lower() in response_text.lower():
                    citations.append(Citation(
                        id=f"topic_location_{city}",
                        type="topic_location_context",
                        source="RAG Context",
                        content=f"Topic-relevant location context: {city}",
                        relevance_score=0.8,
                        timestamp=datetime.now().isoformat(),
                        metadata={
                            "city": city,
                            "state": location_info['state'],
                            "country": location_info['country'],
                            "user_id": location_info['user_id'],
                            "topic_name": topic_name,
                            "is_topic_relevant": True
                        }
                    ))
            
            # Check for regional context if multiple users are in same area
            if len(cities) == 1:
                city = list(cities)[0]
                if city.lower() in response_text.lower():
                    citations.append(Citation(
                        id=f"topic_regional_context_{city}",
                        type="topic_regional_context",
                        source="RAG Context",
                        content=f"All participants in {city} - relevant to {topic_name}",
                        relevance_score=0.9,
                        timestamp=datetime.now().isoformat(),
                        metadata={
                            "city": city,
                            "participant_count": len(participants) - 1,  # Exclude AI bot
                            "topic_name": topic_name,
                            "is_topic_relevant": True
                        }
                    ))
                    
        except Exception as e:
            print(f"Error generating topic-relevant location citations: {e}")
        
        return citations
    
    def _generate_location_citations(self, response_text: str, participants: List[str], topic_name: str) -> List[Citation]:
        """Generate citations based on location context"""
        citations = []
        
        try:
            locations = []
            cities = set()
            
            for user_id in participants:
                if user_id != "ai_bot":
                    location = self.db.get_location_preferences(user_id)
                    if location and location.get('city'):
                        cities.add(location['city'])
                        locations.append({
                            'user_id': user_id,
                            'city': location['city'],
                            'state': location.get('state', ''),
                            'country': location.get('country', '')
                        })
            
            # Check if response mentions any of the cities
            for location_info in locations:
                city = location_info['city']
                if city.lower() in response_text.lower():
                    citations.append(Citation(
                        id=f"location_{city}",
                        type="location_context",
                        source="RAG Context",
                        content=f"Location-based context: {city}",
                        relevance_score=0.7,
                        timestamp=datetime.now().isoformat(),
                        metadata={
                            "city": city,
                            "state": location_info['state'],
                            "country": location_info['country'],
                            "user_id": location_info['user_id']
                        }
                    ))
            
            # Check for regional context if multiple users are in same area
            if len(cities) == 1:
                city = list(cities)[0]
                if city.lower() in response_text.lower():
                    citations.append(Citation(
                        id=f"regional_context_{city}",
                        type="regional_context",
                        source="RAG Context",
                        content=f"All participants in {city}",
                        relevance_score=0.8,
                        timestamp=datetime.now().isoformat(),
                        metadata={
                            "city": city,
                            "participant_count": len(participants) - 1  # Exclude AI bot
                        }
                    ))
                    
        except Exception as e:
            print(f"Error generating location citations: {e}")
        
        return citations
    
    def _generate_topic_interest_citations(self, response_text: str, participants: List[str], topic_name: str) -> List[Citation]:
        """Generate citations for topic-related user interests"""
        citations = []
        
        try:
            topic_lower = topic_name.lower()
            
            # Define topic-interest mappings
            topic_mappings = {
                'technology': ['python', 'programming', 'coding', 'ai', 'machine learning', 'software', 'javascript', 'web development'],
                'food': ['cooking', 'recipe', 'restaurant', 'cuisine', 'chef', 'indian food', 'south indian food', 'north indian food', 'chinese food', 'italian food', 'mexican food', 'thai food', 'japanese food', 'korean food', 'american food', 'fast food', 'street food', 'vegetarian', 'vegan', 'spicy food', 'dessert', 'bakery', 'cafe', 'dining', 'lunch', 'dinner', 'breakfast', 'brunch'],
                'sports': ['fitness', 'exercise', 'gym', 'athletics', 'game', 'football', 'basketball', 'soccer', 'tennis', 'swimming', 'running', 'yoga', 'cycling'],
                'music': ['song', 'artist', 'concert', 'band', 'melody', 'guitar', 'piano', 'singing', 'classical', 'rock', 'pop', 'jazz', 'blues'],
                'travel': ['trip', 'vacation', 'destination', 'tourism', 'explore', 'adventure', 'backpacking', 'cruise', 'flight', 'hotel'],
                'business': ['entrepreneur', 'startup', 'company', 'work', 'career', 'management', 'finance', 'marketing', 'sales'],
                'education': ['learning', 'study', 'course', 'school', 'university', 'college', 'teaching', 'research', 'academic'],
                'health': ['fitness', 'wellness', 'medical', 'exercise', 'nutrition', 'doctor', 'medicine', 'healthy', 'diet']
            }
            
            # Find relevant interests for the topic
            relevant_interests = []
            for topic_key, interests in topic_mappings.items():
                if topic_key in topic_lower:
                    relevant_interests = interests
                    break
            
            if relevant_interests:
                # Check which participants have relevant interests
                for user_id in participants:
                    if user_id != "ai_bot":
                        user_tags = self.db.get_user_tags(user_id)
                        user_profile = self.db.get_user_profile(user_id)
                        
                        for tag in user_tags:
                            if tag.lower() in relevant_interests and tag.lower() in response_text.lower():
                                citations.append(Citation(
                                    id=f"topic_interest_{user_id}_{tag}",
                                    type="topic_interest",
                                    source="RAG Context",
                                    content=f"{user_profile.get('name', 'User')} has {tag} interest relevant to {topic_name}",
                                    relevance_score=0.9,
                                    timestamp=datetime.now().isoformat(),
                                    metadata={
                                        "user_id": user_id,
                                        "tag": tag,
                                        "topic_name": topic_name,
                                        "user_name": user_profile.get('name', 'Unknown')
                                    }
                                ))
                                
        except Exception as e:
            print(f"Error generating topic interest citations: {e}")
        
        return citations
    
    def _generate_web_search_citations(self, response_text: str, user_message: str) -> List[Citation]:
        """Generate citations for web search results"""
        citations = []
        
        try:
            # Check if response might be based on web search
            current_keywords = ['current', 'latest', 'recent', 'now', 'today', '2024', '2025', 'news', 'trending']
            
            if any(keyword in user_message.lower() for keyword in current_keywords):
                citations.append(Citation(
                    id=f"web_search_{datetime.now().timestamp()}",
                    type="web_search",
                    source="Web Search",
                    content="Current information from web search",
                    relevance_score=0.8,
                    timestamp=datetime.now().isoformat(),
                    metadata={
                        "search_triggered": True,
                        "keywords_found": [kw for kw in current_keywords if kw in user_message.lower()]
                    }
                ))
        
        except Exception as e:
            print(f"Error generating web search citations: {e}")
        
        return citations
    
    def _check_content_relevance(self, response_text: str, reference_text: str) -> bool:
        """Check if response content is relevant to reference text"""
        if not response_text or not reference_text:
            return False
        
        # Simple keyword matching (could be enhanced with semantic similarity)
        response_words = set(response_text.lower().split())
        reference_words = set(reference_text.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        response_words -= common_words
        reference_words -= common_words
        
        # Check overlap
        overlap = response_words.intersection(reference_words)
        
        # Consider relevant if there's significant overlap
        return len(overlap) >= 2 or len(overlap) / max(len(response_words), 1) > 0.1


class CitationDisplayManager:
    """Manages citation display in the UI"""
    
    @staticmethod
    def format_citations_for_display(citations: List[Citation]) -> str:
        """Format citations for display in the UI"""
        if not citations:
            return ""
        
        # Create citation links similar to ChatGPT
        citation_links = []
        for i, citation in enumerate(citations, 1):
            citation_links.append(f"[{i}]")
        
        return " " + " ".join(citation_links)
    
    @staticmethod
    def create_citation_details(citations: List[Citation]) -> Dict[str, Any]:
        """Create detailed citation information for expandable display"""
        citation_details = {}
        
        for i, citation in enumerate(citations, 1):
            citation_details[str(i)] = {
                "id": citation.id,
                "type": citation.type,
                "source": citation.source,
                "content": citation.content,
                "relevance_score": citation.relevance_score,
                "timestamp": citation.timestamp,
                "metadata": citation.metadata,
                "display_text": f"**{citation.source}**: {citation.content}"
            }
        
        return citation_details
    
    @staticmethod
    def get_citation_icon(citation_type: str) -> str:
        """Get appropriate icon for citation type"""
        icons = {
            "user_profile": "👤",
            "conversation_context": "💬",
            "group_topic": "📝",
            "language_preference": "🌐",
            "cultural_context": "🏛️",
            "web_search": "🔍",
            "shared_interest": "🤝",
            "location_context": "📍",
            "regional_context": "🌍",
            "topic_interest": "🎯",
            "topic_shared_interest": "🎯🤝",
            "topic_location_context": "🎯📍",
            "topic_regional_context": "🎯🌍"
        }
        return icons.get(citation_type, "📄") 