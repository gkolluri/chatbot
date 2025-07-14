import os
import uuid
from pymongo import MongoClient
import mongomock
import requests
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

class DB:
    def __init__(self):
        uri = os.getenv('MONGODB_ATLAS_URI')
        if uri:
            self.client = MongoClient(uri)
            self.db = self.client['chatbot']
            self.rejected_collection = self.db['rejected_questions']
            self.accepted_collection = self.db['accepted_questions']
            self.users_collection = self.db['users']
            self.conversations_collection = self.db['conversations']
            self.tags_collection = self.db['tags']
            self.user_tags_collection = self.db['user_tags']
            self.group_chats_collection = self.db['group_chats']
            self.group_messages_collection = self.db['group_messages']
            # New collection for vector embeddings
            self.embeddings_collection = self.db['user_embeddings']
        else:
            # Use in-memory mock DB
            self.client = mongomock.MongoClient()
            self.db = self.client['chatbot']
            self.rejected_collection = self.db['rejected_questions']
            self.accepted_collection = self.db['accepted_questions']
            self.users_collection = self.db['users']
            self.conversations_collection = self.db['conversations']
            self.tags_collection = self.db['tags']
            self.user_tags_collection = self.db['user_tags']
            self.group_chats_collection = self.db['group_chats']
            self.group_messages_collection = self.db['group_messages']
            # New collection for vector embeddings
            self.embeddings_collection = self.db['user_embeddings']
        
        # Initialize indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Create necessary indexes including geospatial indexes"""
        try:
            # Get existing indexes
            existing_indexes = [idx['name'] for idx in self.users_collection.list_indexes()]
            
            # Create 2dsphere index for location coordinates if it doesn't exist
            if "location.coordinates_2dsphere" not in existing_indexes:
                self.users_collection.create_index([("location.coordinates", "2dsphere")])
                print("✅ Created 2dsphere index for location.coordinates")
            
            # Create index for user embeddings
            embedding_indexes = [idx['name'] for idx in self.embeddings_collection.list_indexes()]
            if "user_id_1" not in embedding_indexes:
                self.embeddings_collection.create_index([("user_id", 1)])
                print("✅ Created user_id index for embeddings")
            
            if "created_at_-1" not in embedding_indexes:
                self.embeddings_collection.create_index([("created_at", -1)])
                print("✅ Created created_at index for embeddings")
            
            print("Database indexes verified/created successfully")
        except Exception as e:
            print(f"Error creating indexes: {str(e)}")
    
    def _geocode_city_to_coordinates(self, city: str, state: str = None, country: str = None) -> Optional[Dict[str, float]]:
        """Geocode city name to coordinates using OpenStreetMap Nominatim API"""
        try:
            # Build address query
            address_parts = [city]
            if state:
                address_parts.append(state)
            if country:
                address_parts.append(country)
            
            address = ", ".join(address_parts)
            
            # Use OpenStreetMap Nominatim API (free, no API key required)
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': address,
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            }
            
            headers = {
                'User-Agent': 'ChatbotApp/1.0 (contact@example.com)'  # Required by Nominatim
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            if data and len(data) > 0:
                result = data[0]
                return {
                    'latitude': float(result['lat']),
                    'longitude': float(result['lon'])
                }
            
            return None
            
        except Exception as e:
            print(f"Error geocoding {address}: {str(e)}")
            return None
    
    def _ensure_coordinates(self, city: str = None, state: str = None, country: str = None, 
                           coordinates: Dict[str, float] = None) -> Dict[str, float]:
        """Ensure coordinates exist, geocoding city if needed"""
        # If coordinates are provided, use them
        if coordinates and coordinates.get('latitude') and coordinates.get('longitude'):
            return {
                'latitude': float(coordinates['latitude']),
                'longitude': float(coordinates['longitude'])
            }
        
        # If no coordinates but have city, try to geocode
        if city:
            geocoded = self._geocode_city_to_coordinates(city, state, country)
            if geocoded:
                return geocoded
        
        # Return None if no coordinates can be determined
        return None

    def get_or_create_user(self, name):
        """Get existing user by name or create new user with UUID"""
        # Check if user exists
        existing_user = self.users_collection.find_one({'name': name})
        if existing_user:
            return existing_user['user_id'], existing_user['name']
        
        # Create new user
        user_id = str(uuid.uuid4())
        user_doc = {
            'user_id': user_id,
            'name': name,
            'created_at': self._get_timestamp(),
            'profile_updated_at': self._get_timestamp(),
            'preferred_languages': [],  # List of preferred languages
            'native_language': None,    # Primary native language
            'language_comfort_level': 'english',  # Default to English
            # Location information
            'location': {
                'city': None,           # City name
                'state': None,          # State/Province
                'country': None,        # Country
                'timezone': None,       # Timezone
                'coordinates': None,    # GeoJSON Point for 2dsphere index
                'latitude': None,       # Backward compatibility
                'longitude': None,      # Backward compatibility
                'privacy_level': 'city_only',  # 'exact', 'city_only', 'state_only', 'country_only', 'private'
                'last_updated': None
            }
        }
        self.users_collection.insert_one(user_doc)
        return user_id, name

    def update_user_profile(self, user_id, profile_data):
        """Update user profile information"""
        update_data = {
            'profile_updated_at': self._get_timestamp()
        }
        update_data.update(profile_data)
        
        self.users_collection.update_one(
            {'user_id': user_id},
            {'$set': update_data}
        )

    def update_language_preferences(self, user_id, native_language=None, preferred_languages=None, language_comfort_level=None):
        """Update user language preferences"""
        update_data = {
            'profile_updated_at': self._get_timestamp()
        }
        
        if native_language is not None:
            update_data['native_language'] = native_language
        if preferred_languages is not None:
            update_data['preferred_languages'] = preferred_languages
        if language_comfort_level is not None:
            update_data['language_comfort_level'] = language_comfort_level
        
        self.users_collection.update_one(
            {'user_id': user_id},
            {'$set': update_data}
        )

    def get_language_preferences(self, user_id):
        """Get user language preferences"""
        user_profile = self.get_user_profile(user_id)
        if user_profile:
            return {
                'native_language': user_profile.get('native_language'),
                'preferred_languages': user_profile.get('preferred_languages', []),
                'language_comfort_level': user_profile.get('language_comfort_level', 'english')
            }
        return {
            'native_language': None,
            'preferred_languages': [],
            'language_comfort_level': 'english'
        }

    def update_location_preferences(self, user_id, city=None, state=None, country=None, 
                                   timezone=None, coordinates=None, privacy_level=None):
        """Update user location preferences with coordinate fallbacks"""
        update_data = {
            'profile_updated_at': self._get_timestamp()
        }
        
        location_update = {}
        if city is not None:
            location_update['location.city'] = city
        if state is not None:
            location_update['location.state'] = state
        if country is not None:
            location_update['location.country'] = country
        if timezone is not None:
            location_update['location.timezone'] = timezone
        if privacy_level is not None:
            location_update['location.privacy_level'] = privacy_level
        
        # Ensure coordinates exist with fallbacks
        final_coordinates = self._ensure_coordinates(city, state, country, coordinates)
        if final_coordinates:
            # Store coordinates in GeoJSON format for MongoDB 2dsphere index
            location_update['location.coordinates'] = {
                'type': 'Point',
                'coordinates': [final_coordinates['longitude'], final_coordinates['latitude']]
            }
            # Also store as separate fields for backward compatibility
            location_update['location.latitude'] = final_coordinates['latitude']
            location_update['location.longitude'] = final_coordinates['longitude']
        
        if location_update:
            location_update['location.last_updated'] = self._get_timestamp()
            update_data.update(location_update)
        
        self.users_collection.update_one(
            {'user_id': user_id},
            {'$set': update_data}
        )

    def get_location_preferences(self, user_id):
        """Get user location preferences"""
        user_profile = self.get_user_profile(user_id)
        if user_profile and 'location' in user_profile:
            location = user_profile['location']
            return {
                'city': location.get('city'),
                'state': location.get('state'), 
                'country': location.get('country'),
                'timezone': location.get('timezone'),
                'coordinates': location.get('coordinates', {}),
                'privacy_level': location.get('privacy_level', 'city_only'),
                'last_updated': location.get('last_updated')
            }
        return {
            'city': None,
            'state': None,
            'country': None,
            'timezone': None,
            'coordinates': {},
            'privacy_level': 'city_only',
            'last_updated': None
        }

    def find_nearby_users(self, user_id, max_distance_km=50, privacy_level='city_only'):
        """Find users within a certain distance using MongoDB geospatial queries"""
        user_location = self.get_location_preferences(user_id)
        
        # Check if user has coordinates (either GeoJSON or legacy format)
        user_coordinates = None
        if user_location.get('coordinates'):
            if isinstance(user_location['coordinates'], dict):
                if user_location['coordinates'].get('type') == 'Point':
                    # GeoJSON format
                    coords = user_location['coordinates']['coordinates']
                    user_coordinates = [coords[0], coords[1]]  # [longitude, latitude]
                elif user_location['coordinates'].get('latitude') and user_location['coordinates'].get('longitude'):
                    # Legacy format
                    user_coordinates = [user_location['coordinates']['longitude'], user_location['coordinates']['latitude']]
        
        # Fallback to separate lat/lng fields
        if not user_coordinates and user_location.get('latitude') and user_location.get('longitude'):
            user_coordinates = [user_location['longitude'], user_location['latitude']]
        
        if not user_coordinates:
            return []
        
        # Convert km to meters for MongoDB geospatial query
        max_distance_meters = max_distance_km * 1000
        
        # Use MongoDB's $near operator for optimized geospatial search
        try:
            nearby_users = []
            cursor = self.users_collection.find({
                'user_id': {'$ne': user_id},
                'location.coordinates': {
                    '$near': {
                        '$geometry': {
                            'type': 'Point',
                            'coordinates': user_coordinates
                        },
                        '$maxDistance': max_distance_meters
                    }
                },
                'location.privacy_level': {'$in': ['exact', 'city_only']}
            })
            
            for user in cursor:
                # Calculate actual distance for display
                other_coords = user['location']['coordinates']
                if other_coords and other_coords.get('coordinates'):
                    other_lng, other_lat = other_coords['coordinates']
                    distance = self._calculate_distance(
                        user_coordinates[1], user_coordinates[0],  # user lat, lng
                        other_lat, other_lng  # other lat, lng
                    )
                    
                    nearby_users.append({
                        'user_id': user['user_id'],
                        'name': user['name'],
                        'distance_km': round(distance, 2),
                        'city': user['location'].get('city'),
                        'state': user['location'].get('state'),
                        'privacy_level': user['location'].get('privacy_level', 'city_only')
                    })
            
            return nearby_users
            
        except Exception as e:
            print(f"Error in geospatial query: {str(e)}")
            # Fallback to original method if geospatial query fails
            return self._find_nearby_users_fallback(user_id, max_distance_km, privacy_level)
    
    def _find_nearby_users_fallback(self, user_id, max_distance_km=50, privacy_level='city_only'):
        """Fallback method using Haversine calculation for all users"""
        user_location = self.get_location_preferences(user_id)
        
        # Extract coordinates from various formats
        user_lat = user_lng = None
        if user_location.get('coordinates'):
            if isinstance(user_location['coordinates'], dict):
                if user_location['coordinates'].get('type') == 'Point':
                    coords = user_location['coordinates']['coordinates']
                    user_lng, user_lat = coords[0], coords[1]
                elif user_location['coordinates'].get('latitude'):
                    user_lat = user_location['coordinates']['latitude']
                    user_lng = user_location['coordinates']['longitude']
        
        if not user_lat or not user_lng:
            if user_location.get('latitude') and user_location.get('longitude'):
                user_lat = user_location['latitude']
                user_lng = user_location['longitude']
        
        if not user_lat or not user_lng:
            return []
        
        # Get all users with location data
        pipeline = [
            {
                '$match': {
                    'user_id': {'$ne': user_id},
                    '$or': [
                        {'location.coordinates.coordinates': {'$exists': True, '$ne': None}},
                        {'location.latitude': {'$exists': True, '$ne': None}}
                    ],
                    'location.privacy_level': {'$in': ['exact', 'city_only']}
                }
            }
        ]
        
        nearby_users = []
        for user in self.users_collection.aggregate(pipeline):
            other_lat = other_lng = None
            
            # Extract coordinates from various formats
            if user['location'].get('coordinates'):
                if isinstance(user['location']['coordinates'], dict):
                    if user['location']['coordinates'].get('type') == 'Point':
                        coords = user['location']['coordinates']['coordinates']
                        other_lng, other_lat = coords[0], coords[1]
                    elif user['location']['coordinates'].get('latitude'):
                        other_lat = user['location']['coordinates']['latitude']
                        other_lng = user['location']['coordinates']['longitude']
            
            if not other_lat or not other_lng:
                if user['location'].get('latitude') and user['location'].get('longitude'):
                    other_lat = user['location']['latitude']
                    other_lng = user['location']['longitude']
            
            if other_lat and other_lng:
                distance = self._calculate_distance(user_lat, user_lng, other_lat, other_lng)
                
                if distance <= max_distance_km:
                    nearby_users.append({
                        'user_id': user['user_id'],
                        'name': user['name'],
                        'distance_km': round(distance, 2),
                        'city': user['location'].get('city'),
                        'state': user['location'].get('state'),
                        'privacy_level': user['location'].get('privacy_level', 'city_only')
                    })
        
        # Sort by distance
        nearby_users.sort(key=lambda x: x['distance_km'])
        return nearby_users

    def find_users_in_city(self, user_id, city=None, state=None):
        """Find users in the same city or state"""
        if not city and not state:
            user_location = self.get_location_preferences(user_id)
            city = user_location.get('city')
            state = user_location.get('state')
        
        if not city and not state:
            return []
        
        query = {
            'user_id': {'$ne': user_id},
            '$or': []
        }
        
        if city:
            query['$or'].append({'location.city': city})
        if state:
            query['$or'].append({'location.state': state})
        
        users = []
        for user in self.users_collection.find(query):
            if user.get('location', {}).get('privacy_level') not in ['private']:
                users.append({
                    'user_id': user['user_id'],
                    'name': user['name'],
                    'city': user.get('location', {}).get('city'),
                    'state': user.get('location', {}).get('state'),
                    'privacy_level': user.get('location', {}).get('privacy_level', 'city_only')
                })
        
        return users

    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates using Haversine formula"""
        import math
        
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        
        return c * r

    def get_user_profile(self, user_id):
        """Get user profile information"""
        return self.users_collection.find_one({'user_id': user_id})

    def get_all_users(self):
        """Get all users with their profile data"""
        return list(self.users_collection.find())

    def get_all_users_summary(self):
        """Get summary of all users with key information"""
        users = []
        for user in self.users_collection.find():
            user_tags = self.get_user_tags(user['user_id'])
            location = user.get('location', {})
            
            users.append({
                'user_id': user['user_id'],
                'name': user['name'],
                'created_at': user.get('created_at'),
                'profile_updated_at': user.get('profile_updated_at'),
                'native_language': user.get('native_language'),
                'preferred_languages': user.get('preferred_languages', []),
                'language_comfort_level': user.get('language_comfort_level', 'english'),
                'tags': user_tags,
                'location': {
                    'city': location.get('city'),
                    'state': location.get('state'),
                    'country': location.get('country'),
                    'timezone': location.get('timezone'),
                    'coordinates': location.get('coordinates', {}),
                    'privacy_level': location.get('privacy_level', 'city_only'),
                    'last_updated': location.get('last_updated')
                }
            })
        return users

    def add_user_tag(self, user_id, tag, tag_type="manual"):
        """Add a tag to a user (manual or inferred)"""
        tag_doc = {
            'user_id': user_id,
            'tag': tag.lower().strip(),
            'tag_type': tag_type,  # "manual" or "inferred"
            'created_at': self._get_timestamp()
        }
        self.user_tags_collection.insert_one(tag_doc)

    def get_user_tags(self, user_id, tag_type=None):
        """Get tags for a user, optionally filtered by type"""
        query = {'user_id': user_id}
        if tag_type:
            query['tag_type'] = tag_type
        
        return [doc['tag'] for doc in self.user_tags_collection.find(query)]

    def remove_user_tag(self, user_id, tag):
        """Remove a specific tag from a user"""
        self.user_tags_collection.delete_one({
            'user_id': user_id,
            'tag': tag.lower().strip()
        })

    def find_similar_users(self, user_id, min_common_tags=2, include_location=True):
        """Find users with similar tags and optionally consider location"""
        user_tags = set(self.get_user_tags(user_id))
        if not user_tags:
            return []
        
        user_location = self.get_location_preferences(user_id) if include_location else None
        
        # Get all other users and their tags
        all_users = self.users_collection.find({'user_id': {'$ne': user_id}})
        similar_users = []
        
        for user in all_users:
            other_user_tags = set(self.get_user_tags(user['user_id']))
            common_tags = user_tags.intersection(other_user_tags)
            
            if len(common_tags) >= min_common_tags:
                similarity_score = len(common_tags)
                location_bonus = 0
                location_info = {}
                
                # Add location-based similarity bonus
                if include_location and user_location:
                    other_location = user.get('location', {})
                    
                    # Same city bonus
                    if (user_location.get('city') and other_location.get('city') and 
                        user_location['city'].lower() == other_location['city'].lower()):
                        location_bonus += 2
                        location_info['same_city'] = True
                    
                    # Same state bonus
                    elif (user_location.get('state') and other_location.get('state') and 
                          user_location['state'].lower() == other_location['state'].lower()):
                        location_bonus += 1
                        location_info['same_state'] = True
                    
                    # Same country bonus
                    elif (user_location.get('country') and other_location.get('country') and 
                          user_location['country'].lower() == other_location['country'].lower()):
                        location_bonus += 0.5
                        location_info['same_country'] = True
                    
                    location_info.update({
                        'city': other_location.get('city'),
                        'state': other_location.get('state'),
                        'country': other_location.get('country'),
                        'privacy_level': other_location.get('privacy_level', 'city_only')
                    })
                
                similar_users.append({
                    'user_id': user['user_id'],
                    'name': user['name'],
                    'common_tags': list(common_tags),
                    'similarity_score': similarity_score,
                    'location_bonus': location_bonus,
                    'total_score': similarity_score + location_bonus,
                    'location_info': location_info
                })
        
        # Sort by total score (tags + location bonus)
        similar_users.sort(key=lambda x: x['total_score'], reverse=True)
        return similar_users

    def create_group_chat(self, topic_name, user_ids, created_by):
        """Create a new group chat"""
        group_id = str(uuid.uuid4())
        group_doc = {
            'group_id': group_id,
            'topic_name': topic_name,
            'user_ids': user_ids,
            'created_by': created_by,
            'created_at': self._get_timestamp(),
            'is_active': True
        }
        self.group_chats_collection.insert_one(group_doc)
        return group_id

    def get_user_group_chats(self, user_id):
        """Get all group chats for a user"""
        return list(self.group_chats_collection.find({
            'user_ids': user_id,
            'is_active': True
        }))

    def add_group_message(self, group_id, user_id, message, message_type="user", citations=None):
        """Add a message to a group chat with optional citations"""
        message_doc = {
            'group_id': group_id,
            'user_id': user_id,
            'message': message,
            'message_type': message_type,  # "user" or "ai"
            'timestamp': self._get_timestamp(),
            'citations': [],
            'citation_links': '',
            'citation_details': {}
        }
        
        # If citations are provided, convert them to dictionaries for storage
        if citations:
            from citation_system import CitationDisplayManager
            display_manager = CitationDisplayManager()
            
            # Convert Citation objects to dictionaries
            citation_dicts = []
            for citation in citations:
                if hasattr(citation, 'to_dict'):
                    citation_dicts.append(citation.to_dict())
                else:
                    # Fallback: convert to dict manually
                    citation_dicts.append({
                        'id': getattr(citation, 'id', ''),
                        'type': getattr(citation, 'type', ''),
                        'source': getattr(citation, 'source', ''),
                        'content': getattr(citation, 'content', ''),
                        'relevance_score': getattr(citation, 'relevance_score', 0.0),
                        'timestamp': getattr(citation, 'timestamp', ''),
                        'metadata': getattr(citation, 'metadata', {})
                    })
            
            message_doc['citations'] = citation_dicts
            message_doc['citation_links'] = display_manager.format_citations_for_display(citations)
            message_doc['citation_details'] = display_manager.create_citation_details(citations)
        
        self.group_messages_collection.insert_one(message_doc)

    def get_group_messages(self, group_id, limit=50):
        """Get messages for a group chat"""
        messages = self.group_messages_collection.find(
            {'group_id': group_id}
        ).sort('timestamp', 1).limit(limit)
        
        return list(messages)

    def get_group_info(self, group_id):
        """Get group chat information"""
        return self.group_chats_collection.find_one({'group_id': group_id})

    def save_conversation_turn(self, user_id, role, message, conversation_turns):
        """Save a conversation turn for a specific user"""
        conversation_doc = {
            'user_id': user_id,
            'role': role,
            'message': message,
            'conversation_turns': conversation_turns,
            'timestamp': self._get_timestamp()
        }
        self.conversations_collection.insert_one(conversation_doc)

    def get_user_conversation(self, user_id):
        """Get all conversation turns for a specific user"""
        return list(self.conversations_collection.find({'user_id': user_id}).sort('timestamp', 1))

    def get_user_conversations(self, user_id, limit=20):
        """Get recent conversation turns for a specific user"""
        return list(self.conversations_collection.find({'user_id': user_id}).sort('timestamp', -1).limit(limit))

    def add_conversation(self, user_id, role, message, conversation_turns):
        """Add a conversation turn to the database"""
        self.save_conversation_turn(user_id, role, message, conversation_turns)

    # Vector storage methods for MongoDB
    def store_user_embedding(self, user_id: str, embedding: List[float], profile_text: str, 
                           metadata: Dict[str, Any] = None) -> str:
        """Store user profile embedding in MongoDB"""
        try:
            embedding_doc = {
                'user_id': user_id,
                'embedding': embedding,
                'profile_text': profile_text,
                'metadata': metadata or {},
                'created_at': self._get_timestamp(),
                'updated_at': self._get_timestamp()
            }
            
            # Upsert (update if exists, insert if not)
            result = self.embeddings_collection.replace_one(
                {'user_id': user_id},
                embedding_doc,
                upsert=True
            )
            
            return str(result.upserted_id) if result.upserted_id else user_id
            
        except Exception as e:
            print(f"Error storing embedding for user {user_id}: {str(e)}")
            return None
    
    def get_user_embedding(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile embedding from MongoDB"""
        try:
            return self.embeddings_collection.find_one({'user_id': user_id})
        except Exception as e:
            print(f"Error getting embedding for user {user_id}: {str(e)}")
            return None
    
    def delete_user_embedding(self, user_id: str) -> bool:
        """Delete user profile embedding from MongoDB"""
        try:
            result = self.embeddings_collection.delete_one({'user_id': user_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting embedding for user {user_id}: {str(e)}")
            return False
    
    def semantic_search_users(self, query_embedding: List[float], user_id: str = None, 
                            max_results: int = 10, min_similarity: float = 0.7) -> List[Dict[str, Any]]:
        """Perform semantic search for users using cosine similarity"""
        try:
            # Build match criteria
            match_criteria = {}
            if user_id:
                match_criteria['user_id'] = {'$ne': user_id}
            
            # MongoDB aggregation pipeline for cosine similarity
            pipeline = [
                {'$match': match_criteria},
                {
                    '$addFields': {
                        'similarity': {
                            '$let': {
                                'vars': {
                                    'dotProduct': {
                                        '$reduce': {
                                            'input': {'$range': [0, {'$size': '$embedding'}]},
                                            'initialValue': 0,
                                            'in': {
                                                '$add': [
                                                    '$$value',
                                                    {'$multiply': [
                                                        {'$arrayElemAt': ['$embedding', '$$this']},
                                                        {'$arrayElemAt': [query_embedding, '$$this']}
                                                    ]}
                                                ]
                                            }
                                        }
                                    },
                                    'queryMagnitude': {
                                        '$sqrt': {
                                            '$reduce': {
                                                'input': query_embedding,
                                                'initialValue': 0,
                                                'in': {'$add': ['$$value', {'$multiply': ['$$this', '$$this']}]}
                                            }
                                        }
                                    },
                                    'docMagnitude': {
                                        '$sqrt': {
                                            '$reduce': {
                                                'input': '$embedding',
                                                'initialValue': 0,
                                                'in': {'$add': ['$$value', {'$multiply': ['$$this', '$$this']}]}
                                            }
                                        }
                                    }
                                },
                                'in': {
                                    '$cond': {
                                        'if': {'$and': [{'$gt': ['$$queryMagnitude', 0]}, {'$gt': ['$$docMagnitude', 0]}]},
                                        'then': {'$divide': ['$$dotProduct', {'$multiply': ['$$queryMagnitude', '$$docMagnitude']}]},
                                        'else': 0
                                    }
                                }
                            }
                        }
                    }
                },
                {'$match': {'similarity': {'$gte': min_similarity}}},
                {'$sort': {'similarity': -1}},
                {'$limit': max_results}
            ]
            
            results = []
            for doc in self.embeddings_collection.aggregate(pipeline):
                results.append({
                    'user_id': doc['user_id'],
                    'similarity': doc['similarity'],
                    'profile_text': doc['profile_text'],
                    'metadata': doc.get('metadata', {}),
                    'created_at': doc.get('created_at'),
                    'updated_at': doc.get('updated_at')
                })
            
            return results
            
        except Exception as e:
            print(f"Error in semantic search: {str(e)}")
            return []
    
    def hybrid_geospatial_semantic_search(self, query_embedding: List[float], user_id: str,
                                        max_distance_km: float = 50, max_results: int = 10,
                                        min_similarity: float = 0.5, location_weight: float = 0.3) -> List[Dict[str, Any]]:
        """Perform hybrid search combining geospatial and semantic similarity"""
        try:
            user_location = self.get_location_preferences(user_id)
            
            # Get user coordinates
            user_coordinates = None
            if user_location.get('coordinates'):
                if isinstance(user_location['coordinates'], dict):
                    if user_location['coordinates'].get('type') == 'Point':
                        coords = user_location['coordinates']['coordinates']
                        user_coordinates = [coords[0], coords[1]]
            
            if not user_coordinates and user_location.get('latitude') and user_location.get('longitude'):
                user_coordinates = [user_location['longitude'], user_location['latitude']]
            
            if not user_coordinates:
                # Fallback to semantic search only
                return self.semantic_search_users(query_embedding, user_id, max_results, min_similarity)
            
            # Convert km to meters
            max_distance_meters = max_distance_km * 1000
            
            # Use $geoNear aggregation stage for geospatial search
            pipeline = [
                {
                    '$geoNear': {
                        'near': {
                            'type': 'Point',
                            'coordinates': user_coordinates
                        },
                        'distanceField': 'distance',
                        'maxDistance': max_distance_meters,
                        'spherical': True,
                        'query': {'user_id': {'$ne': user_id}}
                    }
                },
                {
                    '$lookup': {
                        'from': 'user_embeddings',
                        'localField': 'user_id',
                        'foreignField': 'user_id',
                        'as': 'embedding_data'
                    }
                },
                {'$unwind': {'path': '$embedding_data', 'preserveNullAndEmptyArrays': True}},
                {
                    '$addFields': {
                        'semantic_similarity': {
                            '$cond': {
                                'if': {'$ne': ['$embedding_data', None]},
                                'then': {
                                    '$let': {
                                        'vars': {
                                            'dotProduct': {
                                                '$reduce': {
                                                    'input': {'$range': [0, {'$size': '$embedding_data.embedding'}]},
                                                    'initialValue': 0,
                                                    'in': {
                                                        '$add': [
                                                            '$$value',
                                                            {'$multiply': [
                                                                {'$arrayElemAt': ['$embedding_data.embedding', '$$this']},
                                                                {'$arrayElemAt': [query_embedding, '$$this']}
                                                            ]}
                                                        ]
                                                    }
                                                }
                                            },
                                            'queryMagnitude': {
                                                '$sqrt': {
                                                    '$reduce': {
                                                        'input': query_embedding,
                                                        'initialValue': 0,
                                                        'in': {'$add': ['$$value', {'$multiply': ['$$this', '$$this']}]}
                                                    }
                                                }
                                            },
                                            'docMagnitude': {
                                                '$sqrt': {
                                                    '$reduce': {
                                                        'input': '$embedding_data.embedding',
                                                        'initialValue': 0,
                                                        'in': {'$add': ['$$value', {'$multiply': ['$$this', '$$this']}]}
                                                    }
                                                }
                                            }
                                        },
                                        'in': {
                                            '$cond': {
                                                'if': {'$and': [{'$gt': ['$$queryMagnitude', 0]}, {'$gt': ['$$docMagnitude', 0]}]},
                                                'then': {'$divide': ['$$dotProduct', {'$multiply': ['$$queryMagnitude', '$$docMagnitude']}]},
                                                'else': 0
                                            }
                                        }
                                    }
                                },
                                'else': 0
                            }
                        },
                        'location_score': {
                            '$divide': [
                                1,
                                {'$add': [1, {'$divide': ['$distance', 10000]}]}
                            ]
                        }
                    }
                },
                {
                    '$addFields': {
                        'combined_score': {
                            '$add': [
                                {'$multiply': [location_weight, '$location_score']},
                                {'$multiply': [{'$subtract': [1, location_weight]}, '$semantic_similarity']}
                            ]
                        }
                    }
                },
                {
                    '$match': {
                        '$and': [
                            {'semantic_similarity': {'$exists': True}},
                            {'semantic_similarity': {'$gte': min_similarity}}
                        ]
                    }
                },
                {'$sort': {'combined_score': -1}},
                {'$limit': max_results}
            ]
            
            results = []
            for doc in self.users_collection.aggregate(pipeline):
                embedding_data = doc.get('embedding_data', {})
                results.append({
                    'user_id': doc['user_id'],
                    'semantic_similarity': doc['semantic_similarity'],
                    'location_score': doc['location_score'],
                    'combined_score': doc['combined_score'],
                    'distance_km': doc.get('distance', 0) / 1000,  # Convert meters to km
                    'profile_text': embedding_data.get('profile_text', ''),
                    'metadata': embedding_data.get('metadata', {}),
                    'user_profile': doc,
                    'created_at': embedding_data.get('created_at'),
                    'updated_at': embedding_data.get('updated_at')
                })
            
            return results
            
        except Exception as e:
            print(f"Error in hybrid search: {str(e)}")
            # Fallback to separate location and semantic searches
            try:
                # Get nearby users using regular geospatial search
                nearby_users = self.find_nearby_users(user_id, max_distance_km)
                
                # Get semantic results
                semantic_results = self.semantic_search_users(query_embedding, user_id, max_results, min_similarity)
                
                # Combine results manually
                combined_results = []
                
                # Add semantic results with location scoring
                for semantic_result in semantic_results:
                    # Find if this user is also nearby
                    location_score = 0.0
                    distance_km = None
                    
                    for nearby_user in nearby_users:
                        if nearby_user['user_id'] == semantic_result['user_id']:
                            location_score = 1.0 / (1.0 + nearby_user['distance_km'] / 10)
                            distance_km = nearby_user['distance_km']
                            break
                    
                    combined_score = (location_weight * location_score) + ((1 - location_weight) * semantic_result['similarity'])
                    
                    combined_results.append({
                        'user_id': semantic_result['user_id'],
                        'semantic_similarity': semantic_result['similarity'],
                        'location_score': location_score,
                        'combined_score': combined_score,
                        'distance_km': distance_km,
                        'profile_text': semantic_result['profile_text'],
                        'metadata': semantic_result['metadata'],
                        'user_profile': {'user_id': semantic_result['user_id'], 'name': semantic_result['metadata'].get('name', '')},
                        'created_at': semantic_result.get('created_at'),
                        'updated_at': semantic_result.get('updated_at')
                    })
                
                # Sort by combined score
                combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
                return combined_results[:max_results]
                
            except Exception as fallback_error:
                print(f"Error in fallback hybrid search: {str(fallback_error)}")
                # Final fallback to semantic search only
                return self.semantic_search_users(query_embedding, user_id, max_results, min_similarity)

    def create_user(self, user_name, user_id=None):
        """Create a new user"""
        if user_id is None:
            user_id = str(uuid.uuid4())
        
        user_doc = {
            'user_id': user_id,
            'name': user_name,
            'created_at': self._get_timestamp(),
            'profile_updated_at': self._get_timestamp(),
            'preferred_languages': [],
            'native_language': None,
            'language_comfort_level': 'english',
            # Location information
            'location': {
                'city': None,           # City name
                'state': None,          # State/Province
                'country': None,        # Country
                'timezone': None,       # Timezone
                'coordinates': {        # GPS coordinates
                    'latitude': None,
                    'longitude': None
                },
                'privacy_level': 'city_only',  # 'exact', 'city_only', 'state_only', 'country_only', 'private'
                'last_updated': None
            }
        }
        self.users_collection.insert_one(user_doc)
        return user_id

    def get_user_stats(self, user_id):
        """Get conversation statistics for a specific user"""
        total_turns = self.conversations_collection.count_documents({'user_id': user_id})
        return {
            'total_turns': total_turns,
            'last_activity': self.conversations_collection.find_one(
                {'user_id': user_id}, 
                sort=[('timestamp', -1)]
            )
        }

    def save_rejected_question(self, question, user_id=None):
        """Save a rejected question to the database"""
        doc = {
            'question': question,
            'timestamp': self._get_timestamp()
        }
        if user_id:
            doc['user_id'] = user_id
        self.rejected_collection.insert_one(doc)

    def save_accepted_question(self, question, user_id=None):
        """Save an accepted question to the database"""
        doc = {
            'question': question,
            'timestamp': self._get_timestamp()
        }
        if user_id:
            doc['user_id'] = user_id
        self.accepted_collection.insert_one(doc)

    def get_rejected_questions(self, user_id=None):
        """Retrieve rejected questions from the database"""
        query = {'user_id': user_id} if user_id else {}
        return [doc['question'] for doc in self.rejected_collection.find(query)]

    def get_accepted_questions(self, user_id=None):
        """Retrieve accepted questions from the database"""
        query = {'user_id': user_id} if user_id else {}
        return [doc['question'] for doc in self.accepted_collection.find(query)]

    def get_question_stats(self, user_id=None):
        """Get statistics about questions"""
        query = {'user_id': user_id} if user_id else {}
        rejected_count = self.rejected_collection.count_documents(query)
        accepted_count = self.accepted_collection.count_documents(query)
        return {
            'rejected_count': rejected_count,
            'accepted_count': accepted_count,
            'total_questions': rejected_count + accepted_count
        }

    def _get_timestamp(self):
        """Get current timestamp"""
        return datetime.now()

    def check_connection(self):
        """Check database connection status"""
        try:
            # Test connection by performing a simple operation
            self.users_collection.find_one()
            return {
                'success': True,
                'message': 'Connected successfully',
                'type': 'MongoDB Atlas' if hasattr(self.client, 'server_info') else 'Mock Database'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Connection failed: {str(e)}',
                'type': 'Unknown'
            }
    
    def get_tag_type(self, tag):
        """Get the type of a tag (manual or inferred)"""
        tag_doc = self.user_tags_collection.find_one({'tag': tag.lower().strip()})
        return tag_doc.get('tag_type', 'unknown') if tag_doc else None
    
    def get_last_question(self, user_id):
        """Get the last follow-up question for a user"""
        # This would typically be stored in a separate collection
        # For now, return None as this is handled by the chatbot
        return None
    
    def get_user_conversation(self, user_id):
        """Get conversation history for a user"""
        conversations = self.conversations_collection.find(
            {'user_id': user_id}
        ).sort('timestamp', 1)
        
        # Convert to (role, message) format
        conversation = []
        for conv in conversations:
            conversation.append((conv['role'], conv['message']))
        
        return conversation

def get_db():
    return DB() 