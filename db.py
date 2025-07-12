import os
import uuid
from pymongo import MongoClient
import mongomock

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
            'language_comfort_level': 'english'  # Default to English
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

    def get_user_profile(self, user_id):
        """Get user profile information"""
        return self.users_collection.find_one({'user_id': user_id})

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

    def find_similar_users(self, user_id, min_common_tags=2):
        """Find users with similar tags"""
        user_tags = set(self.get_user_tags(user_id))
        if not user_tags:
            return []
        
        # Get all other users and their tags
        all_users = self.users_collection.find({'user_id': {'$ne': user_id}})
        similar_users = []
        
        for user in all_users:
            other_user_tags = set(self.get_user_tags(user['user_id']))
            common_tags = user_tags.intersection(other_user_tags)
            
            if len(common_tags) >= min_common_tags:
                similar_users.append({
                    'user_id': user['user_id'],
                    'name': user['name'],
                    'common_tags': list(common_tags),
                    'similarity_score': len(common_tags)
                })
        
        # Sort by similarity score
        similar_users.sort(key=lambda x: x['similarity_score'], reverse=True)
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

    def add_group_message(self, group_id, user_id, message, message_type="user"):
        """Add a message to a group chat"""
        message_doc = {
            'group_id': group_id,
            'user_id': user_id,
            'message': message,
            'message_type': message_type,  # "user" or "ai"
            'timestamp': self._get_timestamp()
        }
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
        conversations = self.conversations_collection.find(
            {'user_id': user_id}
        ).sort('timestamp', 1)
        return [(doc['role'], doc['message']) for doc in conversations]

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
        """Get current timestamp for database records"""
        from datetime import datetime
        return datetime.utcnow()

def get_db():
    return DB() 