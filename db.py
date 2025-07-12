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
        else:
            # Use in-memory mock DB
            self.client = mongomock.MongoClient()
            self.db = self.client['chatbot']
            self.rejected_collection = self.db['rejected_questions']
            self.accepted_collection = self.db['accepted_questions']
            self.users_collection = self.db['users']
            self.conversations_collection = self.db['conversations']

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
            'created_at': self._get_timestamp()
        }
        self.users_collection.insert_one(user_doc)
        return user_id, name

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