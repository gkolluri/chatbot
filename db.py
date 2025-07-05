import os
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
        else:
            # Use in-memory mock DB
            self.client = mongomock.MongoClient()
            self.db = self.client['chatbot']
            self.rejected_collection = self.db['rejected_questions']
            self.accepted_collection = self.db['accepted_questions']

    def save_rejected_question(self, question):
        """Save a rejected question to the database"""
        self.rejected_collection.insert_one({
            'question': question,
            'timestamp': self._get_timestamp()
        })

    def save_accepted_question(self, question):
        """Save an accepted question to the database"""
        self.accepted_collection.insert_one({
            'question': question,
            'timestamp': self._get_timestamp()
        })

    def get_rejected_questions(self):
        """Retrieve all rejected questions from the database"""
        return [doc['question'] for doc in self.rejected_collection.find()]

    def get_accepted_questions(self):
        """Retrieve all accepted questions from the database"""
        return [doc['question'] for doc in self.accepted_collection.find()]

    def get_question_stats(self):
        """Get statistics about questions"""
        rejected_count = self.rejected_collection.count_documents({})
        accepted_count = self.accepted_collection.count_documents({})
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