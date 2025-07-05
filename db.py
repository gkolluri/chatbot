import os
from pymongo import MongoClient
import mongomock

class DB:
    def __init__(self):
        uri = os.getenv('MONGODB_ATLAS_URI')
        if uri:
            self.client = MongoClient(uri)
            self.db = self.client['chatbot']
            self.collection = self.db['rejected_questions']
        else:
            # Use in-memory mock DB
            self.client = mongomock.MongoClient()
            self.db = self.client['chatbot']
            self.collection = self.db['rejected_questions']

    def save_rejected_question(self, question):
        self.collection.insert_one({'question': question})

    def get_rejected_questions(self):
        return [doc['question'] for doc in self.collection.find()]

def get_db():
    return DB() 