import os
from openai import OpenAI
from langchain.schema import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# For simplicity, we use a simple in-memory structure for rejected questions
# In production, this should be persisted (see db.py)

class Chatbot:
    def __init__(self, db=None):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
        self.db = db
        self.rejected_questions = set()
        self.accepted_questions = set()
        if db:
            self.rejected_questions = set(db.get_rejected_questions())
            self.accepted_questions = set(db.get_accepted_questions())
        self.conversation = []  # List of (role, message)
        self.conversation_turns = 0  # Track number of conversation turns
        self.last_question = None  # Store the last follow-up question

    def add_user_message(self, message):
        self.conversation.append(("user", message))
        self.conversation_turns += 1

    def add_bot_message(self, message):
        self.conversation.append(("bot", message))

    def is_rejection(self, message):
        # Simple heuristic for rejection
        return message.strip().lower() in {"no", "skip", "not interested", "nah", "nope"}

    def is_yes(self, message):
        # Check for yes responses
        return message.strip().lower() in {"yes", "y", "yeah", "sure", "okay", "ok"}

    def should_ask_followup(self):
        # Ask follow-up question after every 3 conversation turns
        return self.conversation_turns % 3 == 0 and self.conversation_turns > 0

    def get_followup_question(self, context):
        # Use OpenAI to generate a relevant follow-up question, avoiding rejected ones
        prompt = (
            "Given the conversation so far, suggest ONE relevant follow-up question. "
            "Avoid these rejected questions: " + "; ".join(self.rejected_questions) + ". "
            "Make it a simple yes/no question. "
            "Conversation: " + str(context)
        )
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        question = response.choices[0].message.content.strip()
        return question

    def process_user_message(self, message):
        self.add_user_message(message)
        
        # Check if this is a response to a follow-up question
        if self.last_question and (self.is_yes(message) or self.is_rejection(message)):
            if self.is_rejection(message):
                # Mark the question as rejected
                self.rejected_questions.add(self.last_question)
                if self.db:
                    self.db.save_rejected_question(self.last_question)
                self.last_question = None
                return "Understood. Let me ask something else later."
            else:
                # User said yes, mark as accepted
                self.accepted_questions.add(self.last_question)
                if self.db:
                    self.db.save_accepted_question(self.last_question)
                self.last_question = None
                return "Great! Let's continue our conversation."
        
        # Normal conversation: get OpenAI response
        context = self.conversation
        prompt = "\n".join([f"{r}: {m}" for r, m in context])
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": prompt}]
        )
        bot_reply = response.choices[0].message.content.strip()
        self.add_bot_message(bot_reply)
        
        # Check if we should ask a follow-up question
        if self.should_ask_followup():
            followup = self.get_followup_question(context + [("bot", bot_reply)])
            self.last_question = followup
            return bot_reply, followup
        else:
            return bot_reply

    def get_conversation(self):
        return self.conversation

    def get_conversation_turns(self):
        return self.conversation_turns

    def get_last_question(self):
        return self.last_question

    def get_accepted_questions(self):
        """Get list of accepted questions"""
        return list(self.accepted_questions)

    def get_rejected_questions(self):
        """Get list of rejected questions"""
        return list(self.rejected_questions)

    def get_question_stats(self):
        """Get statistics about questions"""
        if self.db:
            return self.db.get_question_stats()
        else:
            return {
                'rejected_count': len(self.rejected_questions),
                'accepted_count': len(self.accepted_questions),
                'total_questions': len(self.rejected_questions) + len(self.accepted_questions)
            } 