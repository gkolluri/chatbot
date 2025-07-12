import os
from openai import OpenAI
from langchain.schema import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from tag_analyzer import TagAnalyzer

# For simplicity, we use a simple in-memory structure for rejected questions
# In production, this should be persisted (see db.py)

class Chatbot:
    def __init__(self, db=None, user_id=None, user_name=None):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
        self.db = db
        self.user_id = user_id
        self.user_name = user_name
        self.rejected_questions = set()
        self.accepted_questions = set()
        self.conversation = []  # List of (role, message)
        self.conversation_turns = 0  # Track number of conversation turns
        self.last_question = None  # Store the last follow-up question
        self.tag_analyzer = TagAnalyzer()
        self.language_preferences = None
        
        # Load user-specific data if user_id is provided
        if self.db and self.user_id:
            self.rejected_questions = set(self.db.get_rejected_questions(self.user_id))
            self.accepted_questions = set(self.db.get_accepted_questions(self.user_id))
            self.conversation = self.db.get_user_conversation(self.user_id)
            self.conversation_turns = len(self.conversation)
            self.language_preferences = self.db.get_language_preferences(self.user_id)

    def add_user_message(self, message):
        self.conversation.append(("user", message))
        self.conversation_turns += 1
        if self.db and self.user_id:
            self.db.save_conversation_turn(self.user_id, "user", message, self.conversation_turns)

    def add_bot_message(self, message):
        self.conversation.append(("bot", message))
        if self.db and self.user_id:
            self.db.save_conversation_turn(self.user_id, "bot", message, self.conversation_turns)

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
        # Use OpenAI to generate a relevant follow-up question with Indian cultural context
        prompt = (
            "Given the conversation so far, suggest ONE relevant follow-up question. "
            "Avoid these rejected questions: " + "; ".join(self.rejected_questions) + ". "
            "Make it a simple yes/no question. "
            "Consider Indian cultural context, traditions, festivals, cuisine, languages, or contemporary Indian topics when relevant. "
            "Conversation: " + str(context)
        )
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at asking relevant follow-up questions with Indian cultural sensitivity and awareness."},
                {"role": "user", "content": prompt}
            ]
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
                    self.db.save_rejected_question(self.last_question, self.user_id)
                self.last_question = None
                return "Understood. Let me ask something else later."
            else:
                # User said yes, mark as accepted
                self.accepted_questions.add(self.last_question)
                if self.db:
                    self.db.save_accepted_question(self.last_question, self.user_id)
                self.last_question = None
                return "Great! Let's continue our conversation."
        
        # Normal conversation: get OpenAI response with Indian cultural context and language preferences
        context = self.conversation
        prompt = "\n".join([f"{r}: {m}" for r, m in context])
        
        # Build language-aware system prompt
        system_prompt = "You are a helpful assistant designed to connect Indian users and NRIs based on shared interests. You have knowledge of Indian culture, languages, and contemporary topics, but your primary focus is helping users find common ground and shared interests. Be respectful and inclusive of India's diverse cultures while maintaining a professional and friendly tone."
        
        # Add language-specific instructions based on user preferences
        if self.language_preferences:
            native_lang = self.language_preferences.get('native_language')
            preferred_langs = self.language_preferences.get('preferred_languages', [])
            comfort_level = self.language_preferences.get('language_comfort_level', 'english')
            
            if native_lang and native_lang != 'english':
                system_prompt += f"\n\nUser's native language is {native_lang}. You can occasionally use {native_lang} phrases to make the conversation more comfortable, but keep it subtle and professional."
            
            if preferred_langs:
                lang_list = ", ".join(preferred_langs)
                system_prompt += f"\n\nUser prefers languages: {lang_list}. You can incorporate subtle phrases from these languages when appropriate."
            
            if comfort_level == 'native':
                system_prompt += "\n\nUser is comfortable with native language conversations. You can use some native language phrases while keeping the focus on connecting people through shared interests."
            elif comfort_level == 'mixed':
                system_prompt += "\n\nUser is comfortable with mixed language conversations. You can blend English with subtle native language phrases."
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        bot_reply = response.choices[0].message.content.strip()
        self.add_bot_message(bot_reply)
        
        # Analyze conversation for tags after every 5 turns
        if self.conversation_turns % 5 == 0 and self.db and self.user_id:
            self._analyze_and_add_tags()
        
        # Check if we should ask a follow-up question
        if self.should_ask_followup():
            followup = self.get_followup_question(context + [("bot", bot_reply)])
            self.last_question = followup
            return bot_reply, followup
        else:
            return bot_reply

    def _analyze_and_add_tags(self):
        """Analyze conversation and add inferred tags"""
        try:
            # Get current user tags
            current_tags = set(self.db.get_user_tags(self.user_id))
            
            # Analyze conversation for new tags
            inferred_tags = self.tag_analyzer.analyze_conversation_for_tags(self.conversation)
            
            # Add new tags that aren't already present
            for tag in inferred_tags:
                if tag not in current_tags:
                    self.db.add_user_tag(self.user_id, tag, "inferred")
                    current_tags.add(tag)
            
            return inferred_tags
        except Exception as e:
            print(f"Error analyzing tags: {e}")
            return []

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
            return self.db.get_question_stats(self.user_id)
        else:
            return {
                'rejected_count': len(self.rejected_questions),
                'accepted_count': len(self.accepted_questions),
                'total_questions': len(self.rejected_questions) + len(self.accepted_questions)
            }

    def get_user_info(self):
        """Get user information"""
        return {
            'user_id': self.user_id,
            'user_name': self.user_name,
            'conversation_turns': self.conversation_turns
        }

    def get_user_tags(self):
        """Get user tags"""
        if self.db and self.user_id:
            return self.db.get_user_tags(self.user_id)
        return []

    def add_manual_tag(self, tag):
        """Add a manual tag to the user"""
        if self.db and self.user_id:
            cleaned_tag = self.tag_analyzer.clean_tag(tag)
            if self.tag_analyzer.validate_tag(cleaned_tag):
                self.db.add_user_tag(self.user_id, cleaned_tag, "manual")
                return True
        return False

    def remove_tag(self, tag):
        """Remove a tag from the user"""
        if self.db and self.user_id:
            self.db.remove_user_tag(self.user_id, tag)
            return True
        return False

    def update_language_preferences(self, native_language=None, preferred_languages=None, language_comfort_level=None):
        """Update user language preferences"""
        if self.db and self.user_id:
            self.db.update_language_preferences(
                self.user_id, 
                native_language, 
                preferred_languages, 
                language_comfort_level
            )
            # Refresh language preferences
            self.language_preferences = self.db.get_language_preferences(self.user_id)
            return True
        return False

    def get_language_preferences(self):
        """Get user language preferences"""
        if self.db and self.user_id:
            return self.db.get_language_preferences(self.user_id)
        return {
            'native_language': None,
            'preferred_languages': [],
            'language_comfort_level': 'english'
        }

    def get_similar_users(self, min_common_tags=2):
        """Get users with similar tags"""
        if self.db and self.user_id:
            return self.db.find_similar_users(self.user_id, min_common_tags)
        return []

    def suggest_tags(self):
        """Enhanced tag suggestions using AI and conversation analysis"""
        if not self.db or not self.user_id:
            return []
        
        current_tags = self.db.get_user_tags(self.user_id)
        conversation = self.get_conversation()
        
        # Use the enhanced tag suggestion method
        return self.tag_analyzer.suggest_tags_based_on_interests(current_tags, conversation) 