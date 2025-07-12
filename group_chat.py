import os
from openai import OpenAI
from datetime import datetime

class GroupChat:
    def __init__(self, db, group_id, user_id, user_name):
        self.db = db
        self.group_id = group_id
        self.user_id = user_id
        self.user_name = user_name
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
        
    def send_message(self, message):
        """Send a message to the group chat"""
        # Save user message
        self.db.add_group_message(self.group_id, self.user_id, message, "user")
        
        # Get group info and participants
        group_info = self.db.get_group_info(self.group_id)
        if not group_info:
            return "Error: Group not found"
        
        # Get recent messages for context
        recent_messages = self.db.get_group_messages(self.group_id, limit=20)
        
        # Generate AI response
        ai_response = self._generate_ai_response(recent_messages, group_info)
        
        # Save AI response
        self.db.add_group_message(self.group_id, "ai_bot", ai_response, "ai")
        
        return ai_response
    
    def _generate_ai_response(self, recent_messages, group_info):
        """Generate AI response based on group context"""
        # Build conversation context
        context = self._build_conversation_context(recent_messages, group_info)
        
        prompt = f"""
        You are an AI assistant participating in a group chat. 
        
        Group Topic: {group_info['topic_name']}
        Participants: {', '.join([self._get_user_name_by_id(uid) for uid in group_info['user_ids']])}
        
        Recent conversation:
        {context}
        
        Respond naturally as if you're part of the conversation. Keep your response conversational and relevant to the topic.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant participating in a group chat. Be conversational and engaging."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Sorry, I'm having trouble responding right now. Error: {str(e)}"
    
    def _build_conversation_context(self, messages, group_info):
        """Build conversation context from recent messages"""
        context_lines = []
        for msg in messages[-10:]:  # Last 10 messages
            if msg['message_type'] == 'user':
                user_name = self._get_user_name_by_id(msg['user_id'])
                context_lines.append(f"{user_name}: {msg['message']}")
            else:
                context_lines.append(f"AI Assistant: {msg['message']}")
        
        return "\n".join(context_lines)
    
    def _get_user_name_by_id(self, user_id):
        """Get user name by user ID"""
        if user_id == "ai_bot":
            return "AI Assistant"
        
        user_profile = self.db.get_user_profile(user_id)
        return user_profile['name'] if user_profile else "Unknown User"
    
    def get_messages(self, limit=50):
        """Get group chat messages"""
        messages = self.db.get_group_messages(self.group_id, limit)
        formatted_messages = []
        
        for msg in messages:
            if msg['message_type'] == 'user':
                user_name = self._get_user_name_by_id(msg['user_id'])
                formatted_messages.append({
                    'sender': user_name,
                    'message': msg['message'],
                    'timestamp': msg['timestamp'],
                    'is_ai': False
                })
            else:
                formatted_messages.append({
                    'sender': 'AI Assistant',
                    'message': msg['message'],
                    'timestamp': msg['timestamp'],
                    'is_ai': True
                })
        
        return formatted_messages

class GroupChatManager:
    def __init__(self, db):
        self.db = db
    
    def create_group_chat(self, topic_name, user_ids, created_by):
        """Create a new group chat"""
        return self.db.create_group_chat(topic_name, user_ids, created_by)
    
    def get_user_groups(self, user_id):
        """Get all group chats for a user"""
        groups = self.db.get_user_group_chats(user_id)
        formatted_groups = []
        
        for group in groups:
            # Get participant names
            participant_names = []
            for uid in group['user_ids']:
                if uid != "ai_bot":
                    user_profile = self.db.get_user_profile(uid)
                    if user_profile:
                        participant_names.append(user_profile['name'])
            
            # Add AI bot to participants
            participant_names.append("AI Assistant")
            
            formatted_groups.append({
                'group_id': group['group_id'],
                'topic_name': group['topic_name'],
                'participants': participant_names,
                'created_at': group['created_at'],
                'created_by': group['created_by']
            })
        
        return formatted_groups
    
    def get_group_chat(self, group_id, user_id):
        """Get a specific group chat instance"""
        group_info = self.db.get_group_info(group_id)
        if not group_info or user_id not in group_info['user_ids']:
            return None
        
        user_profile = self.db.get_user_profile(user_id)
        user_name = user_profile['name'] if user_profile else "Unknown"
        
        return GroupChat(self.db, group_id, user_id, user_name)
    
    def suggest_group_topics(self, user_tags):
        """Suggest group chat topics based on user tags"""
        if not user_tags:
            return ["General Discussion", "Getting to Know Each Other", "Open Chat"]
        
        # Create topic suggestions based on tags
        topics = []
        for tag in user_tags[:3]:  # Use top 3 tags
            topics.append(f"{tag.title()} Discussion")
            topics.append(f"{tag.title()} Enthusiasts")
        
        # Add some general topics
        topics.extend([
            "General Discussion",
            "Open Chat",
            "Getting to Know Each Other"
        ])
        
        return topics[:5]  # Return top 5 suggestions 