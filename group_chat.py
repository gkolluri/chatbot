import os
import time
from openai import OpenAI
from datetime import datetime
from citation_system import CitationGenerator, CitationDisplayManager
from logging_utils import group_logger, agent_logger, debug, info, error, log_rag_context, log_topic_focus_verification, flow_logger

class GroupChat:
    def __init__(self, db, group_id, user_id, user_name):
        self.db = db
        self.group_id = group_id
        self.user_id = user_id
        self.user_name = user_name
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize citation system
        self.citation_generator = CitationGenerator(db_interface=db)
        self.citation_display_manager = CitationDisplayManager()
        
        info(f"GroupChat initialized for group {group_id}", group_id=group_id, user_id=user_id, user_name=user_name)
    
    def send_message(self, message: str) -> dict:
        """Send a message to the group chat and get AI response"""
        try:
            # Log message sent
            group_logger.message_sent(self.group_id, self.user_id, message)
            
            # Get group info
            group_info = self.db.get_group_info(self.group_id)
            if not group_info:
                return {"success": False, "error": "Group not found"}
            
            topic_name = group_info.get('topic_name', 'General Discussion')
            participant_ids = group_info.get('user_ids', [])
            
            # Build RAG context
            start_time = time.time()
            rag_context = self._build_rag_context(topic_name, participant_ids)
            rag_time = time.time() - start_time
            
            # Log RAG context building
            flow_logger.rag_call_start(
                rag_type="group_chat_context",
                query=f"Group: {topic_name}, Participants: {len(participant_ids)}",
                parameters={"topic_name": topic_name, "participant_count": len(participant_ids)},
                user_id=self.user_id
            )
            
            flow_logger.rag_call_end(
                rag_type="group_chat_context",
                results=[{"context": rag_context}],
                processing_time=rag_time,
                user_id=self.user_id
            )
            
            # Generate AI response
            start_time = time.time()
            ai_response = self._generate_ai_response(message, rag_context, topic_name)
            ai_time = time.time() - start_time
            
            # Log LLM call
            flow_logger.llm_call_start(
                model="gpt-4",
                prompt_length=len(message) + len(rag_context),
                user_id=self.user_id
            )
            
            flow_logger.llm_call_end(
                model="gpt-4",
                response_length=len(ai_response),
                processing_time=ai_time,
                user_id=self.user_id
            )
            
            # Generate citations
            citations = self.citation_generator.generate_citations_for_response(
                ai_response, 
                {"topic_name": topic_name, "participant_count": len(participant_ids)}, 
                message, 
                self.user_id, 
                [str(pid) for pid in participant_ids]
            )
            
            # Log citations
            citation_types = [citation.type for citation in citations]
            flow_logger.rag_call_start(
                rag_type="citation_generation",
                query=ai_response[:100],
                parameters={"citation_count": len(citations)},
                user_id=self.user_id
            )
            
            flow_logger.rag_call_end(
                rag_type="citation_generation",
                results=[{"citations": citations}],
                processing_time=0.1,  # Citation generation is usually fast
                user_id=self.user_id
            )
            
            # Store message in database
            self.db.add_group_message(
                group_id=self.group_id,
                user_id=self.user_id,
                message=message,
                message_type='user'
            )
            
            # Store AI response
            self.db.add_group_message(
                group_id=self.group_id,
                user_id='ai',
                message=ai_response,
                message_type='ai',
                citations=citations
            )
            
            # Log AI response
            group_logger.ai_response_generated(
                self.group_id, ai_response, citations, topic_name, len(participant_ids)
            )
            
            return {
                "success": True,
                "response": ai_response,
                "citations": [citation.to_dict() for citation in citations],
                "topic_name": topic_name,
                "participant_count": len(participant_ids)
            }
            
        except Exception as e:
            error(f"Error sending group message", error=e, group_id=self.group_id, user_id=self.user_id)
            return {"success": False, "error": str(e)}
    
    def get_messages(self):
        """Get formatted messages for this group chat"""
        try:
            # Get raw messages from database
            raw_messages = self.db.get_group_messages(self.group_id)
            
            # Format messages for UI display
            formatted_messages = []
            for msg in raw_messages:
                formatted_msg = {
                    'message': msg.get('message', ''),
                    'sender': msg.get('user_name', msg.get('user_id', 'Unknown')),
                    'user_id': msg.get('user_id', ''),
                    'timestamp': msg.get('timestamp', ''),
                    'is_ai': msg.get('message_type') == 'ai' or msg.get('user_id') == 'ai',
                    'citations': msg.get('citations', []),
                    'citation_links': msg.get('citation_links', ''),
                    'citation_details': msg.get('citation_details', {}),
                    'has_citations': bool(msg.get('citations', []))
                }
                formatted_messages.append(formatted_msg)
            
            return formatted_messages
            
        except Exception as e:
            error(f"Error getting messages for group {self.group_id}", error=e, user_id=self.user_id)
            return []
    
    def _build_rag_context(self, topic_name: str, participant_ids: list) -> str:
        """Build RAG context for group chat"""
        try:
            # Get participant data
            participant_data = []
            for user_id in participant_ids:
                user_info = self.db.get_user_profile(user_id)
                if user_info:
                    # Add user tags to the profile data
                    user_tags = self.db.get_user_tags(user_id)
                    user_info['tags'] = user_tags
                    participant_data.append(user_info)
            
            # Build shared interests
            shared_interests = {}
            for participant in participant_data:
                user_tags = participant.get('tags', [])
                for tag in user_tags:
                    if tag not in shared_interests:
                        shared_interests[tag] = []
                    shared_interests[tag].append(participant.get('name', 'Unknown'))
            
            # Get topic-related tags (enhanced matching)
            topic_related_tags = []
            user_specific_preferences = {}
            for participant in participant_data:
                user_tags = participant.get('tags', [])
                user_name = participant.get('name', 'Unknown')
                user_specific_prefs = []
                
                for tag in user_tags:
                    if self._is_tag_related_to_topic(tag, topic_name):
                        topic_related_tags.append(tag)
                        # Collect specific preferences for each user based on topic
                        if self._is_specific_preference_tag(tag, topic_name):
                            user_specific_prefs.append(tag)
                
                if user_specific_prefs:
                    user_specific_preferences[user_name] = user_specific_prefs
            
            # Build location context
            location_context = ""
            locations = [p.get('location', {}) for p in participant_data if p.get('location')]
            if locations:
                cities = [loc.get('city', '') for loc in locations if loc.get('city')]
                if cities:
                    location_context = f"Participants are from: {', '.join(set(cities))}"
            
            # Log RAG context building
            log_rag_context(
                topic_name=topic_name,
                participant_data=participant_data,
                shared_interests=shared_interests,
                location_context=location_context,
                user_id=self.user_id
            )
            
            # Build context string
            context_parts = [
                f"Group Topic: {topic_name}",
                f"Number of Participants: {len(participant_data)}"
            ]
            
            # Add individual user specific preferences prominently
            if user_specific_preferences:
                context_parts.append("User Specific Preferences:")
                for user_name, specific_prefs in user_specific_preferences.items():
                    context_parts.append(f"  - {user_name}: {', '.join(specific_prefs)}")
            
            if topic_related_tags:
                context_parts.append(f"Topic-related interests: {', '.join(set(topic_related_tags))}")
            
            if shared_interests:
                high_interest_tags = [tag for tag, users in shared_interests.items() if len(users) > 1]
                if high_interest_tags:
                    context_parts.append(f"Shared interests: {', '.join(high_interest_tags)}")
            
            if location_context:
                context_parts.append(location_context)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            error(f"Error building RAG context", error=e, group_id=self.group_id)
            return f"Group Topic: {topic_name}"
    
    def _is_tag_related_to_topic(self, tag: str, topic_name: str) -> bool:
        """Check if a tag is related to the group topic with enhanced matching"""
        tag_lower = tag.lower()
        topic_lower = topic_name.lower()
        
        # Direct match
        if tag_lower in topic_lower or topic_lower in tag_lower:
            return True
        
        # Enhanced topic-tag mappings
        topic_mappings = {
            'technology': ['tech', 'programming', 'coding', 'software', 'computer', 'ai', 'machine learning', 'python', 'javascript', 'web development'],
            'food': ['cooking', 'recipe', 'restaurant', 'cuisine', 'chef', 'indian food', 'south indian food', 'north indian food', 'chinese food', 'italian food', 'mexican food', 'thai food', 'japanese food', 'korean food', 'american food', 'fast food', 'street food', 'vegetarian', 'vegan', 'spicy food', 'dessert', 'bakery', 'cafe', 'dining', 'lunch', 'dinner', 'breakfast', 'brunch'],
            'sports': ['fitness', 'exercise', 'gym', 'athletics', 'game', 'football', 'basketball', 'soccer', 'tennis', 'swimming', 'running', 'yoga', 'cycling'],
            'music': ['song', 'artist', 'concert', 'band', 'melody', 'guitar', 'piano', 'singing', 'classical', 'rock', 'pop', 'jazz', 'blues', 'bollywood', 'hip hop', 'country music', 'electronic music', 'indie music', 'folk music', 'reggae music', 'punk music', 'metal music', 'r&b music', 'soul music', 'gospel music', 'world music', 'ambient music', 'techno music', 'house music', 'trance music', 'drum and bass', 'dubstep', 'trap music', 'classical music', 'rock music', 'pop music', 'jazz music', 'blues music'],
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
    
    def _is_specific_preference_tag(self, tag: str, topic_name: str) -> bool:
        """Check if a tag is a specific preference that should be prioritized based on the topic"""
        tag_lower = tag.lower()
        topic_lower = topic_name.lower()
        
        # Define specific preference tags for different topics
        specific_preferences = {
            'food': [
                'south indian food', 'north indian food', 'indian food', 'indian cuisine',
                'chinese food', 'italian food', 'mexican food', 'thai food', 'japanese food', 
                'korean food', 'american food', 'fast food', 'street food', 'vegetarian', 
                'vegan', 'spicy food', 'dessert', 'cuisine', 'cooking', 'bakery', 'cafe'
            ],
            'music': [
                'bollywood', 'classical music', 'rock music', 'pop music', 'jazz music',
                'hip hop', 'country music', 'electronic music', 'indie music', 'folk music',
                'blues music', 'reggae music', 'punk music', 'metal music', 'r&b music',
                'soul music', 'gospel music', 'world music', 'ambient music', 'techno music',
                'house music', 'trance music', 'drum and bass', 'dubstep', 'trap music'
            ],
            'technology': [
                'python', 'javascript', 'java', 'c++', 'react', 'angular', 'vue', 'node.js',
                'machine learning', 'artificial intelligence', 'data science', 'web development',
                'mobile development', 'cloud computing', 'cybersecurity', 'blockchain',
                'devops', 'software engineering', 'programming', 'coding', 'databases',
                'frontend', 'backend', 'full stack', 'ui/ux design', 'game development'
            ],
            'sports': [
                'football', 'basketball', 'soccer', 'tennis', 'cricket', 'baseball',
                'swimming', 'running', 'cycling', 'gym', 'yoga', 'pilates', 'martial arts',
                'boxing', 'wrestling', 'golf', 'volleyball', 'badminton', 'table tennis',
                'hockey', 'rugby', 'athletics', 'fitness', 'exercise', 'workout'
            ],
            'travel': [
                'adventure travel', 'backpacking', 'luxury travel', 'budget travel',
                'solo travel', 'family travel', 'business travel', 'cultural tourism',
                'eco tourism', 'beach vacation', 'mountain hiking', 'city breaks',
                'road trips', 'cruises', 'camping', 'hostels', 'hotels', 'airbnb'
            ],
            'business': [
                'entrepreneurship', 'startup', 'venture capital', 'marketing', 'sales',
                'finance', 'accounting', 'consulting', 'project management', 'leadership',
                'management', 'business strategy', 'operations', 'human resources',
                'customer service', 'product management', 'business development'
            ],
            'education': [
                'online learning', 'university', 'college', 'courses', 'certification',
                'professional development', 'skill building', 'training', 'workshops',
                'seminars', 'research', 'academic writing', 'teaching', 'tutoring',
                'educational technology', 'e-learning', 'distance learning'
            ],
            'health': [
                'nutrition', 'diet', 'wellness', 'mental health', 'physical fitness',
                'medical', 'healthcare', 'medicine', 'therapy', 'counseling',
                'mindfulness', 'meditation', 'stress management', 'healthy lifestyle',
                'preventive care', 'alternative medicine', 'holistic health'
            ]
        }
        
        # Check if tag matches any topic-specific preferences
        for topic_key, preferences in specific_preferences.items():
            if topic_key in topic_lower:
                if tag_lower in preferences:
                    return True
        
        # Also check for general specific preferences that might be relevant
        # For example, specific brands, names, or detailed preferences
        if len(tag_lower.split()) > 1:  # Multi-word tags are often specific preferences
            return True
        
        return False
    
    def _generate_ai_response(self, message: str, rag_context: str, topic_name: str) -> str:
        """Generate AI response using RAG context"""
        try:
            system_prompt = f"""You are an AI assistant in a group chat about "{topic_name}". 
            
Context about the group:
{rag_context}

Your role:
1. Respond naturally and engagingly to the user's message
2. Keep responses focused on the group's topic when relevant
3. Be helpful, friendly, and encourage group participation
4. Use the context to make responses more relevant to the group's interests
5. Keep responses concise but informative
6. IMPORTANT: When making recommendations, prioritize options that match the user's specific preferences listed in the context above
7. If a user has specific preferences (e.g., "bollywood" for music, "south indian food" for food), prioritize those types in your recommendations
8. If a user has specific preferences, mention those types first and explain why they match the user's interests

User message: {message}"""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error(f"Error generating AI response", error=e, group_id=self.group_id)
            return "I'm having trouble responding right now. Please try again."

class GroupChatManager:
    def __init__(self, db):
        self.db = db
        info("GroupChatManager initialized", operation="manager_init")
    
    def create_group_chat(self, topic_name: str, user_ids: list, created_by: str) -> dict:
        """Create a new group chat"""
        try:
            # Log group creation start
            flow_logger.rag_call_start(
                rag_type="group_creation",
                query=f"Create group: {topic_name}",
                parameters={"topic_name": topic_name, "user_count": len(user_ids)},
                user_id=created_by
            )
            
            group_id = self.db.create_group_chat(topic_name, user_ids, created_by)
            
            # Log group creation end
            flow_logger.rag_call_end(
                rag_type="group_creation",
                results=[{"group_id": group_id}],
                processing_time=0.1,
                user_id=created_by
            )
            
            group_logger.group_created(group_id, topic_name, user_ids, created_by)
            
            return {
                "success": True,
                "group_id": group_id,
                "topic_name": topic_name,
                "participant_count": len(user_ids)
            }
            
        except Exception as e:
            error(f"Error creating group chat", error=e, topic_name=topic_name, created_by=created_by)
            return {"success": False, "error": str(e)}
    
    def get_user_groups(self, user_id: str) -> dict:
        """Get groups for a user"""
        try:
            # Log group retrieval start
            flow_logger.rag_call_start(
                rag_type="user_groups_retrieval",
                query=f"Get groups for user: {user_id}",
                parameters={"user_id": user_id},
                user_id=user_id
            )
            
            groups = self.db.get_user_group_chats(user_id)
            
            # Add participant names to each group
            for group in groups:
                if 'user_ids' in group and 'participants' not in group:
                    participant_names = []
                    for participant_id in group['user_ids']:
                        if participant_id == "ai_bot":
                            participant_names.append("AI Assistant")
                        else:
                            try:
                                user_info = self.db.get_user_profile(participant_id)
                                if user_info and isinstance(user_info, dict):
                                    participant_names.append(user_info.get('name', f'User {participant_id[:8]}'))
                                else:
                                    participant_names.append(f'User {participant_id[:8]}')
                            except Exception as e:
                                error(f"Error getting user profile for {participant_id}", error=e)
                                participant_names.append(f'User {participant_id[:8]}')
                    group['participants'] = participant_names
            
            # Log group retrieval end
            flow_logger.rag_call_end(
                rag_type="user_groups_retrieval",
                results=groups,
                processing_time=0.1,
                user_id=user_id
            )
            
            return {
                "success": True,
                "groups": groups
            }
            
        except Exception as e:
            error(f"Error getting user groups", error=e, user_id=user_id)
            return {"success": False, "error": str(e)} 

    def get_group_chat(self, group_id: str, user_id: str):
        """Get a GroupChat instance for a user and group, or None if not allowed"""
        # Get group info from DB
        group_info = self.db.get_group_info(group_id)
        if not group_info:
            return None
        # Check if user is a participant
        if user_id not in group_info.get('user_ids', []):
            return None
        # Return a GroupChat instance
        return GroupChat(self.db, group_id, user_id, self.db.get_user_profile(user_id).get('name', 'Unknown')) 