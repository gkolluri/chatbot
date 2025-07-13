import re
from collections import Counter
import openai
import os

class TagAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Common topic keywords for tag inference (fallback) - Enhanced with Indian context
        self.topic_keywords = {
            'technology': ['ai', 'machine learning', 'programming', 'software', 'tech', 'computer', 'code', 'algorithm', 'startup', 'digital india', 'upi', 'aadhaar'],
            'sports': ['football', 'basketball', 'soccer', 'tennis', 'gym', 'workout', 'fitness', 'exercise', 'cricket', 'ipl', 'bcci', 'hockey', 'kabaddi', 'badminton'],
            'music': ['music', 'song', 'artist', 'concert', 'album', 'playlist', 'genre', 'instrument', 'bollywood', 'classical', 'carnatic', 'hindustani', 'ghazal', 'qawwali'],
            'food': ['cooking', 'recipe', 'restaurant', 'food', 'cuisine', 'chef', 'ingredients', 'dining', 'indian food', 'north indian', 'south indian', 'street food', 'biryani', 'curry', 'roti', 'dal'],
            'travel': ['travel', 'vacation', 'trip', 'destination', 'hotel', 'flight', 'tourism', 'adventure', 'india travel', 'himalayas', 'goa', 'kerala', 'rajasthan', 'varanasi'],
            'books': ['book', 'reading', 'novel', 'author', 'literature', 'fiction', 'non-fiction', 'library', 'hindi literature', 'bengali literature', 'marathi literature', 'tamil literature'],
            'movies': ['movie', 'film', 'cinema', 'actor', 'director', 'theater', 'streaming', 'series', 'bollywood', 'tollywood', 'kollywood', 'regional cinema', 'art house'],
            'science': ['science', 'research', 'experiment', 'discovery', 'theory', 'physics', 'chemistry', 'biology', 'isro', 'space research', 'ayurveda', 'yoga science'],
            'business': ['business', 'startup', 'entrepreneur', 'marketing', 'finance', 'investment', 'company', 'msme', 'make in india', 'startup india', 'digital payments'],
            'education': ['learning', 'study', 'course', 'university', 'school', 'education', 'academic', 'degree', 'iit', 'iim', 'upsc', 'competitive exams', 'online learning'],
            'health': ['health', 'medical', 'doctor', 'wellness', 'medicine', 'treatment', 'symptoms', 'therapy', 'ayurveda', 'yoga', 'homeopathy', 'allopathy', 'ayush'],
            'art': ['art', 'painting', 'drawing', 'creative', 'design', 'artist', 'gallery', 'exhibition', 'indian art', 'folk art', 'madhubani', 'warli', 'miniature painting'],
            'gaming': ['game', 'gaming', 'video game', 'console', 'player', 'esports', 'streaming', 'mobile gaming', 'pubg', 'free fire', 'indian gaming'],
            'politics': ['politics', 'government', 'policy', 'election', 'democracy', 'law', 'voting', 'parliament', 'state politics', 'local governance', 'panchayat'],
            'environment': ['environment', 'climate', 'sustainability', 'green', 'recycling', 'nature', 'conservation', 'swachh bharat', 'clean india', 'solar energy', 'water conservation'],
            'culture': ['culture', 'tradition', 'festival', 'religion', 'spirituality', 'temple', 'mosque', 'church', 'gurudwara', 'diwali', 'holi', 'eid', 'christmas', 'gurpurab'],
            'languages': ['hindi', 'english', 'bengali', 'telugu', 'marathi', 'tamil', 'gujarati', 'kannada', 'odia', 'punjabi', 'assamese', 'sanskrit', 'urdu', 'regional languages'],
            'fashion': ['fashion', 'clothing', 'style', 'designer', 'saree', 'kurta', 'salwar kameez', 'lehenga', 'ethnic wear', 'western wear', 'jewelry', 'accessories'],
            'spirituality': ['spirituality', 'meditation', 'yoga', 'bhakti', 'guru', 'ashram', 'temple', 'pilgrimage', 'karma', 'dharma', 'moksha', 'enlightenment'],
            'agriculture': ['agriculture', 'farming', 'farmer', 'crops', 'organic', 'pesticides', 'irrigation', 'krishi', 'mandi', 'farmer protests', 'agricultural technology']
        }

    def analyze_conversation_for_tags(self, conversation):
        """Analyze conversation and infer tags based on content"""
        if not conversation:
            return []
        
        # Combine all messages
        all_text = " ".join([msg for _, msg in conversation]).lower()
        
        # Find matching topics using static keywords (fallback)
        inferred_tags = []
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    inferred_tags.append(topic)
                    break
        
        # Use OpenAI to extract additional tags
        try:
            ai_tags = self._extract_tags_with_ai(conversation)
            inferred_tags.extend(ai_tags)
        except Exception as e:
            print(f"Error extracting AI tags: {e}")
        
        # Remove duplicates and return
        return list(set(inferred_tags))

    def _extract_tags_with_ai(self, conversation):
        """Use OpenAI to extract tags from conversation with Indian cultural context"""
        # Prepare conversation text
        conv_text = "\n".join([f"{role}: {msg}" for role, msg in conversation])
        
        prompt = f"""
        Analyze the following conversation and extract 3-5 relevant tags that represent the main topics, interests, or themes discussed.
        Consider Indian cultural context, local languages, regional interests, and cultural nuances.
        Include tags that reflect Indian culture, traditions, languages, regional interests, and contemporary Indian topics.
        Return only the tags as a comma-separated list, no explanations.
        
        Conversation:
        {conv_text}
        
        Tags:"""
        
        try:
            # Use web search for current trends and topics
            web_search_tool = {"type": "web_search_preview"}
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt + "\n\nUse web search to find current trending topics and interests."}],
                tools=[web_search_tool],
                max_tokens=100,
                temperature=0.3
            )
            
            tags_text = response.choices[0].message.content.strip()
            # Parse comma-separated tags
            tags = [tag.strip().lower() for tag in tags_text.split(',') if tag.strip()]
            return tags
        except Exception as e:
            print(f"Error in AI tag extraction: {e}")
            return []

    def generate_dynamic_tag_suggestions(self, user_tags, conversation_history, language_preferences=None, location_preferences=None):
        """Generate dynamic tag suggestions using AI with location context"""
        if not user_tags:
            return []
        
        # Build context for AI
        context_parts = [
            f"User's current tags: {', '.join(user_tags)}",
            f"Conversation context: {' '.join([msg[1] for msg in conversation_history[-5:]])}"
        ]
        
        # Add language context
        if language_preferences:
            native_lang = language_preferences.get('native_language')
            if native_lang:
                context_parts.append(f"User's native language: {native_lang}")
        
        # Add location context
        if location_preferences:
            city = location_preferences.get('city')
            state = location_preferences.get('state')
            country = location_preferences.get('country')
            
            location_context = []
            if city:
                location_context.append(f"City: {city}")
            if state:
                location_context.append(f"State: {state}")
            if country:
                location_context.append(f"Country: {country}")
            
            if location_context:
                context_parts.append(f"User's location: {', '.join(location_context)}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""
        Based on the following user information, suggest 5-8 relevant tags that would interest this user:

        {context}

        Guidelines:
        1. Suggest tags that are related to but not identical to existing tags
        2. Consider the user's conversation topics and interests
        3. Include location-specific interests if relevant (local culture, regional activities, etc.)
        4. Include language-specific interests if relevant (literature, cinema, etc.)
        5. Focus on discoverable interests they might enjoy
        6. Use lowercase, simple phrases
        7. Avoid duplicate or very similar tags to existing ones

        Provide only the tag suggestions, one per line, without explanations.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that suggests relevant interest tags based on user context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            suggestions = response.choices[0].message.content.strip().split('\n')
            # Clean and validate suggestions
            cleaned_suggestions = []
            for suggestion in suggestions:
                cleaned = self.clean_tag(suggestion.strip())
                if cleaned and self.validate_tag(cleaned) and cleaned not in user_tags:
                    cleaned_suggestions.append(cleaned)
            
            return cleaned_suggestions[:8]  # Return max 8 suggestions
            
        except Exception as e:
            print(f"Error generating dynamic suggestions: {e}")
            return []

    def _fallback_tag_suggestions(self, user_tags):
        """Fallback tag suggestions using static mappings"""
        if not user_tags:
            return []
        
        # Analyze conversation for new topics
        conversation_tags = self.analyze_conversation_for_tags([])
        
        # Find related tags based on existing user tags
        related_tags = []
        for tag in user_tags:
            if tag in self.topic_keywords:
                # Add related topics
                for topic, keywords in self.topic_keywords.items():
                    if topic != tag and any(kw in tag for kw in keywords):
                        related_tags.append(topic)
        
        # Combine and remove duplicates
        all_suggestions = conversation_tags + related_tags
        return list(set(all_suggestions))

    def suggest_tags_based_on_interests(self, user_tags, conversation):
        """Enhanced tag suggestions using AI and conversation analysis"""
        # Get AI-generated dynamic suggestions
        ai_suggestions = self.generate_dynamic_tag_suggestions(user_tags, conversation)
        
        # Get conversation-based suggestions
        conversation_suggestions = self.analyze_conversation_for_tags(conversation)
        
        # Combine and prioritize
        all_suggestions = ai_suggestions + conversation_suggestions
        
        # Remove duplicates and existing tags
        existing_tags_set = set(user_tags)
        unique_suggestions = [tag for tag in all_suggestions if tag not in existing_tags_set]
        
        return unique_suggestions[:8]  # Return top 8 suggestions

    def generate_category_suggestions(self, user_tags):
        """Generate category-based tag suggestions"""
        if not user_tags:
            return []
        
        try:
            prompt = f"""
            Based on these user tags: {', '.join(user_tags)}
            
            Generate 5-8 broader category tags that encompass these interests.
            Think of parent categories, industry sectors, or general domains.
            Include Indian cultural categories, regional interests, and traditional domains.
            Consider categories like: Indian culture, regional languages, traditional arts, contemporary India, etc.
            
            Return only the category tags as a comma-separated list.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.5
            )
            
            categories_text = response.choices[0].message.content.strip()
            categories = [cat.strip().lower() for cat in categories_text.split(',') if cat.strip()]
            
            # Filter out existing tags
            existing_tags_set = set(user_tags)
            unique_categories = [cat for cat in categories if cat not in existing_tags_set]
            
            return unique_categories
            
        except Exception as e:
            print(f"Error generating category suggestions: {e}")
            return []

    def generate_synonym_suggestions(self, user_tags):
        """Generate synonym-based tag suggestions"""
        if not user_tags:
            return []
        
        try:
            prompt = f"""
            For each of these tags: {', '.join(user_tags)}
            
            Generate 2-3 synonyms or alternative terms that mean the same thing.
            Include different ways to express the same concept.
            Consider Indian language equivalents (Hindi, regional languages) and cultural variations.
            Include both English and Indian language terms where appropriate.
            
            Return only the synonyms as a comma-separated list.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.6
            )
            
            synonyms_text = response.choices[0].message.content.strip()
            synonyms = [syn.strip().lower() for syn in synonyms_text.split(',') if syn.strip()]
            
            # Filter out existing tags
            existing_tags_set = set(user_tags)
            unique_synonyms = [syn for syn in synonyms if syn not in existing_tags_set]
            
            return unique_synonyms
            
        except Exception as e:
            print(f"Error generating synonym suggestions: {e}")
            return []

    def generate_related_concept_suggestions(self, user_tags):
        """Generate related concept suggestions"""
        if not user_tags:
            return []
        
        try:
            prompt = f"""
            Based on these user interests: {', '.join(user_tags)}
            
            Generate 5-8 closely related concepts, emerging trends, or adjacent topics.
            Think of what someone with these interests might also be interested in.
            Include Indian cultural context, regional interests, traditional practices, and contemporary Indian topics.
            Consider both global and Indian-specific related concepts.
            
            Return only the related concepts as a comma-separated list.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            
            concepts_text = response.choices[0].message.content.strip()
            concepts = [concept.strip().lower() for concept in concepts_text.split(',') if concept.strip()]
            
            # Filter out existing tags
            existing_tags_set = set(user_tags)
            unique_concepts = [concept for concept in concepts if concept not in existing_tags_set]
            
            return unique_concepts
            
        except Exception as e:
            print(f"Error generating related concept suggestions: {e}")
            return []

    def get_location_based_tag_suggestions(self, location_preferences):
        """Get tag suggestions based on user's location"""
        if not location_preferences:
            return []
        
        suggestions = []
        city = location_preferences.get('city', '').lower()
        state = location_preferences.get('state', '').lower()
        country = location_preferences.get('country', '').lower()
        
        # India-specific location tags
        if country == 'india':
            # Add general Indian cultural tags
            suggestions.extend([
                'indian culture', 'festivals', 'classical music', 'traditional arts',
                'regional cuisine', 'bollywood', 'cricket', 'yoga', 'meditation'
            ])
            
            # State-specific tags
            state_tags = {
                'maharashtra': ['marathi culture', 'bollywood', 'ganesh chaturthi', 'vada pav', 'lavani'],
                'kerala': ['malayalam culture', 'kathakali', 'ayurveda', 'backwaters', 'coconut cuisine'],
                'tamil nadu': ['tamil culture', 'bharatanatyam', 'carnatic music', 'temple architecture', 'filter coffee'],
                'karnataka': ['kannada culture', 'yakshagana', 'classical music', 'it industry', 'mysore silk'],
                'west bengal': ['bengali culture', 'durga puja', 'rabindra sangeet', 'fish curry', 'literature'],
                'rajasthan': ['rajasthani culture', 'folk dance', 'desert tourism', 'handicrafts', 'royal heritage'],
                'punjab': ['punjabi culture', 'bhangra', 'sikh heritage', 'punjabi music', 'agriculture'],
                'gujarat': ['gujarati culture', 'garba', 'business', 'textile industry', 'vegetarian cuisine'],
                'goa': ['goan culture', 'beach life', 'seafood', 'portuguese heritage', 'carnival'],
                'andhra pradesh': ['telugu culture', 'kuchipudi', 'spicy food', 'biryani', 'pearls'],
                'telangana': ['telugu culture', 'hyderabadi cuisine', 'tech industry', 'qawwali', 'nizami culture'],
                'odisha': ['odia culture', 'odissi dance', 'jagannath temple', 'classical arts', 'silver work'],
                'assam': ['assamese culture', 'bihu', 'tea culture', 'silk weaving', 'tribal arts'],
                'bihar': ['bihari culture', 'madhubani art', 'chhath puja', 'folk music', 'buddhist heritage'],
                'jharkhand': ['tribal culture', 'folk dance', 'handicrafts', 'forest culture', 'minerals'],
                'uttarakhand': ['pahadi culture', 'spiritual tourism', 'adventure sports', 'hill stations', 'yoga'],
                'himachal pradesh': ['hill culture', 'buddhist heritage', 'adventure tourism', 'apple orchards', 'handicrafts'],
                'jammu and kashmir': ['kashmiri culture', 'handicrafts', 'saffron', 'valley culture', 'shikaras'],
                'delhi': ['delhi culture', 'street food', 'historical monuments', 'political awareness', 'metro culture'],
                'uttar pradesh': ['up culture', 'classical music', 'mughal heritage', 'spiritual tourism', 'handicrafts'],
                'madhya pradesh': ['mp culture', 'tribal arts', 'wildlife', 'khajuraho', 'central indian culture'],
                'chhattisgarh': ['chhattisgarhi culture', 'tribal heritage', 'folk arts', 'rice culture', 'handicrafts'],
                'haryana': ['haryanvi culture', 'folk music', 'sports', 'agricultural lifestyle', 'traditional arts'],
                'manipur': ['manipuri culture', 'manipuri dance', 'martial arts', 'handloom', 'valley culture'],
                'meghalaya': ['khasi culture', 'living root bridges', 'hill stations', 'music culture', 'matrilineal society'],
                'mizoram': ['mizo culture', 'bamboo dance', 'handloom', 'hill culture', 'christian heritage'],
                'nagaland': ['naga culture', 'hornbill festival', 'tribal arts', 'handloom', 'hill culture'],
                'tripura': ['tripuri culture', 'handloom', 'bamboo crafts', 'tribal arts', 'hill culture'],
                'sikkim': ['sikkimese culture', 'buddhist heritage', 'organic farming', 'mountain culture', 'monasteries'],
                'arunachal pradesh': ['arunachali culture', 'tribal heritage', 'buddhist culture', 'handloom', 'hill culture'],
                'ladakh': ['ladakhi culture', 'buddhist monasteries', 'high altitude culture', 'adventure tourism', 'tibetan influence']
            }
            
            if state in state_tags:
                suggestions.extend(state_tags[state])
            
            # City-specific tags
            city_tags = {
                'mumbai': ['mumbai culture', 'bollywood', 'street food', 'local trains', 'business hub'],
                'delhi': ['delhi culture', 'historical monuments', 'street food', 'political hub', 'metro culture'],
                'bangalore': ['bangalore culture', 'it industry', 'pub culture', 'startup ecosystem', 'gardens'],
                'hyderabad': ['hyderabad culture', 'biryani', 'tech industry', 'nizami heritage', 'pearls'],
                'chennai': ['chennai culture', 'classical music', 'filter coffee', 'south indian culture', 'marina beach'],
                'kolkata': ['kolkata culture', 'adda culture', 'literature', 'fish curry', 'cultural capital'],
                'pune': ['pune culture', 'education hub', 'it industry', 'youth culture', 'pleasant weather'],
                'ahmedabad': ['ahmedabad culture', 'business hub', 'textile industry', 'garba', 'heritage'],
                'jaipur': ['jaipur culture', 'pink city', 'royal heritage', 'handicrafts', 'tourism'],
                'lucknow': ['lucknow culture', 'nawabi heritage', 'kebabs', 'chikan work', 'tehzeeb'],
                'kochi': ['kochi culture', 'spices', 'backwaters', 'seafood', 'kerala heritage'],
                'chandigarh': ['chandigarh culture', 'planned city', 'rock garden', 'punjabi culture', 'clean city'],
                'bhubaneswar': ['bhubaneswar culture', 'temple city', 'odissi dance', 'classical arts', 'planned city'],
                'guwahati': ['guwahati culture', 'tea culture', 'silk', 'assamese heritage', 'northeast gateway'],
                'thiruvananthapuram': ['trivandrum culture', 'kerala heritage', 'beaches', 'ayurveda', 'government hub'],
                'indore': ['indore culture', 'street food', 'business hub', 'clean city', 'commercial center'],
                'nagpur': ['nagpur culture', 'oranges', 'central location', 'marathi culture', 'tiger reserves'],
                'patna': ['patna culture', 'bihar heritage', 'ganga river', 'ancient culture', 'educational hub'],
                'bhopal': ['bhopal culture', 'lake city', 'mp heritage', 'cultural mix', 'historical sites'],
                'visakhapatnam': ['vizag culture', 'beaches', 'port city', 'steel industry', 'coastal life'],
                'agra': ['agra culture', 'taj mahal', 'mughal heritage', 'marble work', 'historical monuments'],
                'nashik': ['nashik culture', 'wine culture', 'religious tourism', 'marathi heritage', 'pilgrimage'],
                'varanasi': ['varanasi culture', 'spiritual capital', 'ghats', 'classical music', 'ancient culture'],
                'srinagar': ['srinagar culture', 'dal lake', 'kashmiri heritage', 'handicrafts', 'valley culture'],
                'amritsar': ['amritsar culture', 'golden temple', 'sikh heritage', 'punjabi culture', 'religious tourism']
            }
            
            if city in city_tags:
                suggestions.extend(city_tags[city])
        
        # Remove duplicates and return
        return list(set(suggestions))

    def get_enhanced_tag_suggestions(self, user_tags, conversation_history=None, language_preferences=None, location_preferences=None):
        """Get comprehensive tag suggestions including location-based ones"""
        all_suggestions = []
        
        # Get AI-generated suggestions
        if conversation_history:
            ai_suggestions = self.generate_dynamic_tag_suggestions(
                user_tags, conversation_history, language_preferences, location_preferences
            )
            all_suggestions.extend(ai_suggestions)
        
        # Get location-based suggestions
        if location_preferences:
            location_suggestions = self.get_location_based_tag_suggestions(location_preferences)
            all_suggestions.extend(location_suggestions)
        
        # Get category-based suggestions
        category_suggestions = self.generate_category_suggestions(user_tags)
        all_suggestions.extend(category_suggestions)
        
        # Get synonym suggestions
        synonym_suggestions = self.generate_synonym_suggestions(user_tags)
        all_suggestions.extend(synonym_suggestions)
        
        # Get related concept suggestions
        related_suggestions = self.generate_related_concept_suggestions(user_tags)
        all_suggestions.extend(related_suggestions)
        
        # Remove duplicates and existing tags
        unique_suggestions = []
        seen = set(user_tags)
        
        for suggestion in all_suggestions:
            if suggestion not in seen:
                unique_suggestions.append(suggestion)
                seen.add(suggestion)
        
        return unique_suggestions[:20]  # Return top 20 unique suggestions

    def get_popular_tags(self, db, limit=25):
        """Get most popular tags across all users with diverse interests"""
        # This would need to be implemented in the DB class
        # For now, return a diverse set of tags for swiping
        return [
            # Technology & Digital
            'technology', 'programming', 'ai', 'startup', 'digital', 'mobile apps', 'web development', 'data science',
            
            # Entertainment & Media
            'music', 'movies', 'bollywood', 'gaming', 'streaming', 'podcasts', 'comedy', 'dance',
            
            # Sports & Fitness
            'sports', 'cricket', 'fitness', 'yoga', 'gym', 'running', 'swimming', 'badminton',
            
            # Food & Cuisine
            'food', 'cooking', 'indian food', 'street food', 'biryani', 'desserts', 'healthy eating',
            
            # Travel & Adventure
            'travel', 'adventure', 'hiking', 'photography', 'backpacking', 'road trips', 'international travel',
            
            # Arts & Culture
            'art', 'culture', 'classical music', 'folk art', 'traditional crafts', 'painting', 'dance',
            
            # Business & Career
            'business', 'entrepreneurship', 'career', 'finance', 'investing', 'marketing', 'consulting',
            
            # Education & Learning
            'education', 'learning', 'online courses', 'languages', 'reading', 'writing', 'research',
            
            # Health & Wellness
            'health', 'wellness', 'meditation', 'ayurveda', 'mental health', 'nutrition', 'fitness',
            
            # Lifestyle & Personal
            'fashion', 'beauty', 'lifestyle', 'self-improvement', 'motivation', 'productivity', 'minimalism',
            
            # Social & Community
            'community', 'volunteering', 'social work', 'networking', 'mentoring', 'leadership',
            
            # Creative & Hobbies
            'photography', 'writing', 'poetry', 'music production', 'gardening', 'diy', 'crafts',
            
            # Regional & Cultural
            'regional cinema', 'classical dance', 'folk music', 'traditional festivals', 'heritage',
            
            # Contemporary
            'sustainability', 'environment', 'social media', 'influencer', 'content creation', 'digital nomad'
        ]

    def clean_tag(self, tag):
        """Clean and normalize a tag"""
        return tag.lower().strip()

    def validate_tag(self, tag):
        """Validate if a tag is appropriate"""
        # Basic validation - can be extended
        if len(tag) < 2 or len(tag) > 50:
            return False
        if not re.match(r'^[a-zA-Z0-9\s\-_]+$', tag):
            return False
        return True 