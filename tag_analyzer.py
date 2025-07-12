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
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.3
            )
            
            tags_text = response.choices[0].message.content.strip()
            # Parse comma-separated tags
            tags = [tag.strip().lower() for tag in tags_text.split(',') if tag.strip()]
            return tags
        except Exception as e:
            print(f"Error in AI tag extraction: {e}")
            return []

    def generate_dynamic_tag_suggestions(self, user_tags, conversation=None, language_preferences=None):
        """Generate dynamic tag suggestions using OpenAI LLM with language preferences"""
        if not user_tags:
            return []
        
        try:
            # Prepare context for AI
            context = f"User's current tags: {', '.join(user_tags)}"
            if conversation:
                conv_text = "\n".join([f"{role}: {msg}" for role, msg in conversation[-10:]])  # Last 10 messages
                context += f"\nRecent conversation:\n{conv_text}"
            
            # Add language context
            language_context = ""
            if language_preferences:
                native_lang = language_preferences.get('native_language')
                preferred_langs = language_preferences.get('preferred_languages', [])
                comfort_level = language_preferences.get('language_comfort_level', 'english')
                
                if native_lang:
                    language_context += f"\nUser's native language: {native_lang}"
                if preferred_langs:
                    language_context += f"\nUser's preferred languages: {', '.join(preferred_langs)}"
                language_context += f"\nLanguage comfort level: {comfort_level}"
            
            prompt = f"""
            Based on the user's current tags and context, generate 8-12 additional tag suggestions that help connect people with shared interests.
            Include the following types of suggestions:
            1. Category tags (broader categories the user might be interested in)
            2. Related synonyms (different ways to express similar interests)
            3. Subcategories (more specific areas within their interests)
            4. Indian language equivalents (subtle, for users who prefer native languages)
            5. Contemporary topics and trends
            6. Related concepts (closely related topics)
            7. Emerging trends in their areas of interest
            8. Popular interests among Indian users and NRIs
            
            Focus on interests that help people connect and find common ground.
            Include both English and Indian language terms where appropriate, but keep it subtle.
            
            {language_context}
            
            Context: {context}
            
            Return only the tags as a comma-separated list, no explanations or categories.
            Make suggestions diverse and relevant to the user's interests, prioritizing connection and shared interests.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at understanding user interests and suggesting relevant tags. Provide diverse, relevant suggestions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            suggestions_text = response.choices[0].message.content.strip()
            suggestions = [tag.strip().lower() for tag in suggestions_text.split(',') if tag.strip()]
            
            # Filter out duplicates and existing tags
            existing_tags_set = set(user_tags)
            unique_suggestions = [tag for tag in suggestions if tag not in existing_tags_set]
            
            return unique_suggestions[:10]  # Limit to 10 suggestions
            
        except Exception as e:
            print(f"Error generating dynamic tag suggestions: {e}")
            return self._fallback_tag_suggestions(user_tags)

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