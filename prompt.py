"""
SUPER PROMPT: COMPREHENSIVE SYSTEM DOCUMENTATION
================================================

PROJECT: AI Chatbot Platform for Indian Users - Interest-Based Connection System
================================================================================

OVERVIEW
--------
This is a sophisticated Streamlit-based AI chatbot application designed to connect Indian users and NRIs 
based on shared interests. The platform combines OpenAI's GPT models, MongoDB for data persistence, 
and advanced user profiling with subtle cultural awareness and language support. The system supports 
individual conversations, group chats with AI participation, intelligent user matching based on conversation 
analysis, and language preference management. The primary focus is helping users find common ground and 
shared interests, with Indian cultural context as an underlying layer that enhances the connection experience.

ARCHITECTURE OVERVIEW
====================

Core Components:
1. main.py - Streamlit UI with Indian cultural interface and language preferences
2. chatbot.py - Individual chat logic with language-aware OpenAI integration
3. db.py - MongoDB database layer with language preferences and user session management
4. tag_analyzer.py - AI-powered tag inference with Indian cultural and linguistic context
5. group_chat.py - Multi-user group chat functionality with cultural awareness
6. session_manager.py - Persistent session management system
7. prompt.py - This comprehensive system documentation

DESIGN GUIDELINES
=================

1. INTEREST-BASED CONNECTION WITH CULTURAL AWARENESS
----------------------------------------------------
- Primary focus on connecting users through shared interests
- Subtle Indian cultural context as an underlying layer
- Support for 21 major Indian languages for user comfort
- Respect for diverse traditions and customs
- Professional and inclusive approach to cultural elements

2. MODULAR ARCHITECTURE
-----------------------
- Each component has a single responsibility
- Loose coupling between modules
- Clear interfaces between components
- Easy to extend and maintain
- Language-aware component design

3. USER-CENTRIC DESIGN
----------------------
- Persistent user sessions with UUIDs
- Session persistence across page refreshes
- Personalized experiences based on language preferences
- Intuitive navigation with multiple views
- Real-time feedback and suggestions
- Cultural and linguistic personalization

4. AI INTEGRATION PATTERNS
--------------------------
- OpenAI GPT-3.5-turbo for conversation generation
- Language-aware response generation
- Intelligent tag inference from conversation analysis
- Context-aware responses in group chats
- Adaptive follow-up question generation
- Cultural context integration

5. DATA PERSISTENCE STRATEGY
----------------------------
- MongoDB for scalable data storage
- Mock database for development/testing
- User-specific data isolation
- Conversation history preservation
- Language preferences storage
- Persistent session storage via URL parameters

6. SESSION MANAGEMENT STRATEGY
-----------------------------
- URL parameter-based session storage
- Fallback to Streamlit session state
- Cross-tab session persistence
- Activity tracking and monitoring
- Clean logout functionality
- Language preference persistence

FUNCTIONALITY SCOPE
==================

1. USER AUTHENTICATION & SESSION MANAGEMENT
-------------------------------------------
- Name-based user identification
- UUID generation for unique user tracking
- Persistent sessions across page refreshes
- Cross-tab session synchronization
- User profile management with language preferences
- Activity tracking and session monitoring
- Cultural greeting personalization

2. LANGUAGE PREFERENCES & CULTURAL CONTEXT
-----------------------------------------
- Native language selection (21 Indian languages)
- Preferred languages multi-selection
- Language comfort level settings
- Cultural greeting personalization
- Language-aware AI responses
- Bilingual interface elements
- Regional language support

3. INDIVIDUAL CHAT FUNCTIONALITY
--------------------------------
- OpenAI GPT-3.5-turbo powered conversations
- Language-aware response generation
- Cultural context integration
- Follow-up question generation every 3 turns
- Conversation history tracking
- Question acceptance/rejection learning
- Real-time conversation analysis
- Native language phrase integration

4. ENHANCED TAG MANAGEMENT SYSTEM
---------------------------------
- Manual tag addition by users
- AI-powered tag inference from conversations
- Automatic tag addition with user control
- Tag validation and cleaning
- Tag-based user similarity matching
- Intelligent tag suggestions with cultural context
- Automatic tag analysis every 5 conversation turns
- Language-aware tag suggestions
- Duplicate tag handling and deduplication

5. USER PROFILING & RECOMMENDATIONS
-----------------------------------
- Automatic conversation analysis
- Interest-based user profiling
- Similar user discovery algorithm
- Tag-based matching with similarity scoring
- User recommendation system
- Real-time profile updates
- Cultural and linguistic context consideration

6. GROUP CHAT SYSTEM
--------------------
- Multi-user group chat creation
- AI bot participation in group conversations
- Topic-based chat organization
- Persistent group message history
- Real-time conversation context
- Participant management
- Cultural context awareness

7. NAVIGATION & USER INTERFACE
------------------------------
- Multi-view navigation system
- Responsive Streamlit interface
- Real-time updates and feedback
- Intuitive user experience design
- Persistent view state management
- Bilingual interface elements
- Cultural iconography and greetings

8. SESSION PERSISTENCE
----------------------
- URL parameter-based session storage
- Fallback session state management
- Cross-browser tab synchronization
- Activity timestamp tracking
- Graceful session restoration
- Language preference persistence

IMPLEMENTATION DETAILS
=====================

1. ENHANCED DATABASE SCHEMA
---------------------------

Users Collection:
{
    user_id: UUID (string),
    name: string,
    created_at: datetime,
    profile_updated_at: datetime,
    native_language: string (optional),
    preferred_languages: [string],
    language_comfort_level: "english" | "mixed" | "native"
}

User Tags Collection:
{
    user_id: UUID (string),
    tag: string (lowercase),
    tag_type: "manual" | "inferred",
    created_at: datetime
}

Conversations Collection:
{
    user_id: UUID (string),
    role: "user" | "bot",
    message: string,
    conversation_turns: integer,
    timestamp: datetime
}

Group Chats Collection:
{
    group_id: UUID (string),
    topic_name: string,
    user_ids: [UUID strings],
    created_by: UUID (string),
    created_at: datetime,
    is_active: boolean
}

Group Messages Collection:
{
    group_id: UUID (string),
    user_id: UUID (string) | "ai_bot",
    message: string,
    message_type: "user" | "ai",
    timestamp: datetime
}

2. LANGUAGE PREFERENCES SYSTEM
------------------------------

Language Settings:
- Native Language: Primary language of the user
- Preferred Languages: Multiple languages user is comfortable with
- Comfort Level: How comfortable user is with native language conversations

Supported Languages:
- Hindi, English, Bengali, Telugu, Marathi, Tamil, Gujarati
- Kannada, Odia, Punjabi, Assamese, Sanskrit, Urdu, Malayalam
- Konkani, Manipuri, Nepali, Bodo, Santhali, Dogri, Kashmiri

Comfort Levels:
- English Only: Prefers English conversations
- Mixed Language: Comfortable with English + Native language blend
- Native Language Preferred: Prefers native language with English support

3. ENHANCED TAG INFERENCE ALGORITHM
-----------------------------------

Step 1: Cultural Context Analysis
- Indian cultural keywords and topics
- Regional language terms and transliterations
- Traditional and contemporary Indian topics
- Language-specific cultural references

Step 2: Keyword Matching
- Predefined topic keywords for 20+ categories
- Indian cultural and linguistic context
- Real-time conversation text analysis
- Pattern matching against topic keywords

Step 3: AI-Powered Extraction
- OpenAI GPT-3.5-turbo analysis
- Language-aware tag extraction
- Cultural context consideration
- Comma-separated tag parsing

Step 4: Tag Processing
- Tag cleaning and normalization
- Duplicate removal with source tracking
- Validation against rules
- Language preference alignment

4. AUTOMATIC TAG ADDITION SYSTEM
--------------------------------

Features:
- Automatic addition of all AI suggestions
- Manual control over tag addition
- Duplicate tag handling and deduplication
- Source tracking for tag origins
- Consolidated view of all suggestions
- Individual tag removal capabilities

Process:
1. Generate AI suggestions with cultural context
2. Collect all suggestions from different categories
3. Remove duplicates while tracking sources
4. Auto-add if enabled, or show manual options
5. Provide feedback on added tags
6. Allow individual tag management

5. CULTURAL GREETING SYSTEM
---------------------------

Personalized Greetings:
- Native language greetings for 12 major Indian languages
- Cultural context awareness
- Language preference consideration
- Regional language support

Greeting Examples:
- Hindi: "🙏 नमस्ते! आज मैं आपकी कैसे मदद कर सकता हूं?"
- Bengali: "🙏 নমস্কার! আজ আমি আপনাকে কীভাবে সাহায্য করতে পারি?"
- Telugu: "🙏 నమస్కారం! నేను ఈరోజు మీకు ఎలా సహాయం చేయగలను?"
- And more for other languages...

6. LANGUAGE-AWARE AI RESPONSES
------------------------------

System Prompt Enhancement:
- Base cultural awareness prompt
- Language preference integration
- Native language phrase suggestions
- Cultural context instructions
- Comfort level adaptation

Response Generation:
- Language-aware context building
- Cultural sensitivity integration
- Native language phrase incorporation
- Comfort level-based response style
- Regional cultural references

7. SIMILAR USER ALGORITHM
-------------------------

Input: User ID, Minimum common tags threshold
Process:
1. Get all tags for target user
2. Find all other users in database
3. For each user, calculate tag intersection
4. Filter users with >= minimum common tags
5. Sort by similarity score (number of common tags)
6. Return ranked list with common tags
7. Consider cultural and linguistic context

8. GROUP CHAT AI INTEGRATION
----------------------------

Context Building:
- Recent message history (last 10 messages)
- Group topic and participant information
- Conversation flow analysis
- Cultural context awareness

AI Response Generation:
- Context-aware prompt construction
- Participant-aware responses
- Topic-relevant conversation continuation
- Natural conversation flow maintenance
- Cultural sensitivity integration

9. SESSION PERSISTENCE ALGORITHM
--------------------------------

Login Process:
1. Check URL parameters for existing session
2. Validate session authenticity
3. Restore user context and preferences
4. Initialize chatbot with language preferences
5. Update activity timestamp

Session Storage:
- URL parameters for primary storage
- Session state for fallback
- Cross-tab synchronization
- Language preference persistence

TESTING STRATEGY
================

1. FUNCTIONAL TESTING
---------------------
- User authentication and session management
- Language preference settings and persistence
- Chat functionality with cultural context
- Tag management and automatic addition
- Group chat creation and participation
- Navigation between different views
- Session persistence across page refreshes

2. CULTURAL SENSITIVITY TESTING
-------------------------------
- Native language greeting accuracy
- Cultural context appropriateness
- Language preference integration
- Regional language support
- Cultural reference accuracy

3. LANGUAGE TESTING
-------------------
- All 21 supported Indian languages
- Language preference persistence
- Bilingual interface elements
- Mixed language conversations
- Comfort level adaptation

4. PERFORMANCE TESTING
----------------------
- Database query optimization
- Session management efficiency
- AI response generation speed
- Tag analysis performance
- Memory usage optimization

5. USER EXPERIENCE TESTING
--------------------------
- Intuitive navigation flow
- Responsive interface design
- Real-time feedback systems
- Error handling and recovery
- Accessibility considerations

DEPLOYMENT CONSIDERATIONS
========================

1. ENVIRONMENT SETUP
--------------------
- Python 3.8+ environment
- Required packages: streamlit, openai, pymongo, mongomock
- OpenAI API key configuration
- MongoDB connection setup
- Environment variable management

2. SCALABILITY CONSIDERATIONS
-----------------------------
- Database indexing for user queries
- Session management optimization
- AI response caching strategies
- Tag analysis performance tuning
- Memory usage monitoring

3. SECURITY CONSIDERATIONS
--------------------------
- API key security management
- User data privacy protection
- Session security validation
- Input sanitization and validation
- Cultural sensitivity in content moderation

4. MONITORING & MAINTENANCE
---------------------------
- User activity monitoring
- AI response quality tracking
- Cultural context accuracy
- Language preference usage analytics
- System performance metrics

FUTURE ENHANCEMENTS
===================

1. ADVANCED LANGUAGE FEATURES
-----------------------------
- Real-time language detection
- Automatic language switching
- Voice input and output support
- Regional dialect recognition
- Translation services integration

2. ENHANCED CULTURAL FEATURES
-----------------------------
- Festival and event awareness
- Regional cuisine and culture
- Traditional art and music
- Contemporary Indian topics
- Cultural celebration integration

3. IMPROVED AI CAPABILITIES
---------------------------
- Multi-modal conversation support
- Advanced cultural context understanding
- Personalized learning algorithms
- Enhanced tag suggestion accuracy
- Real-time cultural trend awareness

4. EXPANDED SOCIAL FEATURES
---------------------------
- Community-based recommendations
- Cultural interest groups
- Regional language communities
- Traditional knowledge sharing
- Contemporary Indian discussions

CONCLUSION
==========

This AI Chatbot Platform for Indian Users represents a comprehensive solution that combines
advanced AI capabilities with subtle cultural awareness to help users connect through shared interests.
The system successfully bridges technology with user connection, providing a platform where Indian users
and NRIs can find common ground while maintaining cultural sensitivity as an underlying layer.

The platform's architecture ensures scalability, maintainability, and extensibility,
while its focus on interest-based connection makes it uniquely suited for Indian users and contexts.
The combination of language preferences, automatic tag management, and subtle cultural
awareness creates a personalized experience that prioritizes connection over cultural display.
"""

def get_system_info():
    """Get basic system information"""
    return {
        "project_name": "AI Chatbot Platform for Indian Users",
        "version": "2.0",
        "features": [
            "Interest-Based Connection",
            "Subtle Cultural Awareness",
            "21 Language Support",
            "Automatic Tag Addition",
            "Language Preferences",
            "Enhanced AI Responses",
            "Group Chat System",
            "Session Persistence"
        ]
    }

def get_architecture_overview():
    """Get system architecture overview"""
    return {
        "core_components": [
            "main.py - Streamlit UI with Indian cultural interface",
            "chatbot.py - Language-aware chat logic",
            "db.py - MongoDB with language preferences",
            "tag_analyzer.py - Cultural context tag analysis",
            "group_chat.py - Multi-user group functionality",
            "session_manager.py - Persistent session management"
        ],
        "key_features": [
            "Cultural Sensitivity & Inclusion",
            "Modular Architecture",
            "User-Centric Design",
            "Language-Aware AI Integration",
            "Enhanced Data Persistence",
            "Comprehensive Session Management"
        ]
    }

def get_database_schema():
    """Get enhanced database schema"""
    return {
        "users_collection": {
            "user_id": "UUID (string)",
            "name": "string",
            "created_at": "datetime",
            "profile_updated_at": "datetime",
            "native_language": "string (optional)",
            "preferred_languages": "[string]",
            "language_comfort_level": "english|mixed|native"
        },
        "user_tags_collection": {
            "user_id": "UUID (string)",
            "tag": "string (lowercase)",
            "tag_type": "manual|inferred",
            "created_at": "datetime"
        },
        "conversations_collection": {
            "user_id": "UUID (string)",
            "role": "user|bot",
            "message": "string",
            "conversation_turns": "integer",
            "timestamp": "datetime"
        }
    }

def get_session_management_info():
    """Get session management details"""
    return {
        "primary_storage": "URL Parameters",
        "fallback_storage": "Streamlit Session State",
        "features": [
            "Cross-tab synchronization",
            "Language preference persistence",
            "Activity timestamp tracking",
            "Graceful session restoration",
            "Cultural context preservation"
        ]
    } 