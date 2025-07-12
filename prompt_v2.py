"""
SUPER PROMPT V2: COMPREHENSIVE MULTI-AGENT SYSTEM DOCUMENTATION
===============================================================

PROJECT: Multi-Agent AI Chatbot Platform for Indian Users - LangGraph Architecture
==================================================================================

OVERVIEW
--------
This is a sophisticated Streamlit-based multi-agent AI chatbot application using LangGraph 
architecture to connect Indian users and NRIs based on shared interests. The platform combines 
OpenAI's GPT models, MongoDB for data persistence, and a coordinated multi-agent system with 
specialized LangGraph agents for different functionalities. The system supports individual 
conversations, group chats with AI participation, intelligent user matching based on conversation 
analysis, and language preference management. The primary focus is helping users find common 
ground and shared interests, with Indian cultural context as an underlying layer that enhances 
the connection experience.

ARCHITECTURE OVERVIEW
====================

Core Components:
1. main_v2.py - Streamlit UI with interactive graphs and multi-view navigation
2. multi_agent_chatbot.py - Multi-agent chatbot logic with LangGraph coordination
3. agents/ - LangGraph-based specialized agents for different functionalities
4. db.py - MongoDB database layer with language preferences and user session management
5. session_manager.py - Persistent session management with URL parameter storage
6. prompt_v2.py - This comprehensive system documentation (Multi-Agent version)

Multi-Agent System:
1. ConversationAgent - Handles user conversations and responses
2. TagAnalysisAgent - Analyzes conversations for interest tags
3. UserProfileAgent - Manages user profiles and preferences
4. GroupChatAgent - Handles group chat functionality
5. SessionAgent - Manages user sessions and persistence
6. LanguageAgent - Handles language preferences and cultural context

DESIGN GUIDELINES
=================

1. MULTI-AGENT ARCHITECTURE WITH LANGGRAPH
------------------------------------------
- LangGraph-based agent coordination for scalable processing
- Specialized agents with single responsibilities
- State management and workflow coordination
- Dynamic agent discovery and registration
- Real-time agent status monitoring

2. INTEREST-BASED CONNECTION WITH SUBTLE CULTURAL AWARENESS
-----------------------------------------------------------
- Primary focus on connecting users through shared interests and common ground
- Subtle Indian cultural context as an underlying layer that enhances connections
- Support for 21 major Indian languages for user comfort and accessibility
- Respect for diverse traditions and customs without being overly prominent
- Professional and inclusive approach to cultural elements
- Balance between global interests and Indian cultural nuances

3. MODULAR ARCHITECTURE
-----------------------
- Each agent has a single responsibility and clear interfaces
- Loose coupling between agents for easy maintenance and extension
- Language-aware component design throughout the system
- Scalable database schema with proper indexing
- Session persistence across browser tabs and page refreshes

4. USER-CENTRIC DESIGN
----------------------
- Interactive graphs and analytics dashboard
- Persistent user sessions with UUID-based tracking
- Session persistence across page refreshes using URL parameters
- Personalized experiences based on language preferences
- Intuitive navigation with multiple views (chat, profile, similar users, group chats, system status)
- Real-time feedback and suggestions with immediate visual feedback
- Cultural and linguistic personalization that feels natural

5. AI INTEGRATION PATTERNS
--------------------------
- OpenAI GPT-3.5-turbo for conversation generation and tag analysis
- Language-aware response generation with cultural context
- Intelligent tag inference from conversation analysis every 5 turns
- Dynamic tag suggestions with multiple categories (AI, category, synonym, related)
- Context-aware responses in group chats with AI participation
- Adaptive follow-up question generation with learning from user feedback
- Cultural context integration that enhances rather than dominates

6. DATA PERSISTENCE STRATEGY
----------------------------
- MongoDB for scalable data storage with proper collections
- Mock database for development/testing with seamless switching
- User-specific data isolation with proper indexing
- Conversation history preservation with turn tracking
- Language preferences storage with comfort level settings
- Persistent session storage via URL parameters with fallback

7. SESSION MANAGEMENT STRATEGY
-----------------------------
- URL parameter-based session storage for cross-tab persistence
- Fallback to Streamlit session state for reliability
- Cross-browser tab synchronization
- Activity tracking and monitoring with timestamps
- Clean logout functionality with complete session clearing
- Language preference persistence across sessions

FUNCTIONALITY SCOPE
==================

1. USER AUTHENTICATION & SESSION MANAGEMENT
-------------------------------------------
- Name-based user identification with UUID generation
- Persistent sessions across page refreshes using URL parameters
- Cross-tab session synchronization
- User profile management with language preferences
- Activity tracking and session monitoring
- Subtle cultural greeting personalization
- Complete session cleanup on logout

2. LANGUAGE PREFERENCES & CULTURAL CONTEXT
-----------------------------------------
- Native language selection (21 Indian languages)
- Preferred languages multi-selection
- Language comfort level settings (English Only, Mixed, Native Preferred)
- Subtle cultural greeting personalization
- Language-aware AI responses that feel natural
- Bilingual interface elements where appropriate
- Regional language support with proper transliteration

3. INDIVIDUAL CHAT FUNCTIONALITY
--------------------------------
- OpenAI GPT-3.5-turbo powered conversations
- Language-aware response generation with cultural context
- Follow-up question generation every 3 turns with learning
- Conversation history tracking with turn counting
- Question acceptance/rejection learning for better suggestions
- Real-time conversation analysis for tag inference
- Native language phrase integration when appropriate

4. ENHANCED TAG MANAGEMENT SYSTEM
---------------------------------
- Interactive swipe card interface for tag discovery
- Manual tag addition by users with validation
- AI-powered tag inference from conversations every 5 turns
- Automatic tag addition with user control and feedback
- Tag validation and cleaning with proper formatting
- Tag-based user similarity matching with scoring
- Intelligent tag suggestions with multiple categories:
  * AI-generated dynamic suggestions
  * Category-based suggestions
  * Synonym suggestions
  * Related concept suggestions
- Automatic tag analysis with duplicate handling
- Language-aware tag suggestions with cultural context

5. INTERACTIVE SWIPE INTERFACE
------------------------------
- Card-based tag discovery with emoji icons
- Progress indicators and completion tracking
- Like/Pass/Skip functionality with immediate feedback
- Summary cards showing liked and passed tags
- Diverse tag categories with proper categorization
- Automatic tag addition for liked cards
- Skip all functionality for quick navigation
- Completion celebration with action buttons

6. USER PROFILING & RECOMMENDATIONS
-----------------------------------
- Automatic conversation analysis for interest discovery
- Interest-based user profiling with tag weighting
- Similar user discovery algorithm with similarity scoring
- Tag-based matching with cultural context consideration
- User recommendation system with proper filtering
- Real-time profile updates based on interactions
- Cultural and linguistic context consideration

7. GROUP CHAT SYSTEM
--------------------
- Multi-user group chat creation with topic selection
- AI bot participation in group conversations
- Topic-based chat organization with proper naming
- Persistent group message history with timestamps
- Real-time conversation context for AI responses
- Participant management with proper user identification
- Cultural context awareness in group interactions

8. NAVIGATION & USER INTERFACE
------------------------------
- Multi-view navigation system (chat, profile, similar users, group chats, system status)
- Responsive Streamlit interface with proper styling
- Real-time updates and feedback with immediate response
- Intuitive user experience design with clear navigation
- Persistent view state management across sessions
- Bilingual interface elements where appropriate
- Subtle cultural iconography and greetings

9. SESSION PERSISTENCE
----------------------
- URL parameter-based session storage for maximum persistence
- Fallback session state management for reliability
- Cross-browser tab synchronization
- Activity timestamp tracking with proper formatting
- Graceful session restoration with error handling
- Language preference persistence across sessions

10. SYSTEM STATUS & MONITORING
------------------------------
- Real-time system health monitoring
- Interactive agent node graph with dynamic updates
- Analytics dashboard with usage trends
- Agent performance metrics and status
- Language usage distribution visualization
- System performance metrics tracking
- Real-time metrics with live updates

MULTI-AGENT ARCHITECTURE DETAILS
================================

1. LANGGRAPH AGENT COORDINATION
-------------------------------
- Central coordinator managing all agents
- Dynamic agent registration and discovery
- Request routing based on agent capabilities
- State management across agent interactions
- Error handling and recovery mechanisms
- Agent lifecycle management

2. SPECIALIZED AGENT ROLES
--------------------------

ConversationAgent:
- Handles user conversations and responses
- Manages conversation flow and context
- Generates follow-up questions
- Integrates with other agents for enhanced responses

TagAnalysisAgent:
- Analyzes conversations for interest inference
- Generates dynamic tag suggestions
- Processes conversation history for patterns
- Provides categorized tag recommendations

UserProfileAgent:
- Manages user profiles and preferences
- Handles tag management and updates
- Processes user similarity matching
- Maintains user preference data

GroupChatAgent:
- Handles group chat functionality
- Manages group creation and participation
- Processes group messages and responses
- Coordinates AI participation in groups

SessionAgent:
- Manages user sessions and persistence
- Handles session validation and cleanup
- Tracks user activity and engagement
- Maintains session state across interactions

LanguageAgent:
- Handles language preferences and cultural context
- Provides language-aware responses
- Manages cultural context integration
- Supports multi-language interactions

3. AGENT COMMUNICATION PATTERNS
------------------------------
- Request-response pattern for agent interactions
- State sharing through coordinator
- Asynchronous processing capabilities
- Error propagation and handling
- Performance monitoring and optimization

4. DYNAMIC AGENT DISCOVERY
--------------------------
- Automatic agent registration on startup
- Runtime agent status monitoring
- Dynamic graph updates based on available agents
- Graceful handling of agent failures
- Scalable architecture for new agents

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

Agent Activity Collection:
{
    agent_name: string,
    activity: string,
    timestamp: datetime,
    details: object,
    framework: "LangGraph"
}

2. LANGGRAPH STATE MANAGEMENT
-----------------------------

AgentState TypedDict:
{
    user_id: str,
    user_name: str,
    message: str,
    response: str,
    conversation_history: List[Dict[str, str]],
    language_preferences: Dict[str, Any],
    tags: List[str],
    session_data: Dict[str, Any],
    agent_name: str,
    timestamp: str,
    metadata: Dict[str, Any]
}

3. AGENT WORKFLOW PATTERNS
--------------------------

Standard LangGraph Workflow:
1. process_request_node - Validate and prepare request
2. generate_response_node - Generate response using LLM
3. update_state_node - Update state and metadata
4. END - Complete workflow

4. REQUEST ROUTING MAPPING
--------------------------

Agent Request Mapping:
- send_message → ConversationAgent
- analyze_conversation → TagAnalysisAgent
- get_tag_suggestions → TagAnalysisAgent
- create_profile → UserProfileAgent
- find_similar_users → UserProfileAgent
- create_group_chat → GroupChatAgent
- send_group_message → GroupChatAgent
- get_group_messages → GroupChatAgent
- create_session → SessionAgent
- validate_session → SessionAgent
- get_supported_languages → LanguageAgent
- generate_greeting → LanguageAgent

5. INTERACTIVE GRAPHS & ANALYTICS
---------------------------------

System Status Dashboard:
- Real-time system health metrics
- Interactive agent node graph
- Usage trends and analytics
- Language distribution visualization
- Agent performance metrics
- System performance tracking

Agent Node Graph Features:
- Dynamic agent discovery and positioning
- Real-time status indicators
- Interactive hover information
- Connection visualization with different line styles
- START and END workflow nodes
- Automatic layout adaptation

Analytics Dashboard:
- User growth trends
- Session and conversation metrics
- Language usage distribution
- Agent performance tracking
- System response time monitoring
- Question acceptance rate analysis

6. ENHANCED UI FEATURES
-----------------------

Multi-View Navigation:
- Chat Interface: Main conversation area
- Profile Interface: Tag management and preferences
- Similar Users: User discovery and matching
- Group Chats: Multi-user conversations
- System Status: Monitoring and analytics

Interactive Elements:
- Swipe cards with immediate feedback
- Progress indicators and completion tracking
- Real-time statistics and metrics
- Responsive design for various screen sizes
- Interactive graphs and charts

Visual Design:
- Modern gradient backgrounds
- Emoji icons for visual appeal
- Consistent color scheme
- Clear typography and spacing
- Professional styling throughout

7. ERROR HANDLING & RELIABILITY
-------------------------------

Session Recovery:
- Graceful handling of session loss
- Automatic session restoration
- Fallback mechanisms for data loss
- User-friendly error messages

API Error Handling:
- OpenAI API error management
- Database connection error handling
- Network timeout management
- Graceful degradation of features

Agent Error Handling:
- Individual agent error isolation
- Coordinator-level error recovery
- State consistency maintenance
- Performance monitoring and alerts

Data Validation:
- Tag format validation
- User input sanitization
- Language preference validation
- Session data integrity checks

8. PERFORMANCE OPTIMIZATION
---------------------------

Database Optimization:
- Proper indexing on frequently queried fields
- Efficient query patterns
- Connection pooling
- Caching strategies

UI Performance:
- Lazy loading of components
- Efficient state management
- Optimized re-rendering
- Minimal API calls

Agent Performance:
- Asynchronous processing where possible
- State caching and optimization
- Request batching and optimization
- Memory management and cleanup

Caching Strategy:
- Session data caching
- Tag suggestion caching
- User profile caching
- Conversation history optimization

TECHNICAL SPECIFICATIONS
========================

1. DEPENDENCIES
---------------
- streamlit: Web application framework
- openai: OpenAI API integration
- pymongo: MongoDB database driver
- python-dotenv: Environment variable management
- langchain: LangChain framework
- langgraph: LangGraph for agent coordination
- langchain-openai: OpenAI integration for LangChain
- plotly: Interactive graphs and charts
- pandas: Data manipulation and analysis
- uuid: Unique identifier generation
- datetime: Timestamp handling

2. ENVIRONMENT VARIABLES
------------------------
- OPENAI_API_KEY: OpenAI API key (required)
- MONGODB_ATLAS_URI: MongoDB connection string (optional)

3. FILE STRUCTURE
-----------------
- main_v2.py: Streamlit UI and application entry point (Multi-Agent version)
- multi_agent_chatbot.py: Multi-agent chatbot logic (LangGraph)
- agents/: LangGraph-based agents for different functionalities
  * base_agent.py: Base agent class and coordinator
  * conversation_agent.py: Conversation handling
  * tag_analysis_agent.py: Tag analysis and suggestions
  * user_profile_agent.py: User profile management
  * group_chat_agent.py: Group chat functionality
  * session_agent.py: Session management
  * language_agent.py: Language preferences
- db.py: Database operations and session management
- session_manager.py: Session persistence and management
- prompt_v2.py: System documentation (this file)

4. API INTEGRATIONS
-------------------
- OpenAI GPT-3.5-turbo: Conversation generation and tag analysis
- MongoDB Atlas: Data persistence and user management
- Streamlit: Web interface and session management
- LangGraph: Multi-agent coordination and workflow management

5. SECURITY CONSIDERATIONS
--------------------------
- API key management through environment variables
- User input sanitization and validation
- Session data protection
- Database connection security
- Error message sanitization
- Agent isolation and security

6. SCALABILITY FEATURES
-----------------------
- Modular agent architecture for easy extension
- Database indexing for performance
- Session management for multiple users
- Caching strategies for optimization
- Error handling for reliability
- Dynamic agent discovery and registration

DEPLOYMENT GUIDELINES
====================

1. SETUP INSTRUCTIONS
---------------------
1. Install dependencies: pip install -r requirements.txt
2. Set environment variables in .env file or shell:
   - OPENAI_API_KEY=your_openai_api_key
   - MONGODB_ATLAS_URI=your_mongodb_atlas_uri (optional)
3. Run the app: streamlit run main_v2.py

2. DEVELOPMENT WORKFLOW
-----------------------
- Use mock database for development
- Test with various language preferences
- Validate session persistence across tabs
- Test swipe interface with different tag sets
- Verify group chat functionality
- Check error handling scenarios
- Test agent coordination and routing
- Validate dynamic graph updates

3. PRODUCTION CONSIDERATIONS
---------------------------
- Set up MongoDB Atlas for production
- Configure proper environment variables
- Implement monitoring and logging
- Set up error tracking
- Configure backup strategies
- Implement rate limiting if needed
- Monitor agent performance and health
- Set up alerting for system issues

4. TESTING STRATEGY
-------------------
- Unit tests for individual agents
- Integration tests for agent coordination
- UI tests for user interactions
- Session persistence testing
- Language preference testing
- Tag suggestion accuracy testing
- Agent routing and workflow testing
- Performance and load testing

FUTURE ENHANCEMENTS
===================

1. PLANNED FEATURES
-------------------
- Advanced user matching algorithms
- Real-time notifications
- Voice message support
- Video call integration
- Advanced analytics dashboard
- Mobile app development
- Additional specialized agents
- Enhanced agent coordination

2. SCALABILITY IMPROVEMENTS
---------------------------
- Microservices architecture
- Redis caching layer
- Load balancing
- Database sharding
- CDN integration
- API rate limiting
- Agent clustering and distribution
- Advanced monitoring and alerting

3. USER EXPERIENCE ENHANCEMENTS
------------------------------
- Dark mode support
- Accessibility improvements
- Mobile responsiveness
- Offline functionality
- Push notifications
- Advanced search features
- Enhanced graph visualizations
- Real-time collaboration features

4. AI ENHANCEMENTS
------------------
- Multi-modal AI (text, voice, image)
- Personalized AI assistants
- Advanced conversation analysis
- Predictive user matching
- Sentiment analysis
- Cultural context learning
- Agent learning and adaptation
- Advanced workflow orchestration

5. AGENT SYSTEM ENHANCEMENTS
----------------------------
- Dynamic agent creation and registration
- Agent performance optimization
- Advanced agent communication patterns
- Agent specialization and customization
- Cross-agent learning and knowledge sharing
- Agent health monitoring and recovery
- Scalable agent deployment strategies

This comprehensive system documentation provides a complete overview of the Multi-Agent AI chatbot 
platform, its LangGraph architecture, features, and implementation details. The system is designed 
to connect Indian users and NRIs through shared interests while maintaining subtle cultural awareness 
and providing a modern, engaging user experience with advanced multi-agent capabilities.
"""

def get_system_info():
    """Get basic system information"""
    return {
        "project_name": "Multi-Agent AI Chatbot Platform for Indian Users",
        "version": "2.0.0",
        "description": "Interest-based connection system with LangGraph multi-agent architecture",
        "primary_focus": "Connecting users through shared interests",
        "cultural_approach": "Subtle cultural context as underlying layer",
        "supported_languages": 21,
        "ai_model": "OpenAI GPT-3.5-turbo",
        "database": "MongoDB with mock fallback",
        "framework": "Streamlit + LangGraph",
        "architecture": "Multi-Agent with LangGraph coordination"
    }

def get_architecture_overview():
    """Get architecture overview"""
    return {
        "core_components": [
            "main_v2.py - Streamlit UI with interactive graphs",
            "multi_agent_chatbot.py - Multi-agent chatbot logic",
            "agents/ - LangGraph-based specialized agents",
            "db.py - Database operations",
            "session_manager.py - Session persistence",
            "prompt_v2.py - System documentation"
        ],
        "key_features": [
            "Interactive agent node graph",
            "Multi-agent LangGraph architecture",
            "Dynamic agent discovery",
            "Real-time system monitoring",
            "Analytics dashboard with charts",
            "Persistent session management",
            "Group chat with AI participation",
            "Language preference system",
            "Cultural context awareness"
        ],
        "design_principles": [
            "Multi-agent architecture with LangGraph",
            "Interest-based connection primary",
            "Subtle cultural awareness",
            "Modular agent design",
            "User-centric design",
            "AI-enhanced interactions",
            "Real-time monitoring and analytics"
        ]
    }

def get_agent_system_info():
    """Get multi-agent system information"""
    return {
        "agents": {
            "ConversationAgent": "Handles user conversations and responses",
            "TagAnalysisAgent": "Analyzes conversations for interest tags",
            "UserProfileAgent": "Manages user profiles and preferences",
            "GroupChatAgent": "Handles group chat functionality",
            "SessionAgent": "Manages user sessions and persistence",
            "LanguageAgent": "Handles language preferences and cultural context"
        },
        "framework": "LangGraph v0.0.20",
        "coordination": "Central coordinator with dynamic routing",
        "state_management": "TypedDict-based state sharing",
        "workflow_patterns": "Standard LangGraph workflow with 3 nodes + END",
        "scalability": "Dynamic agent registration and discovery"
    }

def get_database_schema():
    """Get database schema information"""
    return {
        "collections": {
            "users": "User profiles with language preferences",
            "user_tags": "User interest tags with types",
            "conversations": "Chat history with turn tracking",
            "group_chats": "Group chat metadata",
            "group_messages": "Group chat messages",
            "agent_activity": "Agent activity logging"
        },
        "key_fields": {
            "user_id": "UUID for unique identification",
            "native_language": "Primary language preference",
            "preferred_languages": "Multi-language comfort list",
            "language_comfort_level": "English/Mixed/Native preference",
            "tag_type": "Manual or AI-inferred tags",
            "conversation_turns": "Turn counter for analysis",
            "agent_name": "Agent identifier for activity tracking"
        }
    }

def get_session_management_info():
    """Get session management details"""
    return {
        "persistence_strategy": "URL parameters with session state fallback",
        "cross_tab_sync": "URL parameter sharing",
        "session_components": [
            "user_id - Unique identifier",
            "user_name - Display name",
            "authenticated - Session status",
            "last_activity - Timestamp tracking"
        ],
        "cleanup_process": [
            "URL parameter clearing",
            "Session state removal",
            "Database session cleanup",
            "Complete data removal"
        ]
    }

def get_ui_features():
    """Get UI features information"""
    return {
        "navigation_views": [
            "Chat Interface - Main conversation area",
            "Profile Interface - Tag management and preferences",
            "Similar Users - User discovery and matching",
            "Group Chats - Multi-user conversations",
            "System Status - Monitoring and analytics"
        ],
        "interactive_elements": [
            "Swipe cards with immediate feedback",
            "Progress indicators and completion tracking",
            "Real-time statistics and metrics",
            "Interactive graphs and charts",
            "Responsive design for various screen sizes"
        ],
        "visual_design": [
            "Modern gradient backgrounds",
            "Emoji icons for visual appeal",
            "Consistent color scheme",
            "Clear typography and spacing",
            "Professional styling throughout"
        ]
    }

def get_analytics_features():
    """Get analytics and monitoring features"""
    return {
        "system_status": [
            "Real-time system health monitoring",
            "Interactive agent node graph",
            "Usage trends and analytics",
            "Language distribution visualization",
            "Agent performance metrics",
            "System performance tracking"
        ],
        "agent_graph": [
            "Dynamic agent discovery and positioning",
            "Real-time status indicators",
            "Interactive hover information",
            "Connection visualization with different line styles",
            "START and END workflow nodes",
            "Automatic layout adaptation"
        ],
        "analytics_dashboard": [
            "User growth trends",
            "Session and conversation metrics",
            "Language usage distribution",
            "Agent performance tracking",
            "System response time monitoring",
            "Question acceptance rate analysis"
        ]
    } 