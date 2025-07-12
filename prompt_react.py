"""
SUPER PROMPT REACT: COMPREHENSIVE MULTI-AGENT SYSTEM DOCUMENTATION
==================================================================

PROJECT: Multi-Agent AI Chatbot Platform for Indian Users - React AI Pattern Architecture
========================================================================================

OVERVIEW
--------
This is a sophisticated Streamlit-based multi-agent AI chatbot application using the React AI pattern
(Observe-Think-Act loops) to connect Indian users and NRIs based on shared interests. The platform combines
OpenAI's GPT models, MongoDB for data persistence, and a coordinated multi-agent system with specialized React AI
agents for different functionalities. The system supports individual conversations, group chats with AI participation,
intelligent user matching based on conversation analysis, and language preference management. The primary focus is
helping users find common ground and shared interests, with Indian cultural context as an underlying layer that enhances
the connection experience.

ARCHITECTURE OVERVIEW
=====================

Core Components:
1. main_react.py - Streamlit UI with interactive graphs and multi-view navigation (React AI version)
2. react_multi_agent_chatbot.py - Multi-agent chatbot logic with React AI pattern coordination
3. agents/ - React AI pattern-based specialized agents for different functionalities
4. db.py - MongoDB database layer with language preferences and user session management
5. session_manager.py - Persistent session management with URL parameter storage
6. prompt_react.py - This comprehensive system documentation (React AI version)

Multi-Agent System:
1. ConversationAgent - Handles user conversations and responses
2. TagAnalysisAgent - Analyzes conversations for interest tags
3. UserProfileAgent - Manages user profiles and preferences
4. GroupChatAgent - Handles group chat functionality
5. SessionAgent - Manages user sessions and persistence
6. LanguageAgent - Handles language preferences and cultural context

DESIGN GUIDELINES
=================

1. REACT AI PATTERN MULTI-AGENT ARCHITECTURE
--------------------------------------------
- Observe-Think-Act loop for each agent
- Dynamic tool calling and reasoning
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
===================

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
------------------------------------------
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

REACT AI PATTERN ARCHITECTURE DETAILS
=====================================

1. REACT AI AGENT COORDINATION
------------------------------
- Central coordinator managing all agents
- Dynamic agent registration and discovery
- Request routing based on agent capabilities
- State management across agent interactions
- Error handling and recovery mechanisms
- Agent lifecycle management
- Observe-Think-Act loop for each agent:
  * Observe: Analyze current state and context
  * Think: Reason about what action to take
  * Act: Execute the chosen action
  * Reflect: Learn from the outcome

2. SPECIALIZED AGENT ROLES
--------------------------

ConversationAgent:
- Handles user conversations and responses
- Manages conversation flow and context
- Generates follow-up questions
- Integrates with other agents for enhanced responses
- Uses Observe-Think-Act for dynamic reasoning

TagAnalysisAgent:
- Analyzes conversations for interest inference
- Generates dynamic tag suggestions
- Processes conversation history for patterns
- Provides categorized tag recommendations
- Uses Observe-Think-Act for tag reasoning

UserProfileAgent:
- Manages user profiles and preferences
- Handles tag management and updates
- Processes user similarity matching
- Maintains user preference data
- Uses Observe-Think-Act for profile adaptation

GroupChatAgent:
- Handles group chat functionality
- Manages group creation and participation
- Processes group messages and responses
- Coordinates AI participation in groups
- Uses Observe-Think-Act for group coordination

SessionAgent:
- Manages user sessions and persistence
- Handles session validation and cleanup
- Tracks user activity and engagement
- Maintains session state across interactions
- Uses Observe-Think-Act for session management

LanguageAgent:
- Handles language preferences and cultural context
- Provides language-aware responses
- Manages cultural context integration
- Supports multi-language interactions
- Uses Observe-Think-Act for language adaptation

3. AGENT COMMUNICATION PATTERNS
-------------------------------
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
======================

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
    framework: "React AI Pattern"
}

2. REACT AI STATE MANAGEMENT
----------------------------

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

Standard React AI Workflow:
1. Observe: Gather current state and context
2. Think: Reason about the best action
3. Act: Execute the action (tool call, response, update)
4. Reflect: Learn from the outcome and update state
5. END: Complete workflow

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
- Interactive graphs and charts
- Responsive design for various screen sizes

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
- langchain: LangChain framework (optional for tool integration)
- plotly: Interactive graphs and charts
- pandas: Data manipulation and analysis
- uuid: Unique identifier generation
- datetime: Timestamp handling
- numpy: Graph layout calculations

2. ENVIRONMENT VARIABLES
------------------------
- OPENAI_API_KEY: OpenAI API key (required)
- MONGODB_ATLAS_URI: MongoDB connection string (optional)

3. FILE STRUCTURE
-----------------
- main_react.py: Streamlit UI and application entry point (React AI version)
- react_multi_agent_chatbot.py: Multi-agent chatbot logic (React AI pattern)
- agents/: React AI pattern-based agents for different functionalities
  * react_base_agent.py: Base agent class and coordinator
  * react_conversation_agent.py: Conversation handling
  * react_tag_analysis_agent.py: Tag analysis and suggestions
  * react_user_profile_agent.py: User profile management
  * react_group_chat_agent.py: Group chat functionality
  * react_session_agent.py: Session management
  * react_language_agent.py: Language preferences
- db.py: Database operations and session management
- session_manager.py: Session persistence and management
- prompt_react.py: System documentation (this file)

4. API INTEGRATIONS
-------------------
- OpenAI GPT-3.5-turbo: Conversation generation and tag analysis
- MongoDB Atlas: Data persistence and user management
- Streamlit: Web interface and session management
- Plotly: Analytics and graph visualization

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
=====================

1. SETUP INSTRUCTIONS
---------------------
1. Install dependencies: pip install -r requirements.txt
2. Set environment variables in .env file or shell:
   - OPENAI_API_KEY=your_openai_api_key
   - MONGODB_ATLAS_URI=your_mongodb_atlas_uri (optional)
3. Run the app: streamlit run main_react.py

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
----------------------------
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
platform, its React AI pattern architecture, features, and implementation details. The system is designed
to connect Indian users and NRIs through shared interests while maintaining subtle cultural awareness
and providing a modern, engaging user experience with advanced multi-agent capabilities.
""" 