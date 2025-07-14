# 🤖 React AI Pattern Multi-Agent Chatbot with Group Chat & Citations

A sophisticated conversational AI chatbot built with Streamlit, OpenAI GPT-4, React AI patterns, and MongoDB. Features intelligent multi-agent conversations, group chat functionality with AI participation, comprehensive citation system, and advanced user preference-based recommendations.

## ✨ Features

### 🧠 React AI Pattern Architecture
- **Multi-Agent System**: Specialized agents for different conversation types
- **React AI Pattern**: Observe → Think → Act → Reflect conversation flow
- **LangGraph Integration**: Advanced conversation state management
- **Intelligent Agent Selection**: Automatic agent routing based on user needs

### 💬 Group Chat System
- **Multi-User Group Chats**: Create topic-based group conversations
- **AI Participation**: Intelligent AI bot participation in group discussions
- **Topic-Focused Conversations**: Specialized group topics (food, music, technology, etc.)
- **RAG-Enhanced Responses**: Context-aware AI responses using user profiles and preferences

### 📚 Advanced Citation System
- **ChatGPT-Style Citations**: Subtle citation links [1] [2] [3] in AI responses
- **Multiple Citation Types**: User profiles, conversation context, shared interests, location data
- **Contextual Relevance**: Smart citation generation based on response content
- **Expandable Citation Details**: Detailed citation information with sources and metadata

### 🎯 User Preference Prioritization
- **Generic Preference System**: Works across all topics (food, music, technology, sports, etc.)
- **Contextual Recommendations**: AI prioritizes user-specific preferences in recommendations
- **Smart Tag Matching**: Intelligent topic-tag relationship detection
- **Relevance Scoring**: Advanced scoring system for preference matching

### 🗄️ Advanced Data Management
- **MongoDB Integration**: Persistent storage with vector indexing
- **User Profiling**: Comprehensive user profiles with tags and preferences
- **Location-Based Features**: Geospatial queries and location-aware recommendations
- **Session Management**: Persistent sessions across browser tabs

### 🌐 Multi-Language Support
- **21 Indian Languages**: Native language support for Indian users
- **Cultural Context**: Subtle Indian cultural awareness in conversations
- **Language Preferences**: Configurable language comfort levels

## 🏗️ Architecture

### Core Components

- **`main_react.py`**: Main Streamlit application with React AI interface
- **`react_multi_agent_chatbot.py`**: Multi-agent chatbot system with React patterns
- **`group_chat.py`**: Group chat functionality with AI participation
- **`citation_system.py`**: Comprehensive citation generation and management
- **`db.py`**: MongoDB database layer with vector storage
- **`logging_utils.py`**: Advanced logging and monitoring system
- **`agents/`**: Specialized agent implementations
  - `react_base_agent.py`: Base React AI agent
  - `group_chat_agent.py`: Group chat specialized agent
  - `user_profile_agent.py`: User profiling agent
  - `react_rag_nearby_agent.py`: Location-based recommendations agent

### Data Flow

1. **User Input**: Message received via Streamlit interface
2. **Agent Selection**: React AI system selects appropriate agent
3. **Context Building**: RAG system builds context from user profile, location, preferences
4. **AI Processing**: GPT-4 generates response with React AI pattern
5. **Citation Generation**: Comprehensive citations generated for response
6. **Response Delivery**: Response with citations displayed in UI
7. **Storage**: Conversation and citations stored in MongoDB

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- MongoDB Atlas account (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   MONGODB_ATLAS_URI=your_mongodb_atlas_connection_string
   ```

4. **Run the application**
   ```bash
   streamlit run main_react.py
   ```

5. **Open your browser**
   
   Navigate to `http://localhost:8501` to access the chatbot interface.

## 📋 Requirements

### Core Dependencies

- **streamlit**: Web application framework
- **openai**: OpenAI GPT-4 API client
- **langchain**: LangChain ecosystem for AI applications
- **langgraph**: Advanced conversation flow control
- **pymongo**: MongoDB client for data persistence
- **plotly**: Interactive data visualization
- **folium**: Interactive maps for location features
- **geopy**: Geocoding and location services

### Optional Dependencies

- **mongomock**: In-memory MongoDB mock for development

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key | None |
| `MONGODB_ATLAS_URI` | No | MongoDB Atlas connection string | Mock DB |

### Agent Configuration

The system includes several specialized agents:

1. **React Base Agent**: Core React AI pattern implementation
2. **Group Chat Agent**: Handles multi-user group conversations
3. **User Profile Agent**: Manages user profiling and preferences
4. **RAG Nearby Agent**: Location-based recommendations

## 💬 Usage

### Individual Conversations

1. Select conversation type from the sidebar
2. Type your message in the input field
3. AI responds using React AI pattern with citations
4. Citations are displayed as clickable links [1] [2] [3]
5. Expand citation details for source information

### Group Chat

1. Navigate to "Group Chats" tab
2. Create a new group or join existing one
3. Select topic (Food, Music, Technology, etc.)
4. AI participates with context-aware responses
5. User preferences are prioritized in recommendations

### User Preferences

1. Add tags to your profile (food preferences, music genres, etc.)
2. Set location preferences for local recommendations
3. Configure language preferences
4. AI automatically prioritizes your preferences in recommendations

## 🏛️ Project Structure

```
chatbot/
├── main_react.py                    # Main Streamlit application
├── react_multi_agent_chatbot.py     # Multi-agent system
├── group_chat.py                    # Group chat functionality
├── citation_system.py               # Citation generation
├── db.py                           # Database layer
├── logging_utils.py                # Logging system
├── session_manager.py              # Session management
├── tag_analyzer.py                 # Tag analysis
├── chatbot.py                      # Legacy chatbot (for compatibility)
├── agents/                         # Agent implementations
│   ├── react_base_agent.py         # Base React AI agent
│   ├── group_chat_agent.py         # Group chat agent
│   ├── user_profile_agent.py       # User profiling agent
│   └── react_rag_nearby_agent.py   # Location-based agent
├── logs/                           # Application logs
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── LICENSE                         # MIT License
```

## 🎯 Key Features in Detail

### React AI Pattern Implementation

The system implements the React AI pattern:

1. **Observe**: Analyze user input and context
2. **Think**: Reason about the appropriate response strategy
3. **Act**: Execute the chosen action (generate response, query database, etc.)
4. **Reflect**: Evaluate the response and improve future interactions

### Citation System

- **Automatic Generation**: Citations generated based on response content
- **Multiple Sources**: User profiles, conversation context, location data, shared interests
- **Relevance Scoring**: Smart relevance calculation for citation ranking
- **UI Integration**: Subtle citation display similar to ChatGPT

### User Preference Prioritization

- **Generic System**: Works across all domains (food, music, technology, etc.)
- **Contextual Matching**: Intelligent preference-to-topic relationship detection
- **Smart Recommendations**: AI prioritizes user-specific preferences
- **Example**: User with "south indian food" preference gets South Indian restaurants first

### Group Chat Features

- **Topic-Based Groups**: Specialized groups for different interests
- **AI Participation**: Intelligent AI responses in group context
- **Preference-Aware**: AI considers all participants' preferences
- **Citation Support**: Full citation system in group conversations

## 🔍 Advanced Features

### Location-Based Recommendations

- **Geospatial Queries**: MongoDB 2dsphere indexing for location data
- **Nearby Users**: Find users with similar interests nearby
- **Local Recommendations**: Location-aware suggestions (restaurants, events, etc.)
- **Privacy Controls**: Configurable location privacy settings

### Multi-Language Support

- **21 Indian Languages**: Full support for major Indian languages
- **Cultural Context**: Subtle cultural awareness in responses
- **Language Preferences**: Configurable comfort levels
- **Bilingual Interface**: Mixed language support

### Advanced Analytics

- **Conversation Analytics**: Detailed conversation flow analysis
- **User Behavior**: Preference learning and adaptation
- **Performance Metrics**: Response times, citation accuracy, user satisfaction
- **Agent Performance**: Multi-agent system monitoring

## 🧪 Development

### Running in Development Mode

```bash
# Use mock database (no MongoDB required)
streamlit run main_react.py

# With MongoDB Atlas
export MONGODB_ATLAS_URI="your_connection_string"
streamlit run main_react.py
```

### Agent Development

To create a new agent:

1. Extend `ReactBaseAgent` class
2. Implement required methods (`_get_agent_system_prompt`, `_execute_action`)
3. Add agent to the multi-agent system
4. Test with various conversation scenarios

### Citation Development

To add new citation types:

1. Extend `CitationGenerator` class
2. Add new citation type to `_generate_citations_for_response`
3. Update UI to handle new citation display
4. Test citation relevance and accuracy

## 🔒 Security & Privacy

- **API Key Protection**: Secure environment variable management
- **Data Encryption**: MongoDB connection encryption
- **Location Privacy**: Configurable location sharing settings
- **User Data**: Secure user profile and preference storage
- **Session Security**: Secure session management

## 📈 Performance

- **Response Time**: Optimized for sub-2 second responses
- **Scalability**: Designed for multiple concurrent users
- **Caching**: Intelligent caching for frequently accessed data
- **Database Optimization**: Indexed queries for fast retrieval

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new features
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
- Streamlit for the web framework
- LangChain for AI application framework
- MongoDB for data persistence
- React AI pattern inspiration