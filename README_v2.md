# AI Chatbot Platform v2 - LangGraph Multi-Agent System

## Overview

This is a **multi-agent chatbot platform** built with **LangGraph framework**, designed to connect Indian users and NRIs based on shared interests with subtle cultural context. The platform uses a sophisticated multi-agent architecture with enhanced state management and coordination.

## 🚀 Key Features

### Multi-Agent Architecture
- **6 Specialized Agents**: Each handling specific functionality
- **LangGraph Framework**: Advanced state management and workflow coordination
- **Enhanced Scalability**: Better performance and maintainability
- **Improved Error Handling**: Robust error recovery and logging
- **Advanced Workflows**: Stateful conversation management

### Core Functionality
- **Individual Chat Conversations**: AI-powered conversations with cultural awareness
- **Tag-Based Interest System**: AI-driven tag inference and suggestions
- **User Profiling**: Comprehensive user profiles with interest analysis
- **Group Chat**: Multi-user conversations with AI participation
- **Session Management**: Persistent sessions across browser tabs
- **Language Support**: 21 Indian languages with cultural context

## 🤖 Agent Architecture

### 1. ConversationAgent
- Handles individual chat conversations
- Language-aware response generation
- Follow-up question generation
- Conversation history tracking
- Cultural context integration

### 2. TagAnalysisAgent
- AI-powered tag inference from conversations
- Dynamic tag suggestions (AI, category, synonym, related)
- Automatic tag validation and cleaning
- Language-aware tag suggestions
- Duplicate handling and management

### 3. UserProfileAgent
- Automatic conversation analysis for interest discovery
- Interest-based user profiling with tag weighting
- Similar user discovery algorithm
- Tag-based matching with cultural context
- Real-time profile updates

### 4. GroupChatAgent
- Multi-user group chat creation
- AI bot participation in conversations
- Topic-based chat organization
- Persistent group message history
- Real-time conversation context

### 5. SessionAgent
- URL parameter-based session storage
- Cross-browser tab synchronization
- Activity tracking and monitoring
- Clean logout functionality
- Language preference persistence

### 6. LanguageAgent
- Native language selection (21 Indian languages)
- Preferred languages multi-selection
- Language comfort level settings
- Cultural greeting personalization
- Bilingual interface elements

## 🛠️ Technology Stack

### Framework
- **LangGraph v0.0.20**: Advanced multi-agent coordination
- **Streamlit**: Web interface and UI components
- **OpenAI GPT-3.5-turbo**: AI conversation and analysis
- **MongoDB**: Data persistence and storage

### Key Dependencies
```
streamlit>=1.28.0
openai>=1.0.0
python-dotenv>=1.0.0
pymongo>=4.5.0
mongomock>=4.1.0
langchain>=0.1.0
langgraph>=0.0.20
langchain-openai>=0.1.0
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key
- MongoDB (optional, uses mock for development)

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
   Create a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   MONGODB_URI=your_mongodb_uri_here
   ```

4. **Run the application**
   ```bash
   streamlit run main_v2.py
   ```

## 🧪 Testing

### Run System Tests
```bash
python test_multi_agent.py
```

### Test Individual Components
```bash
# Test conversation functionality
python -c "from multi_agent_chatbot import MultiAgentChatbot; chatbot = MultiAgentChatbot(); print(chatbot.process_conversation('test_user', 'Test User', 'Hello!'))"

# Test tag analysis
python -c "from multi_agent_chatbot import MultiAgentChatbot; chatbot = MultiAgentChatbot(); print(chatbot.get_tag_suggestions('test_user', ['technology']))"
```

## 📊 System Architecture

### LangGraph Benefits
- **Stateful Workflows**: Better conversation state management
- **Agent Coordination**: Improved inter-agent communication
- **Message Routing**: Intelligent request routing to appropriate agents
- **Error Handling**: Robust error recovery and logging
- **Activity Tracking**: Comprehensive system monitoring

### Agent Communication Flow
```
User Request → LangGraph Coordinator → Appropriate Agent → Response
```

### State Management
- **Conversation State**: Track conversation context and history
- **User State**: Maintain user preferences and session data
- **System State**: Monitor agent status and system health
- **Workflow State**: Manage complex multi-step processes

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key
- `MONGODB_URI`: MongoDB connection string (optional)
- `LANGCHAIN_TRACING_V2`: Enable LangChain tracing (optional)
- `LANGCHAIN_ENDPOINT`: LangChain endpoint (optional)

### Agent Configuration
Each agent can be configured independently:
- **ConversationAgent**: Model parameters, response length, cultural context
- **TagAnalysisAgent**: Tag categories, suggestion algorithms, validation rules
- **UserProfileAgent**: Profile completeness thresholds, similarity algorithms
- **GroupChatAgent**: Group size limits, AI participation rules
- **SessionAgent**: Session timeout, storage methods
- **LanguageAgent**: Supported languages, cultural context levels

## 📈 Performance Features

### Scalability
- **Modular Design**: Easy to add new agents or modify existing ones
- **State Management**: Efficient conversation and user state handling
- **Caching**: Intelligent caching for frequently accessed data
- **Async Support**: Non-blocking operations for better performance

### Monitoring
- **Agent Status**: Real-time monitoring of all agents
- **System Health**: Comprehensive system status reporting
- **Error Tracking**: Detailed error logging and recovery
- **Performance Metrics**: Response times and throughput monitoring

## 🌟 Advanced Features

### Cultural Context
- **Language Awareness**: Support for 21 Indian languages
- **Cultural Greetings**: Personalized greetings based on language preferences
- **Regional Context**: Consideration of regional cultural nuances
- **Bilingual Support**: Mixed language conversations

### AI Enhancement
- **Follow-up Questions**: Intelligent question generation every 3 turns
- **Interest Discovery**: Automatic tag inference from conversations
- **Similar User Matching**: Advanced algorithms for user recommendations
- **Group Facilitation**: AI participation in group conversations

### User Experience
- **Swipe Interface**: Interactive tag selection with emoji icons
- **Progress Indicators**: Visual feedback for system operations
- **Cross-tab Sync**: Session persistence across browser tabs
- **Responsive Design**: Mobile-friendly interface

## 🔒 Security & Privacy

### Data Protection
- **Session Security**: Secure session management and validation
- **User Privacy**: Respect for user preferences and data
- **API Security**: Secure handling of OpenAI API calls
- **Database Security**: Safe data storage and retrieval

### Error Handling
- **Graceful Degradation**: System continues working even if some agents fail
- **Error Recovery**: Automatic recovery from common errors
- **User Feedback**: Clear error messages and suggestions
- **Logging**: Comprehensive error logging for debugging

## 🚀 Deployment

### Local Development
```bash
streamlit run main_v2.py
```

### Production Deployment
1. Set up MongoDB database
2. Configure environment variables
3. Deploy to your preferred platform (Heroku, AWS, etc.)
4. Set up monitoring and logging

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "main_v2.py"]
```

## 📝 API Reference

### MultiAgentChatbot Class

#### Core Methods
- `process_conversation(user_id, user_name, message, language_preferences)`
- `get_tag_suggestions(user_id, existing_tags, language_preferences)`
- `create_user_profile(user_id, user_name, tags, language_preferences, conversation_history)`
- `find_similar_users(user_id, min_similarity, max_results)`
- `create_group_chat(topic_name, user_id, user_name, language_preferences)`
- `send_group_message(group_id, user_id, user_name, message, language_preferences)`

#### System Methods
- `get_system_status()`: Get comprehensive system status
- `get_agent_capabilities(agent_name)`: Get agent capabilities
- `test_agent_functionality(agent_name)`: Test specific agent
- `get_framework_info()`: Get LangGraph framework information
- `cleanup_system()`: Cleanup system resources

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Testing Guidelines
- Run `python test_multi_agent.py` before submitting
- Ensure all agents are working correctly
- Test with different language preferences
- Verify cultural context integration

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangGraph Team**: For the excellent multi-agent framework
- **OpenAI**: For providing the GPT-3.5-turbo API
- **Streamlit**: For the amazing web framework
- **MongoDB**: For the database solution

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review the test files for examples

---

**Built with ❤️ using LangGraph for enhanced multi-agent coordination** 