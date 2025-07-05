# 🤖 Streamlit Chatbot with OpenAI, LangGraph, and MongoDB

A sophisticated conversational AI chatbot built with Streamlit, OpenAI GPT-3.5, LangGraph for conversation management, and MongoDB for persistent storage. The chatbot features intelligent follow-up questions and learns from user rejections to improve conversation quality.

## ✨ Features

- **Interactive Web Interface**: Clean, responsive Streamlit-based chat interface
- **OpenAI Integration**: Powered by GPT-3.5-turbo for natural language processing
- **Smart Follow-up Questions**: Automatically generates relevant yes/no questions every 3 conversation turns
- **Rejection Learning**: Remembers and avoids previously rejected questions
- **Persistent Storage**: MongoDB integration for storing conversation data and rejected questions
- **Mock Database Support**: Fallback to in-memory storage when MongoDB is unavailable
- **Conversation Tracking**: Real-time metrics showing conversation turns and follow-up timing
- **Session Management**: Maintains conversation state across browser sessions

## 🏗️ Architecture

### Core Components

- **`main.py`**: Streamlit web application entry point and UI
- **`chatbot.py`**: Core chatbot logic with OpenAI integration and conversation management
- **`db.py`**: Database abstraction layer supporting both MongoDB and mock storage

### Data Flow

1. User sends message via Streamlit interface
2. Chatbot processes message through OpenAI API
3. Conversation state is updated and stored
4. Follow-up questions are generated based on conversation context
5. Rejected questions are persisted to avoid repetition

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- MongoDB Atlas account (optional)

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
   
   Or set them in your shell:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   export MONGODB_ATLAS_URI="your_mongodb_atlas_connection_string"
   ```

4. **Run the application**
   ```bash
   streamlit run main.py
   ```

5. **Open your browser**
   
   Navigate to `http://localhost:8501` to access the chatbot interface.

## 📋 Requirements

### Required Dependencies

- **streamlit**: Web application framework
- **openai**: OpenAI API client
- **langchain**: LangGraph conversation management
- **langgraph**: Advanced conversation flow control
- **pymongo**: MongoDB client
- **python-dotenv**: Environment variable management

### Optional Dependencies

- **mongomock**: In-memory MongoDB mock for development/testing

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key | None |
| `MONGODB_ATLAS_URI` | No | MongoDB Atlas connection string | Mock DB |

### Database Configuration

The application supports two database modes:

1. **MongoDB Atlas** (Production): Set `MONGODB_ATLAS_URI` environment variable
2. **Mock Database** (Development): Automatically used when no MongoDB URI is provided

## 💬 Usage

### Basic Conversation

1. Type your message in the text input field
2. Click "Send" or press Enter
3. The chatbot will respond using OpenAI's GPT-3.5-turbo
4. Continue the conversation naturally

### Follow-up Questions

- Every 3 conversation turns, the chatbot will ask a relevant follow-up question
- Click "Yes" to accept the question and continue
- Click "No" to reject the question (it won't be asked again)
- The question will be stored in the database to avoid repetition

### Interface Features

- **Conversation History**: All messages are displayed in chronological order
- **Turn Counter**: Track conversation progress in the sidebar
- **Follow-up Timer**: See when the next follow-up question will appear
- **Debug Mode**: Expand "Show rejected questions" to view stored rejections

## 🏛️ Project Structure

```
chatbot/
├── main.py              # Streamlit app entry point and UI
├── chatbot.py           # Core chatbot logic and OpenAI integration
├── db.py               # Database abstraction layer
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
├── LICENSE            # Project license
└── .gitignore         # Git ignore rules
```

## 🔍 Code Overview

### Main Application (`main.py`)

- Streamlit web interface setup
- Session state management
- UI components (chat display, input fields, buttons)
- Conversation metrics display

### Chatbot Logic (`chatbot.py`)

- OpenAI API integration
- Conversation state management
- Follow-up question generation
- Rejection tracking and learning

### Database Layer (`db.py`)

- MongoDB connection management
- Mock database fallback
- Rejected questions persistence
- Database abstraction interface

## 🧪 Development

### Running in Development Mode

```bash
# Use mock database (no MongoDB required)
streamlit run main.py

# With MongoDB Atlas
export MONGODB_ATLAS_URI="your_connection_string"
streamlit run main.py
```

### Testing

The application includes built-in debugging features:

- Expand "Show rejected questions" to view stored rejections
- Monitor conversation turns in the sidebar
- Check follow-up question timing

### Customization

To customize the chatbot behavior:

1. **Follow-up Frequency**: Modify the `should_ask_followup()` method in `chatbot.py`
2. **Question Generation**: Update the prompt in `get_followup_question()`
3. **Rejection Detection**: Extend the `is_rejection()` and `is_yes()` methods
4. **UI Styling**: Modify the Streamlit components in `main.py`

## 🔒 Security Considerations

- **API Key Protection**: Never commit your OpenAI API key to version control
- **Environment Variables**: Use `.env` files or secure environment variable management
- **Database Security**: Use MongoDB Atlas with proper authentication and network access controls
- **Input Validation**: Consider adding input sanitization for production use

## 🚨 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Ensure `OPENAI_API_KEY` is set correctly
   - Verify the API key has sufficient credits

2. **MongoDB Connection Issues**
   - Check your MongoDB Atlas connection string
   - Verify network access and authentication
   - The app will fall back to mock database if connection fails

3. **Streamlit Port Issues**
   - If port 8501 is busy, Streamlit will automatically use the next available port
   - Check the terminal output for the correct URL

4. **Dependency Installation Issues**
   - Ensure you're using Python 3.8+
   - Try upgrading pip: `pip install --upgrade pip`
   - Install dependencies individually if needed

### Debug Mode

Enable debug information by expanding the "Show rejected questions" section in the app interface.

## 📈 Performance

- **Response Time**: Depends on OpenAI API latency (typically 1-3 seconds)
- **Memory Usage**: Minimal for mock database, varies with MongoDB data size
- **Scalability**: Can handle multiple concurrent users with proper MongoDB setup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing the GPT API
- Streamlit for the web framework
- LangChain/LangGraph for conversation management
- MongoDB for database support

## 📞 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the code comments for implementation details
3. Open an issue on the project repository

---

**Happy Chatting! 🤖💬**