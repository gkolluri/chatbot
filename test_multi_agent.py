"""
Test Script for Multi-Agent Chatbot System using LangGraph
========================================================

This script tests the functionality of the LangGraph-based multi-agent
chatbot system to ensure all components work correctly.
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from multi_agent_chatbot import MultiAgentChatbot
from db import DB as DatabaseInterface

def test_system_initialization():
    """Test system initialization."""
    print("🧪 Testing System Initialization...")
    
    try:
        # Initialize database
        db_interface = DatabaseInterface()
        print("✅ Database interface initialized")
        
        # Initialize multi-agent chatbot
        chatbot = MultiAgentChatbot(db_interface)
        print("✅ Multi-agent chatbot initialized")
        
        # Test system status
        system_status = chatbot.get_system_status()
        print(f"✅ System status: {system_status.get('framework', 'Unknown')}")
        
        return chatbot
    except Exception as e:
        print(f"❌ System initialization failed: {str(e)}")
        return None

def test_agent_capabilities(chatbot):
    """Test agent capabilities."""
    print("\n🧪 Testing Agent Capabilities...")
    
    capabilities = chatbot.get_agent_capabilities()
    
    if capabilities.get('success'):
        print(f"✅ Found {capabilities.get('total_agents', 0)} agents")
        
        for agent_name, agent_capabilities in capabilities.get('all_capabilities', {}).items():
            print(f"  • {agent_name}: {len(agent_capabilities)} capabilities")
    else:
        print(f"❌ Failed to get agent capabilities: {capabilities.get('error')}")

def test_conversation_agent(chatbot):
    """Test conversation agent functionality."""
    print("\n🧪 Testing Conversation Agent...")
    
    # Test conversation processing
    result = chatbot.process_conversation(
        user_id="test_user_1",
        user_name="Test User",
        message="Hello! I'm interested in technology and music.",
        language_preferences={'native_language': 'english'}
    )
    
    if result.get('success'):
        print("✅ Conversation processing successful")
        print(f"  • Bot response: {result.get('bot_response', 'No response')[:100]}...")
    else:
        print(f"❌ Conversation processing failed: {result.get('error')}")

def test_tag_analysis_agent(chatbot):
    """Test tag analysis agent functionality."""
    print("\n🧪 Testing Tag Analysis Agent...")
    
    # Test tag suggestions
    result = chatbot.get_tag_suggestions(
        user_id="test_user_1",
        existing_tags=['technology', 'music'],
        language_preferences={'native_language': 'english'}
    )
    
    if result.get('success'):
        print("✅ Tag suggestions successful")
        suggestions = result.get('suggestions', {})
        for category, tags in suggestions.items():
            if tags:
                print(f"  • {category}: {len(tags)} suggestions")
    else:
        print(f"❌ Tag suggestions failed: {result.get('error')}")

def test_user_profile_agent(chatbot):
    """Test user profile agent functionality."""
    print("\n🧪 Testing User Profile Agent...")
    
    # Test profile creation
    result = chatbot.create_user_profile(
        user_id="test_user_1",
        user_name="Test User",
        tags=['technology', 'music', 'travel'],
        language_preferences={'native_language': 'english'},
        conversation_history=[
            {'role': 'user', 'content': 'I love technology and music'},
            {'role': 'assistant', 'content': 'That sounds great! What kind of music do you enjoy?'}
        ]
    )
    
    if result.get('success'):
        print("✅ Profile creation successful")
        profile = result.get('profile', {})
        print(f"  • Profile completeness: {profile.get('profile_completeness', 0):.2f}")
    else:
        print(f"❌ Profile creation failed: {result.get('error')}")

def test_group_chat_agent(chatbot):
    """Test group chat agent functionality."""
    print("\n🧪 Testing Group Chat Agent...")
    
    # Test group creation
    group_result = chatbot.create_group_chat(
        topic_name="Technology Discussion",
        user_id="test_user_1",
        user_name="Test User",
        language_preferences={'native_language': 'english'}
    )
    
    if group_result.get('success'):
        print("✅ Group creation successful")
        group_id = group_result['group_data']['group_id']
        
        # Test sending a message
        message_result = chatbot.send_group_message(
            group_id=group_id,
            user_id="test_user_1",
            user_name="Test User",
            message="Hello everyone! I'm excited to discuss technology.",
            language_preferences={'native_language': 'english'}
        )
        
        if message_result.get('success'):
            print("✅ Group message successful")
            ai_response = message_result.get('ai_response', '')
            if ai_response:
                print(f"  • AI response: {ai_response[:100]}...")
        else:
            print(f"❌ Group message failed: {message_result.get('error')}")
    else:
        print(f"❌ Group creation failed: {group_result.get('error')}")

def test_session_agent(chatbot):
    """Test session agent functionality."""
    print("\n🧪 Testing Session Agent...")
    
    # Test session creation
    session_result = chatbot.create_user_session(
        user_name="Test User",
        language_preferences={'native_language': 'english', 'comfort_level': 'english_only'}
    )
    
    if session_result.get('success'):
        print("✅ Session creation successful")
        session_data = session_result['session_data']
        print(f"  • User ID: {session_data.get('user_id', 'Unknown')}")
        print(f"  • Session ID: {session_data.get('session_id', 'Unknown')}")
        
        # Test session validation
        user_id = session_data['user_id']
        validation_result = chatbot.validate_session(user_id=user_id)
        
        if validation_result.get('success') and validation_result.get('is_valid'):
            print("✅ Session validation successful")
        else:
            print(f"❌ Session validation failed: {validation_result.get('error')}")
    else:
        print(f"❌ Session creation failed: {session_result.get('error')}")

def test_language_agent(chatbot):
    """Test language agent functionality."""
    print("\n🧪 Testing Language Agent...")
    
    # Test supported languages
    languages_result = chatbot.get_supported_languages()
    
    if languages_result.get('success'):
        print("✅ Language support successful")
        total_languages = languages_result.get('total_languages', 0)
        print(f"  • Supported languages: {total_languages}")
        
        # Test greeting generation
        greeting_result = chatbot.generate_personalized_greeting(
            user_name="Test User",
            language_preferences={'native_language': 'hindi', 'comfort_level': 'mixed_language'},
            greeting_type='cultural'
        )
        
        if greeting_result.get('success'):
            print("✅ Greeting generation successful")
            greeting = greeting_result.get('greeting', '')
            if greeting:
                print(f"  • Greeting: {greeting[:100]}...")
        else:
            print(f"❌ Greeting generation failed: {greeting_result.get('error')}")
    else:
        print(f"❌ Language support failed: {languages_result.get('error')}")

def test_agent_functionality(chatbot):
    """Test individual agent functionality."""
    print("\n🧪 Testing Individual Agent Functionality...")
    
    agents_to_test = [
        'ConversationAgent',
        'TagAnalysisAgent',
        'UserProfileAgent',
        'GroupChatAgent',
        'SessionAgent',
        'LanguageAgent'
    ]
    
    for agent_name in agents_to_test:
        print(f"\n  Testing {agent_name}...")
        result = chatbot.test_agent_functionality(agent_name)
        
        if result.get('success'):
            print(f"    ✅ {agent_name}: PASSED")
        else:
            print(f"    ❌ {agent_name}: FAILED - {result.get('error', 'Unknown error')}")

def test_framework_info(chatbot):
    """Test framework information."""
    print("\n🧪 Testing Framework Information...")
    
    framework_info = chatbot.get_framework_info()
    
    print(f"✅ Framework: {framework_info.get('framework')}")
    print(f"✅ Version: {framework_info.get('version')}")
    print(f"✅ Architecture: {framework_info.get('architecture')}")
    
    print("✅ Features:")
    for feature in framework_info.get('features', []):
        print(f"  • {feature}")
    
    print("✅ Benefits:")
    for benefit in framework_info.get('benefits', []):
        print(f"  • {benefit}")

def main():
    """Main test function."""
    print("🚀 Starting Multi-Agent Chatbot System Tests (LangGraph)")
    print("=" * 60)
    
    # Test system initialization
    chatbot = test_system_initialization()
    if chatbot is None:
        print("❌ Cannot proceed with tests due to initialization failure")
        return
    
    # Test agent capabilities
    test_agent_capabilities(chatbot)
    
    # Test individual agents
    test_conversation_agent(chatbot)
    test_tag_analysis_agent(chatbot)
    test_user_profile_agent(chatbot)
    test_group_chat_agent(chatbot)
    test_session_agent(chatbot)
    test_language_agent(chatbot)
    
    # Test agent functionality
    test_agent_functionality(chatbot)
    
    # Test framework information
    test_framework_info(chatbot)
    
    # Cleanup
    print("\n🧹 Cleaning up system...")
    try:
        chatbot.cleanup_system()
        print("✅ System cleanup completed")
    except Exception as e:
        print(f"❌ Cleanup failed: {str(e)}")
    
    print("\n🎉 All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main() 