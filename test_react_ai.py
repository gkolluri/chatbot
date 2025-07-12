"""
Test Script for React AI Pattern Multi-Agent Chatbot System
=========================================================

This script tests the React AI pattern implementation with Observe-Think-Act loops,
dynamic tool calling, and reasoning capabilities.
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from react_multi_agent_chatbot import ReactMultiAgentChatbot


def test_react_ai_system():
    """Test the React AI pattern system functionality."""
    print("🤖 Testing React AI Pattern Multi-Agent Chatbot System")
    print("=" * 60)
    
    # Initialize the React AI chatbot
    chatbot = ReactMultiAgentChatbot()
    
    # Test 1: System Initialization
    print("\n1. Testing System Initialization")
    print("-" * 30)
    
    status = chatbot.check_status()
    if status.get('success'):
        print("✅ System initialized successfully")
        print(f"   - Database connected: {status.get('database_connected', False)}")
        print(f"   - Agents active: {status.get('agents_active', False)}")
        print(f"   - Coordinator ready: {status.get('coordinator_ready', False)}")
        print(f"   - Framework: {status.get('framework', 'Unknown')}")
    else:
        print("❌ System initialization failed")
        print(f"   Error: {status.get('error', 'Unknown error')}")
        return
    
    # Test 2: User Session Creation
    print("\n2. Testing User Session Creation")
    print("-" * 30)
    
    session_result = chatbot.create_user_session(
        user_name="Test User",
        native_language="Hindi",
        preferred_languages=["English", "Hindi"],
        language_comfort_level="mixed"
    )
    
    if session_result.get('success'):
        user_id = session_result['user_id']
        print("✅ User session created successfully")
        print(f"   - User ID: {user_id}")
        print(f"   - Framework: {session_result.get('framework', 'Unknown')}")
    else:
        print("❌ User session creation failed")
        print(f"   Error: {session_result.get('error', 'Unknown error')}")
        return
    
    # Test 3: React AI Conversation
    print("\n3. Testing React AI Conversation")
    print("-" * 30)
    
    test_messages = [
        "Hello! I love cooking Indian food",
        "I'm interested in technology and startups",
        "Can you suggest some good Indian restaurants?"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\nMessage {i}: {message}")
        
        response_result = chatbot.send_message(
            user_id=user_id,
            message=message,
            language_preferences={
                'native_language': 'Hindi',
                'preferred_languages': ['English', 'Hindi'],
                'language_comfort_level': 'mixed'
            }
        )
        
        if response_result.get('success'):
            print("✅ Response generated successfully")
            print(f"   - Response: {response_result.get('response', 'N/A')[:100]}...")
            print(f"   - Reasoning steps: {response_result.get('reasoning_steps', 0)}")
            print(f"   - Framework: {response_result.get('framework', 'Unknown')}")
            
            # Show reasoning chain if available
            if response_result.get('reasoning_chain'):
                print("   - Reasoning chain:")
                for j, step in enumerate(response_result['reasoning_chain'][:3], 1):
                    print(f"     Step {j}: {step.get('action', 'Unknown')}")
        else:
            print("❌ Response generation failed")
            print(f"   Error: {response_result.get('error', 'Unknown error')}")
    
    # Test 4: React AI Tag Analysis
    print("\n4. Testing React AI Tag Analysis")
    print("-" * 30)
    
    # Get conversation text for analysis
    conversations = chatbot.db.get_user_conversations(user_id) if chatbot.db else []
    conversation_text = " ".join([msg[1] if isinstance(msg, tuple) else msg.get('content', '') 
                                for msg in conversations])
    
    analysis_result = chatbot.analyze_conversation_for_tags(
        user_id=user_id,
        conversation_text=conversation_text,
        language_preferences={
            'native_language': 'Hindi',
            'preferred_languages': ['English', 'Hindi'],
            'language_comfort_level': 'mixed'
        }
    )
    
    if analysis_result.get('success'):
        print("✅ Conversation analysis completed")
        print(f"   - Analysis summary: {analysis_result.get('analysis_summary', 'N/A')}")
        print(f"   - Reasoning steps: {analysis_result.get('reasoning_steps', 0)}")
        print(f"   - Framework: {analysis_result.get('framework', 'Unknown')}")
        
        # Get tag suggestions
        suggestions_result = chatbot.get_tag_suggestions(
            user_id=user_id,
            analysis_result=str(analysis_result.get('analysis_summary', '')),
            existing_tags=[]
        )
        
        if suggestions_result.get('success'):
            suggestions = suggestions_result.get('suggestions', [])
            print(f"   - Suggested tags: {suggestions}")
            print(f"   - Suggestion count: {suggestions_result.get('suggestion_count', 0)}")
        else:
            print("❌ Tag suggestions failed")
            print(f"   Error: {suggestions_result.get('error', 'Unknown error')}")
    else:
        print("❌ Conversation analysis failed")
        print(f"   Error: {analysis_result.get('error', 'Unknown error')}")
    
    # Test 5: React AI System Status
    print("\n5. Testing React AI System Status")
    print("-" * 30)
    
    status_result = chatbot.get_system_status()
    
    if status_result.get('success'):
        print("✅ System status retrieved successfully")
        print(f"   - Total agents: {status_result.get('total_agents', 0)}")
        print(f"   - Framework: {status_result.get('framework', 'Unknown')}")
        
        # Show agent details
        agent_details = status_result.get('agent_details', {})
        for agent_name, details in agent_details.items():
            print(f"   - {agent_name}:")
            print(f"     Status: {'Active' if details['status']['is_active'] else 'Inactive'}")
            print(f"     Tools: {details['status'].get('tools_count', 0)}")
            print(f"     Reasoning history: {details.get('reasoning_history_count', 0)}")
            print(f"     Capabilities: {', '.join(details.get('capabilities', [])[:3])}")
        
        # Show React AI features
        features = status_result.get('react_ai_features', [])
        print(f"   - React AI features: {', '.join(features)}")
    else:
        print("❌ System status retrieval failed")
        print(f"   Error: {status_result.get('error', 'Unknown error')}")
    
    # Test 6: React AI Agent Graph
    print("\n6. Testing React AI Agent Graph")
    print("-" * 30)
    
    graph_result = chatbot.get_agent_graph_data()
    
    if graph_result.get('success'):
        print("✅ Agent graph data retrieved successfully")
        print(f"   - Total nodes: {graph_result.get('total_nodes', 0)}")
        print(f"   - Total edges: {graph_result.get('total_edges', 0)}")
        print(f"   - Framework: {graph_result.get('framework', 'Unknown')}")
        print(f"   - Pattern: {graph_result.get('pattern', 'Unknown')}")
        
        # Show node information
        nodes = graph_result.get('nodes', [])
        for node in nodes:
            print(f"   - Node: {node.get('label', 'Unknown')} ({node.get('type', 'Unknown')})")
    else:
        print("❌ Agent graph data retrieval failed")
        print(f"   Error: {graph_result.get('error', 'Unknown error')}")
    
    # Test 7: React AI Pattern Demonstration
    print("\n7. React AI Pattern Demonstration")
    print("-" * 30)
    
    print("🔄 React AI Pattern: Observe → Think → Act → Reflect")
    print("\nExample conversation flow:")
    print("1. OBSERVE: User says 'I love cooking Indian food'")
    print("2. THINK: This indicates interest in cooking, Indian cuisine, culture")
    print("3. ACT: Generate response about cooking and suggest related tags")
    print("4. REFLECT: Learn that user is interested in culinary topics")
    
    print("\nReact AI Benefits:")
    print("✅ Transparent reasoning process")
    print("✅ Dynamic tool calling")
    print("✅ Cultural context integration")
    print("✅ Adaptive learning")
    print("✅ Scalable architecture")
    
    # Cleanup
    print("\n8. Cleanup")
    print("-" * 30)
    
    chatbot.cleanup()
    print("✅ React AI system cleaned up successfully")
    
    print("\n🎉 React AI Pattern Multi-Agent Chatbot System Test Completed!")
    print("=" * 60)


def demonstrate_react_ai_pattern():
    """Demonstrate the React AI pattern with examples."""
    print("\n🔍 React AI Pattern Demonstration")
    print("=" * 40)
    
    print("\n1. Observe-Think-Act Loop Example:")
    print("   OBSERVE: Analyze user message and context")
    print("   THINK: Reason about user interests and cultural context")
    print("   ACT: Generate appropriate response with cultural awareness")
    print("   REFLECT: Learn from the interaction for future improvements")
    
    print("\n2. Dynamic Tool Calling Example:")
    print("   - analyze_conversation_context: Understand user needs")
    print("   - generate_cultural_response: Create culturally aware responses")
    print("   - generate_followup_question: Create engaging follow-ups")
    print("   - learn_from_response: Improve from user feedback")
    
    print("\n3. Reasoning Chain Example:")
    print("   Step 1: Observe current conversation state")
    print("   Step 2: Think about user's cultural background")
    print("   Step 3: Act by generating culturally appropriate response")
    print("   Step 4: Reflect on response quality and cultural sensitivity")
    
    print("\n4. Cultural Context Integration:")
    print("   - Language preferences consideration")
    print("   - Cultural sensitivity in responses")
    print("   - Regional context awareness")
    print("   - Traditional and contemporary balance")


if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Run tests
    test_react_ai_system()
    
    # Demonstrate React AI pattern
    demonstrate_react_ai_pattern()
    
    print("\n🚀 React AI Pattern System is ready for use!")
    print("Run 'streamlit run main_react.py' to start the application.") 