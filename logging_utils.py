#!/usr/bin/env python3
"""
Enhanced Logging Utilities for Multi-Agent Chatbot System
Provides comprehensive logging for context and agent exchanges
"""

import logging
import json
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import sys
import os

# Configure logging levels
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

@dataclass
class LogContext:
    """Structured log context for better debugging"""
    timestamp: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_name: Optional[str] = None
    operation: Optional[str] = None
    group_id: Optional[str] = None
    topic_name: Optional[str] = None
    message_type: Optional[str] = None
    context_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Handle additional fields not in the dataclass"""
        # Store any additional fields in metadata
        if self.metadata is None:
            self.metadata = {}
    
    @classmethod
    def create_with_extras(cls, **kwargs):
        """Create LogContext with additional fields stored in metadata"""
        # Extract known fields
        known_fields = {
            'timestamp': kwargs.get('timestamp', datetime.now().isoformat()),
            'user_id': kwargs.get('user_id'),
            'session_id': kwargs.get('session_id'),
            'agent_name': kwargs.get('agent_name'),
            'operation': kwargs.get('operation'),
            'group_id': kwargs.get('group_id'),
            'topic_name': kwargs.get('topic_name'),
            'message_type': kwargs.get('message_type'),
            'context_type': kwargs.get('context_type'),
        }
        
        # Store unknown fields in metadata
        metadata = {}
        for key, value in kwargs.items():
            if key not in known_fields and key != 'metadata':
                metadata[key] = value
        
        # Add any explicit metadata
        if 'metadata' in kwargs:
            metadata.update(kwargs['metadata'])
        
        known_fields['metadata'] = metadata
        return cls(**known_fields)

class EnhancedLogger:
    """Enhanced logging system with structured context and debugging capabilities"""
    
    def __init__(self, name: str = "chatbot", level: str = "INFO", 
                 log_to_file: bool = True, log_to_console: bool = True):
        self.name = name
        self.level = LOG_LEVELS.get(level.upper(), logging.INFO)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        self.detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            console_handler.setFormatter(self.simple_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            
            # Main log file
            file_handler = logging.FileHandler(f"{log_dir}/{name}.log")
            file_handler.setLevel(self.level)
            file_handler.setFormatter(self.detailed_formatter)
            self.logger.addHandler(file_handler)
            
            # Debug log file for detailed debugging
            debug_handler = logging.FileHandler(f"{log_dir}/{name}_debug.log")
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(self.detailed_formatter)
            self.logger.addHandler(debug_handler)
    
    def _create_context(self, **kwargs) -> LogContext:
        """Create a log context with current timestamp"""
        return LogContext.create_with_extras(**kwargs)
    
    def _format_context(self, context: LogContext) -> str:
        """Format context for logging"""
        context_dict = asdict(context)
        context_dict.pop('timestamp', None)  # Already in log timestamp
        return f"[{json.dumps(context_dict, default=str)}]"
    
    def debug(self, message: str, **context_kwargs):
        """Log debug message with context"""
        context = self._create_context(**context_kwargs)
        self.logger.debug(f"{self._format_context(context)} {message}")
    
    def info(self, message: str, **context_kwargs):
        """Log info message with context"""
        context = self._create_context(**context_kwargs)
        self.logger.info(f"{self._format_context(context)} {message}")
    
    def warning(self, message: str, **context_kwargs):
        """Log warning message with context"""
        context = self._create_context(**context_kwargs)
        self.logger.warning(f"{self._format_context(context)} {message}")
    
    def error(self, message: str, error: Exception = None, **context_kwargs):
        """Log error message with context and exception details"""
        context = self._create_context(**context_kwargs)
        error_details = ""
        if error:
            error_details = f" | Error: {str(error)} | Traceback: {traceback.format_exc()}"
        self.logger.error(f"{self._format_context(context)} {message}{error_details}")
    
    def critical(self, message: str, error: Exception = None, **context_kwargs):
        """Log critical message with context and exception details"""
        context = self._create_context(**context_kwargs)
        error_details = ""
        if error:
            error_details = f" | Error: {str(error)} | Traceback: {traceback.format_exc()}"
        self.logger.critical(f"{self._format_context(context)} {message}{error_details}")

class AgentLogger:
    """Specialized logger for agent operations and exchanges"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = EnhancedLogger(f"agent.{agent_name}")
    
    def agent_start(self, operation: str, user_id: str = None, **kwargs):
        """Log agent operation start"""
        self.logger.info(
            f"🤖 AGENT START: {operation}",
            agent_name=self.agent_name,
            operation=operation,
            user_id=user_id,
            **kwargs
        )
    
    def agent_end(self, operation: str, result: Any = None, user_id: str = None, **kwargs):
        """Log agent operation end with result"""
        result_summary = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
        self.logger.info(
            f"✅ AGENT END: {operation} | Result: {result_summary}",
            agent_name=self.agent_name,
            operation=operation,
            user_id=user_id,
            metadata={"result_summary": result_summary},
            **kwargs
        )
    
    def agent_error(self, operation: str, error: Exception, user_id: str = None, **kwargs):
        """Log agent operation error"""
        self.logger.error(
            f"❌ AGENT ERROR: {operation}",
            error=error,
            agent_name=self.agent_name,
            operation=operation,
            user_id=user_id,
            **kwargs
        )
    
    def context_building(self, context_type: str, context_data: Dict[str, Any], user_id: str = None, **kwargs):
        """Log context building operations"""
        self.logger.debug(
            f"🔧 CONTEXT BUILDING: {context_type}",
            agent_name=self.agent_name,
            context_type=context_type,
            user_id=user_id,
            metadata={"context_data": context_data},
            **kwargs
        )
    
    def rag_context(self, topic_name: str, participant_count: int, topic_related_tags: List[str], 
                   shared_interests: Dict[str, List[str]], user_id: str = None, **kwargs):
        """Log RAG context building"""
        self.logger.info(
            f"🎯 RAG CONTEXT: Topic='{topic_name}' | Participants={participant_count} | "
            f"Topic Tags={len(topic_related_tags)} | Shared Interests={len(shared_interests)}",
            agent_name=self.agent_name,
            topic_name=topic_name,
            user_id=user_id,
            metadata={
                "participant_count": participant_count,
                "topic_related_tags": topic_related_tags,
                "shared_interests": shared_interests
            },
            **kwargs
        )

class GroupChatLogger:
    """Specialized logger for group chat operations"""
    
    def __init__(self):
        self.logger = EnhancedLogger("group_chat")
    
    def group_created(self, group_id: str, topic_name: str, user_ids: List[str], created_by: str):
        """Log group chat creation"""
        self.logger.info(
            f"📝 GROUP CREATED: '{topic_name}' | ID={group_id} | Participants={len(user_ids)} | Created by={created_by}",
            group_id=group_id,
            topic_name=topic_name,
            operation="group_created",
            metadata={
                "user_ids": user_ids,
                "created_by": created_by,
                "participant_count": len(user_ids)
            }
        )
    
    def message_sent(self, group_id: str, user_id: str, message: str, message_type: str = "user"):
        """Log message sent to group"""
        message_preview = message[:50] + "..." if len(message) > 50 else message
        self.logger.info(
            f"💬 MESSAGE SENT: {message_type.upper()} | '{message_preview}'",
            group_id=group_id,
            user_id=user_id,
            message_type=message_type,
            metadata={"message_preview": message_preview}
        )
    
    def ai_response_generated(self, group_id: str, response: str, citations: List[Any], 
                            topic_name: str, participant_count: int):
        """Log AI response generation"""
        response_preview = response[:100] + "..." if len(response) > 100 else response
        self.logger.info(
            f"🤖 AI RESPONSE: Topic='{topic_name}' | Citations={len(citations)} | '{response_preview}'",
            group_id=group_id,
            topic_name=topic_name,
            message_type="ai",
            metadata={
                "response_preview": response_preview,
                "citation_count": len(citations),
                "participant_count": participant_count
            }
        )
    
    def topic_focus_check(self, group_id: str, topic_name: str, user_message: str, 
                         topic_related_tags: List[str], off_topic_citations: List[Any]):
        """Log topic focus verification"""
        self.logger.info(
            f"🎯 TOPIC FOCUS: '{topic_name}' | Topic Tags={len(topic_related_tags)} | "
            f"Off-topic Citations={len(off_topic_citations)}",
            group_id=group_id,
            topic_name=topic_name,
            operation="topic_focus_check",
            metadata={
                "topic_related_tags": topic_related_tags,
                "off_topic_citations_count": len(off_topic_citations),
                "user_message_preview": user_message[:50]
            }
        )

class CitationLogger:
    """Specialized logger for citation operations"""
    
    def __init__(self):
        self.logger = EnhancedLogger("citations")
    
    def citations_generated(self, citation_count: int, citation_types: List[str], 
                          response_preview: str, user_id: str = None):
        """Log citation generation"""
        self.logger.info(
            f"📚 CITATIONS: {citation_count} generated | Types: {', '.join(citation_types)} | "
            f"Response: '{response_preview[:50]}...'",
            user_id=user_id,
            operation="citations_generated",
            metadata={
                "citation_count": citation_count,
                "citation_types": citation_types,
                "response_preview": response_preview[:100]
            }
        )
    
    def topic_citation(self, citation_type: str, topic_name: str, content: str, 
                      relevance_score: float, user_id: str = None):
        """Log topic-related citation"""
        self.logger.debug(
            f"🎯 TOPIC CITATION: {citation_type} | Topic='{topic_name}' | Score={relevance_score} | "
            f"Content: '{content[:50]}...'",
            user_id=user_id,
            operation="topic_citation",
            metadata={
                "citation_type": citation_type,
                "topic_name": topic_name,
                "relevance_score": relevance_score,
                "content_preview": content[:100]
            }
        )

class DatabaseLogger:
    """Specialized logger for database operations"""
    
    def __init__(self):
        self.logger = EnhancedLogger("database")
    
    def query_executed(self, operation: str, collection: str, query_time: float, 
                      result_count: int = None, user_id: str = None):
        """Log database query execution"""
        self.logger.debug(
            f"🗄️ DB QUERY: {operation} | Collection={collection} | Time={query_time:.3f}s | "
            f"Results={result_count if result_count is not None else 'N/A'}",
            user_id=user_id,
            operation=operation,
            metadata={
                "collection": collection,
                "query_time": query_time,
                "result_count": result_count
            }
        )
    
    def vector_search(self, search_type: str, query: str, result_count: int, 
                     search_time: float, user_id: str = None):
        """Log vector search operations"""
        self.logger.info(
            f"🔍 VECTOR SEARCH: {search_type} | Query='{query[:30]}...' | "
            f"Results={result_count} | Time={search_time:.3f}s",
            user_id=user_id,
            operation="vector_search",
            metadata={
                "search_type": search_type,
                "query": query,
                "result_count": result_count,
                "search_time": search_time
            }
        )

class AgentFlowLogger:
    """Specialized logger for tracking agent flow, RAG calls, and LLM calls"""
    
    def __init__(self):
        self.logger = EnhancedLogger("agent_flow")
        self.flow_id = 0
    
    def start_agent_flow(self, flow_type: str, request: Dict[str, Any], user_id: str = None):
        """Log the start of an agent flow"""
        self.flow_id += 1
        flow_id = f"flow_{self.flow_id}"
        
        self.logger.info(
            f"🚀 AGENT FLOW START: {flow_type} | Flow ID: {flow_id}",
            flow_id=flow_id,
            flow_type=flow_type,
            user_id=user_id,
            operation="flow_start",
            metadata={
                "request_type": request.get('type', ''),
                "request_keys": list(request.keys()),
                "request_preview": str(request)[:200] + "..." if len(str(request)) > 200 else str(request)
            }
        )
        return flow_id
    
    def agent_input(self, agent_name: str, input_data: Dict[str, Any], flow_id: str = None, user_id: str = None):
        """Log agent input"""
        self.logger.info(
            f"📥 AGENT INPUT: {agent_name}",
            agent_name=agent_name,
            flow_id=flow_id,
            user_id=user_id,
            operation="agent_input",
            metadata={
                "input_keys": list(input_data.keys()),
                "input_preview": str(input_data)[:300] + "..." if len(str(input_data)) > 300 else str(input_data)
            }
        )
    
    def agent_output(self, agent_name: str, output_data: Dict[str, Any], flow_id: str = None, user_id: str = None):
        """Log agent output"""
        success = output_data.get('success', False)
        status_icon = "✅" if success else "❌"
        
        self.logger.info(
            f"{status_icon} AGENT OUTPUT: {agent_name} | Success: {success}",
            agent_name=agent_name,
            flow_id=flow_id,
            user_id=user_id,
            operation="agent_output",
            metadata={
                "success": success,
                "output_keys": list(output_data.keys()),
                "output_preview": str(output_data)[:300] + "..." if len(str(output_data)) > 300 else str(output_data),
                "error": output_data.get('error', None) if not success else None
            }
        )
    
    def rag_call_start(self, rag_type: str, query: str, parameters: Dict[str, Any], flow_id: str = None, user_id: str = None):
        """Log RAG call start"""
        self.logger.info(
            f"🔍 RAG CALL START: {rag_type} | Query: '{query[:50]}...'",
            rag_type=rag_type,
            flow_id=flow_id,
            user_id=user_id,
            operation="rag_call_start",
            metadata={
                "query": query,
                "parameters": parameters,
                "query_length": len(query)
            }
        )
    
    def rag_call_end(self, rag_type: str, results: List[Dict], processing_time: float, flow_id: str = None, user_id: str = None):
        """Log RAG call end"""
        self.logger.info(
            f"✅ RAG CALL END: {rag_type} | Results: {len(results)} | Time: {processing_time:.3f}s",
            rag_type=rag_type,
            flow_id=flow_id,
            user_id=user_id,
            operation="rag_call_end",
            metadata={
                "result_count": len(results),
                "processing_time": processing_time,
                "results_preview": [str(r)[:100] + "..." for r in results[:3]] if results else []
            }
        )
    
    def llm_call_start(self, model: str, prompt_length: int, flow_id: str = None, user_id: str = None):
        """Log LLM call start"""
        self.logger.info(
            f"🧠 LLM CALL START: {model} | Prompt Length: {prompt_length}",
            model=model,
            flow_id=flow_id,
            user_id=user_id,
            operation="llm_call_start",
            metadata={
                "prompt_length": prompt_length,
                "model": model
            }
        )
    
    def llm_call_end(self, model: str, response_length: int, processing_time: float, flow_id: str = None, user_id: str = None):
        """Log LLM call end"""
        self.logger.info(
            f"✅ LLM CALL END: {model} | Response Length: {response_length} | Time: {processing_time:.3f}s",
            model=model,
            flow_id=flow_id,
            user_id=user_id,
            operation="llm_call_end",
            metadata={
                "response_length": response_length,
                "processing_time": processing_time,
                "model": model
            }
        )
    
    def agent_chain_start(self, agent_chain: List[str], flow_id: str = None, user_id: str = None):
        """Log the start of an agent chain"""
        self.logger.info(
            f"🔗 AGENT CHAIN START: {' -> '.join(agent_chain)}",
            flow_id=flow_id,
            user_id=user_id,
            operation="agent_chain_start",
            metadata={
                "agent_chain": agent_chain,
                "chain_length": len(agent_chain)
            }
        )
    
    def agent_chain_end(self, agent_chain: List[str], final_result: Dict[str, Any], flow_id: str = None, user_id: str = None):
        """Log the end of an agent chain"""
        success = final_result.get('success', False)
        status_icon = "✅" if success else "❌"
        
        self.logger.info(
            f"{status_icon} AGENT CHAIN END: {' -> '.join(agent_chain)} | Success: {success}",
            flow_id=flow_id,
            user_id=user_id,
            operation="agent_chain_end",
            metadata={
                "agent_chain": agent_chain,
                "final_success": success,
                "final_result_keys": list(final_result.keys()),
                "final_result_preview": str(final_result)[:200] + "..." if len(str(final_result)) > 200 else str(final_result)
            }
        )
    
    def end_agent_flow(self, flow_type: str, final_result: Dict[str, Any], flow_id: str = None, user_id: str = None):
        """Log the end of an agent flow"""
        success = final_result.get('success', False)
        status_icon = "✅" if success else "❌"
        
        self.logger.info(
            f"{status_icon} AGENT FLOW END: {flow_type} | Flow ID: {flow_id} | Success: {success}",
            flow_id=flow_id,
            flow_type=flow_type,
            user_id=user_id,
            operation="flow_end",
            metadata={
                "final_success": success,
                "final_result_keys": list(final_result.keys()),
                "final_result_preview": str(final_result)[:200] + "..." if len(str(final_result)) > 200 else str(final_result)
            }
        )

class ReactAgentLogger:
    """Specialized logger for React AI pattern agents"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = EnhancedLogger(f"react_agent.{agent_name}")
    
    def react_loop_start(self, request: Dict[str, Any], user_id: str = None):
        """Log React AI loop start"""
        self.logger.info(
            f"🔄 REACT LOOP START: {self.agent_name}",
            agent_name=self.agent_name,
            user_id=user_id,
            operation="react_loop_start",
            metadata={
                "request_type": request.get('type', ''),
                "request_keys": list(request.keys()),
                "request_preview": str(request)[:200] + "..." if len(str(request)) > 200 else str(request)
            }
        )
    
    def react_loop_end(self, result: Dict[str, Any], iterations: int, user_id: str = None):
        """Log React AI loop end"""
        success = result.get('success', False)
        status_icon = "✅" if success else "❌"
        
        self.logger.info(
            f"{status_icon} REACT LOOP END: {self.agent_name} | Iterations: {iterations} | Success: {success}",
            agent_name=self.agent_name,
            user_id=user_id,
            operation="react_loop_end",
            metadata={
                "iterations": iterations,
                "success": success,
                "result_keys": list(result.keys()),
                "result_preview": str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            }
        )
    
    def tool_call(self, tool_name: str, tool_input: Any, tool_output: Any, user_id: str = None):
        """Log tool call within React AI loop"""
        self.logger.debug(
            f"🔧 TOOL CALL: {tool_name}",
            agent_name=self.agent_name,
            user_id=user_id,
            operation="tool_call",
            metadata={
                "tool_name": tool_name,
                "tool_input_preview": str(tool_input)[:100] + "..." if len(str(tool_input)) > 100 else str(tool_input),
                "tool_output_preview": str(tool_output)[:100] + "..." if len(str(tool_output)) > 100 else str(tool_output)
            }
        )
    
    def reasoning_step(self, step_type: str, content: str, user_id: str = None):
        """Log reasoning step within React AI loop"""
        self.logger.debug(
            f"💭 REASONING: {step_type}",
            agent_name=self.agent_name,
            user_id=user_id,
            operation="reasoning_step",
            metadata={
                "step_type": step_type,
                "content_preview": content[:100] + "..." if len(content) > 100 else content
            }
        )

# Global logger instances
main_logger = EnhancedLogger("main")
agent_logger = AgentLogger("coordinator")
group_logger = GroupChatLogger()
citation_logger = CitationLogger()
db_logger = DatabaseLogger()
flow_logger = AgentFlowLogger()

def get_react_agent_logger(agent_name: str) -> ReactAgentLogger:
    """Get a React agent logger for a specific agent"""
    return ReactAgentLogger(agent_name)

def log_function_call(func_name: str, **kwargs):
    """Decorator to log function calls"""
    def decorator(func):
        def wrapper(*args, **func_kwargs):
            main_logger.debug(f"🔧 FUNCTION CALL: {func_name}", operation=func_name, **kwargs)
            try:
                result = func(*args, **func_kwargs)
                main_logger.debug(f"✅ FUNCTION SUCCESS: {func_name}", operation=func_name, **kwargs)
                return result
            except Exception as e:
                main_logger.error(f"❌ FUNCTION ERROR: {func_name}", error=e, operation=func_name, **kwargs)
                raise
        return wrapper
    return decorator

def log_rag_context(topic_name: str, participant_data: List[Dict], shared_interests: Dict, 
                   location_context: str, user_id: str = None):
    """Log RAG context building details"""
    agent_logger.rag_context(
        topic_name=topic_name,
        participant_count=len(participant_data),
        topic_related_tags=[tag for p in participant_data for tag in p.get('topic_related_tags', [])],
        shared_interests=shared_interests,
        user_id=user_id
    )

def log_topic_focus_verification(group_id: str, topic_name: str, user_message: str, 
                               citations: List[Any], user_id: str = None):
    """Log topic focus verification"""
    topic_related_citations = [c for c in citations if hasattr(c, 'type') and 'topic' in c.type.lower()]
    off_topic_citations = [c for c in citations if hasattr(c, 'type') and 'topic' not in c.type.lower()]
    
    group_logger.topic_focus_check(
        group_id=group_id,
        topic_name=topic_name,
        user_message=user_message,
        topic_related_tags=[c.content for c in topic_related_citations if hasattr(c, 'content')],
        off_topic_citations=off_topic_citations
    )

# Export main logging functions for easy use
def debug(message: str, **kwargs):
    main_logger.debug(message, **kwargs)

def info(message: str, **kwargs):
    main_logger.info(message, **kwargs)

def warning(message: str, **kwargs):
    main_logger.warning(message, **kwargs)

def error(message: str, error: Exception = None, **kwargs):
    main_logger.error(message, error=error, **kwargs)

def critical(message: str, error: Exception = None, **kwargs):
    main_logger.critical(message, error=error, **kwargs) 