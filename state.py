#!/usr/bin/env python3
"""
LangGraph State Schema for GAIA Agent System
Defines the state structure for agent communication and coordination
"""

from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid

class QuestionType(Enum):
    """Classification of GAIA question types"""
    WIKIPEDIA = "wikipedia"
    WEB_RESEARCH = "web_research"
    YOUTUBE = "youtube"
    FILE_PROCESSING = "file_processing"
    MATHEMATICAL = "mathematical"
    CODE_EXECUTION = "code_execution"
    TEXT_MANIPULATION = "text_manipulation"
    REASONING = "reasoning"
    UNKNOWN = "unknown"

class ModelTier(Enum):
    """Model complexity tiers"""
    ROUTER = "router"     # 7B - Fast classification
    MAIN = "main"         # 32B - Balanced tasks
    COMPLEX = "complex"   # 72B - Complex reasoning

class AgentRole(Enum):
    """Roles of different agents in the system"""
    ROUTER = "router"
    WEB_RESEARCHER = "web_researcher"
    FILE_PROCESSOR = "file_processor"
    CODE_EXECUTOR = "code_executor"
    REASONING_AGENT = "reasoning_agent"
    SYNTHESIZER = "synthesizer"

@dataclass
class ToolResult:
    """Result from a tool execution"""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentResult:
    """Result from an agent's processing"""
    agent_role: AgentRole
    success: bool
    result: str
    confidence: float  # 0.0 to 1.0
    reasoning: str
    tools_used: List[ToolResult] = field(default_factory=list)
    model_used: str = ""
    processing_time: float = 0.0
    cost_estimate: float = 0.0

class GAIAAgentState:
    """
    Central state for the GAIA agent system
    This is passed between all agents in the LangGraph workflow
    """
    
    def __init__(self, question: str, question_id: str = None, file_name: str = None, file_content: bytes = None):
        self.question = question
        self.question_id = question_id or str(uuid.uuid4())
        self.file_name = file_name
        self.file_content = file_content
        
        # Analysis results
        self.question_type: Optional[QuestionType] = None
        self.question_types: List[QuestionType] = []
        self.primary_question_type: Optional[QuestionType] = None
        self.complexity_assessment: str = "medium"
        self.selected_agents: List[AgentRole] = []
        
        # Enhanced router analysis
        self.router_analysis: Optional[Dict[str, Any]] = None
        self.agent_sequence: List[str] = []
        
        # Processing tracking
        self.processing_steps: List[str] = []
        self.agent_results: List[AgentResult] = []
        self.errors: List[str] = []
        self.start_time: float = time.time()
        self.total_cost: float = 0.0
        
        # Final results
        self.final_answer: Optional[str] = None
        self.final_confidence: float = 0.0
        self.synthesis_reasoning: str = ""
        
        # Routing decision tracking
        self.routing_decision: Dict[str, Any] = {}
        
        # Question information
        self.difficulty_level: int = 1  # 1, 2, or 3
        self.file_path: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        
        # Routing decisions
        self.estimated_cost: float = 0.0
        
        # Agent results
        self.tool_results: List[ToolResult] = []
        
        # Final answer
        self.answer_source: str = ""  # Which agent provided the final answer
        
        # System tracking
        self.total_processing_time: float = 0.0
        
        # Status flags
        self.is_complete: bool = False
        self.requires_human_review: bool = False
        self.confidence_threshold_met: bool = False
        
    def add_processing_step(self, step: str):
        """Add a processing step to the history"""
        self.processing_steps.append(f"[{time.time() - self.start_time:.2f}s] {step}")
        
    def add_agent_result(self, result: AgentResult):
        """Add result from an agent"""
        self.agent_results.append(result)
        self.total_cost += result.cost_estimate
        self.total_processing_time += result.processing_time
        self.add_processing_step(f"{result.agent_role.value}: {result.result[:50]}...")
        
    def add_tool_result(self, result: ToolResult):
        """Add result from a tool execution"""
        self.tool_results.append(result)
        self.add_processing_step(f"Tool {result.tool_name}: {'✅' if result.success else '❌'}")
        
    def add_error(self, error_message: str):
        """Add an error message"""
        self.errors.append(error_message)
        self.add_processing_step(f"ERROR: {error_message}")
        
    def get_best_result(self) -> Optional[AgentResult]:
        """Get the agent result with highest confidence"""
        if not self.agent_results:
            return None
        return max(self.agent_results, key=lambda r: r.confidence)
        
    def should_use_complex_model(self) -> bool:
        """Determine if complex model should be used based on state"""
        # Use complex model for:
        # - High difficulty questions
        # - Questions requiring detailed reasoning
        # - When we have budget remaining
        return (
            self.difficulty_level >= 3 or
            self.complexity_assessment == "complex" or
            any("reasoning" in step.lower() for step in self.processing_steps)
        ) and self.total_cost < 0.07  # Keep some budget for complex tasks
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state"""
        return {
            "task_id": self.question_id,
            "question_type": self.question_type.value if self.question_type else "unknown",
            "agents_used": [role.value for role in self.selected_agents],
            "tools_used": [tool.tool_name for tool in self.tool_results],
            "final_answer": self.final_answer,
            "confidence": self.final_confidence,
            "processing_time": self.total_processing_time,
            "total_cost": self.total_cost,
            "steps_count": len(self.processing_steps),
            "is_complete": self.is_complete,
            "error_count": len(self.errors)
        }
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        return {
            "task_id": self.question_id,
            "question": self.question,
            "question_type": self.question_type.value if self.question_type else "unknown",
            "difficulty_level": self.difficulty_level,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "routing_decision": self.routing_decision,
            "selected_agents": [agent.value for agent in self.selected_agents],
            "complexity_assessment": self.complexity_assessment,
            "final_answer": self.final_answer,
            "final_confidence": self.final_confidence,
            "final_reasoning": self.synthesis_reasoning,
            "answer_source": self.answer_source,
            "processing_steps": self.processing_steps,
            "total_cost": self.total_cost,
            "total_processing_time": self.total_processing_time,
            "error_messages": self.errors,
            "is_complete": self.is_complete,
            "summary": self.get_summary(),
            "router_analysis": self.router_analysis,
            "agent_sequence": self.agent_sequence
        }

# Type alias for LangGraph
AgentState = GAIAAgentState 