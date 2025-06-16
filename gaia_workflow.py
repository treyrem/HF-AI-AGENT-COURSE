#!/usr/bin/env python3
"""
GAIA Agent LangGraph Workflow
Main orchestration workflow for the GAIA benchmark agent system
"""

import logging
from typing import Dict, Any, List, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agents.state import GAIAAgentState, AgentRole, QuestionType
from agents.router import RouterAgent
from agents.web_researcher import WebResearchAgent
from agents.file_processor_agent import FileProcessorAgent
from agents.reasoning_agent import ReasoningAgent
from agents.synthesizer import SynthesizerAgent
from models.qwen_client import QwenClient

logger = logging.getLogger(__name__)

class GAIAWorkflow:
    """
    Main GAIA agent workflow using LangGraph
    Orchestrates router → specialized agents → synthesizer pipeline
    """
    
    def __init__(self, llm_client: QwenClient):
        self.llm_client = llm_client
        
        # Initialize all agents
        self.router = RouterAgent(llm_client)
        self.web_researcher = WebResearchAgent(llm_client)
        self.file_processor = FileProcessorAgent(llm_client)
        self.reasoning_agent = ReasoningAgent(llm_client)
        self.synthesizer = SynthesizerAgent(llm_client)
        
        # Create workflow graph
        self.workflow = self._create_workflow()
        
        # Compile workflow with memory
        self.app = self.workflow.compile(checkpointer=MemorySaver())
        
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        # Define the workflow graph
        workflow = StateGraph(GAIAAgentState)
        
        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("web_researcher", self._web_researcher_node)
        workflow.add_node("file_processor", self._file_processor_node)
        workflow.add_node("reasoning_agent", self._reasoning_agent_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        
        # Define entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges from router to agents
        workflow.add_conditional_edges(
            "router",
            self._route_to_agents,
            {
                "web_researcher": "web_researcher",
                "file_processor": "file_processor", 
                "reasoning_agent": "reasoning_agent",
                "multi_agent": "web_researcher",  # Start with web researcher for multi-agent
                "synthesizer": "synthesizer"  # Direct to synthesizer if no agents needed
            }
        )
        
        # Add edges from agents to synthesizer
        workflow.add_edge("web_researcher", "synthesizer")
        workflow.add_edge("file_processor", "synthesizer") 
        workflow.add_edge("reasoning_agent", "synthesizer")
        
        # Add conditional edges for multi-agent scenarios
        workflow.add_conditional_edges(
            "synthesizer",
            self._check_if_complete,
            {
                "complete": END,
                "need_more_agents": "file_processor"  # Route to next agent if needed
            }
        )
        
        return workflow
    
    def _router_node(self, state: GAIAAgentState) -> GAIAAgentState:
        """Router node - classifies question and selects agents"""
        logger.info("🧭 Executing router node")
        return self.router.route_question(state)
    
    def _web_researcher_node(self, state: GAIAAgentState) -> GAIAAgentState:
        """Web researcher node"""
        logger.info("🌐 Executing web researcher node")
        return self.web_researcher.process(state)
    
    def _file_processor_node(self, state: GAIAAgentState) -> GAIAAgentState:
        """File processor node"""
        logger.info("📁 Executing file processor node")
        return self.file_processor.process(state)
    
    def _reasoning_agent_node(self, state: GAIAAgentState) -> GAIAAgentState:
        """Reasoning agent node"""
        logger.info("🧠 Executing reasoning agent node")
        return self.reasoning_agent.process(state)
    
    def _synthesizer_node(self, state: GAIAAgentState) -> GAIAAgentState:
        """Synthesizer node - combines agent results"""
        logger.info("🔗 Executing synthesizer node")
        return self.synthesizer.process(state)
    
    def _route_to_agents(self, state: GAIAAgentState) -> str:
        """Determine which agent(s) to route to based on router decision"""
        
        selected_agents = state.selected_agents
        
        # Remove synthesizer from routing decision (it's always last)
        agent_roles = [agent for agent in selected_agents if agent != AgentRole.SYNTHESIZER]
        
        if not agent_roles:
            # No specific agents selected, go directly to synthesizer
            return "synthesizer"
        elif len(agent_roles) == 1:
            # Single agent selected
            agent = agent_roles[0]
            if agent == AgentRole.WEB_RESEARCHER:
                return "web_researcher"
            elif agent == AgentRole.FILE_PROCESSOR:
                return "file_processor"
            elif agent == AgentRole.REASONING_AGENT:
                return "reasoning_agent"
            else:
                return "synthesizer"
        else:
            # Multiple agents - start with web researcher
            # The workflow will handle additional agents in subsequent steps
            return "multi_agent"
    
    def _check_if_complete(self, state: GAIAAgentState) -> str:
        """Check if processing is complete or if more agents are needed"""
        
        # If synthesis is complete, we're done
        if state.is_complete:
            return "complete"
        
        # Check if we need to run additional agents
        selected_agents = state.selected_agents
        executed_agents = set(result.agent_role for result in state.agent_results)
        
        # Find agents that haven't been executed yet
        remaining_agents = [
            agent for agent in selected_agents 
            if agent not in executed_agents and agent != AgentRole.SYNTHESIZER
        ]
        
        if remaining_agents:
            # Route to next agent
            next_agent = remaining_agents[0]
            if next_agent == AgentRole.FILE_PROCESSOR:
                return "need_more_agents"  # This will route to file_processor
            elif next_agent == AgentRole.REASONING_AGENT:
                return "need_more_agents"  # Would need additional routing logic
            else:
                return "complete"
        else:
            return "complete"
    
    def process_question(self, question: str, file_path: str = None, file_name: str = None, 
                        task_id: str = None, difficulty_level: int = 1) -> GAIAAgentState:
        """
        Process a GAIA question through the complete workflow
        
        Args:
            question: The question to process
            file_path: Optional path to associated file
            file_name: Optional name of associated file
            task_id: Optional task identifier
            difficulty_level: Question difficulty (1-3)
            
        Returns:
            GAIAAgentState with final results
        """
        
        logger.info(f"🚀 Processing question: {question[:100]}...")
        
        # Initialize state
        initial_state = GAIAAgentState(
            question=question,
            question_id=task_id or f"workflow_{hash(question) % 10000}",
            file_name=file_name,
            file_content=None
        )
        initial_state.file_path = file_path
        initial_state.difficulty_level = difficulty_level
        
        try:
            # Execute workflow
            final_state = self.app.invoke(
                initial_state,
                config={"configurable": {"thread_id": initial_state.task_id}}
            )
            
            logger.info(f"✅ Workflow complete: {final_state.final_answer[:100]}...")
            return final_state
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg)
            
            # Create error state
            initial_state.add_error(error_msg)
            initial_state.final_answer = "Workflow execution failed"
            initial_state.final_confidence = 0.0
            initial_state.final_reasoning = error_msg
            initial_state.is_complete = True
            initial_state.requires_human_review = True
            
            return initial_state
    
    def get_workflow_visualization(self) -> str:
        """Get a text representation of the workflow"""
        return """
GAIA Agent Workflow:

┌─────────────┐
│   Router    │ ← Entry Point
└──────┬──────┘
       │
       ├─ Web Researcher ──┐
       ├─ File Processor ──┤
       ├─ Reasoning Agent ─┤
       │                   │
       ▼                   ▼
┌─────────────┐    ┌──────────────┐
│ Synthesizer │ ←──┤ Agent Results │
└──────┬──────┘    └──────────────┘
       │
       ▼
┌─────────────┐
│   END       │
└─────────────┘

Flow:
1. Router classifies question and selects appropriate agent(s)
2. Selected agents process question in parallel/sequence
3. Synthesizer combines results into final answer
4. Workflow completes with final state
"""

# Simplified workflow for cases where we don't need full LangGraph
class SimpleGAIAWorkflow:
    """
    Simplified workflow that doesn't require LangGraph for basic cases
    Useful for testing and lightweight deployments
    """
    
    def __init__(self, llm_client: QwenClient):
        self.llm_client = llm_client
        self.router = RouterAgent(llm_client)
        self.web_researcher = WebResearchAgent(llm_client)
        self.file_processor = FileProcessorAgent(llm_client)
        self.reasoning_agent = ReasoningAgent(llm_client)
        self.synthesizer = SynthesizerAgent(llm_client)
    
    def process_question(self, question: str, file_path: str = None, file_name: str = None, 
                        task_id: str = None, difficulty_level: int = 1) -> GAIAAgentState:
        """Process question with simplified sequential workflow"""
        
        # Initialize state
        state = GAIAAgentState(
            question=question,
            question_id=task_id or f"simple_{hash(question) % 10000}",
            file_name=file_name,
            file_content=None
        )
        state.file_path = file_path
        state.difficulty_level = difficulty_level
        
        try:
            # Step 1: Route
            state = self.router.route_question(state)
            
            # Step 2: Execute agents
            for agent_role in state.selected_agents:
                if agent_role == AgentRole.WEB_RESEARCHER:
                    state = self.web_researcher.process(state)
                elif agent_role == AgentRole.FILE_PROCESSOR:
                    state = self.file_processor.process(state)
                elif agent_role == AgentRole.REASONING_AGENT:
                    state = self.reasoning_agent.process(state)
                # Skip synthesizer for now
            
            # Step 3: Synthesize
            state = self.synthesizer.process(state)
            
            return state
            
        except Exception as e:
            error_msg = f"Simple workflow failed: {str(e)}"
            state.add_error(error_msg)
            state.final_answer = "Processing failed"
            state.final_confidence = 0.0
            state.final_reasoning = error_msg
            state.is_complete = True
            return state 

def create_gaia_workflow(llm_client, tools_dict):
    """
    Create an enhanced GAIA workflow with multi-phase planning and iterative refinement
    """
    
    # Initialize agents with enhanced capabilities
    router = RouterAgent(llm_client)
    web_researcher = WebResearchAgent(llm_client) 
    file_processor = FileProcessorAgent(llm_client)
    reasoning_agent = ReasoningAgent(llm_client)
    synthesizer = SynthesizerAgent(llm_client)
    
    # Enhanced workflow nodes with multi-step processing
    def router_node(state: GAIAAgentState) -> GAIAAgentState:
        """Enhanced router with multi-phase analysis"""
        logger.info("🧭 Router: Starting multi-phase analysis")
        return router.process(state)
    
    def web_researcher_node(state: GAIAAgentState) -> GAIAAgentState:
        """Web researcher with multi-step planning"""
        logger.info("🌐 Web Researcher: Starting enhanced research")
        return web_researcher.process(state)
    
    def file_processor_node(state: GAIAAgentState) -> GAIAAgentState:
        """File processor with step-by-step analysis"""
        logger.info("📁 File Processor: Starting file analysis")
        return file_processor.process(state)
    
    def reasoning_agent_node(state: GAIAAgentState) -> GAIAAgentState:
        """Reasoning agent with systematic approach"""
        logger.info("🧠 Reasoning Agent: Starting analysis")
        return reasoning_agent.process(state)
    
    def synthesizer_node(state: GAIAAgentState) -> GAIAAgentState:
        """Enhanced synthesizer with verification"""
        logger.info("🎯 Synthesizer: Starting GAIA-compliant synthesis")
        return synthesizer.process(state)
    
    def should_continue_to_next_agent(state: GAIAAgentState) -> str:
        """
        Enhanced routing logic that follows the planned agent sequence
        """
        # Get the planned sequence from router
        agent_sequence = getattr(state, 'agent_sequence', [])
        
        if not agent_sequence:
            logger.warning("No agent sequence found, using fallback routing")
            # Fallback to basic routing
            if not state.agent_results:
                return "web_researcher"
            return "synthesizer"
        
        # Count how many agents have been executed
        executed_count = len(state.agent_results)
        
        # Check if we've executed all planned agents
        if executed_count >= len(agent_sequence):
            return "synthesizer"
        
        # Get next agent in sequence
        next_agent = agent_sequence[executed_count]
        
        # Map string names to node names
        agent_mapping = {
            'web_researcher': 'web_researcher',
            'file_processor': 'file_processor', 
            'reasoning_agent': 'reasoning_agent',
            'synthesizer': 'synthesizer'
        }
        
        return agent_mapping.get(next_agent, 'synthesizer')
    
    def check_quality_and_refinement(state: GAIAAgentState) -> str:
        """
        Check if results need refinement before synthesis
        """
        if not state.agent_results:
            return "synthesizer"
        
        # Check overall quality of results
        avg_confidence = sum(r.confidence for r in state.agent_results) / len(state.agent_results)
        
        # If confidence is very low and we haven't tried refinement yet
        if avg_confidence < 0.3 and not getattr(state, 'refinement_attempted', False):
            logger.info(f"Low confidence ({avg_confidence:.2f}), attempting refinement")
            state.refinement_attempted = True
            return "refine_approach"
        
        return "synthesizer"
    
    def refinement_node(state: GAIAAgentState) -> GAIAAgentState:
        """
        Attempt to refine the approach when initial results are poor
        """
        logger.info("🔄 Attempting result refinement")
        state.add_processing_step("Workflow: Attempting refinement due to low confidence")
        
        # Analyze what went wrong and try a different approach
        router_analysis = getattr(state, 'router_analysis', {})
        
        if router_analysis:
            # Try alternative strategy from router analysis
            strategy = router_analysis.get('strategy', {})
            fallback_strategies = strategy.get('fallback_needed', True)
            
            if fallback_strategies:
                # Try web research if it wasn't the primary approach
                if not any(r.agent_role == AgentRole.WEB_RESEARCHER for r in state.agent_results):
                    return web_researcher.process(state)
                # Try reasoning if web search was done
                elif not any(r.agent_role == AgentRole.REASONING_AGENT for r in state.agent_results):
                    return reasoning_agent.process(state)
        
        # Fallback: try reasoning agent for additional analysis
        return reasoning_agent.process(state)
    
    # Create workflow graph with enhanced routing
    workflow = StateGraph(GAIAAgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("web_researcher", web_researcher_node)
    workflow.add_node("file_processor", file_processor_node)
    workflow.add_node("reasoning_agent", reasoning_agent_node)
    workflow.add_node("refine_approach", refinement_node)
    workflow.add_node("synthesizer", synthesizer_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Enhanced routing edges
    workflow.add_conditional_edges(
        "router",
        should_continue_to_next_agent,
        {
            "web_researcher": "web_researcher",
            "file_processor": "file_processor",
            "reasoning_agent": "reasoning_agent", 
            "synthesizer": "synthesizer"
        }
    )
    
    # Progressive routing with quality checks
    workflow.add_conditional_edges(
        "web_researcher",
        should_continue_to_next_agent,
        {
            "file_processor": "file_processor",
            "reasoning_agent": "reasoning_agent",
            "synthesizer": "synthesizer",
            "refine_approach": "refine_approach"
        }
    )
    
    workflow.add_conditional_edges(
        "file_processor", 
        should_continue_to_next_agent,
        {
            "web_researcher": "web_researcher",
            "reasoning_agent": "reasoning_agent",
            "synthesizer": "synthesizer",
            "refine_approach": "refine_approach"
        }
    )
    
    workflow.add_conditional_edges(
        "reasoning_agent",
        should_continue_to_next_agent,
        {
            "web_researcher": "web_researcher",
            "file_processor": "file_processor", 
            "synthesizer": "synthesizer",
            "refine_approach": "refine_approach"
        }
    )
    
    # Quality check before synthesis
    workflow.add_conditional_edges(
        "refine_approach",
        check_quality_and_refinement,
        {
            "synthesizer": "synthesizer",
            "refine_approach": "refine_approach"  # Allow multiple refinement attempts
        }
    )
    
    # Synthesizer is the final step
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()

def create_simple_workflow(llm_client, tools_dict):
    """
    Enhanced simple workflow with better planning and execution
    """
    # Use same agents as complex workflow for consistency
    router = RouterAgent(llm_client)
    web_researcher = WebResearchAgent(llm_client)
    reasoning_agent = ReasoningAgent(llm_client)
    synthesizer = SynthesizerAgent(llm_client)
    
    def process_with_planning(state: GAIAAgentState) -> GAIAAgentState:
        """Simple but systematic processing with planning"""
        
        logger.info("🚀 Starting simple workflow with enhanced planning")
        
        # Step 1: Analyze and plan
        state = router.process(state)
        
        # Step 2: Execute primary research/reasoning
        agent_sequence = getattr(state, 'agent_sequence', ['web_researcher', 'reasoning_agent'])
        
        for agent_name in agent_sequence:
            if agent_name == 'web_researcher':
                state = web_researcher.process(state)
            elif agent_name == 'reasoning_agent':
                state = reasoning_agent.process(state)
            elif agent_name == 'synthesizer':
                break  # Synthesizer is handled separately
            
            # Early exit if we have high confidence result
            if state.agent_results and state.agent_results[-1].confidence > 0.8:
                logger.info("High confidence result achieved, proceeding to synthesis")
                break
        
        # Step 3: Synthesize results
        state = synthesizer.process(state)
        
        return state
    
    # Create simple workflow graph
    workflow = StateGraph(GAIAAgentState)
    workflow.add_node("process", process_with_planning)
    workflow.set_entry_point("process")
    workflow.add_edge("process", END)
    
    return workflow.compile() 