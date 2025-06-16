#!/usr/bin/env python3
"""
Synthesizer Agent for GAIA Agent System
GAIA-Compliant Final Answer Generation for Exact Match Evaluation
"""

import logging
from typing import Dict, List, Optional, Any
from statistics import mean

from agents.state import GAIAAgentState, AgentRole, AgentResult
from models.qwen_client import QwenClient, ModelTier
from tools.final_answer_tool import FinalAnswerTool

logger = logging.getLogger(__name__)

class SynthesizerAgent:
    """
    GAIA-compliant synthesizer that produces EXACT MATCH answers
    Uses 72B model and final answer tool for precise extraction
    """
    
    def __init__(self, llm_client: QwenClient):
        self.llm_client = llm_client
        self.final_answer_tool = FinalAnswerTool(llm_client)
        
    def process(self, state: GAIAAgentState) -> GAIAAgentState:
        """
        Synthesize GAIA-compliant final answer from agent results
        """
        logger.info("🎯 Synthesizer: Starting GAIA-compliant synthesis")
        state.add_processing_step("Synthesizer: Generating GAIA-compliant final answer")
        
        try:
            # Check if we have any agent results
            if not state.agent_results:
                logger.warning("No agent results available for synthesis")
                state.final_answer = "No results available"
                state.final_confidence = 0.0
                state.final_reasoning = "No agent results to synthesize"
                state.is_complete = True
                return state
            
            # Combine all agent results into comprehensive analysis
            combined_analysis = self._combine_agent_results(state)
            
            # Determine question type for specialized extraction
            question_type = self._determine_question_type(state.question)
            
            # Use 72B model for synthesis if we have multiple results or complex question
            if len(state.agent_results) > 1 or state.should_use_complex_model():
                synthesis_result = self._synthesize_with_72b(state, combined_analysis, question_type)
            else:
                synthesis_result = self._synthesize_simple(state, combined_analysis, question_type)
            
            # Extract GAIA-compliant final answer
            final_answer_result = self.final_answer_tool.extract_final_answer(
                question=state.question,
                agent_results=synthesis_result["analysis"],
                question_type=question_type
            )
            
            # Update state with final results
            state.final_answer = final_answer_result["answer"]
            state.final_confidence = final_answer_result["confidence"]
            state.final_reasoning = f"Synthesis: {synthesis_result['reasoning']} | Extraction: {final_answer_result['reasoning']}"
            state.answer_source = "gaia_compliant_synthesis"
            state.is_complete = True
            
            # GAIA compliance check
            if len(state.final_answer) > 100:
                logger.warning(f"Answer may be too long for GAIA: {len(state.final_answer)} chars")
                state.final_confidence *= 0.7  # Reduce confidence for long answers
            
            logger.info(f"✅ GAIA synthesis complete: '{state.final_answer}' (conf: {state.final_confidence:.2f})")
            state.add_processing_step(f"Synthesizer: GAIA answer generated - '{state.final_answer}'")
            
            return state
            
        except Exception as e:
            error_msg = f"GAIA synthesis failed: {str(e)}"
            state.add_error(error_msg)
            logger.error(error_msg)
            
            # Fallback to simple answer
            state.final_answer = "Processing error"
            state.final_confidence = 0.0
            state.final_reasoning = error_msg
            state.answer_source = "error_fallback"
            state.is_complete = True
            
            return state
    
    def _combine_agent_results(self, state: GAIAAgentState) -> str:
        """Combine all agent results into comprehensive analysis"""
        
        analysis_parts = []
        
        # Add successful results first
        successful_results = [r for r in state.agent_results if r.success]
        if successful_results:
            analysis_parts.append("=== SUCCESSFUL AGENT RESULTS ===")
            for result in successful_results:
                analysis_parts.append(f"""
{result.agent_role.value.upper()} (Confidence: {result.confidence:.2f}):
Result: {result.result}
Reasoning: {result.reasoning}
""")
        
        # Add failed results with useful information
        failed_results = [r for r in state.agent_results if not r.success]
        if failed_results:
            analysis_parts.append("\n=== ADDITIONAL CONTEXT ===")
            for result in failed_results:
                if len(result.result) > 10:  # Only include if has some content
                    analysis_parts.append(f"""
{result.agent_role.value.upper()} (Failed):
Attempted: {result.result[:200]}...
""")
        
        return "\n".join(analysis_parts)
    
    def _determine_question_type(self, question: str) -> str:
        """Determine question type for specialized answer extraction"""
        
        question_lower = question.lower()
        
        # Mathematical/counting questions
        if any(word in question_lower for word in ["how many", "count", "number of", "calculate", "sum", "total"]):
            return "mathematical"
        
        # Text manipulation (reversed text, opposites, etc.)
        if any(word in question_lower for word in ["opposite", "reverse", "backwards", "decode"]):
            return "text_manipulation"
        
        # Yes/no questions
        if any(word in question_lower for word in ["yes or no", "true or false", "is it", "does it", "can it"]):
            return "yes_no"
        
        # Name/person questions  
        if any(word in question_lower for word in ["who", "name", "first name", "last name", "surname"]):
            return "name"
        
        # Location questions
        if any(word in question_lower for word in ["where", "city", "country", "location", "place"]):
            return "location"
        
        # File/code questions
        if any(word in question_lower for word in ["file", "image", "code", "python", "attached", "excel"]):
            return "file_processing"
        
        return "general"
    
    def _synthesize_with_72b(self, state: GAIAAgentState, combined_analysis: str, question_type: str) -> Dict[str, Any]:
        """Use 72B model for complex synthesis"""
        
        synthesis_prompt = f"""
CRITICAL: This is GAIA benchmark evaluation requiring EXACT MATCH answers.

Question: {state.question}

Agent Analysis Results:
{combined_analysis}

Your task: Analyze all agent results and provide the most accurate answer.

GAIA COMPLIANCE RULES:
- Your answer must be concise and precise for exact match comparison
- No explanations, no "FINAL ANSWER:" prefix, no extra text
- For numbers: just the number (e.g., "5")
- For yes/no: just "yes" or "no"
- For names: just the name requested
- For locations: just the location name

Question Type: {question_type}

Based on all the agent results above, what is the precise answer to the original question?
Think carefully but respond with ONLY the answer:"""
        
        # Use 72B model for synthesis
        result = self.llm_client.generate(
            synthesis_prompt,
            tier=ModelTier.COMPLEX,  # 72B model
            max_tokens=100
        )
        
        if result.success:
            return {
                "analysis": result.response,
                "reasoning": f"72B synthesis of {len(state.agent_results)} agent results"
            }
        else:
            # Fallback to simple synthesis
            return self._synthesize_simple(state, combined_analysis, question_type)
    
    def _synthesize_simple(self, state: GAIAAgentState, combined_analysis: str, question_type: str) -> Dict[str, Any]:
        """Simple synthesis for single agent results or fallback"""
        
        # Find the best available result
        successful_results = [r for r in state.agent_results if r.success]
        
        if successful_results:
            best_result = max(successful_results, key=lambda r: r.confidence)
            return {
                "analysis": f"Primary result from {best_result.agent_role.value}: {best_result.result}",
                "reasoning": f"Single agent result from {best_result.agent_role.value}"
            }
        else:
            # Try to extract useful info from failures
            all_results = list(state.agent_results)
            if all_results:
                fallback_result = all_results[0]  # Use first available result
                return {
                    "analysis": f"Fallback from {fallback_result.agent_role.value}: {fallback_result.result}",
                    "reasoning": f"Fallback synthesis from {fallback_result.agent_role.value}"
                }
            else:
                return {
                    "analysis": "No agent results available",
                    "reasoning": "No synthesis possible - no results"
                }

# Import regex for LLM response parsing
import re 