#!/usr/bin/env python3
"""
Final Answer Tool for GAIA Agent System
Extracts precise, EXACT MATCH compliant answers from agent results
"""

import re
import logging
from typing import Dict, Any, Optional

from models.qwen_client import QwenClient, ModelTier

logger = logging.getLogger(__name__)

class FinalAnswerTool:
    """
    Tool for extracting precise, GAIA-compliant final answers
    Ensures EXACT MATCH compatibility for Unit 4 API submission
    """
    
    def __init__(self, llm_client: QwenClient):
        self.llm_client = llm_client
        
    def extract_final_answer(self, question: str, agent_results: str, question_type: str = "general") -> Dict[str, Any]:
        """
        Extract GAIA-compliant final answer with enhanced accuracy
        """
        logger.info("🎯 Extracting GAIA-compliant final answer")
        
        try:
            # Create specialized extraction prompt based on question type
            extraction_prompt = self._create_extraction_prompt(question, agent_results, question_type)
            
            # Use 72B model for precise extraction
            llm_result = self.llm_client.generate(
                extraction_prompt,
                tier=ModelTier.COMPLEX,  # Always use most capable model
                max_tokens=100  # Keep answer concise
            )
            
            if llm_result.success:
                # Clean and validate the extracted answer
                raw_answer = llm_result.response.strip()
                final_answer = self._clean_and_validate_answer(raw_answer, question, question_type)
                
                # Assess answer quality
                confidence = self._assess_answer_quality(final_answer, question, agent_results, question_type)
                
                return {
                    "answer": final_answer,
                    "confidence": confidence,
                    "reasoning": f"Extracted from {question_type} analysis using 72B model",
                    "raw_response": raw_answer,
                    "validation_passed": len(final_answer) <= 100 and len(final_answer) > 0
                }
            else:
                # Fallback to simple extraction
                return self._fallback_extraction(question, agent_results)
                
        except Exception as e:
            logger.error(f"Final answer extraction failed: {e}")
            return self._fallback_extraction(question, agent_results)
    
    def _create_extraction_prompt(self, question: str, agent_results: str, question_type: str) -> str:
        """Create specialized extraction prompt based on question type"""
        
        base_instructions = """
        CRITICAL: Extract the exact answer for GAIA benchmark evaluation.
        Your response must be ONLY the answer - no explanations, no prefixes, no extra text.
        
        Question: {question}
        
        Analysis from agents:
        {agent_results}
        
        """
        
        # Specialized instructions based on question type
        if question_type == "mathematical" or "how many" in question.lower():
            type_instructions = """
        This is a counting/mathematical question. Respond with ONLY the number.
        Examples of correct responses: "5", "42", "0"
        Do NOT include words like "albums", "songs", "items", etc.
        """
        
        elif question_type == "yes_no":
            type_instructions = """
        This is a yes/no question. Respond with ONLY "yes" or "no".
        """
        
        elif question_type == "name" or any(word in question.lower() for word in ["who", "name"]):
            type_instructions = """
        This is asking for a name. Respond with ONLY the name requested.
        Examples: "John Smith", "Mike102", "Einstein"
        """
        
        elif question_type == "location":
            type_instructions = """
        This is asking for a location. Respond with ONLY the location name.
        Examples: "Paris", "New York", "LIE", "Hanoi"
        """
        
        elif question_type == "text_manipulation":
            type_instructions = """
        This involves text manipulation. Respond with ONLY the processed text result.
        Examples: "right", "hello", "12345"
        """
        
        else:
            type_instructions = """
        Respond with ONLY the direct answer requested.
        Keep it concise and specific.
        """
        
        ending_instructions = """
        
        EXTRACT ONLY THE ANSWER:"""
        
        return base_instructions.format(
            question=question,
            agent_results=agent_results[:2000]  # Limit input length
        ) + type_instructions + ending_instructions
    
    def _clean_and_validate_answer(self, raw_answer: str, question: str, question_type: str) -> str:
        """Clean and validate the extracted answer"""
        
        # Remove common prefixes and suffixes
        answer = raw_answer.strip()
        
        # Remove common answer prefixes
        prefixes_to_remove = [
            "final answer:", "answer:", "the answer is:", "result:", "conclusion:",
            "based on", "according to", "therefore", "thus", "so", "hence",
            "final answer is", "the result is", "it is", "this is"
        ]
        
        answer_lower = answer.lower()
        for prefix in prefixes_to_remove:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix):].strip()
                answer_lower = answer.lower()
        
        # Remove quotes if they wrap the entire answer
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        elif answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]
        
        # Remove trailing punctuation that's not part of the answer
        while answer and answer[-1] in '.!?:;':
            answer = answer[:-1]
        
        # Special handling for different question types
        if question_type == "mathematical" or "how many" in question.lower():
            # Extract just the number
            numbers = re.findall(r'\b\d+\b', answer)
            if numbers:
                answer = numbers[0]
        
        elif question_type == "yes_no":
            # Normalize yes/no answers
            if any(word in answer.lower() for word in ['yes', 'true', 'correct', 'right']):
                answer = "yes"
            elif any(word in answer.lower() for word in ['no', 'false', 'incorrect', 'wrong']):
                answer = "no"
        
        # Final cleanup
        answer = answer.strip()
        
        # Ensure answer is not empty
        if not answer:
            # Try to extract from the original raw answer
            words = raw_answer.split()
            if words:
                answer = words[-1]  # Take the last word as fallback
        
        return answer
    
    def _assess_answer_quality(self, answer: str, question: str, agent_results: str, question_type: str) -> float:
        """Assess the quality/confidence of the extracted answer"""
        
        confidence = 0.7  # Base confidence
        
        # Factor 1: Answer length appropriateness
        if len(answer) == 0:
            return 0.1  # Very low confidence for empty answers
        elif len(answer) > 100:
            confidence -= 0.2  # Too long for GAIA
        elif 1 <= len(answer) <= 50:
            confidence += 0.1  # Good length
        
        # Factor 2: Question type matching
        question_lower = question.lower()
        
        if ("how many" in question_lower or question_type == "mathematical") and re.match(r'^\d+$', answer):
            confidence += 0.15  # Numeric answer to counting question
        elif ("who" in question_lower or "name" in question_lower) and len(answer.split()) <= 3:
            confidence += 0.1   # Name-like answer to who question
        elif ("where" in question_lower) and len(answer.split()) <= 2:
            confidence += 0.1   # Location-like answer
        elif ("yes or no" in question_lower) and answer.lower() in ["yes", "no"]:
            confidence += 0.15  # Perfect yes/no answer
        
        # Factor 3: Answer appears in agent results (indicates it was found)
        if answer.lower() in agent_results.lower():
            confidence += 0.1
        
        # Factor 4: Answer specificity
        if re.search(r'\b\d{4}\b', answer):  # Contains year
            confidence += 0.05
        if re.search(r'\b[A-Z][a-z]+\b', answer):  # Contains proper noun
            confidence += 0.05
        
        # Factor 5: Common failure patterns
        failure_indicators = ['unknown', 'unclear', 'not found', 'unable to determine', 'no information']
        if any(indicator in answer.lower() for indicator in failure_indicators):
            confidence -= 0.3
        
        return max(0.1, min(0.95, confidence))
    
    def _fallback_extraction(self, question: str, agent_results: str) -> Dict[str, Any]:
        """Simple fallback when LLM extraction fails"""
        
        # Try to extract a reasonable answer from agent results
        lines = agent_results.split('\n')
        
        # Look for lines that might contain answers
        potential_answers = []
        for line in lines:
            line = line.strip()
            if len(line) > 0 and len(line) < 100:
                # Skip lines that are clearly explanatory
                if not any(word in line.lower() for word in ['according', 'based on', 'however', 'therefore', 'because']):
                    potential_answers.append(line)
        
        # Use the first reasonable answer or a fallback
        answer = potential_answers[0] if potential_answers else "Unable to determine"
        
        return {
            "answer": answer,
            "confidence": 0.3,
            "reasoning": "Fallback extraction due to LLM failure",
            "raw_response": agent_results[:100],
            "validation_passed": False
        } 