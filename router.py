#!/usr/bin/env python3
"""
Router Agent for GAIA Question Classification
Analyzes questions and routes them to appropriate specialized agents
"""

import re
import logging
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse

from agents.state import GAIAAgentState, QuestionType, AgentRole, AgentResult
from models.qwen_client import QwenClient, ModelTier

logger = logging.getLogger(__name__)

class RouterAgent:
    """
    Router agent that classifies GAIA questions and determines processing strategy
    """
    
    def __init__(self, llm_client: QwenClient):
        self.llm_client = llm_client
        
    def process(self, state: GAIAAgentState) -> GAIAAgentState:
        """
        Enhanced router processing with improved classification and planning
        """
        logger.info("🧭 Router: Starting enhanced multi-phase analysis")
        state.add_processing_step("Router: Enhanced multi-phase question analysis")
        
        try:
            # Enhanced classification
            classification_result = self._classify_question_enhanced(state.question)
            
            state.question_type = classification_result['question_type']
            state.routing_decision = classification_result['reasoning']
            
            # Select agents based on enhanced classification
            agents = self._select_agents_for_type(classification_result)
            state.selected_agents = agents
            
            # Store enhanced analysis for downstream agents
            state.router_analysis = {
                'classification': classification_result,
                'selected_agents': [a.value for a in agents],
                'confidence': classification_result['confidence']
            }
            
            logger.info(f"✅ Enhanced routing: {classification_result['type']} -> {[a.value for a in agents]}")
            
            return state
            
        except Exception as e:
            error_msg = f"Enhanced router analysis failed: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            
            # Fallback to basic routing
            state.question_type = QuestionType.UNKNOWN
            state.selected_agents = [AgentRole.WEB_RESEARCHER, AgentRole.REASONING_AGENT, AgentRole.SYNTHESIZER]
            state.routing_decision = f"Enhanced routing failed, using fallback: {error_msg}"
            
            return state
    
    def route_question(self, state: GAIAAgentState) -> GAIAAgentState:
        """
        Main routing function - analyzes question and determines processing strategy
        """
        logger.info(f"🧭 Router: Analyzing question type and complexity")
        state.add_processing_step("Router: Analyzing question and selecting agents")
        
        try:
            # Analyze question patterns for classification
            question_types, primary_type = self._classify_question_types(state.question, state.file_name)
            state.question_types = question_types
            state.primary_question_type = primary_type
            
            # Use 72B model for complex routing decisions
            llm_classification = self._get_llm_classification(state.question)
            
            # Combine pattern-based and LLM-based classification
            final_types, final_primary = self._combine_classifications_legacy(
                question_types, primary_type, llm_classification
            )
            
            # Update state with final classification
            state.question_types = final_types
            state.primary_question_type = final_primary
            
            # Select agents based on question types
            selected_agents = self._select_agents(final_types, final_primary, state.question)
            state.selected_agents = selected_agents
            
            logger.info(f"✅ Routing complete: {final_primary.value} -> {[a.value for a in selected_agents]}")
            state.add_processing_step(f"Router: Selected agents - {[a.value for a in selected_agents]}")
            
            return state
            
        except Exception as e:
            error_msg = f"Router failed: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            
            # Fallback to web researcher for unknown questions
            state.selected_agents = [AgentRole.WEB_RESEARCHER]
            state.primary_question_type = QuestionType.WEB_RESEARCH
            
            return state
    
    def _classify_question_types(self, question: str, file_name: str = None) -> Tuple[List[QuestionType], QuestionType]:
        """
        Enhanced classification that can detect multiple question types
        Returns: (all_detected_types, primary_type)
        """
        
        question_lower = question.lower()
        detected_types = []
        
        # File processing questions (highest priority when file is present)
        if file_name:
            file_ext = file_name.lower().split('.')[-1] if '.' in file_name else ""
            
            if file_ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg']:
                detected_types.append(QuestionType.FILE_PROCESSING)
            elif file_ext in ['mp3', 'wav', 'ogg', 'flac', 'm4a']:
                detected_types.append(QuestionType.FILE_PROCESSING)
            elif file_ext in ['xlsx', 'xls', 'csv']:
                detected_types.append(QuestionType.FILE_PROCESSING)
            elif file_ext in ['py', 'js', 'java', 'cpp', 'c']:
                detected_types.append(QuestionType.CODE_EXECUTION)
            else:
                detected_types.append(QuestionType.FILE_PROCESSING)
        
        # Enhanced URL-based classification
        url_patterns = {
            QuestionType.WIKIPEDIA: [
                r'wikipedia\.org', r'featured article', r'promoted.*wikipedia',
                r'english wikipedia', r'wiki.*article'
            ],
            QuestionType.YOUTUBE: [
                r'youtube\.com', r'youtu\.be', r'watch\?v=', r'video.*youtube',
                r'https://www\.youtube\.com/watch'
            ]
        }
        
        for question_type, patterns in url_patterns.items():
            if any(re.search(pattern, question_lower) for pattern in patterns):
                detected_types.append(question_type)
        
        # Enhanced content-based classification with better patterns
        classification_patterns = {
            QuestionType.MATHEMATICAL: [
                # Counting/quantity questions
                r'\bhow many\b', r'\bhow much\b', r'\bcount\b', r'\bnumber of\b',
                r'\btotal\b', r'\bsum\b', r'\baverage\b', r'\bmean\b',
                # Calculations
                r'\bcalculate\b', r'\bcompute\b', r'\bsolve\b', 
                # Mathematical operations
                r'\d+\s*[\+\-\*/]\s*\d+', r'\bsquare root\b', r'\bpercentage\b',
                # Table analysis
                r'\btable\b.*\bdefining\b', r'\bgiven.*table\b', r'\boperation table\b',
                # Specific math terms
                r'\bequation\b', r'\bformula\b', r'\bratio\b', r'\bfactorial\b',
                # Economic/statistical
                r'\binterest\b', r'\bcompound\b', r'\bstatistics\b'
            ],
            QuestionType.TEXT_MANIPULATION: [
                # Text operations
                r'\breverse\b', r'\bbackwards\b', r'\bencode\b', r'\bdecode\b',
                r'\btransform\b', r'\bconvert\b', r'\buppercase\b', r'\blowercase\b',
                r'\breplace\b', r'\bextract\b', r'\bopposite\b',
                # Pattern recognition for backwards text
                r'[a-z]+\s+[a-z]+\s+[a-z]+.*\.',  # Potential backwards sentence
                # Specific text manipulation clues
                r'\.rewsna\b', r'\bword.*opposite\b'
            ],
            QuestionType.CODE_EXECUTION: [
                r'\bcode\b', r'\bprogram\b', r'\bscript\b', r'\bfunction\b', r'\balgorithm\b',
                r'\bexecute\b', r'\brun.*code\b', r'\bpython\b', r'\bjavascript\b',
                r'\battached.*code\b', r'\bfinal.*output\b', r'\bnumeric output\b'
            ],
            QuestionType.REASONING: [
                # Logical reasoning
                r'\bwhy\b', r'\bexplain\b', r'\banalyze\b', r'\breasoning\b', r'\blogic\b',
                r'\brelationship\b', r'\bcompare\b', r'\bcontrast\b', r'\bconclusion\b',
                # Complex analysis
                r'\bexamine\b', r'\bidentify\b', r'\bdetermine\b', r'\bassess\b',
                r'\bevaluate\b', r'\binterpret\b'
            ],
            QuestionType.WEB_RESEARCH: [
                # General research
                r'\bsearch\b', r'\bfind.*information\b', r'\bresearch\b', r'\blook up\b',
                r'\bwebsite\b', r'\bonline\b', r'\binternet\b',
                # Who/what/when/where questions
                r'\bwho\s+(?:is|was|are|were|did|does)\b',
                r'\bwhat\s+(?:is|was|are|were)\b', r'\bwhen\s+(?:is|was|did|does)\b',
                r'\bwhere\s+(?:is|was|are|were)\b',
                # Factual queries
                r'\bauthor\b', r'\bpublished\b', r'\bhistory\b', r'\bhistorical\b',
                r'\bcentury\b', r'\byear\b', r'\bbiography\b', r'\bwinner\b',
                # Specific research indicators
                r'\bstudio albums\b', r'\brecipient\b', r'\bcompetition\b', r'\bspecimens\b'
            ]
        }
        
        # Score each category with enhanced scoring
        type_scores = {}
        for question_type, patterns in classification_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, question_lower)
                score += len(matches)
                # Give extra weight to certain patterns
                if question_type == QuestionType.MATHEMATICAL and pattern in [r'\bhow many\b', r'\bhow much\b']:
                    score += 2  # Boost counting questions
                elif question_type == QuestionType.TEXT_MANIPULATION and any(special in pattern for special in ['opposite', 'reverse', 'backwards']):
                    score += 1  # Reduced further to avoid over-weighting
            if score > 0:
                type_scores[question_type] = score
        
        # Special handling for specific question patterns
        
        # Detect backwards/scrambled text (strong indicator) - only for clearly backwards text
        if re.search(r'\.rewsna\b|etirw\b|dnatsrednu\b', question_lower):
            type_scores[QuestionType.TEXT_MANIPULATION] = type_scores.get(QuestionType.TEXT_MANIPULATION, 0) + 3
            
        # Detect code execution patterns (strong indicator)
        if re.search(r'\bfinal.*output\b|\bnumeric.*output\b|\battached.*code\b', question_lower):
            type_scores[QuestionType.CODE_EXECUTION] = type_scores.get(QuestionType.CODE_EXECUTION, 0) + 4
            
        # Detect mathematical operations with numbers (boost mathematical score)
        if re.search(r'\b\d+.*\b(?:studio albums|between|and)\b.*\d+', question_lower):
            type_scores[QuestionType.MATHEMATICAL] = type_scores.get(QuestionType.MATHEMATICAL, 0) + 3
            
        # Detect table/grid operations
        if re.search(r'\btable.*defining.*\*', question_lower) or '|*|' in question:
            type_scores[QuestionType.MATHEMATICAL] = type_scores.get(QuestionType.MATHEMATICAL, 0) + 4
            
        # Multi-step questions that need research AND calculation
        if ('how many' in question_lower or 'how much' in question_lower) and \
           any(term in question_lower for term in ['between', 'from', 'during', 'published', 'released']):
            type_scores[QuestionType.WEB_RESEARCH] = type_scores.get(QuestionType.WEB_RESEARCH, 0) + 3  # Increased from 2
            type_scores[QuestionType.MATHEMATICAL] = type_scores.get(QuestionType.MATHEMATICAL, 0) + 3  # Increased from 2
            
        # Detect factual research questions (boost web research)
        if any(pattern in question_lower for pattern in ['who is', 'who was', 'who did', 'what is', 'when did', 'where', 'which']):
            type_scores[QuestionType.WEB_RESEARCH] = type_scores.get(QuestionType.WEB_RESEARCH, 0) + 2
            
        # Detect image/file references
        if any(term in question_lower for term in ['image', 'picture', 'photo', 'file', 'attached', 'provided']):
            type_scores[QuestionType.FILE_PROCESSING] = type_scores.get(QuestionType.FILE_PROCESSING, 0) + 4  # Increased from 3
            
        # Detect Wikipedia-specific questions
        if any(term in question_lower for term in ['wikipedia', 'featured article', 'english wikipedia']):
            type_scores[QuestionType.WIKIPEDIA] = type_scores.get(QuestionType.WIKIPEDIA, 0) + 4
        
        # Add detected types based on scores
        for qtype, score in type_scores.items():
            if score > 0 and qtype not in detected_types:
                detected_types.append(qtype)
        
        # If no types detected, default to web research
        if not detected_types:
            detected_types.append(QuestionType.WEB_RESEARCH)
        
        # Determine primary type (highest scoring)
        if type_scores:
            primary_type = max(type_scores.keys(), key=lambda t: type_scores[t])
        else:
            primary_type = detected_types[0] if detected_types else QuestionType.WEB_RESEARCH
        
        return detected_types, primary_type
    
    def _assess_complexity(self, question: str) -> str:
        """Assess question complexity with enhanced logic"""
        
        question_lower = question.lower()
        
        # Complex indicators
        complex_indicators = [
            'multi-step', 'multiple', 'several', 'complex', 'detailed',
            'analyze', 'explain why', 'reasoning', 'relationship',
            'compare and contrast', 'comprehensive', 'thorough',
            'between.*and', 'table.*defining', 'attached.*file'
        ]
        
        # Simple indicators  
        simple_indicators = [
            'what is', 'who is', 'when did', 'where is', 'yes or no',
            'true or false', 'simple', 'quick', 'name'
        ]
        
        complex_score = sum(1 for indicator in complex_indicators if re.search(indicator, question_lower))
        simple_score = sum(1 for indicator in simple_indicators if re.search(indicator, question_lower))
        
        # Additional complexity factors
        if len(question) > 200:
            complex_score += 1
        if len(question.split()) > 30:
            complex_score += 1
        if question.count('?') > 1:  # Multiple questions
            complex_score += 1
        if '|' in question and '*' in question:  # Tables
            complex_score += 2
        if re.search(r'\d+.*between.*\d+', question_lower):  # Date ranges
            complex_score += 1
        
        # Determine complexity
        if complex_score >= 3:
            return "complex"
        elif complex_score >= 1 and simple_score == 0:
            return "medium"
        elif simple_score >= 2 and complex_score == 0:
            return "simple"
        else:
            return "medium"
    
    def _select_agents_enhanced(self, question_types: List[QuestionType], primary_type: QuestionType, 
                              has_file: bool, complexity: str) -> List[AgentRole]:
        """
        Enhanced agent selection that can choose multiple agents for complex workflows
        """
        
        agents = []
        
        # Always include synthesizer at the end for final answer compilation
        # (We'll add it at the end to ensure proper ordering)
        
        # Multi-agent selection based on detected question types
        agent_priorities = {
            QuestionType.FILE_PROCESSING: [AgentRole.FILE_PROCESSOR],
            QuestionType.CODE_EXECUTION: [AgentRole.CODE_EXECUTOR],
            QuestionType.WIKIPEDIA: [AgentRole.WEB_RESEARCHER],
            QuestionType.YOUTUBE: [AgentRole.WEB_RESEARCHER],
            QuestionType.WEB_RESEARCH: [AgentRole.WEB_RESEARCHER],
            QuestionType.MATHEMATICAL: [AgentRole.REASONING_AGENT],
            QuestionType.TEXT_MANIPULATION: [AgentRole.REASONING_AGENT],
            QuestionType.REASONING: [AgentRole.REASONING_AGENT]
        }
        
        # Add agents based on all detected question types
        for qtype in question_types:
            if qtype in agent_priorities:
                for agent in agent_priorities[qtype]:
                    if agent not in agents:
                        agents.append(agent)
        
        # Special combinations for multi-step questions
        
        # For CODE_EXECUTION as primary type, prioritize code executor
        if primary_type == QuestionType.CODE_EXECUTION:
            # Ensure code executor is first, followed by any other needed agents
            ordered_agents = []
            if AgentRole.CODE_EXECUTOR not in ordered_agents:
                ordered_agents.append(AgentRole.CODE_EXECUTOR)
            # Add other agents if needed for multi-type questions
            for agent in agents:
                if agent != AgentRole.CODE_EXECUTOR and agent not in ordered_agents:
                    ordered_agents.append(agent)
            agents = ordered_agents
        
        # Research + Math combinations (e.g., "How many albums between 2000-2009?")
        elif (QuestionType.WEB_RESEARCH in question_types and QuestionType.MATHEMATICAL in question_types):
            # Ensure proper order: Research first, then math
            ordered_agents = []
            if AgentRole.WEB_RESEARCHER not in ordered_agents:
                ordered_agents.append(AgentRole.WEB_RESEARCHER)
            if AgentRole.REASONING_AGENT not in ordered_agents:
                ordered_agents.append(AgentRole.REASONING_AGENT)
            agents = ordered_agents
        
        # File + Analysis combinations
        elif has_file and len(question_types) > 1:
            # File processing should come first
            ordered_agents = []
            if AgentRole.FILE_PROCESSOR not in ordered_agents:
                ordered_agents.append(AgentRole.FILE_PROCESSOR)
            # Then add other agents
            for agent in agents:
                if agent != AgentRole.FILE_PROCESSOR and agent not in ordered_agents:
                    ordered_agents.append(agent)
            agents = ordered_agents
        
        # For complex questions, add reasoning if not already present
        if complexity == "complex" and AgentRole.REASONING_AGENT not in agents:
            agents.append(AgentRole.REASONING_AGENT)
        
        # Fallback for unknown/unclear questions - use multiple agents
        if primary_type == QuestionType.UNKNOWN or not agents:
            agents = [AgentRole.WEB_RESEARCHER, AgentRole.REASONING_AGENT]
        
        # Always add synthesizer at the end
        agents.append(AgentRole.SYNTHESIZER)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_agents = []
        for agent in agents:
            if agent not in seen:
                seen.add(agent)
                unique_agents.append(agent)
                
        return unique_agents
    
    def _estimate_cost(self, complexity: str, agents: List[AgentRole]) -> float:
        """Estimate processing cost based on complexity and agents"""
        
        base_costs = {
            "simple": 0.005,   # Router model mostly
            "medium": 0.015,   # Mix of router and main
            "complex": 0.035   # Include complex model usage
        }
        
        base_cost = base_costs.get(complexity, 0.015)
        
        # Additional cost per agent (more agents = more processing)
        agent_cost = len(agents) * 0.008
        
        return base_cost + agent_cost
    
    def _get_routing_reasoning(self, primary_type: QuestionType, complexity: str, 
                             agents: List[AgentRole], all_types: List[QuestionType]) -> str:
        """Generate human-readable reasoning for routing decision"""
        
        reasons = []
        
        # Primary type reasoning
        type_descriptions = {
            QuestionType.WIKIPEDIA: "References Wikipedia content",
            QuestionType.YOUTUBE: "Involves YouTube video analysis", 
            QuestionType.FILE_PROCESSING: "Requires file processing",
            QuestionType.MATHEMATICAL: "Involves mathematical computation/counting",
            QuestionType.CODE_EXECUTION: "Requires code execution",
            QuestionType.TEXT_MANIPULATION: "Involves text transformation/manipulation",
            QuestionType.REASONING: "Requires logical reasoning/analysis",
            QuestionType.WEB_RESEARCH: "Needs web research for factual information"
        }
        
        if primary_type in type_descriptions:
            reasons.append(type_descriptions[primary_type])
        
        # Multi-type questions
        if len(all_types) > 1:
            other_types = [t for t in all_types if t != primary_type]
            reasons.append(f"Also involves: {', '.join([t.value for t in other_types])}")
        
        # Complexity reasoning
        if complexity == "complex":
            reasons.append("Complex multi-step reasoning required")
        elif complexity == "simple":
            reasons.append("Straightforward question")
        
        # Agent workflow reasoning
        agent_names = [agent.value.replace('_', ' ') for agent in agents]
        if len(agents) > 2:  # More than synthesizer + one agent
            reasons.append(f"Multi-agent workflow: {' → '.join(agent_names)}")
        else:
            reasons.append(f"Single-agent workflow: {', '.join(agent_names)}")
        
        return "; ".join(reasons)
    
    def _llm_enhanced_routing(self, state: GAIAAgentState) -> GAIAAgentState:
        """Use LLM for enhanced routing analysis of complex/unknown questions"""
        
        prompt = f"""
        Analyze this GAIA benchmark question and provide routing guidance:
        
        Question: {state.question}
        File attached: {state.file_name if state.file_name else "None"}
        Detected types: {state.routing_decision.get('all_types', [])}
        Primary classification: {state.question_type.value}
        Current complexity: {state.complexity_assessment}
        Selected agents: {[a.value for a in state.selected_agents]}
        
        Does this question need:
        1. Web research to find factual information?
        2. Mathematical calculation or counting?
        3. Text manipulation or decoding?
        4. File processing or analysis?
        5. Logical reasoning or analysis?
        
        Should the agent selection be adjusted? If so, provide specific recommendations.
        Keep response concise and focused on routing decisions.
        """
        
        try:
            # Use main model (32B) for better routing decisions
            tier = ModelTier.MAIN
            result = self.llm_client.generate(prompt, tier=tier, max_tokens=300)
            
            if result.success:
                state.add_processing_step("Router: Enhanced with LLM analysis")
                state.routing_decision["llm_analysis"] = result.response
                logger.info("✅ LLM enhanced routing completed")
            else:
                state.add_error(f"LLM routing enhancement failed: {result.error}")
                
        except Exception as e:
            state.add_error(f"LLM routing error: {str(e)}")
            logger.error(f"LLM routing failed: {e}")
        
        return state 

    def _get_llm_classification(self, question: str) -> Dict[str, Any]:
        """Use 72B model for intelligent question classification"""
        
        classification_prompt = f"""
Analyze this GAIA benchmark question and classify it for agent routing.

Question: {question}

Determine:
1. Primary question type (mathematical, text_manipulation, web_research, file_processing, reasoning, factual_lookup)
2. Required capabilities (research, calculation, file_analysis, text_processing, logical_reasoning)
3. Complexity level (simple, moderate, complex)
4. Expected answer type (number, text, yes_no, name, location, list)

Provide your analysis in this format:
PRIMARY_TYPE: [type]
CAPABILITIES: [cap1, cap2, cap3]
COMPLEXITY: [level]
ANSWER_TYPE: [type]
REASONING: [brief explanation]
"""
        
        # Use 72B model for classification
        result = self.llm_client.generate(
            classification_prompt,
            tier=ModelTier.COMPLEX,  # 72B model for better reasoning
            max_tokens=200
        )
        
        if result.success:
            return self._parse_llm_classification(result.response)
        else:
            logger.warning("LLM classification failed, using pattern-based only")
            return {"primary_type": "unknown", "capabilities": [], "complexity": "moderate"}
    
    def _parse_llm_classification(self, response: str) -> Dict[str, Any]:
        """Parse LLM classification response"""
        
        parsed = {
            "primary_type": "unknown",
            "capabilities": [],
            "complexity": "moderate",
            "answer_type": "text",
            "reasoning": ""
        }
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("PRIMARY_TYPE:"):
                parsed["primary_type"] = line.split(":", 1)[1].strip().lower()
            elif line.startswith("CAPABILITIES:"):
                caps_text = line.split(":", 1)[1].strip()
                parsed["capabilities"] = [c.strip().lower() for c in caps_text.split(",")]
            elif line.startswith("COMPLEXITY:"):
                parsed["complexity"] = line.split(":", 1)[1].strip().lower()
            elif line.startswith("ANSWER_TYPE:"):
                parsed["answer_type"] = line.split(":", 1)[1].strip().lower()
            elif line.startswith("REASONING:"):
                parsed["reasoning"] = line.split(":", 1)[1].strip()
        
        return parsed
    
    def _combine_classifications_legacy(self, pattern_types: List[QuestionType], pattern_primary: QuestionType,
                                       llm_classification: Dict[str, Any]) -> Tuple[List[QuestionType], QuestionType]:
        """Combine pattern-based and LLM-based classifications"""
        
        # Map LLM classification to our enum types
        llm_type_mapping = {
            "mathematical": QuestionType.MATHEMATICAL,
            "text_manipulation": QuestionType.TEXT_MANIPULATION,
            "web_research": QuestionType.WEB_RESEARCH,
            "file_processing": QuestionType.FILE_PROCESSING,
            "reasoning": QuestionType.REASONING,
            "factual_lookup": QuestionType.WEB_RESEARCH,
            "code_execution": QuestionType.CODE_EXECUTION
        }
        
        llm_primary = llm_type_mapping.get(llm_classification["primary_type"], QuestionType.WEB_RESEARCH)
        
        # Combine types - prefer LLM classification for primary, merge for secondary types
        combined_types = list(pattern_types)
        if llm_primary not in combined_types:
            combined_types.insert(0, llm_primary)  # Add LLM primary to front
        
        # Use LLM primary if it's confident, otherwise stick with pattern
        if llm_classification["complexity"] in ["complex", "moderate"] and llm_primary != QuestionType.WEB_RESEARCH:
            final_primary = llm_primary
        else:
            final_primary = pattern_primary
        
        logger.info(f"🤖 Combined classification: Pattern={pattern_primary.value}, LLM={llm_primary.value}, Final={final_primary.value}")
        
        return combined_types, final_primary
    
    def _select_agents_for_type(self, classification_result: Dict[str, Any]) -> List[AgentRole]:
        """Select appropriate agents based on enhanced classification"""
        
        question_type = classification_result['type']
        confidence = classification_result['confidence']
        
        # Agent selection based on question type
        if question_type == 'mathematical':
            agents = [AgentRole.WEB_RESEARCHER, AgentRole.REASONING_AGENT]
        elif question_type == 'text_manipulation':
            agents = [AgentRole.REASONING_AGENT]
        elif question_type == 'file_processing':
            agents = [AgentRole.FILE_PROCESSOR, AgentRole.REASONING_AGENT]
        elif question_type == 'web_research':
            agents = [AgentRole.WEB_RESEARCHER]
        elif question_type == 'reasoning':
            agents = [AgentRole.REASONING_AGENT, AgentRole.WEB_RESEARCHER]
        elif question_type == 'factual_lookup':
            agents = [AgentRole.WEB_RESEARCHER]
        else:
            # General questions - try multiple approaches
            agents = [AgentRole.WEB_RESEARCHER, AgentRole.REASONING_AGENT]
        
        # Always add synthesizer
        agents.append(AgentRole.SYNTHESIZER)
        
        # If confidence is low, add more agents for better coverage
        if confidence < 0.6:
            if AgentRole.WEB_RESEARCHER not in agents:
                agents.insert(-1, AgentRole.WEB_RESEARCHER)  # Insert before synthesizer
        
        return agents 

    def _analyze_question_structure(self, question: str) -> Dict[str, Any]:
        """
        Phase 1: Analyze the structural components of the question
        """
        structure = {
            'type': 'unknown',
            'complexity': 'medium',
            'components': [],
            'data_sources': [],
            'temporal_aspects': [],
            'quantitative_aspects': []
        }
        
        question_lower = question.lower()
        
        # Identify question type
        if any(word in question_lower for word in ['how many', 'count', 'number of', 'quantity']):
            structure['type'] = 'quantitative'
        elif any(word in question_lower for word in ['who is', 'who was', 'who did', 'name of']):
            structure['type'] = 'identification'
        elif any(word in question_lower for word in ['where', 'location', 'place']):
            structure['type'] = 'location'
        elif any(word in question_lower for word in ['when', 'date', 'time', 'year']):
            structure['type'] = 'temporal'
        elif any(word in question_lower for word in ['what is', 'define', 'explain']):
            structure['type'] = 'definition'
        elif any(word in question_lower for word in ['calculate', 'compute', 'solve']):
            structure['type'] = 'mathematical'
        elif any(word in question_lower for word in ['compare', 'difference', 'versus']):
            structure['type'] = 'comparison'
        elif 'file' in question_lower or 'attached' in question_lower:
            structure['type'] = 'file_analysis'
        else:
            structure['type'] = 'complex_reasoning'
        
        # Identify data sources needed
        if any(term in question_lower for term in ['wikipedia', 'article', 'page']):
            structure['data_sources'].append('wikipedia')
        if any(term in question_lower for term in ['video', 'youtube', 'watch']):
            structure['data_sources'].append('video')
        if any(term in question_lower for term in ['file', 'attached', 'document']):
            structure['data_sources'].append('file')
        if any(term in question_lower for term in ['recent', 'latest', 'current', '2024', '2025']):
            structure['data_sources'].append('web_search')
        
        # Identify temporal aspects
        import re
        years = re.findall(r'\b(?:19|20)\d{2}\b', question)
        dates = re.findall(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b', question_lower)
        structure['temporal_aspects'] = years + dates
        
        # Identify quantitative aspects
        quantities = re.findall(r'\b\d+(?:\.\d+)?\b', question)
        structure['quantitative_aspects'] = quantities
        
        # Assess complexity
        complexity_factors = [
            len(question.split()) > 25,  # Long question
            len(structure['data_sources']) > 1,  # Multiple sources
            len(structure['temporal_aspects']) > 1,  # Multiple time periods
            'and' in question_lower and 'or' in question_lower,  # Multiple conditions
            question.count('?') > 1,  # Multiple questions
        ]
        
        if sum(complexity_factors) >= 3:
            structure['complexity'] = 'high'
        elif sum(complexity_factors) >= 1:
            structure['complexity'] = 'medium'
        else:
            structure['complexity'] = 'low'
        
        return structure
    
    def _analyze_information_needs(self, question: str, structural: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 2: Analyze what specific information is needed to answer the question
        """
        needs = {
            'primary_need': 'factual_lookup',
            'information_types': [],
            'precision_required': 'medium',
            'verification_needed': False,
            'synthesis_complexity': 'simple'
        }
        
        # Determine primary information need
        if structural['type'] == 'quantitative':
            needs['primary_need'] = 'numerical_data'
            needs['precision_required'] = 'high'
        elif structural['type'] == 'identification':
            needs['primary_need'] = 'entity_identification'
        elif structural['type'] == 'mathematical':
            needs['primary_need'] = 'computation'
            needs['precision_required'] = 'high'
        elif structural['type'] == 'file_analysis':
            needs['primary_need'] = 'file_processing'
        elif structural['type'] == 'comparison':
            needs['primary_need'] = 'comparative_analysis'
            needs['verification_needed'] = True
        else:
            needs['primary_need'] = 'factual_lookup'
        
        # Determine information types needed
        if 'wikipedia' in structural['data_sources']:
            needs['information_types'].append('encyclopedic')
        if 'video' in structural['data_sources']:
            needs['information_types'].append('multimedia_content')
        if 'web_search' in structural['data_sources']:
            needs['information_types'].append('current_information')
        if 'file' in structural['data_sources']:
            needs['information_types'].append('document_analysis')
        
        # Assess synthesis complexity
        if structural['complexity'] == 'high' or len(needs['information_types']) > 2:
            needs['synthesis_complexity'] = 'complex'
        elif len(needs['information_types']) > 1:
            needs['synthesis_complexity'] = 'moderate'
        
        return needs
    
    def _plan_execution_strategy(self, question: str, structural: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 3: Plan the execution strategy based on analysis
        """
        strategy = {
            'approach': 'sequential',
            'parallel_possible': False,
            'iterative_refinement': False,
            'fallback_needed': True,
            'verification_steps': []
        }
        
        # Determine approach
        if requirements['primary_need'] == 'file_processing':
            strategy['approach'] = 'file_first'
        elif requirements['primary_need'] == 'computation':
            strategy['approach'] = 'reasoning_first'
        elif len(requirements['information_types']) > 2:
            strategy['approach'] = 'multi_source'
            strategy['parallel_possible'] = True
        elif 'current_information' in requirements['information_types']:
            strategy['approach'] = 'web_first'
        else:
            strategy['approach'] = 'knowledge_first'
        
        # Determine if iterative refinement is needed
        if (structural['complexity'] == 'high' or 
            requirements['precision_required'] == 'high' or
            requirements['verification_needed']):
            strategy['iterative_refinement'] = True
        
        # Plan verification steps
        if requirements['verification_needed']:
            strategy['verification_steps'] = ['cross_reference', 'consistency_check']
        if requirements['precision_required'] == 'high':
            strategy['verification_steps'].append('precision_validation')
        
        return strategy
    
    def _select_agent_sequence(self, strategy: Dict[str, Any], requirements: Dict[str, Any]) -> List[str]:
        """
        Phase 4: Select the optimal sequence of agents based on strategy
        """
        sequence = []
        
        # Base sequence based on approach
        if strategy['approach'] == 'file_first':
            sequence = ['file_processor', 'reasoning_agent', 'synthesizer']
        elif strategy['approach'] == 'reasoning_first':
            sequence = ['reasoning_agent', 'web_researcher', 'synthesizer']
        elif strategy['approach'] == 'web_first':
            sequence = ['web_researcher', 'reasoning_agent', 'synthesizer']
        elif strategy['approach'] == 'knowledge_first':
            sequence = ['web_researcher', 'reasoning_agent', 'synthesizer']
        elif strategy['approach'] == 'multi_source':
            sequence = ['web_researcher', 'file_processor', 'reasoning_agent', 'synthesizer']
        else:  # sequential
            sequence = ['reasoning_agent', 'web_researcher', 'synthesizer']
        
        # Add verification agents if needed
        if strategy['iterative_refinement']:
            # Insert reasoning agent before synthesizer for verification
            if 'reasoning_agent' in sequence:
                sequence.remove('reasoning_agent')
            sequence.insert(-1, 'reasoning_agent')  # Before synthesizer
        
        # Ensure synthesizer is always last
        if 'synthesizer' in sequence:
            sequence.remove('synthesizer')
        sequence.append('synthesizer')
        
        return sequence 

    def _classify_question_enhanced(self, question: str) -> Dict[str, Any]:
        """Enhanced question classification using better pattern matching and LLM analysis"""
        
        question_lower = question.lower()
        
        # Enhanced pattern classification
        pattern_classification = self._classify_by_enhanced_patterns(question_lower, question)
        
        # LLM-based classification for complex cases
        llm_classification = self._classify_with_llm(question)
        
        # Combine both approaches
        final_classification = self._combine_classifications(pattern_classification, llm_classification, question)
        
        logger.info(f"🤖 Enhanced classification: Pattern={pattern_classification['type']}, LLM={llm_classification['type']}, Final={final_classification['type']}")
        
        return final_classification
    
    def _classify_by_enhanced_patterns(self, question_lower: str, original_question: str) -> Dict[str, Any]:
        """Enhanced pattern-based classification with better accuracy"""
        
        # Mathematical/counting questions (high confidence patterns)
        mathematical_patterns = [
            r'\bhow many\b',
            r'\bcount\b.*\b(of|the)\b',
            r'\bnumber of\b',
            r'\btotal\b.*\b(of|number)\b',
            r'\bcalculate\b',
            r'\bsum\b.*\bof\b',
            r'\bhow much\b',
            r'\bquantity\b'
        ]
        
        if any(re.search(pattern, question_lower) for pattern in mathematical_patterns):
            # Check for temporal constraints
            temporal_indicators = ['between', 'from', 'during', 'in', r'\b(19|20)\d{2}\b']
            has_temporal = any(re.search(indicator, question_lower) for indicator in temporal_indicators)
            
            return {
                'type': 'mathematical',
                'confidence': 0.9,
                'subtype': 'temporal_counting' if has_temporal else 'general_counting',
                'reasoning': 'Strong mathematical/counting indicators found'
            }
        
        # Text manipulation questions
        text_manipulation_patterns = [
            r'\bopposite\b',
            r'\breverse\b',
            r'\bbackwards\b',
            r'\bdecode\b',
            r'\btranslate\b',
            r'\bconvert\b',
            r'\.rewsna',  # Common in reversed text questions
            r'\bcipher\b',
            r'\bencrypt\b'
        ]
        
        if any(re.search(pattern, question_lower) for pattern in text_manipulation_patterns):
            return {
                'type': 'text_manipulation',
                'confidence': 0.85,
                'subtype': 'text_processing',
                'reasoning': 'Text manipulation patterns detected'
            }
        
        # File/code processing questions  
        file_patterns = [
            r'\battached\b.*\b(file|image|document|excel|csv|python|code)\b',
            r'\bfile\b.*\b(contains|attached|uploaded)\b',
            r'\b(image|photo|picture)\b.*\b(shows|contains|attached)\b',
            r'\bcode\b.*\b(attached|file|script)\b',
            r'\bspreadsheet\b',
            r'\b\.py\b|\b\.csv\b|\b\.xlsx\b|\b\.png\b|\b\.jpg\b'
        ]
        
        if any(re.search(pattern, question_lower) for pattern in file_patterns):
            return {
                'type': 'file_processing',
                'confidence': 0.9,
                'subtype': 'file_analysis',
                'reasoning': 'File processing indicators found'
            }
        
        # Web research questions (specific indicators)
        web_research_patterns = [
            r'\bwikipedia\b.*\barticle\b',
            r'\bfeatured article\b',
            r'\bpromoted\b.*\b(in|during)\b.*\b(19|20)\d{2}\b',
            r'\bnominated\b.*\bby\b',
            r'\byoutube\b.*\bvideo\b',
            r'\bwatch\?v=\b',
            r'\bhttps?://\b',
            r'\bwebsite\b|\burl\b'
        ]
        
        if any(re.search(pattern, question_lower) for pattern in web_research_patterns):
            return {
                'type': 'web_research',
                'confidence': 0.8,
                'subtype': 'specific_lookup',
                'reasoning': 'Web-specific content indicators found'
            }
        
        # Reasoning/analysis questions
        reasoning_patterns = [
            r'\banalyze\b|\banalysis\b',
            r'\bcompare\b|\bcomparison\b',
            r'\bexplain\b|\bexplanation\b',
            r'\bwhy\b.*\b(is|are|was|were|do|does|did)\b',
            r'\bhow\b.*\b(does|do|did|can|could|would)\b',
            r'\bwhat.*difference\b',
            r'\bwhat.*relationship\b'
        ]
        
        if any(re.search(pattern, question_lower) for pattern in reasoning_patterns):
            return {
                'type': 'reasoning',
                'confidence': 0.7,
                'subtype': 'analytical_reasoning',
                'reasoning': 'Reasoning/analysis patterns detected'
            }
        
        # General factual questions
        factual_patterns = [
            r'\bwho\b.*\b(is|was|are|were)\b',
            r'\bwhat\b.*\b(is|was|are|were)\b',
            r'\bwhen\b.*\b(did|was|were|is|are)\b',
            r'\bwhere\b.*\b(is|was|are|were)\b',
            r'\bwhich\b.*\b(is|was|are|were)\b'
        ]
        
        if any(re.search(pattern, question_lower) for pattern in factual_patterns):
            return {
                'type': 'factual_lookup',
                'confidence': 0.6,
                'subtype': 'general_factual',
                'reasoning': 'General factual question patterns'
            }
        
        # Default classification
        return {
            'type': 'general',
            'confidence': 0.4,
            'subtype': 'unclassified',
            'reasoning': 'No specific patterns matched'
        }
    
    def _classify_with_llm(self, question: str) -> Dict[str, Any]:
        """LLM-based classification for complex questions"""
        
        classification_prompt = f"""
        Analyze this question and classify it into one of these categories:

        Categories:
        - mathematical: Questions asking for counts, calculations, quantities
        - text_manipulation: Questions involving text reversal, encoding, word puzzles
        - file_processing: Questions about attached files, images, code, data
        - web_research: Questions requiring web search, Wikipedia lookup, current information
        - reasoning: Questions requiring analysis, comparison, logical deduction
        - factual_lookup: Simple fact-based questions about people, places, events

        Question: {question}

        Respond with just the category name and a brief reason (max 10 words).
        Format: category_name: reason

        Classification:"""
        
        try:
            llm_result = self.llm_client.generate(
                classification_prompt,
                tier=ModelTier.ROUTER,  # Use fast model for classification
                max_tokens=50
            )
            
            if llm_result.success:
                response = llm_result.response.strip().lower()
                
                # Parse the response
                if ':' in response:
                    category, reason = response.split(':', 1)
                    category = category.strip()
                    reason = reason.strip()
                else:
                    category = response.split()[0] if response.split() else 'general'
                    reason = 'llm classification'
                
                # Validate category
                valid_categories = ['mathematical', 'text_manipulation', 'file_processing', 'web_research', 'reasoning', 'factual_lookup']
                if category not in valid_categories:
                    category = 'general'
                
                return {
                    'type': category,
                    'confidence': 0.7,
                    'reasoning': f'LLM: {reason}'
                }
            else:
                return {
                    'type': 'general',
                    'confidence': 0.3,
                    'reasoning': 'LLM classification failed'
                }
                
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return {
                'type': 'general',
                'confidence': 0.3,
                'reasoning': 'LLM classification error'
            }
    
    def _select_agents(self, question_types: List[QuestionType], primary_type: QuestionType, question: str) -> List[AgentRole]:
        """Select agents based on combined classification"""
        
        agents = []
        
        # Primary agent based on primary type
        primary_agent_map = {
            QuestionType.MATHEMATICAL: AgentRole.REASONING_AGENT,
            QuestionType.TEXT_MANIPULATION: AgentRole.REASONING_AGENT,
            QuestionType.WEB_RESEARCH: AgentRole.WEB_RESEARCHER,
            QuestionType.FILE_PROCESSING: AgentRole.FILE_PROCESSOR,
            QuestionType.REASONING: AgentRole.REASONING_AGENT,
            QuestionType.CODE_EXECUTION: AgentRole.CODE_EXECUTOR
        }
        
        primary_agent = primary_agent_map.get(primary_type, AgentRole.WEB_RESEARCHER)
        if primary_agent not in agents:
            agents.append(primary_agent)
        
        # Add secondary agents based on all detected types
        for qtype in question_types:
            if qtype != primary_type:  # Don't duplicate primary
                secondary_agent = primary_agent_map.get(qtype)
                if secondary_agent and secondary_agent not in agents:
                    agents.append(secondary_agent)
        
        # Always add synthesizer at the end
        if AgentRole.SYNTHESIZER not in agents:
            agents.append(AgentRole.SYNTHESIZER)
        
        return agents 

    def _combine_classifications(self, pattern_result: Dict[str, Any], llm_result: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Combine pattern and LLM classifications for final decision"""
        
        pattern_type = pattern_result['type']
        pattern_confidence = pattern_result['confidence']
        llm_type = llm_result['type']
        llm_confidence = llm_result['confidence']
        
        # If pattern matching has high confidence, trust it
        if pattern_confidence >= 0.8:
            final_type = pattern_type
            final_confidence = pattern_confidence
            reasoning = f"High confidence pattern match: {pattern_result['reasoning']}"
        
        # If both agree, boost confidence
        elif pattern_type == llm_type:
            final_type = pattern_type
            final_confidence = min(0.95, (pattern_confidence + llm_confidence) / 2 + 0.1)
            reasoning = f"Pattern and LLM agree: {pattern_type}"
        
        # If they disagree, use the one with higher confidence
        elif pattern_confidence > llm_confidence:
            final_type = pattern_type
            final_confidence = pattern_confidence * 0.9  # Slight penalty for disagreement
            reasoning = f"Pattern-based: {pattern_result['reasoning']}"
        else:
            final_type = llm_type
            final_confidence = llm_confidence * 0.9  # Slight penalty for disagreement
            reasoning = f"LLM-based: {llm_result['reasoning']}"
        
        # Map to question types
        type_mapping = {
            'mathematical': QuestionType.MATHEMATICAL,
            'text_manipulation': QuestionType.TEXT_MANIPULATION,
            'file_processing': QuestionType.FILE_PROCESSING,
            'web_research': QuestionType.WEB_RESEARCH,
            'reasoning': QuestionType.REASONING,
            'factual_lookup': QuestionType.WEB_RESEARCH,  # Map to web_research
            'general': QuestionType.UNKNOWN
        }
        
        question_type = type_mapping.get(final_type, QuestionType.UNKNOWN)
        
        return {
            'type': final_type,
            'question_type': question_type,
            'confidence': final_confidence,
            'reasoning': reasoning,
            'pattern_result': pattern_result,
            'llm_result': llm_result
        } 