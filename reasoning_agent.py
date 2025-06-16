#!/usr/bin/env python3
"""
Reasoning Agent for GAIA Agent System
Handles mathematical, logical, and analytical reasoning questions
"""

import re
import logging
from typing import Dict, List, Optional, Any, Union

from agents.state import GAIAAgentState, AgentRole, AgentResult, ToolResult
from models.qwen_client import QwenClient, ModelTier
from tools.calculator import CalculatorTool

logger = logging.getLogger(__name__)

class ReasoningAgent:
    """
    Specialized agent for reasoning tasks
    Handles mathematical calculations, logical deduction, and analytical problems
    """
    
    def __init__(self, llm_client: QwenClient):
        self.llm_client = llm_client
        self.calculator = CalculatorTool()
        
    def process(self, state: GAIAAgentState) -> GAIAAgentState:
        """
        Process reasoning questions using mathematical and logical analysis
        """
        logger.info(f"Reasoning agent processing: {state.question[:100]}...")
        state.add_processing_step("Reasoning Agent: Starting analysis")
        
        try:
            # Determine reasoning strategy
            strategy = self._determine_reasoning_strategy(state.question)
            state.add_processing_step(f"Reasoning Agent: Strategy = {strategy}")
            
            # Execute reasoning with enhanced error handling
            result = None
            try:
                # Execute reasoning based on strategy
                if strategy == "mathematical":
                    result = self._process_mathematical(state)
                elif strategy == "statistical":
                    result = self._process_statistical(state)
                elif strategy == "unit_conversion":
                    result = self._process_unit_conversion(state)
                elif strategy == "logical_deduction":
                    result = self._process_logical_deduction(state)
                elif strategy == "pattern_analysis":
                    result = self._process_pattern_analysis(state)
                elif strategy == "step_by_step":
                    result = self._process_step_by_step(state)
                elif strategy == "general_reasoning":
                    result = self._process_general_reasoning(state)
                else:
                    result = self._process_general_reasoning(state)
                    
            except Exception as strategy_error:
                logger.warning(f"Strategy {strategy} failed: {strategy_error}, trying fallback")
                # Try fallback reasoning
                try:
                    result = self._process_fallback_reasoning(state, strategy, str(strategy_error))
                except Exception as fallback_error:
                    logger.error(f"Fallback reasoning also failed: {fallback_error}")
                    result = self._create_graceful_failure_result(state, f"Reasoning failed: {fallback_error}")
            
            # Ensure we always have a valid result
            if not result or not isinstance(result, AgentResult):
                result = self._create_graceful_failure_result(state, "No reasoning results available")
            
            # Add result to state
            state.add_agent_result(result)
            state.add_processing_step(f"Reasoning Agent: Completed with confidence {result.confidence:.2f}")
            
            return state
            
        except Exception as e:
            error_msg = f"Reasoning failed: {str(e)}"
            state.add_error(error_msg)
            logger.error(error_msg)
            
            # Create failure result but ensure system continues
            failure_result = AgentResult(
                agent_role=AgentRole.REASONING_AGENT,
                success=False,
                result=f"Processing encountered difficulties: Reasoning failed",
                confidence=0.1,  # Very low but not zero to allow synthesis
                reasoning=f"Exception during reasoning: {str(e)}",
                tools_used=[],
                model_used="error",
                processing_time=0.0,
                cost_estimate=0.0
            )
            state.add_agent_result(failure_result)
            return state
    
    def _determine_reasoning_strategy(self, question: str) -> str:
        """Determine the best reasoning strategy for the question"""
        
        question_lower = question.lower()
        
        # Mathematical calculations
        math_indicators = [
            'calculate', 'compute', 'solve', 'equation', 'formula',
            'multiply', 'divide', 'add', 'subtract', 'sum', 'total',
            'percentage', 'percent', 'ratio', 'proportion'
        ]
        if any(indicator in question_lower for indicator in math_indicators):
            return "mathematical"
        
        # Statistical analysis
        stats_indicators = [
            'average', 'mean', 'median', 'mode', 'standard deviation',
            'variance', 'correlation', 'distribution', 'sample'
        ]
        if any(indicator in question_lower for indicator in stats_indicators):
            return "statistical"
        
        # Unit conversions
        unit_indicators = [
            'convert', 'to', 'from', 'meter', 'feet', 'celsius', 'fahrenheit',
            'gram', 'pound', 'liter', 'gallon', 'hour', 'minute'
        ]
        conversion_pattern = r'\d+\s*\w+\s+to\s+\w+'
        if (any(indicator in question_lower for indicator in unit_indicators) or 
            re.search(conversion_pattern, question_lower)):
            return "unit_conversion"
        
        # Logical deduction
        logic_indicators = [
            'if', 'then', 'therefore', 'because', 'since', 'given that',
            'prove', 'demonstrate', 'conclude', 'infer', 'deduce'
        ]
        if any(indicator in question_lower for indicator in logic_indicators):
            return "logical_deduction"
        
        # Pattern analysis
        pattern_indicators = [
            'pattern', 'sequence', 'series', 'next', 'continues',
            'follows', 'trend', 'progression'
        ]
        if any(indicator in question_lower for indicator in pattern_indicators):
            return "pattern_analysis"
        
        # Step-by-step problems
        step_indicators = [
            'step', 'process', 'procedure', 'method', 'approach',
            'how to', 'explain how', 'show how'
        ]
        if any(indicator in question_lower for indicator in step_indicators):
            return "step_by_step"
        
        # Default to general reasoning
        return "general_reasoning"
    
    def _process_mathematical(self, state: GAIAAgentState) -> AgentResult:
        """Process mathematical calculation questions"""
        
        logger.info("Processing mathematical calculation")
        
        # Extract mathematical expressions from the question
        expressions = self._extract_mathematical_expressions(state.question)
        
        if expressions:
            # Try to solve with calculator
            calc_results = []
            for expr in expressions:
                calc_result = self.calculator.execute(expr)
                calc_results.append(calc_result)
            
            # Use LLM to interpret results and provide answer
            if calc_results and any(r.success for r in calc_results):
                return self._analyze_calculation_results(state, calc_results)
            else:
                # Fallback to LLM-only mathematical reasoning
                return self._llm_mathematical_reasoning(state)
        else:
            # No clear expressions, use LLM reasoning
            return self._llm_mathematical_reasoning(state)
    
    def _process_statistical(self, state: GAIAAgentState) -> AgentResult:
        """Process statistical analysis questions"""
        
        logger.info("Processing statistical analysis")
        
        # Extract numerical data from question
        numbers = self._extract_numbers(state.question)
        
        if len(numbers) >= 2:
            # Perform statistical calculations
            stats_data = {"operation": "statistics", "data": numbers}
            calc_result = self.calculator.execute(stats_data)
            
            if calc_result.success:
                return self._analyze_statistical_results(state, calc_result, numbers)
            else:
                return self._llm_statistical_reasoning(state, numbers)
        else:
            # Use LLM for statistical reasoning without clear data
            return self._llm_statistical_reasoning(state, [])
    
    def _process_unit_conversion(self, state: GAIAAgentState) -> AgentResult:
        """Process unit conversion questions"""
        
        logger.info("Processing unit conversion")
        
        # Extract conversion details
        conversion_info = self._extract_conversion_info(state.question)
        
        if conversion_info:
            value, from_unit, to_unit = conversion_info
            conversion_data = {
                "operation": "convert",
                "value": value,
                "from_unit": from_unit,
                "to_unit": to_unit
            }
            
            calc_result = self.calculator.execute(conversion_data)
            
            if calc_result.success:
                return self._analyze_conversion_results(state, calc_result, conversion_info)
            else:
                return self._llm_conversion_reasoning(state, conversion_info)
        else:
            # Use LLM for conversion reasoning
            return self._llm_conversion_reasoning(state, None)
    
    def _process_logical_deduction(self, state: GAIAAgentState) -> AgentResult:
        """Process logical reasoning and deduction questions"""
        
        logger.info("Processing logical deduction")
        
        # Use complex model for logical reasoning
        reasoning_prompt = f"""
        Please solve this logical reasoning problem step by step:
        
        Question: {state.question}
        
        Approach this systematically:
        1. Identify the given information
        2. Identify what needs to be determined
        3. Apply logical rules and deduction
        4. State your conclusion clearly
        
        Please provide a clear, logical answer.
        """
        
        model_tier = ModelTier.COMPLEX  # Use best model for complex reasoning
        llm_result = self.llm_client.generate(reasoning_prompt, tier=model_tier, max_tokens=600)
        
        if llm_result.success:
            return AgentResult(
                agent_role=AgentRole.REASONING_AGENT,
                success=True,
                result=llm_result.response,
                confidence=0.80,
                reasoning="Applied logical deduction and reasoning",
                model_used=llm_result.model_used,
                processing_time=llm_result.response_time,
                cost_estimate=llm_result.cost_estimate
            )
        else:
            return self._create_failure_result("Logical reasoning failed")
    
    def _process_pattern_analysis(self, state: GAIAAgentState) -> AgentResult:
        """Process pattern recognition and analysis questions"""
        
        logger.info("Processing pattern analysis")
        
        # Extract sequences or patterns from question
        numbers = self._extract_numbers(state.question)
        
        pattern_prompt = f"""
        Analyze this pattern or sequence problem:
        
        Question: {state.question}
        
        {"Numbers found: " + str(numbers) if numbers else ""}
        
        Please:
        1. Identify the pattern or rule
        2. Explain the logic
        3. Provide the answer
        
        Be systematic and show your reasoning.
        """
        
        model_tier = ModelTier.COMPLEX  # Use 72B model for pattern analysis
        llm_result = self.llm_client.generate(pattern_prompt, tier=model_tier, max_tokens=500)
        
        if llm_result.success:
            confidence = 0.85 if numbers else 0.75  # Higher confidence with numerical data
            return AgentResult(
                agent_role=AgentRole.REASONING_AGENT,
                success=True,
                result=llm_result.response,
                confidence=confidence,
                reasoning="Analyzed patterns and sequences with 72B model",
                model_used=llm_result.model_used,
                processing_time=llm_result.response_time,
                cost_estimate=llm_result.cost_estimate
            )
        else:
            return self._create_failure_result("Pattern analysis failed")
    
    def _process_step_by_step(self, state: GAIAAgentState) -> AgentResult:
        """Process questions requiring step-by-step explanation"""
        
        logger.info("Processing step-by-step reasoning")
        
        step_prompt = f"""
        Please solve this problem with a clear step-by-step approach:
        
        Question: {state.question}
        
        Structure your response as:
        Step 1: [First step and reasoning]
        Step 2: [Second step and reasoning]
        ...
        Final Answer: [Clear conclusion]
        
        Be thorough and explain each step.
        """
        
        model_tier = ModelTier.COMPLEX  # Use 72B model for step-by-step reasoning
        llm_result = self.llm_client.generate(step_prompt, tier=model_tier, max_tokens=600)
        
        if llm_result.success:
            return AgentResult(
                agent_role=AgentRole.REASONING_AGENT,
                success=True,
                result=llm_result.response,
                confidence=0.85,  # Higher confidence with 72B model
                reasoning="Provided step-by-step solution with 72B model",
                model_used=llm_result.model_used,
                processing_time=llm_result.response_time,
                cost_estimate=llm_result.cost_estimate
            )
        else:
            return self._create_failure_result("Step-by-step reasoning failed")
    
    def _process_general_reasoning(self, state: GAIAAgentState) -> AgentResult:
        """Process general reasoning questions"""
        
        logger.info("Processing general reasoning")
        
        reasoning_prompt = f"""
        Please analyze and answer this reasoning question:
        
        Question: {state.question}
        
        Think through this carefully and provide a well-reasoned answer.
        Consider all aspects of the question and explain your reasoning.
        """
        
        model_tier = ModelTier.COMPLEX  # Use 72B model for general reasoning
        llm_result = self.llm_client.generate(reasoning_prompt, tier=model_tier, max_tokens=500)
        
        if llm_result.success:
            return AgentResult(
                agent_role=AgentRole.REASONING_AGENT,
                success=True,
                result=llm_result.response,
                confidence=0.80,  # Higher confidence with 72B model
                reasoning="Applied general reasoning and analysis with 72B model",
                model_used=llm_result.model_used,
                processing_time=llm_result.response_time,
                cost_estimate=llm_result.cost_estimate
            )
        else:
            return self._create_failure_result("General reasoning failed")
    
    def _extract_mathematical_expressions(self, question: str) -> List[str]:
        """Extract mathematical expressions from question text"""
        expressions = []
        
        # Look for explicit mathematical expressions
        math_patterns = [
            r'\d+\s*[\+\-\*/]\s*\d+',
            r'\d+\s*\^\s*\d+',
            r'sqrt\(\d+\)',
            r'\d+\s*%',
            r'\d+\s*factorial',
        ]
        
        for pattern in math_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            expressions.extend(matches)
        
        return expressions
    
    def _extract_numbers(self, question: str) -> List[float]:
        """Extract numerical values from question text"""
        numbers = []
        
        # Find all numbers (integers and floats)
        number_pattern = r'[-+]?\d*\.?\d+'
        matches = re.findall(number_pattern, question)
        
        for match in matches:
            try:
                if '.' in match:
                    numbers.append(float(match))
                else:
                    numbers.append(float(int(match)))
            except ValueError:
                continue
        
        return numbers
    
    def _extract_conversion_info(self, question: str) -> Optional[tuple]:
        """Extract unit conversion information from question"""
        
        # Pattern for "X unit to unit" format
        conversion_pattern = r'(\d+(?:\.\d+)?)\s*(\w+)\s+to\s+(\w+)'
        match = re.search(conversion_pattern, question.lower())
        
        if match:
            value, from_unit, to_unit = match.groups()
            return float(value), from_unit, to_unit
        
        return None
    
    def _analyze_calculation_results(self, state: GAIAAgentState, calc_results: List) -> AgentResult:
        """Analyze calculator results and provide answer"""
        
        successful_results = [r for r in calc_results if r.success]
        
        if successful_results:
            result_summaries = []
            total_cost = 0.0
            total_time = 0.0
            
            for calc_result in successful_results:
                if calc_result.result.get('success'):
                    calc_data = calc_result.result['calculation']
                    result_summaries.append(f"{calc_data['expression']} = {calc_data['result']}")
                    total_cost += calc_result.result.get('cost_estimate', 0)
                    total_time += calc_result.execution_time
            
            analysis_prompt = f"""
            Based on these calculations, please answer the original question:
            
            Question: {state.question}
            
            Calculation Results:
            {chr(10).join(result_summaries)}
            
            Please provide a direct answer incorporating these calculations.
            """
            
            llm_result = self.llm_client.generate(analysis_prompt, tier=ModelTier.COMPLEX, max_tokens=400)
            
            if llm_result.success:
                return AgentResult(
                    agent_role=AgentRole.REASONING_AGENT,
                    success=True,
                    result=llm_result.response,
                    confidence=0.85,
                    reasoning="Performed calculations and analyzed results",
                    tools_used=[ToolResult(
                        tool_name="calculator",
                        success=True,
                        result=result_summaries,
                        execution_time=total_time
                    )],
                    model_used=llm_result.model_used,
                    processing_time=total_time + llm_result.response_time,
                    cost_estimate=total_cost + llm_result.cost_estimate
                )
        
        return self._create_failure_result("Mathematical calculations failed")
    
    def _analyze_statistical_results(self, state: GAIAAgentState, calc_result, numbers: List[float]) -> AgentResult:
        """Analyze statistical calculation results"""
        
        if calc_result.success and calc_result.result.get('success'):
            stats = calc_result.result['statistics']
            
            analysis_prompt = f"""
            Based on this statistical analysis, please answer the question:
            
            Question: {state.question}
            
            Data: {numbers}
            Statistical Results:
            - Count: {stats.get('count')}
            - Mean: {stats.get('mean')}
            - Median: {stats.get('median')}
            - Min: {stats.get('min')}
            - Max: {stats.get('max')}
            - Standard Deviation: {stats.get('stdev', 'N/A')}
            
            Please provide a direct answer based on this statistical analysis.
            """
            
            llm_result = self.llm_client.generate(analysis_prompt, tier=ModelTier.COMPLEX, max_tokens=400)
            
            if llm_result.success:
                return AgentResult(
                    agent_role=AgentRole.REASONING_AGENT,
                    success=True,
                    result=llm_result.response,
                    confidence=0.85,
                    reasoning="Performed statistical analysis",
                    tools_used=[ToolResult(
                        tool_name="calculator",
                        success=True,
                        result=stats,
                        execution_time=calc_result.execution_time
                    )],
                    model_used=llm_result.model_used,
                    processing_time=calc_result.execution_time + llm_result.response_time,
                    cost_estimate=llm_result.cost_estimate
                )
        
        return self._create_failure_result("Statistical analysis failed")
    
    def _analyze_conversion_results(self, state: GAIAAgentState, calc_result, conversion_info: tuple) -> AgentResult:
        """Analyze unit conversion results"""
        
        if calc_result.success and calc_result.result.get('success'):
            conversion_data = calc_result.result['conversion']
            value, from_unit, to_unit = conversion_info
            
            analysis_prompt = f"""
            Based on this unit conversion, please answer the question:
            
            Question: {state.question}
            
            Conversion: {value} {from_unit} = {conversion_data['result']} {conversion_data['units']}
            
            Please provide a direct answer incorporating this conversion.
            """
            
            llm_result = self.llm_client.generate(analysis_prompt, tier=ModelTier.COMPLEX, max_tokens=400)
            
            if llm_result.success:
                return AgentResult(
                    agent_role=AgentRole.REASONING_AGENT,
                    success=True,
                    result=llm_result.response,
                    confidence=0.90,
                    reasoning="Performed unit conversion",
                    tools_used=[ToolResult(
                        tool_name="calculator",
                        success=True,
                        result=conversion_data,
                        execution_time=calc_result.execution_time
                    )],
                    model_used=llm_result.model_used,
                    processing_time=calc_result.execution_time + llm_result.response_time,
                    cost_estimate=llm_result.cost_estimate
                )
        
        return self._create_failure_result("Unit conversion failed")
    
    def _llm_mathematical_reasoning(self, state: GAIAAgentState) -> AgentResult:
        """Fallback to LLM-only mathematical reasoning"""
        
        math_prompt = f"""
        Please solve this mathematical problem:
        
        Question: {state.question}
        
        Show your mathematical reasoning and calculations step by step.
        Provide a clear numerical answer.
        """
        
        model_tier = ModelTier.COMPLEX  # Use 72B model for mathematical reasoning
        llm_result = self.llm_client.generate(math_prompt, tier=model_tier, max_tokens=500)
        
        if llm_result.success:
            return AgentResult(
                agent_role=AgentRole.REASONING_AGENT,
                success=True,
                result=llm_result.response,
                confidence=0.70,
                reasoning="Applied mathematical reasoning (LLM-only)",
                model_used=llm_result.model_used,
                processing_time=llm_result.response_time,
                cost_estimate=llm_result.cost_estimate
            )
        else:
            return self._create_failure_result("Mathematical reasoning failed")
    
    def _llm_statistical_reasoning(self, state: GAIAAgentState, numbers: List[float]) -> AgentResult:
        """Fallback to LLM-only statistical reasoning"""
        
        stats_prompt = f"""
        Please analyze this statistical problem:
        
        Question: {state.question}
        
        {"Numbers identified: " + str(numbers) if numbers else ""}
        
        Apply statistical reasoning and provide a clear answer.
        """
        
        model_tier = ModelTier.COMPLEX
        llm_result = self.llm_client.generate(stats_prompt, tier=model_tier, max_tokens=400)
        
        if llm_result.success:
            return AgentResult(
                agent_role=AgentRole.REASONING_AGENT,
                success=True,
                result=llm_result.response,
                confidence=0.65,
                reasoning="Applied statistical reasoning (LLM-only)",
                model_used=llm_result.model_used,
                processing_time=llm_result.response_time,
                cost_estimate=llm_result.cost_estimate
            )
        else:
            return self._create_failure_result("Statistical reasoning failed")
    
    def _llm_conversion_reasoning(self, state: GAIAAgentState, conversion_info: Optional[tuple]) -> AgentResult:
        """Fallback to LLM-only conversion reasoning"""
        
        conversion_prompt = f"""
        Please solve this unit conversion problem:
        
        Question: {state.question}
        
        {f"Conversion detected: {conversion_info}" if conversion_info else ""}
        
        Apply conversion reasoning and provide a clear answer.
        """
        
        model_tier = ModelTier.COMPLEX
        llm_result = self.llm_client.generate(conversion_prompt, tier=model_tier, max_tokens=300)
        
        if llm_result.success:
            return AgentResult(
                agent_role=AgentRole.REASONING_AGENT,
                success=True,
                result=llm_result.response,
                confidence=0.65,
                reasoning="Applied conversion reasoning (LLM-only)",
                model_used=llm_result.model_used,
                processing_time=llm_result.response_time,
                cost_estimate=llm_result.cost_estimate
            )
        else:
            return self._create_failure_result("Conversion reasoning failed")
    
    def _create_failure_result(self, error_message: str) -> AgentResult:
        """Create a failure result"""
        return AgentResult(
            agent_role=AgentRole.REASONING_AGENT,
            success=False,
            result=error_message,
            confidence=0.0,
            reasoning=error_message,
            model_used="error",
            processing_time=0.0,
            cost_estimate=0.0
        )
    
    def _process_fallback_reasoning(self, state: GAIAAgentState, original_strategy: str, error_msg: str) -> AgentResult:
        """Enhanced fallback reasoning when primary strategy fails"""
        
        logger.info(f"Executing fallback reasoning after {original_strategy} failure")
        
        # Try simple general reasoning as fallback
        try:
            fallback_prompt = f"""
            Please answer this question using basic reasoning:
            
            Question: {state.question}
            
            Note: Original strategy '{original_strategy}' failed with: {error_msg}
            
            Please provide the best answer you can using simple analysis and reasoning.
            Focus on extracting key information from the question and providing a helpful response.
            """
            
            # Use main model for fallback
            llm_result = self.llm_client.generate(fallback_prompt, tier=ModelTier.COMPLEX, max_tokens=400)
            
            if llm_result.success:
                return AgentResult(
                    agent_role=AgentRole.REASONING_AGENT,
                    success=True,
                    result=llm_result.response,
                    confidence=0.3,  # Lower confidence for fallback
                    reasoning=f"Fallback reasoning after {original_strategy} failed: {error_msg}",
                    tools_used=[],
                    model_used=llm_result.model_used,
                    processing_time=llm_result.response_time,
                    cost_estimate=llm_result.cost_estimate
                )
            else:
                raise Exception(f"Fallback LLM reasoning failed: {llm_result.error}")
                
        except Exception as fallback_error:
            logger.error(f"Fallback reasoning failed: {fallback_error}")
            return self._create_graceful_failure_result(state, f"All reasoning methods failed: {fallback_error}")
    
    def _create_graceful_failure_result(self, state: GAIAAgentState, error_context: str) -> AgentResult:
        """Create a graceful failure result that allows the system to continue"""
        
        # Try to extract any useful information from the question itself
        question_analysis = f"Question analysis: {state.question[:200]}"
        
        return AgentResult(
            agent_role=AgentRole.REASONING_AGENT,
            success=False,
            result=f"Processing encountered difficulties: {error_context}",
            confidence=0.1,
            reasoning=f"Reasoning failed: {error_context}",
            tools_used=[],
            model_used="none",
            processing_time=0.0,
            cost_estimate=0.0
        ) 