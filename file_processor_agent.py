#!/usr/bin/env python3
"""
File Processor Agent for GAIA Agent System
Handles file-based questions with intelligent processing strategies
"""

import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from agents.state import GAIAAgentState, AgentRole, AgentResult, ToolResult
from models.qwen_client import QwenClient, ModelTier
from tools.file_processor import FileProcessorTool
from tools.calculator import CalculatorTool

logger = logging.getLogger(__name__)

class FileProcessorAgent:
    """
    Specialized agent for file processing tasks
    Handles images, audio, CSV/Excel, Python code, and other file types
    """
    
    def __init__(self, llm_client: QwenClient):
        self.llm_client = llm_client
        self.file_processor = FileProcessorTool()
        self.calculator = CalculatorTool()  # For data analysis
        
    def process(self, state: GAIAAgentState) -> GAIAAgentState:
        """
        Process file-based questions using file analysis tools
        """
        logger.info(f"File processor processing: {state.question[:100]}...")
        state.add_processing_step("File Processor: Starting file analysis")
        
        try:
            # Check if file exists
            if not state.file_path or not os.path.exists(state.file_path):
                error_msg = f"File not found: {state.file_path}"
                state.add_error(error_msg)
                result = self._create_failure_result(error_msg)
                state.add_agent_result(result)
                return state
            
            # Determine processing strategy
            strategy = self._determine_processing_strategy(state.question, state.file_path)
            state.add_processing_step(f"File Processor: Strategy = {strategy}")
            
            # Execute processing based on strategy
            if strategy == "image_analysis":
                result = self._process_image(state)
            elif strategy == "data_analysis":
                result = self._process_data_file(state)
            elif strategy == "code_analysis":
                result = self._process_code_file(state)
            elif strategy == "audio_analysis":
                result = self._process_audio_file(state)
            elif strategy == "text_analysis":
                result = self._process_text_file(state)
            else:
                result = self._process_generic_file(state)
            
            # Add result to state
            state.add_agent_result(result)
            state.add_processing_step(f"File Processor: Completed with confidence {result.confidence:.2f}")
            
            return state
            
        except Exception as e:
            error_msg = f"File processing failed: {str(e)}"
            state.add_error(error_msg)
            logger.error(error_msg)
            
            # Create failure result
            failure_result = self._create_failure_result(error_msg)
            state.add_agent_result(failure_result)
            return state
    
    def _determine_processing_strategy(self, question: str, file_path: str) -> str:
        """Determine the best processing strategy based on file type and question"""
        
        file_extension = Path(file_path).suffix.lower()
        question_lower = question.lower()
        
        # Image file analysis
        if file_extension in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}:
            return "image_analysis"
        
        # Audio file analysis
        if file_extension in {'.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac'}:
            return "audio_analysis"
        
        # Data file analysis
        if file_extension in {'.csv', '.xlsx', '.xls', '.json'}:
            return "data_analysis"
        
        # Code file analysis
        if file_extension in {'.py', '.js', '.java', '.cpp', '.c', '.html', '.css'}:
            return "code_analysis"
        
        # Text file analysis
        if file_extension in {'.txt', '.md', '.rst'}:
            return "text_analysis"
        
        # Default to generic processing
        return "generic_analysis"
    
    def _process_image(self, state: GAIAAgentState) -> AgentResult:
        """Process image files and answer questions about them"""
        
        logger.info(f"Processing image: {state.file_path}")
        
        # Analyze image with file processor
        file_result = self.file_processor.execute(state.file_path)
        
        if file_result.success and file_result.result.get('success'):
            file_data = file_result.result['result']
            
            # Create analysis prompt based on image metadata and question
            analysis_prompt = f"""
            Based on this image analysis, please answer the following question:
            
            Question: {state.question}
            
            Image Information:
            - File: {file_data.get('file_path', '')}
            - Type: {file_data.get('file_type', '')}
            - Content Description: {file_data.get('content', '')}
            - Metadata: {file_data.get('metadata', {})}
            
            Please provide a direct answer based on the image analysis.
            If the question asks about specific details that cannot be determined from the metadata alone, 
            please indicate what information is available and what would require visual analysis.
            """
            
            # Use main model for image analysis
            model_tier = ModelTier.MAIN
            llm_result = self.llm_client.generate(analysis_prompt, tier=model_tier, max_tokens=400)
            
            if llm_result.success:
                confidence = 0.75  # Good confidence for image metadata analysis
                return AgentResult(
                    agent_role=AgentRole.FILE_PROCESSOR,
                    success=True,
                    result=llm_result.response,
                    confidence=confidence,
                    reasoning="Analyzed image metadata and properties",
                    tools_used=[ToolResult(
                        tool_name="file_processor",
                        success=True,
                        result=file_data,
                        execution_time=file_result.execution_time
                    )],
                    model_used=llm_result.model_used,
                    processing_time=file_result.execution_time + llm_result.response_time,
                    cost_estimate=llm_result.cost_estimate
                )
            else:
                # Fallback to metadata description
                return AgentResult(
                    agent_role=AgentRole.FILE_PROCESSOR,
                    success=True,
                    result=file_data.get('content', 'Image analyzed'),
                    confidence=0.60,
                    reasoning="Image processed but analysis failed",
                    tools_used=[ToolResult(
                        tool_name="file_processor",
                        success=True,
                        result=file_data,
                        execution_time=file_result.execution_time
                    )],
                    model_used="fallback",
                    processing_time=file_result.execution_time,
                    cost_estimate=0.0
                )
        else:
            return self._create_failure_result("Image processing failed")
    
    def _process_data_file(self, state: GAIAAgentState) -> AgentResult:
        """Process CSV/Excel files and perform data analysis"""
        
        logger.info(f"Processing data file: {state.file_path}")
        
        # Analyze data file
        file_result = self.file_processor.execute(state.file_path)
        
        if file_result.success and file_result.result.get('success'):
            file_data = file_result.result['result']
            metadata = file_data.get('metadata', {})
            content = file_data.get('content', {})
            
            # Check if question requires calculations
            question_lower = state.question.lower()
            needs_calculation = any(term in question_lower for term in [
                'calculate', 'sum', 'total', 'average', 'mean', 'count',
                'maximum', 'minimum', 'how many', 'what is the'
            ])
            
            if needs_calculation and 'sample_data' in content:
                return self._perform_data_calculations(state, file_data, file_result)
            else:
                return self._analyze_data_structure(state, file_data, file_result)
        else:
            return self._create_failure_result("Data file processing failed")
    
    def _perform_data_calculations(self, state: GAIAAgentState, file_data: Dict, file_result: ToolResult) -> AgentResult:
        """Perform calculations on data file content"""
        
        metadata = file_data.get('metadata', {})
        content = file_data.get('content', {})
        
        # Extract data for calculations
        sample_data = content.get('sample_data', [])
        
        # Use LLM to determine what calculations to perform
        calculation_prompt = f"""
        Based on this data file and question, determine what calculations are needed:
        
        Question: {state.question}
        
        Data Structure:
        - Columns: {metadata.get('columns', [])}
        - Rows: {metadata.get('row_count', 0)}
        - Sample Data: {sample_data[:3]}  # First 3 rows
        
        Please specify what calculations should be performed and on which columns.
        Respond with specific calculation instructions.
        """
        
        llm_result = self.llm_client.generate(calculation_prompt, tier=ModelTier.MAIN, max_tokens=200)
        
        if llm_result.success:
            # For now, provide data summary with LLM analysis
            analysis_prompt = f"""
            Based on this data analysis, please answer the question:
            
            Question: {state.question}
            
            Data Summary:
            - File: {metadata.get('shape', [])} (rows x columns)
            - Columns: {metadata.get('columns', [])}
            - Numeric columns: {metadata.get('numeric_columns', [])}
            - Statistics: {metadata.get('numeric_stats', {})}
            - Sample data: {sample_data}
            
            Calculation guidance: {llm_result.response}
            
            Please provide the answer based on the data.
            """
            
            analysis_result = self.llm_client.generate(analysis_prompt, tier=ModelTier.MAIN, max_tokens=400)
            
            if analysis_result.success:
                return AgentResult(
                    agent_role=AgentRole.FILE_PROCESSOR,
                    success=True,
                    result=analysis_result.response,
                    confidence=0.80,
                    reasoning="Performed data analysis and calculations",
                    tools_used=[file_result],
                    model_used=analysis_result.model_used,
                    processing_time=file_result.execution_time + llm_result.response_time + analysis_result.response_time,
                    cost_estimate=llm_result.cost_estimate + analysis_result.cost_estimate
                )
        
        # Fallback to basic data summary
        return self._analyze_data_structure(state, file_data, file_result)
    
    def _analyze_data_structure(self, state: GAIAAgentState, file_data: Dict, file_result: ToolResult) -> AgentResult:
        """Analyze data file structure and content"""
        
        metadata = file_data.get('metadata', {})
        content = file_data.get('content', {})
        
        analysis_prompt = f"""
        Based on this data file analysis, please answer the question:
        
        Question: {state.question}
        
        Data File Information:
        - Structure: {metadata.get('shape', [])} (rows x columns)
        - Columns: {metadata.get('columns', [])}
        - Data types: {metadata.get('data_types', {})}
        - Description: {content.get('description', '')}
        - Sample data: {content.get('sample_data', [])}
        
        Please provide a direct answer based on the data structure and content.
        """
        
        model_tier = ModelTier.MAIN
        llm_result = self.llm_client.generate(analysis_prompt, tier=model_tier, max_tokens=400)
        
        if llm_result.success:
            return AgentResult(
                agent_role=AgentRole.FILE_PROCESSOR,
                success=True,
                result=llm_result.response,
                confidence=0.75,
                reasoning="Analyzed data file structure and content",
                tools_used=[file_result],
                model_used=llm_result.model_used,
                processing_time=file_result.execution_time + llm_result.response_time,
                cost_estimate=llm_result.cost_estimate
            )
        else:
            return AgentResult(
                agent_role=AgentRole.FILE_PROCESSOR,
                success=True,
                result=content.get('description', 'Data file analyzed'),
                confidence=0.60,
                reasoning="Data file processed but analysis failed",
                tools_used=[file_result],
                model_used="fallback",
                processing_time=file_result.execution_time,
                cost_estimate=0.0
            )
    
    def _process_code_file(self, state: GAIAAgentState) -> AgentResult:
        """Process code files and analyze their content"""
        
        logger.info(f"Processing code file: {state.file_path}")
        
        # Analyze code file
        file_result = self.file_processor.execute(state.file_path)
        
        if file_result.success and file_result.result.get('success'):
            file_data = file_result.result['result']
            metadata = file_data.get('metadata', {})
            content = file_data.get('content', {})
            
            analysis_prompt = f"""
            Based on this code analysis, please answer the question:
            
            Question: {state.question}
            
            Code File Information:
            - Type: {file_data.get('file_type', '')}
            - Description: {content.get('description', '')}
            - Metadata: {metadata}
            - Code snippet: {content.get('code_snippet', '')}
            
            Please analyze the code and provide a direct answer.
            """
            
            model_tier = ModelTier.MAIN
            llm_result = self.llm_client.generate(analysis_prompt, tier=model_tier, max_tokens=500)
            
            if llm_result.success:
                return AgentResult(
                    agent_role=AgentRole.FILE_PROCESSOR,
                    success=True,
                    result=llm_result.response,
                    confidence=0.80,
                    reasoning="Analyzed code structure and content",
                    tools_used=[ToolResult(
                        tool_name="file_processor",
                        success=True,
                        result=file_data,
                        execution_time=file_result.execution_time
                    )],
                    model_used=llm_result.model_used,
                    processing_time=file_result.execution_time + llm_result.response_time,
                    cost_estimate=llm_result.cost_estimate
                )
            else:
                return AgentResult(
                    agent_role=AgentRole.FILE_PROCESSOR,
                    success=True,
                    result=content.get('description', 'Code file analyzed'),
                    confidence=0.60,
                    reasoning="Code file processed but analysis failed",
                    tools_used=[ToolResult(
                        tool_name="file_processor",
                        success=True,
                        result=file_data,
                        execution_time=file_result.execution_time
                    )],
                    model_used="fallback",
                    processing_time=file_result.execution_time,
                    cost_estimate=0.0
                )
        else:
            return self._create_failure_result("Code file processing failed")
    
    def _process_audio_file(self, state: GAIAAgentState) -> AgentResult:
        """Process audio files (basic metadata for now)"""
        
        logger.info(f"Processing audio file: {state.file_path}")
        
        # Analyze audio file
        file_result = self.file_processor.execute(state.file_path)
        
        if file_result.success and file_result.result.get('success'):
            file_data = file_result.result['result']
            
            analysis_prompt = f"""
            Based on this audio file information, please answer the question:
            
            Question: {state.question}
            
            Audio File Information:
            - Content: {file_data.get('content', '')}
            - Metadata: {file_data.get('metadata', {})}
            
            Please provide an answer based on the available audio file information.
            Note: Full audio transcription is not currently available, but file metadata is provided.
            """
            
            model_tier = ModelTier.ROUTER  # Use lighter model for basic audio metadata
            llm_result = self.llm_client.generate(analysis_prompt, tier=model_tier, max_tokens=300)
            
            if llm_result.success:
                return AgentResult(
                    agent_role=AgentRole.FILE_PROCESSOR,
                    success=True,
                    result=llm_result.response,
                    confidence=0.50,  # Lower confidence due to limited audio processing
                    reasoning="Analyzed audio file metadata (transcription not available)",
                    tools_used=[ToolResult(
                        tool_name="file_processor",
                        success=True,
                        result=file_data,
                        execution_time=file_result.execution_time
                    )],
                    model_used=llm_result.model_used,
                    processing_time=file_result.execution_time + llm_result.response_time,
                    cost_estimate=llm_result.cost_estimate
                )
        
        return self._create_failure_result("Audio file processing not fully supported")
    
    def _process_text_file(self, state: GAIAAgentState) -> AgentResult:
        """Process text files and analyze their content"""
        
        logger.info(f"Processing text file: {state.file_path}")
        
        # Analyze text file
        file_result = self.file_processor.execute(state.file_path)
        
        if file_result.success and file_result.result.get('success'):
            file_data = file_result.result['result']
            content = file_data.get('content', {})
            
            analysis_prompt = f"""
            Based on this text file content, please answer the question:
            
            Question: {state.question}
            
            Text Content:
            {content.get('text', '')[:2000]}...
            
            File Statistics:
            - Word count: {file_data.get('metadata', {}).get('word_count', 0)}
            - Line count: {file_data.get('metadata', {}).get('line_count', 0)}
            
            Please analyze the text and provide a direct answer.
            """
            
            model_tier = ModelTier.MAIN
            llm_result = self.llm_client.generate(analysis_prompt, tier=model_tier, max_tokens=400)
            
            if llm_result.success:
                return AgentResult(
                    agent_role=AgentRole.FILE_PROCESSOR,
                    success=True,
                    result=llm_result.response,
                    confidence=0.85,
                    reasoning="Analyzed text file content",
                    tools_used=[ToolResult(
                        tool_name="file_processor",
                        success=True,
                        result=file_data,
                        execution_time=file_result.execution_time
                    )],
                    model_used=llm_result.model_used,
                    processing_time=file_result.execution_time + llm_result.response_time,
                    cost_estimate=llm_result.cost_estimate
                )
        
        return self._create_failure_result("Text file processing failed")
    
    def _process_generic_file(self, state: GAIAAgentState) -> AgentResult:
        """Process unknown file types with generic analysis"""
        
        logger.info(f"Processing generic file: {state.file_path}")
        
        # Try generic file processing
        file_result = self.file_processor.execute(state.file_path)
        
        if file_result.success:
            file_data = file_result.result
            
            # Create basic response about file
            basic_info = f"File analyzed: {state.file_path}. "
            if file_data.get('success'):
                basic_info += f"File type: {file_data.get('result', {}).get('file_type', 'unknown')}. "
                basic_info += "Generic file analysis completed."
            else:
                basic_info += f"Analysis result: {file_data.get('message', 'Processing completed')}"
            
            return AgentResult(
                agent_role=AgentRole.FILE_PROCESSOR,
                success=True,
                result=basic_info,
                confidence=0.40,
                reasoning="Generic file processing attempted",
                tools_used=[ToolResult(
                    tool_name="file_processor",
                    success=True,
                    result=file_data,
                    execution_time=file_result.execution_time
                )],
                model_used="basic",
                processing_time=file_result.execution_time,
                cost_estimate=0.0
            )
        else:
            return self._create_failure_result("Generic file processing failed")
    
    def _create_failure_result(self, error_message: str) -> AgentResult:
        """Create a failure result"""
        return AgentResult(
            agent_role=AgentRole.FILE_PROCESSOR,
            success=False,
            result=error_message,
            confidence=0.0,
            reasoning=error_message,
            model_used="error",
            processing_time=0.0,
            cost_estimate=0.0
        ) 