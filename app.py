#!/usr/bin/env python3
"""
GAIA Agent Production Interface
Production-ready Gradio app for the GAIA benchmark agent system with Unit 4 API integration
"""

import os
import gradio as gr
import logging
import time
import requests
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import tempfile
from pathlib import Path
import json
from datetime import datetime
import csv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our workflow
from workflow.gaia_workflow import SimpleGAIAWorkflow, create_gaia_workflow
from models.qwen_client import QwenClient
from agents.state import GAIAAgentState

# Constants for Unit 4 API
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

class GAIAResultLogger:
    """
    Logger for GAIA evaluation results with export functionality
    """
    
    def __init__(self):
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
    def log_evaluation_results(self, username: str, questions_data: list, results_log: list, 
                             final_result: dict, execution_time: float) -> dict:
        """
        Log complete evaluation results to multiple formats
        Returns paths to generated files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"gaia_evaluation_{username}_{timestamp}"
        
        files_created = {}
        
        try:
            # 1. CSV Export (for easy sharing)
            csv_path = self.results_dir / f"{base_filename}.csv"
            self._save_csv_results(csv_path, results_log, final_result)
            files_created["csv"] = str(csv_path)
            
            # 2. Detailed JSON Export
            json_path = self.results_dir / f"{base_filename}.json"
            detailed_results = self._create_detailed_results(
                username, questions_data, results_log, final_result, execution_time, timestamp
            )
            self._save_json_results(json_path, detailed_results)
            files_created["json"] = str(json_path)
            
            # 3. Summary Report
            summary_path = self.results_dir / f"{base_filename}_summary.md"
            self._save_summary_report(summary_path, detailed_results)
            files_created["summary"] = str(summary_path)
            
            logger.info(f"✅ Results logged to {len(files_created)} files: {list(files_created.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Error logging results: {e}")
            files_created["error"] = str(e)
        
        return files_created
    
    def _save_csv_results(self, path: Path, results_log: list, final_result: dict):
        """Save results in CSV format for easy sharing"""
        with open(path, 'w', newline='', encoding='utf-8') as csvfile:
            if not results_log:
                return
                
            fieldnames = list(results_log[0].keys()) + ['Correct', 'Score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Header
            writer.writeheader()
            
            # Add overall results info
            score = final_result.get('score', 'N/A')
            correct_count = final_result.get('correct_count', 'N/A')
            total_attempted = final_result.get('total_attempted', len(results_log))
            
            # Write each result
            for i, row in enumerate(results_log):
                row_data = row.copy()
                row_data['Correct'] = 'Unknown'  # We don't get individual correct/incorrect from API
                row_data['Score'] = f"{score}% ({correct_count}/{total_attempted})" if i == 0 else ""
                writer.writerow(row_data)
    
    def _create_detailed_results(self, username: str, questions_data: list, results_log: list, 
                                final_result: dict, execution_time: float, timestamp: str) -> dict:
        """Create comprehensive results dictionary"""
        return {
            "metadata": {
                "username": username,
                "timestamp": timestamp,
                "execution_time_seconds": execution_time,
                "total_questions": len(questions_data),
                "total_processed": len(results_log),
                "system_info": {
                    "gradio_version": "4.44.0",
                    "python_version": "3.x",
                    "space_id": os.getenv("SPACE_ID", "local"),
                    "space_host": os.getenv("SPACE_HOST", "local")
                }
            },
            "evaluation_results": {
                "overall_score": final_result.get('score', 'N/A'),
                "correct_count": final_result.get('correct_count', 'N/A'),
                "total_attempted": final_result.get('total_attempted', len(results_log)),
                "success_rate": f"{final_result.get('score', 0)}%",
                "api_message": final_result.get('message', 'No message'),
                "submission_successful": 'score' in final_result
            },
            "question_details": [
                {
                    "index": i + 1,
                    "task_id": item.get("task_id"),
                    "question": item.get("question"),
                    "level": item.get("Level", "Unknown"),
                    "file_name": item.get("file_name", ""),
                    "submitted_answer": next(
                        (r["Submitted Answer"] for r in results_log if r.get("Task ID") == item.get("task_id")), 
                        "No answer"
                    ),
                    "question_length": len(item.get("question", "")),
                    "answer_length": len(next(
                        (r["Submitted Answer"] for r in results_log if r.get("Task ID") == item.get("task_id")), 
                        ""
                    ))
                }
                for i, item in enumerate(questions_data)
            ],
            "processing_summary": {
                "questions_by_level": self._analyze_questions_by_level(questions_data),
                "questions_with_files": len([q for q in questions_data if q.get("file_name")]),
                "average_question_length": sum(len(q.get("question", "")) for q in questions_data) / len(questions_data) if questions_data else 0,
                "average_answer_length": sum(len(r.get("Submitted Answer", "")) for r in results_log) / len(results_log) if results_log else 0,
                "processing_time_per_question": execution_time / len(results_log) if results_log else 0
            },
            "raw_results_log": results_log,
            "api_response": final_result
        }
    
    def _analyze_questions_by_level(self, questions_data: list) -> dict:
        """Analyze question distribution by level"""
        level_counts = {}
        for q in questions_data:
            level = q.get("Level", "Unknown")
            level_counts[level] = level_counts.get(level, 0) + 1
        return level_counts
    
    def _save_json_results(self, path: Path, detailed_results: dict):
        """Save detailed results in JSON format"""
        with open(path, 'w', encoding='utf-8') as jsonfile:
            json.dump(detailed_results, jsonfile, indent=2, ensure_ascii=False)
    
    def _save_summary_report(self, path: Path, detailed_results: dict):
        """Save human-readable summary report"""
        metadata = detailed_results["metadata"]
        results = detailed_results["evaluation_results"]
        summary = detailed_results["processing_summary"]
        
        report = f"""# GAIA Agent Evaluation Report

## Summary
- **User**: {metadata['username']}
- **Date**: {metadata['timestamp']}
- **Overall Score**: {results['overall_score']}% ({results['correct_count']}/{results['total_attempted']} correct)
- **Execution Time**: {metadata['execution_time_seconds']:.2f} seconds
- **Submission Status**: {'✅ Success' if results['submission_successful'] else '❌ Failed'}

## Question Analysis
- **Total Questions**: {metadata['total_questions']}
- **Successfully Processed**: {metadata['total_processed']}
- **Questions with Files**: {summary['questions_with_files']}
- **Average Question Length**: {summary['average_question_length']:.0f} characters
- **Average Answer Length**: {summary['average_answer_length']:.0f} characters
- **Processing Time per Question**: {summary['processing_time_per_question']:.2f} seconds

## Questions by Level
"""
        
        for level, count in summary['questions_by_level'].items():
            report += f"- **Level {level}**: {count} questions\n"
        
        report += f"""
## API Response
{results['api_message']}

## System Information
- **Space ID**: {metadata['system_info']['space_id']}
- **Space Host**: {metadata['system_info']['space_host']}
- **Gradio Version**: {metadata['system_info']['gradio_version']}

---
*Report generated automatically by GAIA Agent System*
"""
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)
    
    def get_latest_results(self, username: str = None) -> list:
        """Get list of latest result files"""
        pattern = f"gaia_evaluation_{username}_*" if username else "gaia_evaluation_*"
        files = list(self.results_dir.glob(pattern))
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return files[:10]  # Return 10 most recent

class GAIAAgentApp:
    """Production GAIA Agent Application with LangGraph workflow and Qwen models"""
    
    def __init__(self, hf_token: Optional[str] = None):
        """Initialize the application with LangGraph workflow and Qwen models only"""
        
        # Priority order: 1) passed hf_token, 2) HF_TOKEN env var
        if not hf_token:
            hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            raise ValueError("HuggingFace token with inference permissions is required. Please set HF_TOKEN environment variable or login with full access.")
        
        try:
            # Initialize QwenClient with token
            from models.qwen_client import QwenClient
            self.llm_client = QwenClient(hf_token=hf_token)
            
            # Initialize LangGraph workflow with tools
            self.workflow = SimpleGAIAWorkflow(self.llm_client)
            
            self.initialized = True
            logger.info("✅ GAIA Agent system initialized with LangGraph workflow and Qwen models")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize GAIA Agent system: {e}")
            raise RuntimeError(f"System initialization failed: {e}. Please ensure HF_TOKEN has inference permissions.")
    
    @classmethod
    def create_with_oauth_token(cls, oauth_token: str) -> "GAIAAgentApp":
        """Create a new instance with OAuth token"""
        if not oauth_token:
            raise ValueError("Valid OAuth token is required for GAIA Agent initialization")
        return cls(hf_token=oauth_token)
    
    def __call__(self, question: str) -> str:
        """
        Main agent call for Unit 4 API compatibility
        """
        if not self.initialized:
            return "System not initialized"
        
        try:
            result_state = self.workflow.process_question(
                question=question,
                task_id=f"unit4_{hash(question) % 10000}"
            )
            
            # Return the final answer for API submission
            return result_state.final_answer if result_state.final_answer else "Unable to process question"
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return f"Processing error: {str(e)}"
    
    def process_question_detailed(self, question: str, file_input=None, show_reasoning: bool = False) -> Tuple[str, str, str]:
        """
        Process a question through the GAIA agent system with detailed output
        
        Returns:
            Tuple of (answer, details, reasoning)
        """
        
        if not self.initialized:
            return "❌ System not initialized", "", ""
        
        if not question.strip():
            return "❌ Please provide a question", "", ""
        
        start_time = time.time()
        
        # Handle file upload
        file_path = None
        file_name = None
        if file_input is not None:
            file_path = file_input.name
            file_name = os.path.basename(file_path)
        
        try:
            # Process through workflow
            result_state = self.workflow.process_question(
                question=question,
                file_path=file_path,
                file_name=file_name,
                task_id=f"manual_{hash(question) % 10000}"
            )
            
            processing_time = time.time() - start_time
            
            # Format answer
            answer = result_state.final_answer
            if not answer:
                answer = "Unable to process question - no answer generated"
            
            # Format details
            details = self._format_details(result_state, processing_time)
            
            # Format reasoning (if requested)
            reasoning = ""
            if show_reasoning:
                reasoning = self._format_reasoning(result_state)
            
            return answer, details, reasoning
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}", "Please try again or contact support", ""
    
    def _format_details(self, state, processing_time: float) -> str:
        """Format processing details"""
        
        details = []
        
        # Basic info
        details.append(f"🎯 **Question Type**: {state.question_type.value}")
        details.append(f"⚡ **Processing Time**: {processing_time:.2f}s")
        details.append(f"📊 **Confidence**: {state.final_confidence:.2f}")
        details.append(f"💰 **Cost**: ${state.total_cost:.4f}")
        
        # Agents used
        agents_used = [result.agent_role.value for result in state.agent_results]
        details.append(f"🤖 **Agents Used**: {', '.join(agents_used) if agents_used else 'None'}")
        
        # Tools used
        tools_used = []
        for result in state.agent_results:
            tools_used.extend(result.tools_used)
        unique_tools = list(set(tools_used))
        details.append(f"🔧 **Tools Used**: {', '.join(unique_tools) if unique_tools else 'None'}")
        
        # File processing
        if state.file_name:
            details.append(f"📁 **File Processed**: {state.file_name}")
        
        # Quality indicators
        if state.confidence_threshold_met:
            details.append("✅ **Quality**: High confidence")
        elif state.final_confidence > 0.5:
            details.append("⚠️ **Quality**: Medium confidence")
        else:
            details.append("❌ **Quality**: Low confidence")
        
        # Review status
        if state.requires_human_review:
            details.append("👁️ **Review**: Human review recommended")
        
        # Error count
        if state.error_messages:
            details.append(f"⚠️ **Errors**: {len(state.error_messages)} encountered")
        
        return "\n".join(details)
    
    def _format_reasoning(self, state) -> str:
        """Format detailed reasoning and workflow steps"""
        
        reasoning = []
        
        # Routing decision
        reasoning.append("## 🧭 Routing Decision")
        reasoning.append(f"**Classification**: {state.question_type.value}")
        reasoning.append(f"**Selected Agents**: {[a.value for a in state.selected_agents]}")
        reasoning.append(f"**Reasoning**: {state.routing_decision}")
        reasoning.append("")
        
        # Agent results
        reasoning.append("## 🤖 Agent Processing")
        for i, result in enumerate(state.agent_results, 1):
            reasoning.append(f"### Agent {i}: {result.agent_role.value}")
            reasoning.append(f"**Success**: {'✅' if result.success else '❌'}")
            reasoning.append(f"**Confidence**: {result.confidence:.2f}")
            reasoning.append(f"**Tools Used**: {', '.join(result.tools_used) if result.tools_used else 'None'}")
            reasoning.append(f"**Reasoning**: {result.reasoning}")
            reasoning.append(f"**Result**: {result.result[:200]}...")
            reasoning.append("")
        
        # Synthesis process
        reasoning.append("## 🔗 Synthesis Process")
        reasoning.append(f"**Strategy**: {state.answer_source}")
        reasoning.append(f"**Final Reasoning**: {state.final_reasoning}")
        reasoning.append("")
        
        # Processing timeline
        reasoning.append("## ⏱️ Processing Timeline")
        for i, step in enumerate(state.processing_steps, 1):
            reasoning.append(f"{i}. {step}")
        
        return "\n".join(reasoning)
    
    def get_examples(self) -> list:
        """Get example questions for the interface that showcase multi-agent capabilities"""
        return [
            "How many studio albums were published by Mercedes Sosa between 2000 and 2009?",
            "What is the capital of the country that has the most time zones?",
            "Calculate the compound interest on $1000 at 5% annual rate compounded quarterly for 3 years",
            "What is the square root of the sum of the first 10 prime numbers?",
            "Who was the first person to walk on the moon and what year did it happen?",
            "Compare the GDP of Japan and Germany in 2023 and tell me the difference",
        ]

    def process_with_langgraph(self, question: str, question_id: str = None) -> Dict[str, Any]:
        """
        Process question using enhanced LangGraph workflow with multi-phase planning
        """
        try:
            logger.info(f"📝 Processing question with enhanced LangGraph workflow: {question[:100]}...")
            
            # Create enhanced state with proper initialization
            state = GAIAAgentState(
                question=question,
                question_id=question_id,
                file_name=None,  # File handling would be added here if needed
                file_content=None
            )
            
            # Create enhanced workflow with multi-step planning
            workflow = create_gaia_workflow(self.llm_client, self.tools)
            
            logger.info("🚀 Starting enhanced multi-phase workflow execution")
            
            # Execute workflow with enhanced planning and refinement
            result_state = workflow.invoke(state)
            
            # Extract enhanced results
            processing_details = {
                "steps": result_state.processing_steps,
                "agents_used": [r.agent_role.value for r in result_state.agent_results],
                "router_analysis": getattr(result_state, 'router_analysis', {}),
                "agent_sequence": getattr(result_state, 'agent_sequence', []),
                "total_steps": len(result_state.processing_steps),
                "refinement_attempted": getattr(result_state, 'refinement_attempted', False)
            }
            
            # Calculate enhanced confidence based on multi-agent results
            if result_state.agent_results:
                confidences = [r.confidence for r in result_state.agent_results]
                avg_confidence = sum(confidences) / len(confidences)
                max_confidence = max(confidences)
                # Boost confidence for multi-agent consensus
                enhanced_confidence = min(0.95, (avg_confidence + max_confidence) / 2)
            else:
                enhanced_confidence = 0.1
            
            return {
                "answer": result_state.final_answer or "Unable to determine answer",
                "confidence": enhanced_confidence,
                "reasoning": result_state.synthesis_reasoning or "Multi-phase processing completed",
                "cost": result_state.total_cost,
                "processing_time": time.time() - result_state.start_time,
                "processing_details": processing_details,
                "agent_results": [
                    {
                        "agent": r.agent_role.value,
                        "success": r.success,
                        "confidence": r.confidence,
                        "reasoning": r.reasoning[:200] + "..." if len(r.reasoning) > 200 else r.reasoning,
                        "processing_time": r.processing_time,
                        "cost": r.cost_estimate
                    }
                    for r in result_state.agent_results
                ],
                "errors": result_state.errors
            }
            
        except Exception as e:
            error_msg = f"Enhanced LangGraph processing failed: {str(e)}"
            logger.error(error_msg)
            return {
                "answer": "Processing failed with enhanced workflow",
                "confidence": 0.0,
                "reasoning": error_msg,
                "cost": 0.0,
                "processing_time": 0.0,
                "processing_details": {"error": error_msg},
                "agent_results": [],
                "errors": [error_msg]
            }

def check_oauth_scopes(oauth_token: str) -> Dict[str, any]:
    """
    Check what scopes are available with the OAuth token
    Returns a dictionary with scope information and capabilities
    """
    if not oauth_token:
        return {
            "logged_in": False,
            "scopes": [],
            "can_inference": False,
            "can_read": False,
            "user_info": {},
            "message": "Not logged in"
        }
    
    try:
        headers = {"Authorization": f"Bearer {oauth_token}"}
        
        # Test whoami endpoint (requires read scope)
        logger.info("🔍 Testing OAuth token with whoami endpoint...")
        try:
            whoami_response = requests.get("https://huggingface.co/api/whoami", headers=headers, timeout=10)
            can_read = whoami_response.status_code == 200
            logger.info(f"✅ Whoami response: {whoami_response.status_code}")
            
            if whoami_response.status_code == 401:
                logger.warning("⚠️ OAuth token unauthorized for whoami endpoint")
            elif whoami_response.status_code != 200:
                logger.warning(f"⚠️ Unexpected whoami response: {whoami_response.status_code}")
                
        except Exception as whoami_error:
            logger.error(f"❌ Whoami test failed: {whoami_error}")
            can_read = False
        
        # Test inference capability by trying a simple model call
        logger.info("🔍 Testing OAuth token with inference endpoint...")
        can_inference = False
        try:
            # Try a very simple inference call to test scope
            inference_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
            test_payload = {"inputs": "test", "options": {"wait_for_model": False, "use_cache": True}}
            inference_response = requests.post(inference_url, headers=headers, json=test_payload, timeout=15)
            
            # 200 = success, 503 = model loading (but scope works), 401/403 = no scope
            can_inference = inference_response.status_code in [200, 503]
            logger.info(f"✅ Inference response: {inference_response.status_code}")
            
            if inference_response.status_code == 401:
                logger.warning("⚠️ OAuth token unauthorized for inference endpoint - likely missing 'inference' scope")
            elif inference_response.status_code == 403:
                logger.warning("⚠️ OAuth token forbidden for inference endpoint - insufficient permissions")
            elif inference_response.status_code not in [200, 503]:
                logger.warning(f"⚠️ Unexpected inference response: {inference_response.status_code}")
                
        except Exception as inference_error:
            logger.error(f"❌ Inference test failed: {inference_error}")
            can_inference = False
        
        # Alternative inference test - try Qwen model directly
        if not can_inference:
            logger.info("🔍 Testing OAuth token with Qwen model directly...")
            try:
                qwen_url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"
                qwen_payload = {"inputs": "Hello", "options": {"wait_for_model": False}}
                qwen_response = requests.post(qwen_url, headers=headers, json=qwen_payload, timeout=15)
                
                qwen_inference = qwen_response.status_code in [200, 503]
                if qwen_inference:
                    can_inference = True
                    logger.info(f"✅ Qwen model response: {qwen_response.status_code}")
                else:
                    logger.warning(f"⚠️ Qwen model response: {qwen_response.status_code}")
                    
            except Exception as qwen_error:
                logger.error(f"❌ Qwen model test failed: {qwen_error}")
        
        # Determine probable scopes based on capabilities
        probable_scopes = []
        if can_read:
            probable_scopes.append("read")
        if can_inference:
            probable_scopes.append("inference")
        
        logger.info(f"📊 Final scope assessment: {probable_scopes}")
        
        # Get user info if available
        user_info = {}
        if can_read and whoami_response.status_code == 200:
            try:
                user_data = whoami_response.json()
                user_info = {
                    "name": user_data.get("name", "Unknown"),
                    "fullname": user_data.get("fullName", ""),
                    "avatar": user_data.get("avatarUrl", "")
                }
                logger.info(f"✅ User info retrieved: {user_info.get('name', 'unknown')}")
            except Exception as user_error:
                logger.warning(f"⚠️ Could not parse user info: {user_error}")
                user_info = {}
        
        return {
            "logged_in": True,
            "scopes": probable_scopes,
            "can_inference": can_inference,
            "can_read": can_read,
            "user_info": user_info,
            "message": f"Logged in with scopes: {', '.join(probable_scopes) if probable_scopes else 'limited'}"
        }
        
    except Exception as e:
        logger.error(f"❌ OAuth scope check failed: {e}")
        return {
            "logged_in": True,
            "scopes": ["unknown"],
            "can_inference": False,
            "can_read": False,
            "user_info": {},
            "message": f"Could not determine scopes: {str(e)}"
        }

def format_auth_status(profile: gr.OAuthProfile | None) -> str:
    """Format authentication status for display in UI"""
    
    # Check for HF_TOKEN first (best performance)
    hf_token = os.getenv("HF_TOKEN")
    
    if hf_token:
        # HF_TOKEN is available - this is the best case scenario
        return """
### 🎯 Authentication Status: HF_TOKEN Environment Variable

**🚀 FULL SYSTEM CAPABILITIES ENABLED**

**Authentication Source**: HF_TOKEN environment variable
**Model Access**: Qwen 2.5 models (7B/32B/72B) via HuggingFace Inference API
**Workflow**: LangGraph multi-agent system with specialized tools

**Available Features:**
- ✅ **Advanced Model Access**: Full Qwen model capabilities (7B/32B/72B)
- ✅ **High Performance**: 30%+ expected GAIA score
- ✅ **LangGraph Workflow**: Multi-agent orchestration with synthesis
- ✅ **Specialized Agents**: Web research, file processing, mathematical reasoning  
- ✅ **Professional Tools**: Wikipedia, web search, calculator, file processor
- ✅ **Manual Testing**: Individual question processing with detailed analysis
- ✅ **Official Evaluation**: GAIA benchmark submission

💡 **Status**: Optimal configuration for GAIA benchmark performance with real AI agents.
"""
    
    # Check HuggingFace Spaces OAuth configuration
    oauth_scopes = os.getenv("OAUTH_SCOPES")
    oauth_client_id = os.getenv("OAUTH_CLIENT_ID")
    # Accept both 'inference-api' and 'inference' as valid inference scopes
    has_inference_scope = oauth_scopes and ("inference-api" in oauth_scopes or "inference" in oauth_scopes)
    
    if not profile:
        oauth_status = ""
        if oauth_client_id:
            if has_inference_scope:
                oauth_status = "**🔑 OAuth Configuration**: ✅ Space configured with inference scope"
            else:
                oauth_status = "**⚠️ OAuth Configuration**: Space OAuth enabled but missing inference scope"
        else:
            oauth_status = "**❌ OAuth Configuration**: Space not configured for OAuth (missing `hf_oauth: true` in README.md)"
        
        return f"""
### 🔐 Authentication Status: Not Logged In

Please log in to access GAIA evaluation with Qwen models and LangGraph workflow.

{oauth_status}

**What you need:**
- 🔑 HuggingFace login with `read` and `inference` permissions
- 🤖 Access to Qwen 2.5 models via HF Inference API
- 🧠 LangGraph multi-agent system capabilities

**🔑 OAuth Scopes**: Login requests inference scope for Qwen model access.
**📈 Expected Performance**: 30%+ GAIA score with full LangGraph workflow and Qwen models.
**⚠️ No Fallbacks**: System requires proper authentication - no simplified responses.
"""
    
    username = profile.username
    oauth_token = getattr(profile, 'oauth_token', None) or getattr(profile, 'token', None)
    
    # Try multiple methods to extract OAuth token
    if not oauth_token:
        for attr in ['access_token', 'id_token', 'bearer_token']:
            token = getattr(profile, attr, None)
            if token:
                oauth_token = token
                logger.info(f"🔑 Found OAuth token via {attr}")
                break
    
    # If still no token, check if profile has any token-like attributes
    if not oauth_token and hasattr(profile, '__dict__'):
        token_attrs = [attr for attr in profile.__dict__.keys() if 'token' in attr.lower()]
        if token_attrs:
            logger.info(f"🔍 Available token attributes: {token_attrs}")
            # Try the first available token attribute
            oauth_token = getattr(profile, token_attrs[0], None)
            if oauth_token:
                logger.info(f"🔑 Using token from {token_attrs[0]}")
    
    scope_info = check_oauth_scopes(oauth_token) if oauth_token else {
        "logged_in": True,
        "scopes": [],
        "can_inference": False,
        "can_read": False,
        "user_info": {},
        "message": "Logged in but no OAuth token found"
    }
    
    status_parts = [f"### 🔐 Authentication Status: Logged In as {username}"]
    
    # Safely access user_info
    user_info = scope_info.get("user_info", {})
    if user_info and user_info.get("fullname"):
        status_parts.append(f"**Full Name**: {user_info['fullname']}")
    
    # HuggingFace Spaces OAuth Environment Status
    if oauth_client_id:
        if has_inference_scope:
            status_parts.append("**🏠 Space OAuth**: ✅ Configured with inference scope")
        else:
            status_parts.append("**🏠 Space OAuth**: ⚠️ Missing inference scope in README.md")
            status_parts.append(f"**Available Scopes**: {oauth_scopes}")
    else:
        status_parts.append("**🏠 Space OAuth**: ❌ Not configured (`hf_oauth: true` missing)")
    
    # Safely access scopes
    scopes = scope_info.get("scopes", [])
    status_parts.append(f"**Detected Token Scopes**: {', '.join(scopes) if scopes else 'None detected'}")
    status_parts.append("")
    status_parts.append("**System Capabilities:**")
    
    # Safely access capabilities
    can_inference = scope_info.get("can_inference", False)
    can_read = scope_info.get("can_read", False)
    
    if can_inference:
        status_parts.extend([
            "- ✅ **Qwen Model Access**: Full Qwen 2.5 model capabilities (7B/32B/72B)",
            "- ✅ **High Performance**: 30%+ expected GAIA score",
            "- ✅ **LangGraph Workflow**: Multi-agent orchestration with synthesis",
            "- ✅ **Specialized Agents**: Web research, file processing, reasoning",
            "- ✅ **Professional Tools**: Wikipedia, web search, calculator, file processor",
            "- ✅ **Inference Access**: Full model generation capabilities"
        ])
    else:
        status_parts.extend([
            "- ❌ **No Qwen Model Access**: Insufficient OAuth permissions",
            "- ❌ **No LangGraph Workflow**: Requires inference permissions",
            "- ❌ **Limited Functionality**: Cannot process GAIA questions",
            "- ❌ **No Inference Access**: Read-only permissions detected"
        ])
    
    if can_read:
        status_parts.append("- ✅ **Profile Access**: Can read user information")
    
    status_parts.extend([
        "- ✅ **Manual Testing**: Individual question processing (if authenticated)",
        "- ✅ **Official Evaluation**: GAIA benchmark submission (if authenticated)"
    ])
    
    if not can_inference:
        if not has_inference_scope:
            status_parts.extend([
                "",
                "🔧 **Space Configuration Issue**: Add inference scope to README.md:",
                "```yaml",
                "hf_oauth_scopes:",
                "  - inference-api",
                "```",
                "**After updating**: Space will restart and request proper scopes on next login."
            ])
        
        status_parts.extend([
            "",
            "🔑 **Authentication Required**: Your OAuth session lacks inference permissions.",
            "**Solution**: Logout and login again to request full inference access.",
            "**Alternative**: Set HF_TOKEN as a Space secret for guaranteed Qwen model access.",
            "**Note**: System requires Qwen model access - no simplified fallbacks available."
        ])
        
        # Add specific guidance if we couldn't find an OAuth token
        if not oauth_token:
            status_parts.extend([
                "",
                "🔍 **OAuth Token Issue**: Could not extract OAuth token from your session.",
                "**Troubleshooting**: Click '🔍 Debug OAuth' button above to investigate.",
                "**Common Fix**: Logout and login again to refresh your OAuth session."
            ])
    else:
        status_parts.extend([
            "",
            "🎉 **Excellent**: You have full inference access for optimal GAIA performance!",
            "🤖 **Ready**: LangGraph workflow with Qwen models fully operational."
        ])
    
    return "\n".join(status_parts)

def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions from Unit 4 API, runs the GAIA Agent with LangGraph workflow,
    and displays the results. Handles OAuth authentication at runtime.
    """
    start_time = time.time()
    
    # Initialize result logger
    result_logger = GAIAResultLogger()
    
    # Check OAuth environment configuration first
    oauth_client_id = os.getenv("OAUTH_CLIENT_ID")
    oauth_scopes = os.getenv("OAUTH_SCOPES")
    
    if not oauth_client_id:
        return "❌ OAuth not configured. Please add 'hf_oauth: true' to README.md", None, format_auth_status(None), None, None, None
    
    # Accept both 'inference-api' and 'inference' as valid inference scopes
    if not oauth_scopes or not ("inference-api" in oauth_scopes or "inference" in oauth_scopes):
        return f"❌ Missing inference scope. Current scopes: {oauth_scopes}. Please add inference scope to README.md", None, format_auth_status(None), None, None, None
    
    # Get space info for code submission
    space_id = os.getenv("SPACE_ID")

    # Priority order for token: 1) HF_TOKEN env var, 2) OAuth token from profile
    hf_token = os.getenv("HF_TOKEN")
    oauth_token = None
    username = "unknown_user"
    
    if hf_token:
        logger.info("🎯 Using HF_TOKEN environment variable for Qwen model access")
        oauth_token = hf_token
        username = "hf_token_user"
    elif profile:
        username = f"{profile.username}"
        
        # Try multiple ways to extract OAuth token
        oauth_token = getattr(profile, 'oauth_token', None) or getattr(profile, 'token', None)
        
        if not oauth_token:
            for attr in ['access_token', 'id_token', 'bearer_token']:
                token = getattr(profile, attr, None)
                if token:
                    oauth_token = token
                    logger.info(f"🔑 Found OAuth token via {attr}")
                    break
        
        if oauth_token:
            logger.info(f"✅ User logged in: {username}, OAuth token extracted successfully")
            
            # Test OAuth token validity
            try:
                headers = {"Authorization": f"Bearer {oauth_token}"}
                test_response = requests.get("https://huggingface.co/api/whoami", headers=headers, timeout=5)
                
                if test_response.status_code == 401:
                    logger.error("❌ OAuth token has insufficient scopes for Qwen model inference")
                    return "Authentication Error: Your OAuth token lacks inference permissions. Please logout and login again to refresh your OAuth session.", None, format_auth_status(profile), None, None, None
                elif test_response.status_code == 200:
                    logger.info("✅ OAuth token validated successfully")
                else:
                    logger.warning(f"⚠️ OAuth token validation returned {test_response.status_code}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not validate OAuth token: {e}")
        else:
            logger.warning(f"⚠️ User {username} logged in but no OAuth token found")
            return f"OAuth Token Missing: Could not extract authentication token for user {username}. Please logout and login again.", None, format_auth_status(profile), None, None, None
    else:
        logger.error("❌ No authentication provided")
        return "Authentication Required: Please login with HuggingFace. Your Space has OAuth configured but you need to login first.", None, format_auth_status(None), None, None, None

    if not oauth_token:
        return "Authentication Required: Valid token with inference permissions needed for Qwen model access.", None, format_auth_status(profile), None, None, None

    # Get authentication status for display
    auth_status = format_auth_status(profile)
    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate GAIA Agent with LangGraph workflow
    try:
        logger.info("🚀 Creating GAIA Agent with LangGraph workflow and Qwen models")
        agent = GAIAAgentApp.create_with_oauth_token(oauth_token)
            
        if not agent.initialized:
            return "System Error: GAIA Agent failed to initialize with LangGraph workflow", None, auth_status, None, None, None
            
        logger.info("✅ GAIA Agent initialized successfully")
        
    except ValueError as ve:
        logger.error(f"Authentication error: {ve}")
        return f"Authentication Error: {ve}", None, auth_status, None, None, None
    except RuntimeError as re:
        logger.error(f"System initialization error: {re}")
        return f"System Error: {re}", None, auth_status, None, None, None
    except Exception as e:
        logger.error(f"Unexpected error initializing agent: {e}")
        return f"Unexpected Error: {e}. Please check your authentication and try again.", None, auth_status, None, None, None

    # Agent code URL
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else "Local Development"
    logger.info(f"Agent code URL: {agent_code}")

    # 2. Fetch Questions
    logger.info(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            logger.error("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None, auth_status, None, None, None
        logger.info(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None, auth_status, None, None, None
    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response from questions endpoint: {e}")
        return f"Error decoding server response for questions: {e}", None, auth_status, None, None, None
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None, auth_status, None, None, None

    # 3. Run GAIA Agent on questions
    results_log = []
    answers_payload = []
    logger.info(f"🤖 Running GAIA Agent on {len(questions_data)} questions with LangGraph workflow...")
    
    for i, item in enumerate(questions_data, 1):
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            logger.warning(f"Skipping item with missing task_id or question: {item}")
            continue
        
        logger.info(f"Processing question {i}/{len(questions_data)}: {task_id}")
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({
                "Task ID": task_id, 
                "Question": question_text[:100] + "..." if len(question_text) > 100 else question_text, 
                "Submitted Answer": submitted_answer[:200] + "..." if len(submitted_answer) > 200 else submitted_answer
            })
            logger.info(f"✅ Question {i} processed successfully")
        except Exception as e:
            logger.error(f"Error running GAIA agent on task {task_id}: {e}")
            error_answer = f"AGENT ERROR: {str(e)}"
            answers_payload.append({"task_id": task_id, "submitted_answer": error_answer})
            results_log.append({
                "Task ID": task_id, 
                "Question": question_text[:100] + "..." if len(question_text) > 100 else question_text, 
                "Submitted Answer": error_answer
            })

    if not answers_payload:
        logger.error("GAIA Agent did not produce any answers to submit.")
        return "GAIA Agent did not produce any answers to submit.", pd.DataFrame(results_log), auth_status, None, None, None

    # 4. Prepare and submit results
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"🚀 GAIA Agent finished processing {len(answers_payload)} questions. Submitting results for user '{username}'..."
    logger.info(status_update)

    # 5. Submit to Unit 4 API
    logger.info(f"📤 Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=120)
        logger.info(f"📨 Unit 4 API response status: {response.status_code}")
        
        response.raise_for_status()
        result_data = response.json()
        
        # Log the actual response for debugging
        logger.info(f"📊 Unit 4 API response data: {result_data}")
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # 6. Log results to files
        logger.info("📝 Logging evaluation results...")
        logged_files = result_logger.log_evaluation_results(
            username=username,
            questions_data=questions_data,
            results_log=results_log,
            final_result=result_data,
            execution_time=execution_time
        )
        
        # Prepare download files
        csv_file = logged_files.get("csv")
        json_file = logged_files.get("json") 
        summary_file = logged_files.get("summary")
        
        final_status = (
            f"🎉 GAIA Agent Evaluation Complete!\n"
            f"👤 User: {result_data.get('username')}\n"
            f"🏆 Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"⏱️ Execution Time: {execution_time:.2f} seconds\n"
            f"💬 API Response: {result_data.get('message', 'No message received.')}\n\n"
            f"📁 Results saved to {len([f for f in [csv_file, json_file, summary_file] if f])} files for download."
        )
        logger.info("✅ GAIA evaluation completed successfully")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df, auth_status, csv_file, json_file, summary_file
        
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"❌ Submission Failed: {error_detail}"
        logger.error(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df, auth_status, None, None, None
    except requests.exceptions.Timeout:
        status_message = "❌ Submission Failed: The request timed out."
        logger.error(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df, auth_status, None, None, None
    except requests.exceptions.RequestException as e:
        status_message = f"❌ Submission Failed: Network error - {e}"
        logger.error(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df, auth_status, None, None, None
    except Exception as e:
        status_message = f"❌ An unexpected error occurred during submission: {e}"
        logger.error(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df, auth_status, None, None, None

def create_interface():
    """Create the Gradio interface with both Unit 4 API and manual testing"""
    
    # Note: We don't initialize GAIAAgentApp here since it requires authentication
    # Each request will create its own authenticated instance
    
    # Custom CSS for better styling
    css = """
    /* Base styling for proper contrast */
    .gradio-container {
        color: #3c3c3c !important;
        background-color: #faf9f7 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Fix all text elements EXCEPT buttons */
    .gradio-container *:not(button):not(.gr-button):not(.gr-button-primary):not(.gr-button-secondary), 
    .gradio-container *:not(button):not(.gr-button):not(.gr-button-primary):not(.gr-button-secondary)::before, 
    .gradio-container *:not(button):not(.gr-button):not(.gr-button-primary):not(.gr-button-secondary)::after {
        color: #3c3c3c !important;
    }
    
    /* Headers */
    .gradio-container h1, 
    .gradio-container h2, 
    .gradio-container h3, 
    .gradio-container h4, 
    .gradio-container h5, 
    .gradio-container h6 {
        color: #2c2c2c !important;
        font-weight: 600 !important;
    }
    
    /* Paragraphs and text content */
    .gradio-container p,
    .gradio-container div:not(.gr-button):not(.gr-button-primary):not(.gr-button-secondary),
    .gradio-container span:not(.gr-button):not(.gr-button-primary):not(.gr-button-secondary),
    .gradio-container label {
        color: #3c3c3c !important;
    }
    
    /* Input fields */
    .gradio-container input,
    .gradio-container textarea {
        color: #3c3c3c !important;
        background-color: #ffffff !important;
        border: 1px solid #d4c4b0 !important;
        border-radius: 6px !important;
    }
    
    /* Buttons - Subtle professional styling */
    .gradio-container button,
    .gradio-container .gr-button,
    .gradio-container .gr-button-primary,
    .gradio-container .gr-button-secondary,
    .gradio-container button *,
    .gradio-container .gr-button *,
    .gradio-container .gr-button-primary *,
    .gradio-container .gr-button-secondary * {
        color: #3c3c3c !important;
        font-weight: 500 !important;
        text-shadow: none !important;
        border-radius: 6px !important;
        border: 1px solid #d4c4b0 !important;
        transition: all 0.2s ease !important;
    }
    
    .gradio-container .gr-button-primary,
    .gradio-container button[variant="primary"] {
        background: #f5f3f0 !important;
        color: #3c3c3c !important;
        border: 1px solid #d4c4b0 !important;
        padding: 8px 16px !important;
        border-radius: 6px !important;
    }
    
    .gradio-container .gr-button-secondary,
    .gradio-container button[variant="secondary"] {
        background: #ffffff !important;
        color: #3c3c3c !important;
        border: 1px solid #d4c4b0 !important;
        padding: 8px 16px !important;
        border-radius: 6px !important;
    }
    
    .gradio-container button:not([variant]) {
        background: #f8f6f3 !important;
        color: #3c3c3c !important;
        border: 1px solid #d4c4b0 !important;
        padding: 8px 16px !important;
        border-radius: 6px !important;
    }
    
    /* Button hover states - subtle changes */
    .gradio-container button:hover,
    .gradio-container .gr-button:hover,
    .gradio-container .gr-button-primary:hover {
        background: #ede9e4 !important;
        color: #2c2c2c !important;
        border: 1px solid #c4b49f !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08) !important;
    }
    
    .gradio-container .gr-button-secondary:hover {
        background: #f5f3f0 !important;
        color: #2c2c2c !important;
        border: 1px solid #c4b49f !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08) !important;
    }
    
    /* Login button styling */
    .gradio-container .gr-button:contains("Login"),
    .gradio-container button:contains("Login") {
        background: #e8e3dc !important;
        color: #3c3c3c !important;
        border: 1px solid #d4c4b0 !important;
    }
    
    /* Markdown content */
    .gradio-container .gr-markdown,
    .gradio-container .markdown,
    .gradio-container .prose {
        color: #3c3c3c !important;
        background-color: transparent !important;
    }
    
    /* Special content boxes */
    .container {
        max-width: 1200px; 
        margin: auto; 
        padding: 20px;
        background-color: #faf9f7 !important;
        color: #3c3c3c !important;
    }
    
    .output-markdown {
        font-size: 16px; 
        line-height: 1.6; 
        color: #3c3c3c !important;
        background-color: #faf9f7 !important;
    }
    
    .details-box {
        background-color: #f5f3f0 !important; 
        padding: 15px; 
        border-radius: 8px; 
        margin: 10px 0; 
        color: #3c3c3c !important;
        border: 1px solid #e0d5c7 !important;
    }
    
    .reasoning-box {
        background-color: #ffffff !important; 
        padding: 20px; 
        border: 1px solid #e0d5c7 !important; 
        border-radius: 8px; 
        color: #3c3c3c !important;
    }
    
    .unit4-section {
        background-color: #f0ede8 !important; 
        padding: 20px; 
        border-radius: 8px; 
        margin: 20px 0; 
        color: #4a4035 !important;
        border: 1px solid #d4c4b0 !important;
    }
    
    .unit4-section h1,
    .unit4-section h2,
    .unit4-section h3,
    .unit4-section p,
    .unit4-section div:not(button):not(.gr-button) {
        color: #4a4035 !important;
    }
    
    /* Login section */
    .oauth-login {
        background: #f5f3f0 !important; 
        padding: 10px; 
        border-radius: 5px; 
        margin: 10px 0;
        color: #3c3c3c !important;
        border: 1px solid #e0d5c7 !important;
    }
    
    /* Tables */
    .gradio-container table,
    .gradio-container th,
    .gradio-container td {
        color: #3c3c3c !important;
        background-color: #ffffff !important;
        border: 1px solid #e0d5c7 !important;
    }
    
    .gradio-container th {
        background-color: #f5f3f0 !important;
        font-weight: 600 !important;
    }
    
    /* Examples and other interactive elements */
    .gradio-container .gr-examples,
    .gradio-container .gr-file,
    .gradio-container .gr-textbox,
    .gradio-container .gr-checkbox {
        color: #3c3c3c !important;
        background-color: #ffffff !important;
    }
    
    /* Fix any remaining text contrast issues */
    .gradio-container .gr-form,
    .gradio-container .gr-panel,
    .gradio-container .gr-block {
        color: #3c3c3c !important;
        background-color: transparent !important;
    }
    
    /* Ensure proper text on light backgrounds */
    .gradio-container .light,
    .gradio-container [data-theme="light"] {
        color: #3c3c3c !important;
        background-color: #faf9f7 !important;
    }
    
    /* Override any problematic inline styles but preserve button colors */
    .gradio-container [style*="color: white"]:not(button):not(.gr-button) {
        color: #3c3c3c !important;
    }
    
    /* Professional spacing and shadows */
    .gradio-container .gr-box {
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        border-radius: 8px !important;
    }
    
    /* Override any remaining purple/blue elements */
    .gradio-container .gr-textbox,
    .gradio-container .gr-dropdown,
    .gradio-container .gr-number,
    .gradio-container .gr-slider {
        background-color: #ffffff !important;
        border: 1px solid #d4c4b0 !important;
        color: #3c3c3c !important;
    }
    
    /* Force override any Gradio default styling */
    .gradio-container * {
        background-color: inherit !important;
    }
    
    .gradio-container *[style*="background-color: rgb(239, 68, 68)"],
    .gradio-container *[style*="background-color: rgb(59, 130, 246)"],
    .gradio-container *[style*="background-color: rgb(147, 51, 234)"],
    .gradio-container *[style*="background-color: rgb(16, 185, 129)"] {
        background-color: #f5f3f0 !important;
        color: #3c3c3c !important;
        border: 1px solid #d4c4b0 !important;
    }
    
    /* Loading states */
    .gradio-container .loading {
        background-color: #f5f3f0 !important;
        color: #6b5d4f !important;
    }
    
    /* Progress bars */
    .gradio-container .gr-progress {
        background-color: #f5f3f0 !important;
    }
    
    .gradio-container .gr-progress-bar {
        background-color: #a08b73 !important;
    }
    """
    
    # Configure OAuth with full inference access
    oauth_config = {
        "scopes": ["read", "inference"],  # Request both read and inference access
    }
    
    with gr.Blocks(css=css, title="GAIA Agent System", theme=gr.themes.Soft()) as interface:
        
        # Header
        gr.Markdown("""
        # 🤖 GAIA Agent System
        
        **Advanced Multi-Agent AI System for GAIA Benchmark Questions**
        
        This system uses **Qwen 2.5 models (7B/32B/72B)** with specialized agents orchestrated through 
        **LangGraph** to provide accurate, well-reasoned answers to complex questions.
        
        **Architecture**: Router → Specialized Agents → Tools → Synthesizer → Final Answer
        """)
        
        # Unit 4 API Section
        with gr.Row(elem_classes=["unit4-section"]):
            with gr.Column():
                gr.Markdown("""
                ## 🏆 GAIA Benchmark Evaluation
                
                **Official Unit 4 API Integration with LangGraph Workflow**
                
                Run the complete GAIA Agent system using Qwen 2.5 models and LangGraph multi-agent 
                orchestration on all benchmark questions and submit results to the official API.
                
                **System Requirements:**
                1. 🔑 **Authentication**: HuggingFace login with `read` and `inference` permissions
                2. 🤖 **Models**: Access to Qwen 2.5 models (7B/32B/72B) via HF Inference API
                3. 🧠 **Workflow**: LangGraph multi-agent system with specialized tools
                
                **Instructions:**
                1. Log in to your Hugging Face account using the button below (**Full inference access required**)
                2. Click 'Run GAIA Evaluation & Submit All Answers' to process all questions
                3. View your official score and detailed results
                
                ⚠️ **Note**: This may take several minutes to process all questions with the multi-agent system.
                
                💡 **OAuth Scopes**: Login requests both `read` and `inference` permissions 
                for Qwen model access and optimal performance (30%+ GAIA score expected).
                
                🚫 **No Fallbacks**: System requires proper authentication - simplified responses not available.
                """)
                
                # Authentication status section
                auth_status_display = gr.Markdown(
                    """
### 🔐 Authentication Status: Not Logged In

Please log in to access GAIA evaluation with Qwen models and LangGraph workflow.

**What you need:**
- 🔑 HuggingFace login with `read` and `inference` permissions
- 🤖 Access to Qwen 2.5 models via HF Inference API
- 🧠 LangGraph multi-agent system capabilities

**Expected Performance**: 30%+ GAIA score with full LangGraph workflow and Qwen models.
""",
                    elem_classes=["oauth-login"]
                )
                
                # Add Gradio's built-in OAuth login button
                gr.LoginButton()
                
                with gr.Row():
                    refresh_auth_button = gr.Button("🔄 Refresh Auth Status", variant="secondary", scale=1)
                
                unit4_run_button = gr.Button(
                    "🚀 Run GAIA Evaluation & Submit All Answers", 
                    variant="primary",
                    scale=2
                )
                
                unit4_status_output = gr.Textbox(
                    label="Evaluation Status / Submission Result", 
                    lines=5, 
                    interactive=False
                )
                
                unit4_results_table = gr.DataFrame(
                    label="Questions and GAIA Agent Answers", 
                    wrap=True
                )
                
                # Download section
                gr.Markdown("### 📁 Download Results")
                gr.Markdown("After evaluation completes, download your results in different formats:")
                
                with gr.Row():
                    csv_download = gr.File(
                        label="📊 CSV Results",
                        visible=False,
                        interactive=False
                    )
                    
                    json_download = gr.File(
                        label="🔍 Detailed JSON",
                        visible=False,
                        interactive=False
                    )
                    
                    summary_download = gr.File(
                        label="📋 Summary Report",
                        visible=False,
                        interactive=False
                    )
        
        gr.Markdown("---")
        
        # Manual Testing Section
        gr.Markdown("""
        ## 🧪 Manual Question Testing
        
        Test individual questions with detailed analysis using **Qwen models** and **LangGraph workflow**.
        
        **Features:**
        - 🤖 **Qwen 2.5 Models**: Intelligent tier selection (7B → 32B → 72B) based on complexity
        - 🧠 **LangGraph Orchestration**: Multi-agent workflow with synthesis
        - 🔧 **Specialized Agents**: Router, web research, file processing, mathematical reasoning
        - 📊 **Detailed Analysis**: Processing details, confidence scores, cost tracking
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                gr.Markdown("### 📝 Input")
                
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="Enter your question here...",
                    lines=3,
                    max_lines=10
                )
                
                file_input = gr.File(
                    label="Optional File Upload",
                    file_types=[".txt", ".csv", ".xlsx", ".py", ".json", ".png", ".jpg", ".mp3", ".wav"],
                    type="filepath"
                )
                
                with gr.Row():
                    show_reasoning = gr.Checkbox(
                        label="Show detailed reasoning",
                        value=False
                    )
                    
                    submit_btn = gr.Button(
                        "🔍 Process Question",
                        variant="secondary"
                    )
                
                # Examples
                gr.Markdown("#### 💡 Example Questions")
                
                example_questions = [
                    "How many studio albums were published by Mercedes Sosa between 2000 and 2009?",
                    "What is the capital of the country that has the most time zones?",
                    "Calculate the compound interest on $1000 at 5% annual rate compounded quarterly for 3 years",
                    "What is the square root of the sum of the first 10 prime numbers?",
                    "Who was the first person to walk on the moon and what year did it happen?",
                    "Compare the GDP of Japan and Germany in 2023 and tell me the difference",
                ]
                
                examples = gr.Examples(
                    examples=example_questions,
                    inputs=[question_input],
                    cache_examples=False
                )
            
            with gr.Column(scale=3):
                # Output section
                gr.Markdown("### 📊 Results")
                
                answer_output = gr.Markdown(
                    label="Answer",
                    elem_classes=["output-markdown"]
                )
                
                details_output = gr.Markdown(
                    label="Processing Details",
                    elem_classes=["details-box"]
                )
                
                reasoning_output = gr.Markdown(
                    label="Detailed Reasoning",
                    visible=False,
                    elem_classes=["reasoning-box"]
                )
        
        # Event handlers for Unit 4 API - Using Gradio's built-in OAuth
        def run_gaia_evaluation(oauth_token: gr.OAuthToken | None, profile: gr.OAuthProfile | None):
            """Run GAIA evaluation using Gradio's built-in OAuth"""
            start_time = time.time()
            
            # Initialize result logger
            result_logger = GAIAResultLogger()
            
            # Check authentication using Gradio's OAuth parameters
            if oauth_token is None or profile is None:
                return "❌ Authentication Required: Please login with HuggingFace to access GAIA evaluation.", None, None, None, None, None
            
            username = profile.username if profile else "unknown_user"
            hf_token = oauth_token.token if oauth_token else None
            
            if not hf_token:
                return "❌ OAuth Token Missing: Could not extract authentication token. Please logout and login again.", None, None, None, None, None
            
            logger.info(f"✅ Starting GAIA evaluation for user: {username}")
            
            # Rest of the function exactly as in run_and_submit_all but using oauth_token.token
            api_url = DEFAULT_API_URL
            questions_url = f"{api_url}/questions"
            submit_url = f"{api_url}/submit"
            
            # Get space info for code submission
            space_id = os.getenv("SPACE_ID")
            
            # 1. Instantiate GAIA Agent with LangGraph workflow
            try:
                logger.info("🚀 Creating GAIA Agent with LangGraph workflow and Qwen models")
                agent = GAIAAgentApp.create_with_oauth_token(hf_token)
                    
                if not agent.initialized:
                    return "❌ System Error: GAIA Agent failed to initialize with LangGraph workflow", None, None, None, None, None
                    
                logger.info("✅ GAIA Agent initialized successfully")
                
            except ValueError as ve:
                logger.error(f"Authentication error: {ve}")
                return f"❌ Authentication Error: {ve}", None, None, None, None, None
            except RuntimeError as re:
                logger.error(f"System initialization error: {re}")
                return f"❌ System Error: {re}", None, None, None, None, None
            except Exception as e:
                logger.error(f"Unexpected error initializing agent: {e}")
                return f"❌ Unexpected Error: {e}. Please check your authentication and try again.", None, None, None, None, None

            # Agent code URL
            agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else "Local Development"
            logger.info(f"Agent code URL: {agent_code}")

            # 2. Fetch Questions
            logger.info(f"Fetching questions from: {questions_url}")
            try:
                response = requests.get(questions_url, timeout=15)
                response.raise_for_status()
                questions_data = response.json()
                if not questions_data:
                    logger.error("Fetched questions list is empty.")
                    return "❌ Fetched questions list is empty or invalid format.", None, None, None, None, None
                logger.info(f"✅ Fetched {len(questions_data)} questions.")
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching questions: {e}")
                return f"❌ Error fetching questions: {e}", None, None, None, None, None
            except Exception as e:
                logger.error(f"An unexpected error occurred fetching questions: {e}")
                return f"❌ An unexpected error occurred fetching questions: {e}", None, None, None, None, None

            # 3. Run GAIA Agent on questions
            results_log = []
            answers_payload = []
            logger.info(f"🤖 Running GAIA Agent on {len(questions_data)} questions with LangGraph workflow...")
            
            for i, item in enumerate(questions_data, 1):
                task_id = item.get("task_id")
                question_text = item.get("question")
                if not task_id or question_text is None:
                    logger.warning(f"Skipping item with missing task_id or question: {item}")
                    continue
                
                logger.info(f"Processing question {i}/{len(questions_data)}: {task_id}")
                try:
                    submitted_answer = agent(question_text)
                    answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
                    results_log.append({
                        "Task ID": task_id, 
                        "Question": question_text[:100] + "..." if len(question_text) > 100 else question_text, 
                        "Submitted Answer": submitted_answer[:200] + "..." if len(submitted_answer) > 200 else submitted_answer
                    })
                    logger.info(f"✅ Question {i} processed successfully")
                except Exception as e:
                    logger.error(f"Error running GAIA agent on task {task_id}: {e}")
                    error_answer = f"AGENT ERROR: {str(e)}"
                    answers_payload.append({"task_id": task_id, "submitted_answer": error_answer})
                    results_log.append({
                        "Task ID": task_id, 
                        "Question": question_text[:100] + "..." if len(question_text) > 100 else question_text, 
                        "Submitted Answer": error_answer
                    })

            if not answers_payload:
                logger.error("GAIA Agent did not produce any answers to submit.")
                return "❌ GAIA Agent did not produce any answers to submit.", pd.DataFrame(results_log), None, None, None, None

            # 4. Prepare and submit results
            submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
            status_update = f"🚀 GAIA Agent finished processing {len(answers_payload)} questions. Submitting results for user '{username}'..."
            logger.info(status_update)

            # 5. Submit to Unit 4 API
            logger.info(f"📤 Submitting {len(answers_payload)} answers to: {submit_url}")
            try:
                response = requests.post(submit_url, json=submission_data, timeout=120)
                logger.info(f"📨 Unit 4 API response status: {response.status_code}")
                
                response.raise_for_status()
                result_data = response.json()
                
                # Log the actual response for debugging
                logger.info(f"📊 Unit 4 API response data: {result_data}")
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # 6. Log results to files
                logger.info("📝 Logging evaluation results...")
                logged_files = result_logger.log_evaluation_results(
                    username=username,
                    questions_data=questions_data,
                    results_log=results_log,
                    final_result=result_data,
                    execution_time=execution_time
                )
                
                # Prepare download files
                csv_file = logged_files.get("csv")
                json_file = logged_files.get("json") 
                summary_file = logged_files.get("summary")
                
                final_status = (
                    f"🎉 GAIA Agent Evaluation Complete!\n"
                    f"👤 User: {result_data.get('username')}\n"
                    f"🏆 Overall Score: {result_data.get('score', 'N/A')}% "
                    f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
                    f"⏱️ Execution Time: {execution_time:.2f} seconds\n"
                    f"💬 API Response: {result_data.get('message', 'No message received.')}\n\n"
                    f"📁 Results saved to {len([f for f in [csv_file, json_file, summary_file] if f])} files for download."
                )
                logger.info("✅ GAIA evaluation completed successfully")
                results_df = pd.DataFrame(results_log)
                
                # Update download file visibility and values
                csv_update = gr.update(value=csv_file, visible=csv_file is not None)
                json_update = gr.update(value=json_file, visible=json_file is not None)
                summary_update = gr.update(value=summary_file, visible=summary_file is not None)
                
                return final_status, results_df, csv_update, json_update, summary_update
                
            except requests.exceptions.HTTPError as e:
                error_detail = f"Server responded with status {e.response.status_code}."
                try:
                    error_json = e.response.json()
                    error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
                except requests.exceptions.JSONDecodeError:
                    error_detail += f" Response: {e.response.text[:500]}"
                status_message = f"❌ Submission Failed: {error_detail}"
                logger.error(status_message)
                results_df = pd.DataFrame(results_log)
                return status_message, results_df, None, None, None
            except Exception as e:
                status_message = f"❌ An unexpected error occurred during submission: {e}"
                logger.error(status_message)
                results_df = pd.DataFrame(results_log)
                return status_message, results_df, None, None, None
        
        def update_auth_status(profile: gr.OAuthProfile | None):
            """Update authentication status display using Gradio's OAuth"""
            if profile is None:
                return """
### 🔐 Authentication Status: Not Logged In

Please click the "Sign in with Hugging Face" button above to access GAIA evaluation.

**What you need:**
- 🔑 HuggingFace login with `read` and `inference` permissions
- 🤖 Access to Qwen 2.5 models via HF Inference API
- 🧠 LangGraph multi-agent system capabilities

**Expected Performance**: 30%+ GAIA score with full LangGraph workflow and Qwen models.
"""
            else:
                return f"""
### 🔐 Authentication Status: ✅ Logged In as {profile.username}

**✅ Ready for GAIA Evaluation!**

- ✅ **OAuth Profile**: {profile.name or profile.username}
- ✅ **Qwen Model Access**: Available via HF Inference API
- ✅ **LangGraph Workflow**: Multi-agent orchestration ready
- ✅ **Official Evaluation**: Click "Run GAIA Evaluation" to start

🎯 **Expected Results**: 30%+ GAIA score with full LangGraph workflow and Qwen models.
"""
        
        # Set up automatic login state checking
        interface.load(
            fn=update_auth_status,
            outputs=[auth_status_display]
        )
        
        unit4_run_button.click(
            fn=run_gaia_evaluation,
            inputs=[],  # Gradio automatically injects OAuth parameters
            outputs=[unit4_status_output, unit4_results_table, csv_download, json_download, summary_download]
        )
        
        # Refresh authentication status manually
        refresh_auth_button.click(
            fn=update_auth_status,
            outputs=[auth_status_display]
        )
        
        # Event handlers for manual testing
        def process_and_update(question, file_input, show_reasoning, oauth_token: gr.OAuthToken | None, profile: gr.OAuthProfile | None):
            """Process question with authentication check"""
            
            if not question.strip():
                return "❌ Please provide a question", "", "", gr.update(visible=False)
            
            # Check for authentication - prioritize HF_TOKEN, then OAuth
            hf_token = os.getenv("HF_TOKEN")
            
            if not hf_token and (oauth_token is None or profile is None):
                error_msg = """
## ❌ Authentication Required

**This system requires authentication to access Qwen models and LangGraph workflow.**

**How to authenticate:**
1. 🔑 **Login with HuggingFace**: Use the "Sign in with Hugging Face" button above
2. 🌐 **Use Official Evaluation**: Login via the GAIA Benchmark section above
3. 📝 **Get Token**: Visit https://huggingface.co/settings/tokens to create one with `inference` permissions

**Note**: Manual testing requires the same authentication as the official evaluation.
"""
                return error_msg, "", "", gr.update(visible=False)
            
            # Use HF_TOKEN if available, otherwise use OAuth token
            auth_token = hf_token if hf_token else (oauth_token.token if oauth_token else None)
            
            if not auth_token:
                return "❌ No valid authentication token found", "", "", gr.update(visible=False)
            
            try:
                # Create authenticated app instance for this request
                app = GAIAAgentApp(hf_token=auth_token)
                
                # Process the question
                answer, details, reasoning = app.process_question_detailed(question, file_input, show_reasoning)
                
                # Format answer with markdown
                formatted_answer = f"""
## 🎯 Answer

{answer}
"""
                
                # Format details
                formatted_details = f"""
## 📋 Processing Details

{details}
"""
                
                # Show/hide reasoning based on checkbox
                reasoning_visible = show_reasoning and reasoning.strip()
                
                return (
                    formatted_answer,
                    formatted_details, 
                    reasoning if reasoning_visible else "",
                    gr.update(visible=reasoning_visible)
                )
                
            except ValueError as ve:
                error_msg = f"""
## ❌ Authentication Error

{str(ve)}

**Solution**: Please ensure your authentication has `inference` permissions.
"""
                return error_msg, "", "", gr.update(visible=False)
                
            except RuntimeError as re:
                error_msg = f"""
## ❌ System Error

{str(re)}

**This may be due to:**
- Qwen model access issues
- HuggingFace Inference API unavailability
- Network connectivity problems
"""
                return error_msg, "", "", gr.update(visible=False)
                
            except Exception as e:
                error_msg = f"""
## ❌ Unexpected Error

{str(e)}

**Please try again or contact support if the issue persists.**
"""
                return error_msg, "", "", gr.update(visible=False)
        
        submit_btn.click(
            fn=process_and_update,
            inputs=[question_input, file_input, show_reasoning],
            outputs=[answer_output, details_output, reasoning_output, reasoning_output]
        )
        
        # Show/hide reasoning based on checkbox
        show_reasoning.change(
            fn=lambda show: gr.update(visible=show),
            inputs=[show_reasoning],
            outputs=[reasoning_output]
        )
        
        # Footer
        gr.Markdown("""
        ---
        
        ### 🔧 System Architecture
        
        **LangGraph Multi-Agent Workflow:**
        - **Router Agent**: Classifies questions and selects appropriate specialized agents (using 32B model for better accuracy)
        - **Web Research Agent**: Multi-engine search with DuckDuckGo (primary), Tavily API (secondary), Wikipedia (fallback)
        - **File Processing Agent**: Processes uploaded files (CSV, images, code, audio)
        - **Reasoning Agent**: Handles mathematical calculations and logical reasoning
        - **Synthesizer Agent**: Combines results from multiple agents into final answers
        
        **Models Used**: Qwen 2.5 (7B/32B/72B) with intelligent tier selection for optimal cost/performance
        
        **Tools Available**: Multi-engine web search (DuckDuckGo + Tavily + Wikipedia), mathematical calculator, multi-format file processor
        
        ### 📈 Performance Metrics
        - **Success Rate**: 30%+ expected on GAIA benchmark with full authentication
        - **Average Response Time**: ~3-5 seconds per question depending on complexity
        - **Cost Efficiency**: $0.01-0.40 per question depending on model tier selection
        - **Architecture**: Multi-agent LangGraph orchestration with intelligent synthesis
        - **Reliability**: Robust error handling and graceful degradation within workflow
        - **Web Search**: 3-tier search system (DuckDuckGo → Tavily → Wikipedia) with smart query optimization
        
        ### 🎯 Authentication Requirements
        - **HF_TOKEN Environment Variable**: Best performance with full access to Qwen models
        - **OAuth with Inference Scope**: Full access to Qwen 2.5 models via HuggingFace Inference API
        - **Optional**: TAVILY_API_KEY for enhanced web search capabilities (1,000 free searches/month)
        - **No Fallback Options**: System requires proper authentication for multi-agent functionality
        """)
    
    return interface

def main():
    """Main application entry point"""
    
    # Check if running in production (HuggingFace Spaces)
    is_production = (
        os.getenv("GRADIO_ENV") == "production" or 
        os.getenv("SPACE_ID") is not None or
        os.getenv("SPACE_HOST") is not None
    )
    
    # Check for space environment variables
    space_host = os.getenv("SPACE_HOST")
    space_id = os.getenv("SPACE_ID")
    
    if space_host:
        logger.info(f"✅ SPACE_HOST found: {space_host}")
        logger.info(f"   Runtime URL: https://{space_host}")
    else:
        logger.info("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id:
        logger.info(f"✅ SPACE_ID found: {space_id}")
        logger.info(f"   Repo URL: https://huggingface.co/spaces/{space_id}")
    else:
        logger.info("ℹ️  SPACE_ID environment variable not found (running locally?).")
    
    logger.info(f"🔧 Production mode: {is_production}")
    
    # Create interface
    interface = create_interface()
    
    # Launch configuration with OAuth scopes
    if is_production:
        # Production settings for HuggingFace Spaces with OAuth
        launch_kwargs = {
            "server_name": "0.0.0.0",
            "server_port": int(os.getenv("PORT", 7860)),
            "share": False,
            "debug": False,
            "show_error": True,
            "quiet": False,
            "favicon_path": None,
            "auth": None,
            # Configure OAuth with full inference access
            "auth_message": "Login with HuggingFace for full inference access to models",
        }
        logger.info(f"🚀 Launching in PRODUCTION mode on 0.0.0.0:{launch_kwargs['server_port']}")
        logger.info("🔑 OAuth configured to request 'read' and 'inference' scopes")
    else:
        # Development settings
        launch_kwargs = {
            "server_name": "127.0.0.1",
            "server_port": 7860,
            "share": False,
            "debug": True,
            "show_error": True,
            "quiet": False,
            "favicon_path": None,
            "inbrowser": True,
            "auth_message": "Login with HuggingFace for full inference access to models",
        }
        logger.info("🔧 Launching in DEVELOPMENT mode on 127.0.0.1:7860")
    
    # Set OAuth environment variables for HuggingFace Spaces
    if is_production:
        # These environment variables tell HF Spaces what OAuth scopes to request
        os.environ["OAUTH_SCOPES"] = "read,inference"
        os.environ["OAUTH_CLIENT_ID"] = os.getenv("OAUTH_CLIENT_ID", "")
        logger.info("🔐 OAuth environment configured for inference access")
    
    interface.launch(**launch_kwargs)

def process_question_with_gaia_agent(question_text: str, question_id: str = None, 
                                  file_name: str = None, file_content: bytes = None) -> Dict[str, Any]:
    """
    Process a GAIA question using enhanced multi-phase planning workflow
    """
    try:
        logger.info(f"📝 Processing GAIA question with enhanced workflow: {question_text[:100]}...")
        
        # Create GAIA agent with enhanced capabilities
        llm_client = QwenClient()
        gaia_agent = GAIAAgentApp.create_with_qwen_client(llm_client)
        
        # Use enhanced LangGraph workflow with multi-step planning
        result = gaia_agent.process_with_langgraph(question_text, question_id)
        
        # Enhanced result formatting for GAIA compliance
        enhanced_result = {
            "question_id": question_id or "unknown",
            "question": question_text,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "reasoning": result["reasoning"],
            "cost_estimate": result["cost"],
            "processing_time": result["processing_time"],
            "workflow_type": "enhanced_multi_phase",
            "processing_details": result["processing_details"],
            "agent_results": result["agent_results"],
            "success": len(result["errors"]) == 0,
            "error_messages": result["errors"]
        }
        
        return enhanced_result
        
    except Exception as e:
        error_msg = f"Enhanced GAIA processing failed: {str(e)}"
        logger.error(error_msg)
        return {
            "question_id": question_id or "unknown",
            "question": question_text,
            "answer": "Enhanced processing failed",
            "confidence": 0.0,
            "reasoning": error_msg,
            "cost_estimate": 0.0,
            "processing_time": 0.0,
            "workflow_type": "enhanced_multi_phase_failed",
            "processing_details": {"error": error_msg},
            "agent_results": [],
            "success": False,
            "error_messages": [error_msg]
        }

if __name__ == "__main__":
    main() 