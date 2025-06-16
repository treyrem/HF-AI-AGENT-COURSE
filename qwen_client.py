#!/usr/bin/env python3
"""
HuggingFace Qwen 2.5 Model Client
Handles inference for router, main, and complex models with cost tracking
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.language_models.llms import LLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTier(Enum):
    """Model complexity tiers for cost optimization"""
    ROUTER = "router"     # Fast, cheap routing decisions
    MAIN = "main"         # Balanced performance  
    COMPLEX = "complex"   # Best performance for hard tasks

@dataclass
class ModelConfig:
    """Configuration for each model"""
    name: str
    tier: ModelTier
    max_tokens: int
    temperature: float
    cost_per_token: float  # Estimated cost per token
    timeout: int
    requires_special_auth: bool = False  # For Nebius API models

@dataclass
class InferenceResult:
    """Result of model inference with metadata"""
    response: str
    model_used: str
    tokens_used: int
    cost_estimate: float
    response_time: float
    success: bool
    error: Optional[str] = None

class QwenClient:
    """HuggingFace client with fallback model support"""
    
    def __init__(self, hf_token: Optional[str] = None):
        """Initialize the client with HuggingFace token for Qwen models only"""
        # Debug: Print environment variables for verification
        print("\n=== Environment Variables ===")
        for key in ["HUGGINGFACE_TOKEN", "HF_TOKEN"]:
            val = os.getenv(key)
            print(f"{key}: {'***' + val[-4:] if val else 'Not set'}")
        print("==========================\n")
        
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HuggingFace token is required for model access. Please provide HF_TOKEN or login with inference permissions.")
        
        print(f"Using HuggingFace token ending with: {self.hf_token[-8:]}")
        
        # Initialize cost tracking first
        self.total_cost = 0.0
        self.request_count = 0
        self.budget_limit = 0.10  # $0.10 total budget
            
        # Define model configurations using more accessible models
        self.models = {
            ModelTier.ROUTER: ModelConfig(
                name="mistralai/Mistral-7B-Instruct-v0.2",
                tier=ModelTier.ROUTER,
                max_tokens=512,
                temperature=0.3,
                cost_per_token=0.0002,
                timeout=30,
                requires_special_auth=False
            ),
            ModelTier.MAIN: ModelConfig(
                name="mistralai/Mixtral-8x7B-Instruct-v0.1",
                tier=ModelTier.MAIN,
                max_tokens=1024,
                temperature=0.2,
                cost_per_token=0.0005,
                timeout=45,
                requires_special_auth=False
            ),
            ModelTier.COMPLEX: ModelConfig(
                name="google/gemma-7b-it",
                tier=ModelTier.COMPLEX,
                max_tokens=2048,
                temperature=0.1,
                cost_per_token=0.0007,
                timeout=60,
                requires_special_auth=False
            )
        }
        
        # Initialize clients
        self.inference_clients = {}
        self.langchain_clients = {}
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize HuggingFace clients for the models"""
        
        logger.info("🎯 Initializing models via HuggingFace Inference API...")
        success = self._try_initialize_models(self.models, "Model")
        
        if not success:
            raise RuntimeError("Failed to initialize any models. Please check your HF_TOKEN has inference permissions and try again.")
        
        # Test the main model to ensure it's working
        logger.info("🧪 Testing model connectivity...")
        try:
            test_result = self.generate("Hello, please respond with 'Connection successful' if you can read this.", max_tokens=20)
            if test_result.success and test_result.response.strip():
                logger.info(f"✅ Models ready: '{test_result.response.strip()}'")
            else:
                logger.error(f"❌ Model test failed: {test_result}")
                raise RuntimeError("Model connectivity test failed")
        except Exception as e:
            logger.error(f"❌ Model test exception: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
    
    def _try_initialize_models(self, model_configs: Dict, model_type: str) -> bool:
        """Try to initialize models"""
        success_count = 0
        
        for tier, config in model_configs.items():
            try:
                logger.info(f"🔄 Initializing {tier.value} model: {config.name}")
                
                # Initialize the clients with detailed error handling
                try:
                    self.inference_clients[tier] = InferenceClient(
                        model=config.name,
                        token=self.hf_token,
                        timeout=config.timeout
                    )
                    logger.info(f"  ✅ Successfully created InferenceClient for {config.name}")
                except Exception as e:
                    logger.error(f"  ❌ Failed to create InferenceClient for {config.name}: {str(e)}")
                    raise
                
                try:
                    self.langchain_clients[tier] = HuggingFaceEndpoint(
                        repo_id=config.name,
                        max_new_tokens=config.max_tokens,
                        temperature=config.temperature,
                        huggingfacehub_api_token=self.hf_token,
                        timeout=config.timeout
                    )
                    logger.info(f"  ✅ Successfully created HuggingFaceEndpoint for {config.name}")
                except Exception as e:
                    logger.error(f"  ❌ Failed to create HuggingFaceEndpoint for {config.name}: {str(e)}")
                    raise
                
                # Quick test to verify authentication and model access
                try:
                    logger.info(f"  🧪 Testing model with a simple prompt...")
                    test_messages = [{"role": "user", "content": "Hello, please respond with 'OK'"}]
                    test_response = self.inference_clients[tier].chat_completion(
                        messages=test_messages,
                        model=config.name,
                        max_tokens=5,
                        temperature=0.1
                    )
                    logger.info(f"  ✅ Model test successful. Response: {test_response}")
                    success_count += 1
                    logger.info(f"✅ {tier.value} model fully initialized: {config.name}")
                except Exception as auth_error:
                    logger.error(f"❌ {tier.value} model test failed for {config.name}")
                    logger.error(f"   Error details: {str(auth_error)}")
                    logger.error(f"   Response type: {type(test_response) if 'test_response' in locals() else 'N/A'}")
                    logger.error(f"   Response content: {test_response if 'test_response' in locals() else 'N/A'}")
                    self.inference_clients[tier] = None
                    self.langchain_clients[tier] = None
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize {tier.value} model: {config.name}")
                logger.error(f"   Error type: {type(e).__name__}")
                logger.error(f"   Error details: {str(e)}")
                import traceback
                logger.error(f"   Stack trace: {traceback.format_exc()}")
                self.inference_clients[tier] = None
                self.langchain_clients[tier] = None
        
        return success_count > 0
    
    def get_model_status(self) -> Dict[str, bool]:
        """Check which models are available"""
        status = {}
        for tier in ModelTier:
            status[tier.value] = (
                self.inference_clients.get(tier) is not None and 
                self.langchain_clients.get(tier) is not None
            )
        return status
    
    def select_model_tier(self, complexity: str = "medium", budget_conscious: bool = True, question_text: str = "") -> ModelTier:
        """Smart model selection based on task complexity, budget, and question analysis"""
        
        # Check budget constraints
        budget_used_percent = (self.total_cost / self.budget_limit) * 100
        
        if budget_conscious and budget_used_percent > 80:
            logger.warning(f"Budget critical ({budget_used_percent:.1f}% used), forcing router model")
            return ModelTier.ROUTER
        elif budget_conscious and budget_used_percent > 60:
            logger.warning(f"Budget warning ({budget_used_percent:.1f}% used), limiting complex model usage")
            complexity = "simple" if complexity == "complex" else complexity
            
        # Enhanced complexity analysis based on question content
        if question_text:
            question_lower = question_text.lower()
            
            # Indicators for complex reasoning (use 72B model)
            complex_indicators = [
                "analyze", "explain why", "reasoning", "logic", "complex", "difficult",
                "multi-step", "calculate and explain", "compare and contrast",
                "what is the relationship", "how does", "why is", "prove that",
                "step by step", "detailed analysis", "comprehensive"
            ]
            
            # Indicators for simple tasks (use 7B model)  
            simple_indicators = [
                "what is", "who is", "when", "where", "simple", "quick",
                "yes or no", "true or false", "list", "name", "find"
            ]
            
            # Math and coding indicators (use 32B model - good balance)
            math_indicators = [
                "calculate", "compute", "solve", "equation", "formula", "math",
                "number", "total", "sum", "average", "percentage", "code", "program"
            ]
            
            # File processing indicators (use 32B+ models)
            file_indicators = [
                "image", "picture", "photo", "audio", "sound", "video", "file",
                "document", "excel", "csv", "data", "chart", "graph"
            ]
            
            # Count indicators
            complex_score = sum(1 for indicator in complex_indicators if indicator in question_lower)
            simple_score = sum(1 for indicator in simple_indicators if indicator in question_lower)
            math_score = sum(1 for indicator in math_indicators if indicator in question_lower)
            file_score = sum(1 for indicator in file_indicators if indicator in question_lower)
            
            # Auto-detect complexity based on content
            if complex_score >= 2 or len(question_text) > 200:
                complexity = "complex"
            elif file_score >= 1 or math_score >= 2:
                complexity = "medium"
            elif simple_score >= 2 and complex_score == 0:
                complexity = "simple"
                
        # Select based on complexity with budget awareness
        if complexity == "complex" and budget_used_percent < 70:
            selected_tier = ModelTier.COMPLEX
        elif complexity == "simple" or budget_used_percent > 75:
            selected_tier = ModelTier.ROUTER
        else:
            selected_tier = ModelTier.MAIN
            
        # Fallback if selected model unavailable
        if not self.inference_clients.get(selected_tier):
            logger.warning(f"Selected model {selected_tier.value} unavailable, falling back")
            for fallback in [ModelTier.MAIN, ModelTier.ROUTER, ModelTier.COMPLEX]:
                if self.inference_clients.get(fallback):
                    selected_tier = fallback
                    break
            else:
                raise RuntimeError("No models available")
        
        # Log selection reasoning
        logger.info(f"Selected {selected_tier.value} model (complexity: {complexity}, budget: {budget_used_percent:.1f}%)")
        return selected_tier
    
    async def generate_async(self, 
                           prompt: str, 
                           tier: Optional[ModelTier] = None,
                           max_tokens: Optional[int] = None) -> InferenceResult:
        """Async text generation with models via HuggingFace Inference API"""
        
        if tier is None:
            tier = self.select_model_tier(question_text=prompt)
            
        config = self.models[tier]
        client = self.inference_clients.get(tier)
        
        if not client:
            return InferenceResult(
                response="",
                model_used=config.name,
                tokens_used=0,
                cost_estimate=0.0,
                response_time=0.0,
                success=False,
                error=f"Model {tier.value} not available"
            )
        
        start_time = time.time()
        
        try:
            # Use specified max_tokens or model default
            tokens = max_tokens or config.max_tokens
            
            # Prepare messages for chat completion
            messages = [{"role": "user", "content": prompt}]
            
            logger.info(f"🤖 Generating with {config.name} (max_tokens={tokens}, temp={config.temperature})...")
            
            # Get the raw response
            response = client.chat_completion(
                messages=messages,
                model=config.name,
                max_tokens=tokens,
                temperature=config.temperature,
                stop_sequences=["\n"]
            )
            
            # Extract response text - handle different response formats
            response_text = ""
            if isinstance(response, dict):
                if 'choices' in response and len(response['choices']) > 0:
                    choice = response['choices'][0]
                    if isinstance(choice, dict) and 'message' in choice and 'content' in choice['message']:
                        response_text = choice['message']['content'].strip()
                    else:
                        response_text = str(choice.get('text', '')).strip()
                elif 'generated_text' in response:
                    response_text = response['generated_text'].strip()
            elif isinstance(response, str):
                response_text = response.strip()
            
            if not response_text:
                raise ValueError("Empty response from model")
                
            # Estimate tokens (this is approximate)
            estimated_tokens = len(prompt.split()) + len(response_text.split())
            
            # Calculate cost
            cost = (estimated_tokens * config.cost_per_token) / 1000  # Convert to cost per token
            
            # Update total cost
            self.total_cost += cost
            self.request_count += 1
            
            logger.debug(f"Generated response: {response_text[:200]}...")
            
            return InferenceResult(
                response=response_text,
                model_used=config.name,
                tokens_used=estimated_tokens,
                cost_estimate=cost,
                response_time=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Generation failed: {error_msg}")
            logger.exception("Generation error details:")
            return InferenceResult(
                response="",
                model_used=config.name,
                tokens_used=0,
                cost_estimate=0.0,
                response_time=time.time() - start_time,
                success=False,
                error=error_msg
            )
    
    def generate(self, 
                prompt: str, 
                tier: Optional[ModelTier] = None,
                max_tokens: Optional[int] = None) -> InferenceResult:
        """Synchronous text generation (wrapper for async)"""
        import asyncio
        
        # Create event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(
            self.generate_async(prompt, tier, max_tokens)
        )
    
    def get_langchain_llm(self, tier: ModelTier) -> Optional[LLM]:
        """Get LangChain LLM instance for agent integration"""
        return self.langchain_clients.get(tier)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage and cost statistics"""
        return {
            "total_cost": self.total_cost,
            "request_count": self.request_count,
            "budget_limit": self.budget_limit,
            "budget_remaining": self.budget_limit - self.total_cost,
            "budget_used_percent": (self.total_cost / self.budget_limit) * 100,
            "average_cost_per_request": self.total_cost / max(self.request_count, 1),
            "models_available": self.get_model_status()
        }
    
    def reset_usage_tracking(self):
        """Reset usage statistics (for testing/development)"""
        self.total_cost = 0.0
        self.request_count = 0
        logger.info("Usage tracking reset")

# Test functions
def test_model_connection(client: QwenClient, tier: ModelTier):
    """Test connection to a specific model tier"""
    test_prompt = "Hello! Please respond with 'Connection successful' if you can read this."
    
    logger.info(f"Testing {tier.value} model...")
    result = client.generate(test_prompt, tier=tier, max_tokens=50)
    
    if result.success:
        logger.info(f"✅ {tier.value} model test successful: {result.response[:50]}...")
        logger.info(f"   Response time: {result.response_time:.2f}s")
        logger.info(f"   Cost estimate: ${result.cost_estimate:.6f}")
    else:
        logger.error(f"❌ {tier.value} model test failed: {result.error}")
    
    return result.success

def test_all_models():
    """Test all available models"""
    logger.info("🧪 Testing all Qwen models...")
    
    client = QwenClient()
    
    results = {}
    for tier in ModelTier:
        results[tier] = test_model_connection(client, tier)
    
    logger.info("📊 Test Results Summary:")
    for tier, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"   {tier.value:8}: {status}")
    
    logger.info("💰 Usage Statistics:")
    stats = client.get_usage_stats()
    for key, value in stats.items():
        if key != "models_available":
            logger.info(f"   {key}: {value}")
    
    return results

if __name__ == "__main__":
    # Load environment variables for testing
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run tests when script executed directly
    test_all_models() 