#!/usr/bin/env python3
"""
Test Production Fixes for GAIA Agent System
Quick validation that error handling improvements are working
"""

import logging
import time
from typing import List, Dict, Any

from models.qwen_client import QwenClient
from workflow.gaia_workflow import SimpleGAIAWorkflow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionFixTester:
    """Test the production fixes for error handling and robustness"""
    
    def __init__(self):
        try:
            self.llm_client = QwenClient()
            self.workflow = SimpleGAIAWorkflow(self.llm_client)
            logger.info("✅ Test environment initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize test environment: {e}")
            raise
    
    def test_error_handling_scenarios(self) -> Dict[str, Any]:
        """Test various error scenarios that were causing production failures"""
        
        test_scenarios = [
            {
                "name": "Wikipedia Research Failure Simulation",
                "question": "What is the most obscure fictional character from the imaginary book 'Zzzzz12345NonExistent'?",
                "expected_behavior": "Should fail gracefully and provide fallback response"
            },
            {
                "name": "Mathematical Reasoning with Complex Data",
                "question": "Calculate the square root of negative infinity divided by zero plus the factorial of pi",
                "expected_behavior": "Should handle impossible math gracefully"
            },
            {
                "name": "Conversion with Invalid Units",
                "question": "Convert 50 zorkples to flibbers using the international zorkple standard",
                "expected_behavior": "Should recognize invalid units and respond appropriately"
            },
            {
                "name": "Web Research with Rate Limiting Simulation",
                "question": "What are the current stock prices for all Fortune 500 companies as of this exact moment?",
                "expected_behavior": "Should handle external API limitations gracefully"
            },
            {
                "name": "Complex Multi-Agent Question",
                "question": "Analyze the correlation between quantum entanglement and the price of tea in 17th century Mongolia while also calculating the fibonacci sequence backwards from infinity",
                "expected_behavior": "Should route to multiple agents and synthesize results"
            }
        ]
        
        results = {
            "test_summary": {
                "total_tests": len(test_scenarios),
                "passed": 0,
                "failed": 0,
                "errors": []
            },
            "detailed_results": []
        }
        
        for i, scenario in enumerate(test_scenarios, 1):
            logger.info(f"\n🧪 Test {i}/{len(test_scenarios)}: {scenario['name']}")
            logger.info(f"Question: {scenario['question']}")
            
            start_time = time.time()
            
            try:
                # Process the question
                result_state = self.workflow.process_question(
                    question=scenario['question'],
                    task_id=f"fix_test_{i}"
                )
                
                processing_time = time.time() - start_time
                
                # Analyze the result
                test_result = self._analyze_test_result(scenario, result_state, processing_time)
                results["detailed_results"].append(test_result)
                
                if test_result["passed"]:
                    results["test_summary"]["passed"] += 1
                    logger.info(f"✅ PASSED: {test_result['reason']}")
                else:
                    results["test_summary"]["failed"] += 1
                    logger.warning(f"❌ FAILED: {test_result['reason']}")
                
                # Log key metrics
                logger.info(f"   📊 Confidence: {result_state.final_confidence:.2f}")
                logger.info(f"   ⏱️  Time: {processing_time:.2f}s")
                logger.info(f"   💰 Cost: ${result_state.total_cost:.4f}")
                logger.info(f"   🎯 Answer: {result_state.final_answer[:100]}...")
                
            except Exception as e:
                error_msg = f"Exception in test {i}: {str(e)}"
                logger.error(f"❌ ERROR: {error_msg}")
                results["test_summary"]["errors"].append(error_msg)
                results["test_summary"]["failed"] += 1
                
                results["detailed_results"].append({
                    "test_name": scenario['name'],
                    "passed": False,
                    "reason": f"Test exception: {str(e)}",
                    "processing_time": time.time() - start_time,
                    "confidence": 0.0,
                    "answer": "Test failed with exception"
                })
        
        return results
    
    def _analyze_test_result(self, scenario: Dict[str, Any], result_state, processing_time: float) -> Dict[str, Any]:
        """Analyze if a test result meets expectations for error handling"""
        
        test_result = {
            "test_name": scenario['name'],
            "passed": False,
            "reason": "",
            "processing_time": processing_time,
            "confidence": result_state.final_confidence,
            "answer": result_state.final_answer,
            "agents_used": [role.value for role in result_state.agent_results.keys()],
            "error_count": len(result_state.error_messages)
        }
        
        # Check for catastrophic failures
        if result_state.final_answer is None or result_state.final_answer == "":
            test_result["reason"] = "Critical failure: No answer generated"
            return test_result
        
        # Check for system crash indicators
        crash_indicators = [
            "system not initialized",
            "workflow execution failed",
            "unable to process question - no agent results available"
        ]
        
        answer_lower = result_state.final_answer.lower()
        if any(indicator in answer_lower for indicator in crash_indicators):
            test_result["reason"] = "System crash detected in response"
            return test_result
        
        # Check for graceful error handling
        graceful_indicators = [
            "processing encountered difficulties",
            "research sources failed", 
            "reasoning failed",
            "conversion failed",
            "mathematical complexity",
            "limited information available"
        ]
        
        has_graceful_handling = any(indicator in answer_lower for indicator in graceful_indicators)
        
        # Evaluate based on scenario expectations
        if has_graceful_handling and result_state.final_confidence >= 0.1:
            test_result["passed"] = True
            test_result["reason"] = "Graceful error handling with reasonable confidence"
        elif not has_graceful_handling and result_state.final_confidence >= 0.3:
            test_result["passed"] = True
            test_result["reason"] = "Provided meaningful answer with acceptable confidence"
        elif result_state.final_confidence > 0.0 and len(result_state.agent_results) > 0:
            test_result["passed"] = True
            test_result["reason"] = "System remained stable and attempted processing"
        else:
            test_result["reason"] = f"Insufficient error handling or system instability (confidence: {result_state.final_confidence:.2f})"
        
        return test_result
    
    def run_comprehensive_test(self) -> None:
        """Run comprehensive test and report results"""
        
        logger.info("🚀 Starting Production Fix Validation Tests")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            results = self.test_error_handling_scenarios()
            total_time = time.time() - start_time
            
            # Print summary
            summary = results["test_summary"]
            logger.info("\n" + "=" * 60)
            logger.info("📋 TEST SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total Tests: {summary['total_tests']}")
            logger.info(f"✅ Passed: {summary['passed']}")
            logger.info(f"❌ Failed: {summary['failed']}")
            logger.info(f"⚠️  Errors: {len(summary['errors'])}")
            logger.info(f"📊 Success Rate: {summary['passed']/summary['total_tests']*100:.1f}%")
            logger.info(f"⏱️  Total Time: {total_time:.2f}s")
            
            # Success threshold
            success_rate = summary['passed'] / summary['total_tests']
            if success_rate >= 0.8:  # 80% success rate for error handling
                logger.info("🎉 PRODUCTION FIXES VALIDATION: PASSED")
                logger.info("System demonstrates robust error handling and graceful degradation")
            else:
                logger.warning("⚠️  PRODUCTION FIXES VALIDATION: NEEDS IMPROVEMENT")
                logger.warning(f"Success rate {success_rate*100:.1f}% below 80% threshold")
            
            # Print any errors
            if summary['errors']:
                logger.error("\n🔥 ERRORS ENCOUNTERED:")
                for error in summary['errors']:
                    logger.error(f"   - {error}")
            
        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {str(e)}")
            raise

def main():
    """Main test execution"""
    try:
        tester = ProductionFixTester()
        tester.run_comprehensive_test()
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        exit(1)

if __name__ == "__main__":
    main() 