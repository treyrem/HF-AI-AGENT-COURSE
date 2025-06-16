#!/usr/bin/env python3
"""
Complete Integration Test for GAIA Agent System
Tests the full pipeline: Router -> Agents -> Tools -> Results
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from agents.state import GAIAAgentState, QuestionType, AgentRole
from agents.router import RouterAgent
from agents.web_researcher import WebResearchAgent
from agents.file_processor_agent import FileProcessorAgent
from agents.reasoning_agent import ReasoningAgent
from models.qwen_client import QwenClient

def test_complete_pipeline():
    """Test the complete GAIA agent pipeline"""
    
    print("🚀 GAIA Complete Integration Test")
    print("=" * 50)
    
    # Initialize system
    try:
        llm_client = QwenClient()
        router = RouterAgent(llm_client)
        web_agent = WebResearchAgent(llm_client)
        file_agent = FileProcessorAgent(llm_client)
        reasoning_agent = ReasoningAgent(llm_client)
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return False
    
    # End-to-end test cases
    test_cases = [
        {
            "question": "What is the population of Paris?",
            "description": "Simple Wikipedia/web research question",
            "expected_agent": AgentRole.WEB_RESEARCHER
        },
        {
            "question": "Calculate the area of a circle with radius 5 meters",
            "description": "Mathematical reasoning with unit conversion",
            "expected_agent": AgentRole.REASONING_AGENT
        },
        {
            "question": "What is the average of these numbers: 10, 20, 30, 40, 50?",
            "description": "Statistical calculation",
            "expected_agent": AgentRole.REASONING_AGENT
        }
    ]
    
    results = []
    total_cost = 0.0
    start_time = time.time()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['description']}")
        print(f"   Question: {test_case['question']}")
        
        try:
            # Step 1: Initialize state
            state = GAIAAgentState()
            state.task_id = f"test_{i}"
            state.question = test_case["question"]
            
            # Step 2: Route question
            routed_state = router.route_question(state)
            print(f"   ✅ Router: {routed_state.question_type.value} -> {[a.value for a in routed_state.selected_agents]}")
            
            # Step 3: Process with appropriate agent
            if test_case["expected_agent"] in routed_state.selected_agents:
                if test_case["expected_agent"] == AgentRole.WEB_RESEARCHER:
                    processed_state = web_agent.process(routed_state)
                elif test_case["expected_agent"] == AgentRole.REASONING_AGENT:
                    processed_state = reasoning_agent.process(routed_state)
                elif test_case["expected_agent"] == AgentRole.FILE_PROCESSOR:
                    processed_state = file_agent.process(routed_state)
                else:
                    print(f"   ⚠️  Agent {test_case['expected_agent'].value} not implemented in test")
                    continue
                
                # Check results
                if processed_state.agent_results:
                    agent_result = list(processed_state.agent_results.values())[-1]
                    success = agent_result.success
                    confidence = agent_result.confidence
                    cost = processed_state.total_cost
                    processing_time = processed_state.total_processing_time
                    
                    print(f"   ✅ Agent: {agent_result.agent_role.value}")
                    print(f"   ✅ Result: {agent_result.result[:100]}...")
                    print(f"   📊 Confidence: {confidence:.2f}")
                    print(f"   💰 Cost: ${cost:.4f}")
                    print(f"   ⏱️  Time: {processing_time:.2f}s")
                    
                    total_cost += cost
                    results.append(success)
                    
                    print(f"   🎯 Overall: {'✅ PASS' if success else '❌ FAIL'}")
                else:
                    print(f"   ❌ No agent results produced")
                    results.append(False)
            else:
                print(f"   ⚠️  Expected agent {test_case['expected_agent'].value} not selected")
                results.append(False)
                
        except Exception as e:
            print(f"   ❌ Pipeline failed: {e}")
            results.append(False)
    
    # File processing test with actual file
    print(f"\n🧪 Test 4: File Processing with CSV")
    print(f"   Description: Complete file analysis pipeline")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test CSV
            csv_path = os.path.join(temp_dir, "sales_data.csv")
            with open(csv_path, 'w') as f:
                f.write("product,sales,price\nWidget A,100,25.50\nWidget B,150,30.00\nWidget C,80,22.75")
            
            # Initialize state with file
            state = GAIAAgentState()
            state.task_id = "test_file"
            state.question = "What is the total sales value across all products?"
            state.file_name = "sales_data.csv"
            state.file_path = csv_path
            
            # Route and process
            routed_state = router.route_question(state)
            processed_state = file_agent.process(routed_state)
            
            if processed_state.agent_results:
                agent_result = list(processed_state.agent_results.values())[-1]
                success = agent_result.success
                total_cost += processed_state.total_cost
                results.append(success)
                
                print(f"   ✅ Router: {routed_state.question_type.value}")
                print(f"   ✅ Agent: File processor")
                print(f"   ✅ Result: {agent_result.result[:100]}...")
                print(f"   💰 Cost: ${processed_state.total_cost:.4f}")
                print(f"   🎯 Overall: {'✅ PASS' if success else '❌ FAIL'}")
            else:
                print(f"   ❌ File processing failed")
                results.append(False)
                
    except Exception as e:
        print(f"   ❌ File test failed: {e}")
        results.append(False)
    
    # Final summary
    total_time = time.time() - start_time
    passed = sum(results)
    total = len(results)
    pass_rate = (passed / total) * 100
    
    print("\n" + "=" * 50)
    print("📊 COMPLETE INTEGRATION RESULTS")
    print("=" * 50)
    print(f"🎯 Tests Passed: {passed}/{total} ({pass_rate:.1f}%)")
    print(f"💰 Total Cost: ${total_cost:.4f}")
    print(f"⏱️  Total Time: {total_time:.2f} seconds")
    print(f"📈 Average Cost per Test: ${total_cost/total:.4f}")
    print(f"⚡ Average Time per Test: {total_time/total:.2f}s")
    
    # Budget analysis
    monthly_budget = 0.10  # $0.10/month
    if total_cost <= monthly_budget:
        remaining_budget = monthly_budget - total_cost
        estimated_questions = int(remaining_budget / (total_cost / total))
        print(f"💰 Budget Status: ✅ ${remaining_budget:.4f} remaining (~{estimated_questions} more tests)")
    else:
        print(f"💰 Budget Status: ⚠️  Over budget by ${total_cost - monthly_budget:.4f}")
    
    # Success criteria
    if pass_rate >= 80 and total_cost <= 0.05:  # 80% success, reasonable cost
        print("\n🚀 INTEGRATION SUCCESS! System ready for GAIA benchmark!")
        return True
    elif pass_rate >= 80:
        print("\n✅ FUNCTIONALITY SUCCESS! (Higher cost than ideal)")
        return True
    else:
        print("\n⚠️  INTEGRATION ISSUES! Check individual test failures")
        return False

if __name__ == "__main__":
    success = test_complete_pipeline()
    sys.exit(0 if success else 1) 