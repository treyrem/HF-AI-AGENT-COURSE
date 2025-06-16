#!/usr/bin/env python3
"""
Integration test for GAIA Agents
Tests Web Researcher, File Processor, and Reasoning agents
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from agents.state import GAIAAgentState, QuestionType
from agents.web_researcher import WebResearchAgent
from agents.file_processor_agent import FileProcessorAgent
from agents.reasoning_agent import ReasoningAgent
from models.qwen_client import QwenClient

def test_agents():
    """Test all implemented agents"""
    
    print("🤖 GAIA Agents Integration Test")
    print("=" * 50)
    
    # Initialize LLM client
    try:
        llm_client = QwenClient()
    except Exception as e:
        print(f"❌ Failed to initialize LLM client: {e}")
        return False
    
    results = []
    start_time = time.time()
    
    # Test 1: Web Research Agent
    print("\n🌐 Testing Web Research Agent...")
    web_agent = WebResearchAgent(llm_client)
    
    web_test_cases = [
        {
            "question": "What is the capital of France?",
            "question_type": QuestionType.WIKIPEDIA,
            "complexity": "simple"
        },
        {
            "question": "Find information about Python programming language",
            "question_type": QuestionType.WEB_RESEARCH,
            "complexity": "medium"
        }
    ]
    
    for i, test_case in enumerate(web_test_cases, 1):
        state = GAIAAgentState()
        state.question = test_case["question"]
        state.question_type = test_case["question_type"]
        state.complexity_assessment = test_case["complexity"]
        
        try:
            result_state = web_agent.process(state)
            success = len(result_state.agent_results) > 0 and list(result_state.agent_results.values())[-1].success
            results.append(('Web Research', f'Test {i}', success, list(result_state.agent_results.values())[-1].processing_time if result_state.agent_results else 0))
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   Test {i}: {status}")
            
        except Exception as e:
            results.append(('Web Research', f'Test {i}', False, 0))
            print(f"   Test {i}: ❌ FAIL ({e})")
    
    # Test 2: File Processor Agent
    print("\n📁 Testing File Processor Agent...")
    file_agent = FileProcessorAgent(llm_client)
    
    # Create test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CSV test file
        csv_path = os.path.join(temp_dir, "test.csv")
        with open(csv_path, 'w') as f:
            f.write("name,age,salary\nAlice,25,50000\nBob,30,60000\nCharlie,35,70000")
        
        # Create Python test file  
        py_path = os.path.join(temp_dir, "test.py")
        with open(py_path, 'w') as f:
            f.write("def calculate_sum(a, b):\n    return a + b\n\nresult = calculate_sum(5, 3)")
        
        file_test_cases = [
            {
                "question": "What is the average salary in this data?",
                "file_path": csv_path,
                "question_type": QuestionType.FILE_PROCESSING,
                "complexity": "medium"
            },
            {
                "question": "What does this Python code do?",
                "file_path": py_path,
                "question_type": QuestionType.FILE_PROCESSING,
                "complexity": "simple"
            }
        ]
        
        for i, test_case in enumerate(file_test_cases, 1):
            state = GAIAAgentState()
            state.question = test_case["question"]
            state.file_path = test_case["file_path"]
            state.question_type = test_case["question_type"]
            state.complexity_assessment = test_case["complexity"]
            
            try:
                result_state = file_agent.process(state)
                success = len(result_state.agent_results) > 0 and list(result_state.agent_results.values())[-1].success
                results.append(('File Processor', f'Test {i}', success, list(result_state.agent_results.values())[-1].processing_time if result_state.agent_results else 0))
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"   Test {i}: {status}")
                
            except Exception as e:
                results.append(('File Processor', f'Test {i}', False, 0))
                print(f"   Test {i}: ❌ FAIL ({e})")
    
    # Test 3: Reasoning Agent
    print("\n🧠 Testing Reasoning Agent...")
    reasoning_agent = ReasoningAgent(llm_client)
    
    reasoning_test_cases = [
        {
            "question": "Calculate 15% of 200",
            "question_type": QuestionType.REASONING,
            "complexity": "simple"
        },
        {
            "question": "Convert 100 celsius to fahrenheit",
            "question_type": QuestionType.REASONING,
            "complexity": "simple"
        },
        {
            "question": "What is the average of 10, 15, 20, 25, 30?",
            "question_type": QuestionType.REASONING,
            "complexity": "medium"
        }
    ]
    
    for i, test_case in enumerate(reasoning_test_cases, 1):
        state = GAIAAgentState()
        state.question = test_case["question"]
        state.question_type = test_case["question_type"]
        state.complexity_assessment = test_case["complexity"]
        
        try:
            result_state = reasoning_agent.process(state)
            success = len(result_state.agent_results) > 0 and list(result_state.agent_results.values())[-1].success
            results.append(('Reasoning', f'Test {i}', success, list(result_state.agent_results.values())[-1].processing_time if result_state.agent_results else 0))
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   Test {i}: {status}")
            
        except Exception as e:
            results.append(('Reasoning', f'Test {i}', False, 0))
            print(f"   Test {i}: ❌ FAIL ({e})")
    
    # Summary
    total_time = time.time() - start_time
    passed_tests = sum(1 for _, _, success, _ in results if success)
    total_tests = len(results)
    
    print("\n" + "=" * 50)
    print("📊 AGENT TEST RESULTS")
    print("=" * 50)
    
    # Results by agent
    agents = {}
    for agent, test, success, exec_time in results:
        if agent not in agents:
            agents[agent] = {'passed': 0, 'total': 0, 'time': 0}
        agents[agent]['total'] += 1
        agents[agent]['time'] += exec_time
        if success:
            agents[agent]['passed'] += 1
    
    for agent, stats in agents.items():
        pass_rate = (stats['passed'] / stats['total']) * 100
        avg_time = stats['time'] / stats['total']
        status = "✅" if pass_rate == 100 else "⚠️" if pass_rate >= 80 else "❌"
        print(f"{status} {agent:15}: {stats['passed']}/{stats['total']} ({pass_rate:5.1f}%) - Avg: {avg_time:.3f}s")
    
    # Overall results
    overall_pass_rate = (passed_tests / total_tests) * 100
    print(f"\n🎯 OVERALL: {passed_tests}/{total_tests} tests passed ({overall_pass_rate:.1f}%)")
    print(f"⏱️  TOTAL TIME: {total_time:.2f} seconds")
    
    # Success criteria
    if overall_pass_rate >= 80:
        print("🚀 AGENTS READY! Multi-agent system is working correctly!")
        return True
    else:
        print("⚠️  ISSUES FOUND! Check individual agent failures above")
        return False

if __name__ == "__main__":
    success = test_agents()
    sys.exit(0 if success else 1) 