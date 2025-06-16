#!/usr/bin/env python3
"""
Test Router Agent for GAIA Agent System
Tests question classification and agent selection logic
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from agents.state import GAIAAgentState, QuestionType, AgentRole
from agents.router import RouterAgent
from models.qwen_client import QwenClient

def test_router_agent():
    """Test the router agent with various question types"""
    
    print("🧭 GAIA Router Agent Test")
    print("=" * 40)
    
    # Initialize LLM client and router
    try:
        llm_client = QwenClient()
        router = RouterAgent(llm_client)
    except Exception as e:
        print(f"❌ Failed to initialize router: {e}")
        return False
    
    # Test cases covering all question types
    test_cases = [
        {
            "question": "What is the capital of France?",
            "expected_type": [QuestionType.WIKIPEDIA, QuestionType.WEB_RESEARCH, QuestionType.UNKNOWN],  # Allow multiple valid types
            "expected_agents": [AgentRole.WEB_RESEARCHER]
        },
        {
            "question": "Calculate 25% of 400 and add 50",
            "expected_type": [QuestionType.MATHEMATICAL],
            "expected_agents": [AgentRole.REASONING_AGENT]
        },
        {
            "question": "What information can you extract from this CSV file?",
            "expected_type": [QuestionType.FILE_PROCESSING],
            "expected_agents": [AgentRole.FILE_PROCESSOR],
            "has_file": True
        },
        {
            "question": "Search for recent news about artificial intelligence",
            "expected_type": [QuestionType.WEB_RESEARCH],
            "expected_agents": [AgentRole.WEB_RESEARCHER]
        },
        {
            "question": "What does this Python code do and how can it be improved?",
            "expected_type": [QuestionType.CODE_EXECUTION, QuestionType.FILE_PROCESSING],  # Both are valid
            "expected_agents": [AgentRole.FILE_PROCESSOR, AgentRole.CODE_EXECUTOR],  # Either is acceptable
            "has_file": True
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['question'][:50]}... ---")
        
        # Create state
        state = GAIAAgentState()
        state.question = test_case["question"]
        if test_case.get("has_file"):
            state.file_name = "test_file.csv"
            state.file_path = "/tmp/test_file.csv"
        
        try:
            # Process with router
            result_state = router.route_question(state)
            
            # Check results
            type_correct = result_state.question_type in test_case["expected_type"]
            agents_correct = any(agent in result_state.selected_agents for agent in test_case["expected_agents"])
            
            success = type_correct and agents_correct
            results.append(success)
            
            print(f"   Question Type: {result_state.question_type.value} ({'✅' if type_correct else '❌'})")
            print(f"   Selected Agents: {[a.value for a in result_state.selected_agents]} ({'✅' if agents_correct else '❌'})")
            print(f"   Complexity: {result_state.complexity_assessment}")
            print(f"   Overall: {'✅ PASS' if success else '❌ FAIL'}")
            
        except Exception as e:
            print(f"   ❌ FAIL: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    pass_rate = (passed / total) * 100
    
    print("\n" + "=" * 40)
    print(f"🎯 ROUTER RESULTS: {passed}/{total} ({pass_rate:.1f}%)")
    
    if pass_rate >= 80:
        print("🚀 Router working correctly!")
        return True
    else:
        print("⚠️ Router needs improvement")
        return False

if __name__ == "__main__":
    success = test_router_agent()
    sys.exit(0 if success else 1) 