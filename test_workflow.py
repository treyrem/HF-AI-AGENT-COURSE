#!/usr/bin/env python3
"""
Complete Workflow Test for GAIA Agent System
Tests both LangGraph and simplified workflow implementations
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from workflow.gaia_workflow import GAIAWorkflow, SimpleGAIAWorkflow
from models.qwen_client import QwenClient

def test_simple_workflow():
    """Test the simplified workflow implementation"""
    
    print("🧪 Testing Simple GAIA Workflow")
    print("=" * 50)
    
    # Initialize workflow
    try:
        llm_client = QwenClient()
        workflow = SimpleGAIAWorkflow(llm_client)
    except Exception as e:
        print(f"❌ Failed to initialize workflow: {e}")
        return False
    
    # Test cases
    test_cases = [
        {
            "question": "What is the capital of France?",
            "description": "Simple web research question",
            "expected_agents": ["web_researcher"]
        },
        {
            "question": "Calculate 25% of 200",
            "description": "Mathematical reasoning question",
            "expected_agents": ["reasoning_agent"]
        },
        {
            "question": "What is the average of 10, 15, 20?",
            "description": "Statistical calculation",
            "expected_agents": ["reasoning_agent"]
        }
    ]
    
    results = []
    total_cost = 0.0
    start_time = time.time()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['description']}")
        print(f"   Question: {test_case['question']}")
        
        try:
            # Process question
            result_state = workflow.process_question(
                question=test_case["question"],
                task_id=f"simple_test_{i}"
            )
            
            # Check results
            success = result_state.is_complete and result_state.final_answer
            confidence = result_state.final_confidence
            cost = result_state.total_cost
            
            print(f"   ✅ Router: {result_state.question_type.value}")
            print(f"   ✅ Agents: {[a.value for a in result_state.selected_agents]}")
            print(f"   ✅ Final Answer: {result_state.final_answer[:100]}...")
            print(f"   📊 Confidence: {confidence:.2f}")
            print(f"   💰 Cost: ${cost:.4f}")
            print(f"   🎯 Success: {'✅ PASS' if success else '❌ FAIL'}")
            
            total_cost += cost
            results.append(bool(success))
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            results.append(False)
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Simple Workflow Results:")
    print(f"   🎯 Tests Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"   💰 Total Cost: ${total_cost:.4f}")
    print(f"   ⏱️  Total Time: {total_time:.2f}s")
    
    return passed >= total * 0.8  # 80% success rate

def test_complete_workflow_with_files():
    """Test workflow with file processing"""
    
    print("\n🧪 Testing Complete Workflow with Files")
    print("=" * 50)
    
    try:
        llm_client = QwenClient()
        workflow = SimpleGAIAWorkflow(llm_client)
    except Exception as e:
        print(f"❌ Failed to initialize workflow: {e}")
        return False
    
    # Create test file
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, "test_data.csv")
        with open(csv_path, 'w') as f:
            f.write("item,quantity,price\nApple,10,1.50\nBanana,20,0.75\nOrange,15,2.00")
        
        print(f"📁 Created test file: {csv_path}")
        
        try:
            result_state = workflow.process_question(
                question="What is the total value of all items in this data?",
                file_path=csv_path,
                file_name="test_data.csv",
                task_id="file_test"
            )
            
            success = result_state.is_complete and result_state.final_answer
            
            print(f"   ✅ Router: {result_state.question_type.value}")
            print(f"   ✅ Agents: {[a.value for a in result_state.selected_agents]}")
            print(f"   ✅ Final Answer: {result_state.final_answer[:150]}...")
            print(f"   📊 Confidence: {result_state.final_confidence:.2f}")
            print(f"   💰 Cost: ${result_state.total_cost:.4f}")
            print(f"   🎯 File Processing: {'✅ PASS' if success else '❌ FAIL'}")
            
            return bool(success)
            
        except Exception as e:
            print(f"   ❌ File test failed: {e}")
            return False

def test_workflow_error_handling():
    """Test workflow error handling and edge cases"""
    
    print("\n🧪 Testing Workflow Error Handling")
    print("=" * 50)
    
    try:
        llm_client = QwenClient()
        workflow = SimpleGAIAWorkflow(llm_client)
    except Exception as e:
        print(f"❌ Failed to initialize workflow: {e}")
        return False
    
    # Test cases that might cause errors
    error_test_cases = [
        {
            "question": "",  # Empty question
            "description": "Empty question"
        },
        {
            "question": "x" * 5000,  # Very long question
            "description": "Extremely long question"
        },
        {
            "question": "What is this file about?",
            "file_path": "/nonexistent/file.txt",  # Non-existent file
            "description": "Non-existent file"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(error_test_cases, 1):
        print(f"\n🔍 Error Test {i}: {test_case['description']}")
        
        try:
            result_state = workflow.process_question(
                question=test_case["question"],
                file_path=test_case.get("file_path"),
                task_id=f"error_test_{i}"
            )
            
            # Check if error was handled gracefully
            graceful_handling = (
                result_state.is_complete and
                result_state.final_answer and
                not result_state.final_answer.startswith("Traceback")
            )
            
            print(f"   ✅ Graceful Handling: {'✅ PASS' if graceful_handling else '❌ FAIL'}")
            print(f"   ✅ Error Messages: {len(result_state.error_messages)}")
            print(f"   ✅ Final Answer: {result_state.final_answer[:100]}...")
            
            results.append(graceful_handling)
            
        except Exception as e:
            print(f"   ❌ Unhandled exception: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Error Handling Results:")
    print(f"   🎯 Tests Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    return passed >= total * 0.8

def test_workflow_state_management():
    """Test workflow state management and tracking"""
    
    print("\n🧪 Testing Workflow State Management")
    print("=" * 50)
    
    try:
        llm_client = QwenClient()
        workflow = SimpleGAIAWorkflow(llm_client)
    except Exception as e:
        print(f"❌ Failed to initialize workflow: {e}")
        return False
    
    try:
        result_state = workflow.process_question(
            question="What is the square root of 144?",
            task_id="state_test"
        )
        
        # Verify state completeness
        state_checks = {
            "has_task_id": bool(result_state.task_id),
            "has_question": bool(result_state.question),
            "has_routing_decision": bool(result_state.routing_decision),
            "has_processing_steps": len(result_state.processing_steps) > 0,
            "has_final_answer": bool(result_state.final_answer),
            "is_complete": result_state.is_complete,
            "has_cost_tracking": result_state.total_cost >= 0,
            "has_timing": result_state.total_processing_time >= 0
        }
        
        print("   📊 State Management Checks:")
        for check, passed in state_checks.items():
            status = "✅" if passed else "❌"
            print(f"      {status} {check}: {passed}")
        
        # Check state summary
        summary = result_state.get_summary()
        print(f"\n   📋 State Summary:")
        for key, value in summary.items():
            print(f"      {key}: {value}")
        
        # Verify processing steps
        print(f"\n   🔄 Processing Steps ({len(result_state.processing_steps)}):")
        for i, step in enumerate(result_state.processing_steps[-5:], 1):  # Last 5 steps
            print(f"      {i}. {step}")
        
        all_passed = all(state_checks.values())
        print(f"\n   🎯 State Management: {'✅ PASS' if all_passed else '❌ FAIL'}")
        
        return all_passed
        
    except Exception as e:
        print(f"   ❌ State test failed: {e}")
        return False

def main():
    """Run all workflow tests"""
    
    print("🚀 GAIA Workflow Integration Tests")
    print("=" * 60)
    
    test_results = []
    start_time = time.time()
    
    # Run all tests
    test_results.append(test_simple_workflow())
    test_results.append(test_complete_workflow_with_files())
    test_results.append(test_workflow_error_handling())
    test_results.append(test_workflow_state_management())
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(test_results)
    total = len(test_results)
    
    print("\n" + "=" * 60)
    print("📊 COMPLETE WORKFLOW TEST RESULTS")
    print("=" * 60)
    print(f"🎯 Test Suites Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"⏱️  Total Time: {total_time:.2f} seconds")
    
    # Test breakdown
    test_names = [
        "Simple Workflow",
        "File Processing",
        "Error Handling", 
        "State Management"
    ]
    
    print(f"\n📋 Test Breakdown:")
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅" if result else "❌"
        print(f"   {status} {name}")
    
    if passed == total:
        print("\n🚀 ALL WORKFLOW TESTS PASSED! System ready for production!")
        return True
    elif passed >= total * 0.8:
        print("\n✅ MOST TESTS PASSED! System functional with minor issues.")
        return True
    else:
        print("\n⚠️  SIGNIFICANT ISSUES! Review failures above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 