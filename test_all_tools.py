#!/usr/bin/env python3
"""
Integration test for all GAIA Agent tools
Tests Wikipedia, Web Search, Calculator, and File Processor tools
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from tools.wikipedia_tool import WikipediaTool
from tools.web_search_tool import WebSearchTool  
from tools.calculator import CalculatorTool
from tools.file_processor import FileProcessorTool

def test_all_tools():
    """Comprehensive test of all GAIA agent tools"""
    
    print("🧪 GAIA Agent Tools Integration Test")
    print("=" * 50)
    
    results = []
    start_time = time.time()
    
    # Test 1: Wikipedia Tool
    print("\n📚 Testing Wikipedia Tool...")
    wikipedia_tool = WikipediaTool()
    test_cases = [
        "Albert Einstein",
        {"query": "Machine Learning", "action": "summary"}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        result = wikipedia_tool.execute(test_case)
        success = result.success and result.result.get('found', False)
        results.append(('Wikipedia', f'Test {i}', success, result.execution_time))
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   Test {i}: {status} ({result.execution_time:.2f}s)")
    
    # Test 2: Web Search Tool  
    print("\n🔍 Testing Web Search Tool...")
    web_search_tool = WebSearchTool()
    test_cases = [
        "Python programming",
        {"query": "https://www.python.org", "action": "extract"}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        result = web_search_tool.execute(test_case)
        success = result.success and result.result.get('found', False)
        results.append(('Web Search', f'Test {i}', success, result.execution_time))
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   Test {i}: {status} ({result.execution_time:.2f}s)")
    
    # Test 3: Calculator Tool
    print("\n🧮 Testing Calculator Tool...")
    calculator_tool = CalculatorTool()
    test_cases = [
        "2 + 3 * 4",
        {"operation": "statistics", "data": [1, 2, 3, 4, 5]},
        {"operation": "convert", "value": 100, "from_unit": "cm", "to_unit": "m"}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        result = calculator_tool.execute(test_case)
        success = result.success and result.result.get('success', False)
        results.append(('Calculator', f'Test {i}', success, result.execution_time))
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   Test {i}: {status} ({result.execution_time:.3f}s)")
    
    # Test 4: File Processor Tool
    print("\n📁 Testing File Processor Tool...")
    file_processor_tool = FileProcessorTool()
    
    # Create test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CSV test file
        csv_path = os.path.join(temp_dir, "test.csv")
        with open(csv_path, 'w') as f:
            f.write("name,value\nTest,42\nData,100")
        
        # Create Python test file  
        py_path = os.path.join(temp_dir, "test.py")
        with open(py_path, 'w') as f:
            f.write("def test_function():\n    return 'Hello, World!'")
        
        test_files = [csv_path, py_path]
        
        for i, file_path in enumerate(test_files, 1):
            result = file_processor_tool.execute(file_path)
            success = result.success and result.result.get('success', False)
            results.append(('File Processor', f'Test {i}', success, result.execution_time))
            status = "✅ PASS" if success else "❌ FAIL"
            file_type = os.path.splitext(file_path)[1]
            print(f"   Test {i} ({file_type}): {status} ({result.execution_time:.3f}s)")
    
    # Summary
    total_time = time.time() - start_time
    passed_tests = sum(1 for _, _, success, _ in results if success)
    total_tests = len(results)
    
    print("\n" + "=" * 50)
    print("📊 INTEGRATION TEST RESULTS")
    print("=" * 50)
    
    # Results by tool
    tools = {}
    for tool, test, success, exec_time in results:
        if tool not in tools:
            tools[tool] = {'passed': 0, 'total': 0, 'time': 0}
        tools[tool]['total'] += 1
        tools[tool]['time'] += exec_time
        if success:
            tools[tool]['passed'] += 1
    
    for tool, stats in tools.items():
        pass_rate = (stats['passed'] / stats['total']) * 100
        avg_time = stats['time'] / stats['total']
        status = "✅" if pass_rate == 100 else "⚠️" if pass_rate >= 80 else "❌"
        print(f"{status} {tool:15}: {stats['passed']}/{stats['total']} ({pass_rate:5.1f}%) - Avg: {avg_time:.3f}s")
    
    # Overall results
    overall_pass_rate = (passed_tests / total_tests) * 100
    print(f"\n🎯 OVERALL: {passed_tests}/{total_tests} tests passed ({overall_pass_rate:.1f}%)")
    print(f"⏱️  TOTAL TIME: {total_time:.2f} seconds")
    
    # Success criteria
    if overall_pass_rate >= 90:
        print("🚀 EXCELLENT! All tools working correctly - Ready for agent integration!")
        return True
    elif overall_pass_rate >= 80:
        print("✅ GOOD! Most tools working - Minor issues to address")
        return True
    else:
        print("⚠️  NEEDS WORK! Significant issues found - Check individual tool failures")
        return False

def test_tool_coordination():
    """Test how tools can work together in a coordinated workflow"""
    
    print("\n🤝 Testing Tool Coordination...")
    print("-" * 30)
    
    # Scenario: Research Python programming, then calculate some metrics
    try:
        # Step 1: Get information about Python
        wiki_tool = WikipediaTool()
        wiki_result = wiki_tool.execute("Python (programming language)")
        
        if wiki_result.success:
            print("✅ Step 1: Wikipedia lookup successful")
            
            # Step 2: Get additional web information
            web_tool = WebSearchTool()
            web_result = web_tool.execute("Python programming language features")
            
            if web_result.success:
                print("✅ Step 2: Web search successful")
                
                # Step 3: Calculate some metrics
                calc_tool = CalculatorTool()
                search_count = len(web_result.result.get('results', []))
                calc_result = calc_tool.execute(f"sqrt({search_count}) * 10")
                
                if calc_result.success:
                    print("✅ Step 3: Calculation successful")
                    print(f"   Coordinated result: Found {search_count} web results, computed metric: {calc_result.result['calculation']['result']}")
                    return True
    
    except Exception as e:
        print(f"❌ Coordination test failed: {e}")
    
    return False

if __name__ == "__main__":
    success = test_all_tools()
    coordination_success = test_tool_coordination()
    
    if success and coordination_success:
        print("\n🎉 ALL TESTS PASSED! Tools are ready for agent integration!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Check output above.")
        sys.exit(1) 