#!/usr/bin/env python3
"""
Calculator Tool for GAIA Agent System
Handles mathematical calculations, unit conversions, and statistical operations
"""

import re
import math
import statistics
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from tools import BaseTool

logger = logging.getLogger(__name__)

@dataclass
class CalculationResult:
    """Container for calculation results"""
    expression: str
    result: Union[float, int, str]
    result_type: str
    steps: List[str]
    units: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "expression": self.expression,
            "result": self.result,
            "result_type": self.result_type,
            "steps": self.steps,
            "units": self.units
        }

class CalculatorTool(BaseTool):
    """
    Calculator tool for mathematical operations
    Supports basic math, advanced functions, statistics, and unit conversions
    """
    
    def __init__(self):
        super().__init__("calculator")
        
        # Safe mathematical functions
        self.safe_functions = {
            # Basic functions
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'len': len,
            
            # Math module functions
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
            'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
            'exp': math.exp, 'log': math.log, 'log10': math.log10,
            'sqrt': math.sqrt, 'pow': pow, 'ceil': math.ceil, 'floor': math.floor,
            'factorial': math.factorial, 'gcd': math.gcd,
            
            # Constants
            'pi': math.pi, 'e': math.e,
            
            # Statistics functions
            'mean': statistics.mean, 'median': statistics.median,
            'mode': statistics.mode, 'stdev': statistics.stdev,
            'variance': statistics.variance
        }
        
        # Unit conversion factors (to base units)
        self.unit_conversions = {
            # Length (to meters)
            'length': {
                'mm': 0.001, 'cm': 0.01, 'dm': 0.1, 'm': 1,
                'km': 1000, 'in': 0.0254, 'ft': 0.3048,
                'yd': 0.9144, 'mi': 1609.344
            },
            # Weight (to grams)
            'weight': {
                'mg': 0.001, 'g': 1, 'kg': 1000,
                'oz': 28.3495, 'lb': 453.592, 'ton': 1000000
            },
            # Temperature (special handling)
            'temperature': {
                'celsius': 'celsius', 'fahrenheit': 'fahrenheit',
                'kelvin': 'kelvin', 'c': 'celsius', 'f': 'fahrenheit', 'k': 'kelvin'
            },
            # Time (to seconds)
            'time': {
                's': 1, 'min': 60, 'h': 3600, 'hr': 3600,
                'day': 86400, 'week': 604800, 'month': 2629746, 'year': 31556952
            },
            # Area (to square meters)
            'area': {
                'mm2': 0.000001, 'cm2': 0.0001, 'm2': 1,
                'km2': 1000000, 'in2': 0.00064516, 'ft2': 0.092903
            },
            # Volume (to liters)
            'volume': {
                'ml': 0.001, 'l': 1, 'gal': 3.78541, 'qt': 0.946353,
                'pt': 0.473176, 'cup': 0.236588, 'fl_oz': 0.0295735
            }
        }
    
    def _execute_impl(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Execute calculator operations based on input type
        
        Args:
            input_data: Can be:
                - str: Mathematical expression
                - dict: {"expression": str, "operation": str, "data": list, "units": dict}
        """
        
        if isinstance(input_data, str):
            return self._evaluate_expression(input_data)
            
        elif isinstance(input_data, dict):
            operation = input_data.get("operation", "evaluate")
            
            if operation == "evaluate":
                expression = input_data.get("expression", "")
                return self._evaluate_expression(expression)
            elif operation == "statistics":
                data = input_data.get("data", [])
                return self._calculate_statistics(data)
            elif operation == "convert":
                value = input_data.get("value", 0)
                from_unit = input_data.get("from_unit", "")
                to_unit = input_data.get("to_unit", "")
                return self._convert_units(value, from_unit, to_unit)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _evaluate_expression(self, expression: str) -> Dict[str, Any]:
        """
        Safely evaluate a mathematical expression
        """
        try:
            # Clean the expression
            original_expression = expression
            expression = self._clean_expression(expression)
            
            steps = [f"Original: {original_expression}", f"Cleaned: {expression}"]
            
            # Check for unit conversion patterns
            unit_match = re.search(r'(\d+\.?\d*)\s*(\w+)\s+to\s+(\w+)', expression)
            if unit_match:
                value, from_unit, to_unit = unit_match.groups()
                return self._convert_units(float(value), from_unit, to_unit)
            
            # Replace common mathematical expressions
            expression = self._replace_math_expressions(expression)
            steps.append(f"With functions: {expression}")
            
            # Validate expression safety
            if not self._is_safe_expression(expression):
                raise ValueError("Expression contains unsafe operations")
            
            # Create safe evaluation environment
            safe_dict = {
                "__builtins__": {},
                **self.safe_functions
            }
            
            # Evaluate the expression
            result = eval(expression, safe_dict)
            
            # Determine result type and format
            if isinstance(result, (int, float)):
                if result == int(result):
                    result = int(result)
                    result_type = "integer"
                else:
                    result = round(result, 10)  # Avoid floating point errors
                    result_type = "float"
            else:
                result_type = type(result).__name__
            
            calc_result = CalculationResult(
                expression=original_expression,
                result=result,
                result_type=result_type,
                steps=steps
            )
            
            return {
                "success": True,
                "calculation": calc_result.to_dict(),
                "message": f"Successfully evaluated: {result}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "expression": expression,
                "message": f"Calculation failed: {str(e)}",
                "error_type": type(e).__name__
            }
    
    def _clean_expression(self, expression: str) -> str:
        """Clean and normalize mathematical expression"""
        # Remove extra whitespace
        expression = re.sub(r'\s+', ' ', expression.strip())
        
        # Replace common text with symbols
        replacements = {
            ' plus ': '+', ' minus ': '-', ' times ': '*', ' multiply ': '*',
            ' divided by ': '/', ' divide ': '/', ' power ': '**', ' to the power of ': '**'
        }
        
        for text, symbol in replacements.items():
            expression = expression.replace(text, symbol)
        
        # Handle percentage
        expression = re.sub(r'(\d+\.?\d*)%', r'(\1/100)', expression)
        
        return expression
    
    def _replace_math_expressions(self, expression: str) -> str:
        """Replace mathematical function names with proper calls"""
        # Handle square root
        expression = re.sub(r'sqrt\s*\(([^)]+)\)', r'sqrt(\1)', expression)
        expression = re.sub(r'square root of (\d+\.?\d*)', r'sqrt(\1)', expression)
        
        # Handle logarithms
        expression = re.sub(r'log\s*\(([^)]+)\)', r'log(\1)', expression)
        expression = re.sub(r'ln\s*\(([^)]+)\)', r'log(\1)', expression)
        
        # Handle trigonometric functions
        trig_functions = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan']
        for func in trig_functions:
            expression = re.sub(f'{func}\\s*\\(([^)]+)\\)', f'{func}(\\1)', expression)
        
        return expression
    
    def _is_safe_expression(self, expression: str) -> bool:
        """Check if expression is safe to evaluate"""
        # Forbidden patterns
        forbidden_patterns = [
            r'__.*__',  # Dunder methods
            r'import\s',  # Import statements
            r'exec\s*\(',  # Exec function
            r'eval\s*\(',  # Eval function
            r'open\s*\(',  # File operations
            r'file\s*\(',  # File operations
            r'input\s*\(',  # Input function
            r'raw_input\s*\(',  # Raw input
        ]
        
        for pattern in forbidden_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                return False
        
        return True
    
    def _calculate_statistics(self, data: List[float]) -> Dict[str, Any]:
        """Calculate statistical measures for a dataset"""
        try:
            if not data:
                raise ValueError("Empty dataset provided")
            
            data = [float(x) for x in data]  # Ensure all values are numeric
            
            stats = {
                "count": len(data),
                "sum": sum(data),
                "mean": statistics.mean(data),
                "median": statistics.median(data),
                "min": min(data),
                "max": max(data),
                "range": max(data) - min(data)
            }
            
            # Add standard deviation and variance if enough data points
            if len(data) > 1:
                stats["stdev"] = statistics.stdev(data)
                stats["variance"] = statistics.variance(data)
            
            # Add mode if applicable
            try:
                stats["mode"] = statistics.mode(data)
            except statistics.StatisticsError:
                stats["mode"] = "No unique mode"
            
            return {
                "success": True,
                "statistics": stats,
                "data": data,
                "message": f"Calculated statistics for {len(data)} data points"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Statistics calculation failed: {str(e)}",
                "error_type": type(e).__name__
            }
    
    def _convert_units(self, value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
        """Convert between different units"""
        try:
            from_unit = from_unit.lower()
            to_unit = to_unit.lower()
            
            # Find the unit type
            unit_type = None
            for utype, units in self.unit_conversions.items():
                if from_unit in units and to_unit in units:
                    unit_type = utype
                    break
            
            if not unit_type:
                raise ValueError(f"Cannot convert between {from_unit} and {to_unit}")
            
            # Special handling for temperature
            if unit_type == 'temperature':
                result = self._convert_temperature(value, from_unit, to_unit)
            else:
                # Standard unit conversion
                from_factor = self.unit_conversions[unit_type][from_unit]
                to_factor = self.unit_conversions[unit_type][to_unit]
                result = value * from_factor / to_factor
            
            # Round to reasonable precision
            if result == int(result):
                result = int(result)
            else:
                result = round(result, 6)
            
            conversion_result = CalculationResult(
                expression=f"{value} {from_unit} to {to_unit}",
                result=result,
                result_type="conversion",
                steps=[
                    f"Convert {value} {from_unit} to {to_unit}",
                    f"Result: {result} {to_unit}"
                ],
                units=to_unit
            )
            
            return {
                "success": True,
                "conversion": conversion_result.to_dict(),
                "message": f"Converted {value} {from_unit} = {result} {to_unit}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Unit conversion failed: {str(e)}",
                "error_type": type(e).__name__
            }
    
    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert temperature between Celsius, Fahrenheit, and Kelvin"""
        # Normalize unit names
        unit_map = {'c': 'celsius', 'f': 'fahrenheit', 'k': 'kelvin'}
        from_unit = unit_map.get(from_unit, from_unit)
        to_unit = unit_map.get(to_unit, to_unit)
        
        # Convert to Celsius first
        if from_unit == 'fahrenheit':
            celsius = (value - 32) * 5/9
        elif from_unit == 'kelvin':
            celsius = value - 273.15
        else:  # Already Celsius
            celsius = value
        
        # Convert from Celsius to target unit
        if to_unit == 'fahrenheit':
            return celsius * 9/5 + 32
        elif to_unit == 'kelvin':
            return celsius + 273.15
        else:  # Stay in Celsius
            return celsius

def test_calculator_tool():
    """Test the calculator tool with various operations"""
    tool = CalculatorTool()
    
    # Test cases
    test_cases = [
        "2 + 3 * 4",
        "sqrt(16) + 2^3",
        "sin(pi/2) + cos(0)",
        {"operation": "statistics", "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        {"operation": "convert", "value": 100, "from_unit": "cm", "to_unit": "m"},
        {"operation": "convert", "value": 32, "from_unit": "f", "to_unit": "c"},
        "10 factorial",
        "mean([1, 2, 3, 4, 5])",
        "15% of 200"
    ]
    
    print("🧪 Testing Calculator Tool...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case} ---")
        try:
            result = tool.execute(test_case)
            
            if result.success:
                if 'calculation' in result.result:
                    calc = result.result['calculation']
                    print(f"✅ Result: {calc['result']} ({calc['result_type']})")
                elif 'statistics' in result.result:
                    stats = result.result['statistics']
                    print(f"✅ Mean: {stats['mean']}, Median: {stats['median']}, StDev: {stats.get('stdev', 'N/A')}")
                elif 'conversion' in result.result:
                    conv = result.result['conversion']
                    print(f"✅ Conversion: {conv['result']} {conv['units']}")
                print(f"   Message: {result.result.get('message', 'No message')}")
            else:
                print(f"❌ Error: {result.result.get('message', 'Unknown error')}")
            
            print(f"   Execution time: {result.execution_time:.3f}s")
            
        except Exception as e:
            print(f"❌ Exception: {str(e)}")

if __name__ == "__main__":
    # Test when run directly
    test_calculator_tool() 