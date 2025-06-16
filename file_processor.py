#!/usr/bin/env python3
"""
File Processing Tool for GAIA Agent System
Handles multiple file formats: images, audio, Excel/CSV, Python code
"""

import os
import re
import io
import logging
import mimetypes
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pandas as pd
from PIL import Image
import ast

from tools import BaseTool

logger = logging.getLogger(__name__)

class FileProcessingResult:
    """Container for file processing results"""
    
    def __init__(self, file_path: str, file_type: str, success: bool, 
                 content: Any = None, metadata: Dict[str, Any] = None):
        self.file_path = file_path
        self.file_type = file_type
        self.success = success
        self.content = content
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_type": self.file_type,
            "success": self.success,
            "content": self.content,
            "metadata": self.metadata
        }

class FileProcessorTool(BaseTool):
    """
    File processor tool for multiple file formats
    Supports images, audio, Excel/CSV, and Python code analysis
    """
    
    def __init__(self):
        super().__init__("file_processor")
        
        # Supported file types
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        self.audio_extensions = {'.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac'}
        self.data_extensions = {'.csv', '.xlsx', '.xls', '.json', '.txt'}
        self.code_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.html', '.css'}
        
    def _execute_impl(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Execute file processing operations based on input type
        
        Args:
            input_data: Can be:
                - str: File path to process
                - dict: {"file_path": str, "operation": str, "options": dict}
        """
        
        if isinstance(input_data, str):
            return self._process_file(input_data)
            
        elif isinstance(input_data, dict):
            file_path = input_data.get("file_path", "")
            operation = input_data.get("operation", "auto")
            options = input_data.get("options", {})
            
            if operation == "auto":
                return self._process_file(file_path, **options)
            elif operation == "analyze_image":
                return self._analyze_image(file_path, **options)
            elif operation == "process_data":
                return self._process_data_file(file_path, **options)
            elif operation == "analyze_code":
                return self._analyze_code(file_path, **options)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _process_file(self, file_path: str, **options) -> Dict[str, Any]:
        """
        Auto-detect file type and process accordingly
        """
        try:
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "message": f"File not found: {file_path}",
                    "error_type": "file_not_found"
                }
            
            # Detect file type
            file_extension = Path(file_path).suffix.lower()
            file_type = self._detect_file_type(file_path, file_extension)
            
            logger.info(f"Processing {file_type} file: {file_path}")
            
            # Route to appropriate processor
            if file_type == "image":
                return self._analyze_image(file_path, **options)
            elif file_type == "audio":
                return self._analyze_audio(file_path, **options)
            elif file_type == "data":
                return self._process_data_file(file_path, **options)
            elif file_type == "code":
                return self._analyze_code(file_path, **options)
            elif file_type == "text":
                return self._process_text_file(file_path, **options)
            else:
                return {
                    "success": False,
                    "message": f"Unsupported file type: {file_type}",
                    "file_path": file_path,
                    "detected_type": file_type
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"File processing failed: {str(e)}",
                "file_path": file_path,
                "error_type": type(e).__name__
            }
    
    def _detect_file_type(self, file_path: str, extension: str) -> str:
        """Detect file type based on extension and MIME type"""
        
        if extension in self.image_extensions:
            return "image"
        elif extension in self.audio_extensions:
            return "audio"
        elif extension in self.data_extensions:
            return "data"
        elif extension in self.code_extensions:
            return "code"
        elif extension in {'.txt', '.md', '.rst'}:
            return "text"
        else:
            # Try MIME type detection
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type:
                if mime_type.startswith('image/'):
                    return "image"
                elif mime_type.startswith('audio/'):
                    return "audio"
                elif mime_type.startswith('text/'):
                    return "text"
            
            return "unknown"
    
    def _analyze_image(self, file_path: str, **options) -> Dict[str, Any]:
        """
        Analyze image files and extract metadata
        """
        try:
            with Image.open(file_path) as img:
                # Basic image information
                metadata = {
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "width": img.width,
                    "height": img.height,
                    "file_size": os.path.getsize(file_path)
                }
                
                # EXIF data if available
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    if exif:
                        metadata["exif_data"] = dict(list(exif.items())[:10])  # First 10 EXIF entries
                
                # Color analysis
                if img.mode in ['RGB', 'RGBA']:
                    colors = img.getcolors(maxcolors=10)
                    if colors:
                        dominant_colors = sorted(colors, reverse=True)[:5]
                        metadata["dominant_colors"] = [
                            {"count": count, "rgb": color} 
                            for count, color in dominant_colors
                        ]
                
                # Basic content description
                content_description = self._describe_image_content(img, metadata)
                
                result = FileProcessingResult(
                    file_path=file_path,
                    file_type="image",
                    success=True,
                    content=content_description,
                    metadata=metadata
                )
                
                return {
                    "success": True,
                    "result": result.to_dict(),
                    "message": f"Successfully analyzed image: {img.width}x{img.height} {img.format}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Image analysis failed: {str(e)}",
                "file_path": file_path,
                "error_type": type(e).__name__
            }
    
    def _describe_image_content(self, img: Image.Image, metadata: Dict[str, Any]) -> str:
        """Generate basic description of image content"""
        description_parts = []
        
        # Size description
        width, height = img.size
        if width > height:
            orientation = "landscape"
        elif height > width:
            orientation = "portrait"
        else:
            orientation = "square"
        
        description_parts.append(f"{orientation} {img.format} image")
        description_parts.append(f"Dimensions: {width} x {height} pixels")
        
        # Color information
        if img.mode == 'RGB':
            description_parts.append("Full color RGB image")
        elif img.mode == 'RGBA':
            description_parts.append("RGB image with transparency")
        elif img.mode == 'L':
            description_parts.append("Grayscale image")
        elif img.mode == '1':
            description_parts.append("Black and white image")
        
        # File size
        file_size = metadata.get("file_size", 0)
        if file_size > 0:
            size_mb = file_size / (1024 * 1024)
            if size_mb >= 1:
                description_parts.append(f"File size: {size_mb:.1f} MB")
            else:
                size_kb = file_size / 1024
                description_parts.append(f"File size: {size_kb:.1f} KB")
        
        return ". ".join(description_parts)
    
    def _analyze_audio(self, file_path: str, **options) -> Dict[str, Any]:
        """
        Analyze audio files (basic metadata for now)
        """
        try:
            # Basic file information
            file_size = os.path.getsize(file_path)
            file_extension = Path(file_path).suffix.lower()
            
            metadata = {
                "file_extension": file_extension,
                "file_size": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2)
            }
            
            # For now, provide basic file info
            # In a full implementation, you might use libraries like:
            # - pydub for audio processing
            # - speech_recognition for transcription
            # - librosa for audio analysis
            
            content_description = f"Audio file ({file_extension}) - {metadata['file_size_mb']} MB"
            
            result = FileProcessingResult(
                file_path=file_path,
                file_type="audio",
                success=True,
                content=content_description,
                metadata=metadata
            )
            
            return {
                "success": True,
                "result": result.to_dict(),
                "message": f"Audio file detected: {metadata['file_size_mb']} MB {file_extension}",
                "note": "Full audio transcription requires additional setup"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Audio analysis failed: {str(e)}",
                "file_path": file_path,
                "error_type": type(e).__name__
            }
    
    def _process_data_file(self, file_path: str, **options) -> Dict[str, Any]:
        """
        Process Excel, CSV, and other data files
        """
        try:
            file_extension = Path(file_path).suffix.lower()
            
            # Read data based on file type
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_extension == '.json':
                df = pd.read_json(file_path)
            else:
                # Try as text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return self._process_text_content(content, file_path)
            
            # Analyze DataFrame
            metadata = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "column_count": len(df.columns),
                "row_count": len(df),
                "data_types": df.dtypes.to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "has_missing_values": df.isnull().any().any()
            }
            
            # Basic statistics for numeric columns
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_columns:
                metadata["numeric_columns"] = numeric_columns
                metadata["numeric_stats"] = df[numeric_columns].describe().to_dict()
            
            # Sample data (first few rows)
            sample_data = df.head(5).to_dict(orient='records')
            
            # Generate content description
            content_description = self._describe_data_content(df, metadata)
            
            result = FileProcessingResult(
                file_path=file_path,
                file_type="data",
                success=True,
                content={
                    "description": content_description,
                    "sample_data": sample_data,
                    "full_data": df.to_dict(orient='records') if len(df) <= 100 else None
                },
                metadata=metadata
            )
            
            return {
                "success": True,
                "result": result.to_dict(),
                "message": f"Successfully processed data file: {df.shape[0]} rows, {df.shape[1]} columns"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Data file processing failed: {str(e)}",
                "file_path": file_path,
                "error_type": type(e).__name__
            }
    
    def _describe_data_content(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> str:
        """Generate description of data file content"""
        description_parts = []
        
        # Basic structure
        rows, cols = df.shape
        description_parts.append(f"Data table with {rows} rows and {cols} columns")
        
        # Column information
        if cols <= 10:
            column_names = ", ".join(df.columns.tolist())
            description_parts.append(f"Columns: {column_names}")
        else:
            description_parts.append(f"Columns include: {', '.join(df.columns.tolist()[:5])}... and {cols-5} more")
        
        # Data types
        numeric_cols = len(metadata.get("numeric_columns", []))
        if numeric_cols > 0:
            description_parts.append(f"{numeric_cols} numeric columns")
        
        # Missing values
        if metadata.get("has_missing_values"):
            description_parts.append("Contains missing values")
        
        return ". ".join(description_parts)
    
    def _analyze_code(self, file_path: str, **options) -> Dict[str, Any]:
        """
        Analyze code files (focusing on Python for now)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.py':
                return self._analyze_python_code(code_content, file_path)
            else:
                return self._analyze_generic_code(code_content, file_path, file_extension)
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Code analysis failed: {str(e)}",
                "file_path": file_path,
                "error_type": type(e).__name__
            }
    
    def _analyze_python_code(self, code_content: str, file_path: str) -> Dict[str, Any]:
        """Analyze Python code structure and content"""
        try:
            # Parse the Python code
            tree = ast.parse(code_content)
            
            # Extract code elements
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args]
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        "name": node.name,
                        "line": node.lineno
                    })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    else:
                        module = node.module or ""
                        for alias in node.names:
                            imports.append(f"{module}.{alias.name}")
            
            # Code statistics
            lines = code_content.split('\n')
            metadata = {
                "total_lines": len(lines),
                "non_empty_lines": len([line for line in lines if line.strip()]),
                "comment_lines": len([line for line in lines if line.strip().startswith('#')]),
                "function_count": len(functions),
                "class_count": len(classes),
                "import_count": len(imports),
                "functions": functions[:10],  # First 10 functions
                "classes": classes[:10],      # First 10 classes
                "imports": list(set(imports))  # Unique imports
            }
            
            # Generate description
            content_description = self._describe_python_code(metadata)
            
            result = FileProcessingResult(
                file_path=file_path,
                file_type="python_code",
                success=True,
                content={
                    "description": content_description,
                    "code_snippet": code_content[:1000] + "..." if len(code_content) > 1000 else code_content,
                    "full_code": code_content
                },
                metadata=metadata
            )
            
            return {
                "success": True,
                "result": result.to_dict(),
                "message": f"Python code analyzed: {metadata['function_count']} functions, {metadata['class_count']} classes"
            }
            
        except SyntaxError as e:
            return {
                "success": False,
                "message": f"Python syntax error: {str(e)}",
                "file_path": file_path,
                "error_type": "syntax_error"
            }
    
    def _describe_python_code(self, metadata: Dict[str, Any]) -> str:
        """Generate description of Python code"""
        description_parts = []
        
        # Basic statistics
        total_lines = metadata.get("total_lines", 0)
        non_empty_lines = metadata.get("non_empty_lines", 0)
        description_parts.append(f"Python file with {total_lines} total lines ({non_empty_lines} non-empty)")
        
        # Functions and classes
        func_count = metadata.get("function_count", 0)
        class_count = metadata.get("class_count", 0)
        
        if func_count > 0:
            description_parts.append(f"{func_count} functions defined")
        if class_count > 0:
            description_parts.append(f"{class_count} classes defined")
        
        # Imports
        imports = metadata.get("imports", [])
        if imports:
            if len(imports) <= 5:
                description_parts.append(f"Imports: {', '.join(imports)}")
            else:
                description_parts.append(f"Imports {len(imports)} modules including: {', '.join(imports[:3])}...")
        
        return ". ".join(description_parts)
    
    def _analyze_generic_code(self, code_content: str, file_path: str, extension: str) -> Dict[str, Any]:
        """Analyze non-Python code files"""
        lines = code_content.split('\n')
        
        metadata = {
            "file_extension": extension,
            "total_lines": len(lines),
            "non_empty_lines": len([line for line in lines if line.strip()]),
            "file_size": len(code_content),
        }
        
        # Basic content analysis
        content_description = f"{extension.upper()} code file with {metadata['total_lines']} lines"
        
        result = FileProcessingResult(
            file_path=file_path,
            file_type="code",
            success=True,
            content={
                "description": content_description,
                "code_snippet": code_content[:500] + "..." if len(code_content) > 500 else code_content
            },
            metadata=metadata
        )
        
        return {
            "success": True,
            "result": result.to_dict(),
            "message": f"Code file analyzed: {metadata['total_lines']} lines of {extension.upper()} code"
        }
    
    def _process_text_file(self, file_path: str, **options) -> Dict[str, Any]:
        """Process plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self._process_text_content(content, file_path)
            
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                return self._process_text_content(content, file_path)
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Text file processing failed: {str(e)}",
                    "file_path": file_path,
                    "error_type": type(e).__name__
                }
    
    def _process_text_content(self, content: str, file_path: str) -> Dict[str, Any]:
        """Process text content and extract metadata"""
        lines = content.split('\n')
        words = content.split()
        
        metadata = {
            "character_count": len(content),
            "word_count": len(words),
            "line_count": len(lines),
            "non_empty_lines": len([line for line in lines if line.strip()]),
            "average_line_length": sum(len(line) for line in lines) / max(len(lines), 1)
        }
        
        # Generate preview
        preview = content[:500] + "..." if len(content) > 500 else content
        
        result = FileProcessingResult(
            file_path=file_path,
            file_type="text",
            success=True,
            content={
                "text": content,
                "preview": preview
            },
            metadata=metadata
        )
        
        return {
            "success": True,
            "result": result.to_dict(),
            "message": f"Text file processed: {metadata['word_count']} words, {metadata['line_count']} lines"
        }

def test_file_processor_tool():
    """Test the file processor tool with various file types"""
    tool = FileProcessorTool()
    
    # Create test files for demonstration
    test_files = []
    
    # Create a simple CSV file
    csv_content = """name,age,city
John,25,New York
Jane,30,San Francisco
Bob,35,Chicago"""
    
    csv_path = "/tmp/test_data.csv"
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    test_files.append(csv_path)
    
    # Create a simple Python file
    py_content = """#!/usr/bin/env python3
import os
import sys

def hello_world():
    '''Simple greeting function'''
    return "Hello, World!"

class TestClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value

if __name__ == "__main__":
    print(hello_world())
"""
    
    py_path = "/tmp/test_script.py"
    with open(py_path, 'w') as f:
        f.write(py_content)
    test_files.append(py_path)
    
    print("🧪 Testing File Processor Tool...")
    
    for i, file_path in enumerate(test_files, 1):
        print(f"\n--- Test {i}: {file_path} ---")
        try:
            result = tool.execute(file_path)
            
            if result.success:
                file_result = result.result['result']
                print(f"✅ Success: {file_result['file_type']} file")
                print(f"   Message: {result.result.get('message', 'No message')}")
                if 'metadata' in file_result:
                    metadata = file_result['metadata']
                    print(f"   Metadata: {list(metadata.keys())}")
            else:
                print(f"❌ Error: {result.result.get('message', 'Unknown error')}")
            
            print(f"   Execution time: {result.execution_time:.3f}s")
            
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
    
    # Clean up test files
    for file_path in test_files:
        try:
            os.remove(file_path)
        except:
            pass

if __name__ == "__main__":
    # Test when run directly
    test_file_processor_tool() 