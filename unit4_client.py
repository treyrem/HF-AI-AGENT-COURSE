#!/usr/bin/env python3
"""
Unit 4 API Client for GAIA Benchmark Questions
Handles question fetching, file downloads, and answer submission
"""

import os
import requests
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import json
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GAIAQuestion:
    """GAIA benchmark question data structure"""
    task_id: str
    question: str
    level: int  # 1, 2, or 3 (difficulty level)
    final_answer: Optional[str] = None
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SubmissionResult:
    """Result of answer submission"""
    task_id: str
    submitted_answer: str
    success: bool
    score: Optional[float] = None
    feedback: Optional[str] = None
    error: Optional[str] = None

class Unit4APIClient:
    """Client for Unit 4 API to fetch GAIA questions and submit answers"""
    
    def __init__(self, base_url: str = "https://agents-course-unit4-scoring.hf.space"):
        """Initialize Unit 4 API client"""
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GAIA-Agent-System/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        # Create downloads directory
        self.downloads_dir = Path("downloads")
        self.downloads_dir.mkdir(exist_ok=True)
        
        # Track API usage
        self.requests_made = 0
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # Seconds between requests
        
    def _rate_limit(self):
        """Implement basic rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
        self.requests_made += 1
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with rate limiting and error handling"""
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            logger.debug(f"Making {method} request to {url}")
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def get_questions(self, level: Optional[int] = None, limit: Optional[int] = None) -> List[GAIAQuestion]:
        """Fetch GAIA questions from the API"""
        
        endpoint = "/questions"
        params = {}
        
        if level is not None:
            params['level'] = level
        if limit is not None:
            params['limit'] = limit
            
        try:
            response = self._make_request('GET', endpoint, params=params)
            data = response.json()
            
            questions = []
            
            # Handle different response formats
            if isinstance(data, list):
                question_list = data
            elif isinstance(data, dict) and 'questions' in data:
                question_list = data['questions']
            else:
                question_list = [data]  # Single question
                
            for q_data in question_list:
                question = GAIAQuestion(
                    task_id=q_data.get('task_id', ''),
                    question=q_data.get('question', ''),
                    level=q_data.get('level', 1),
                    final_answer=q_data.get('final_answer'),
                    file_name=q_data.get('file_name'),
                    metadata=q_data
                )
                questions.append(question)
                
            logger.info(f"✅ Fetched {len(questions)} questions from API")
            return questions
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch questions: {e}")
            return []
    
    def get_random_question(self, level: Optional[int] = None) -> Optional[GAIAQuestion]:
        """Fetch a random question from the API"""
        
        endpoint = "/random-question"
        params = {}
        
        if level is not None:
            params['level'] = level
            
        try:
            response = self._make_request('GET', endpoint, params=params)
            data = response.json()
            
            question = GAIAQuestion(
                task_id=data.get('task_id', ''),
                question=data.get('question', ''),
                level=data.get('level', 1),
                final_answer=data.get('final_answer'),
                file_name=data.get('file_name'),
                metadata=data
            )
            
            logger.info(f"✅ Fetched random question: {question.task_id}")
            return question
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch random question: {e}")
            return None
    
    def download_file(self, task_id: str, file_name: Optional[str] = None) -> Optional[str]:
        """Download file associated with a question"""
        
        if not task_id:
            logger.error("Task ID required for file download")
            return None
            
        endpoint = f"/files/{task_id}"
        
        try:
            response = self._make_request('GET', endpoint, stream=True)
            
            # Determine filename
            if file_name:
                filename = file_name
            else:
                # Try to get filename from response headers
                content_disposition = response.headers.get('content-disposition', '')
                if 'filename=' in content_disposition:
                    filename = content_disposition.split('filename=')[1].strip('"')
                else:
                    # Use task_id as fallback
                    filename = f"{task_id}_file"
                    
            # Save file
            file_path = self.downloads_dir / filename
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            logger.info(f"✅ Downloaded file: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to download file for {task_id}: {e}")
            return None
    
    def submit_answer(self, task_id: str, answer: str) -> SubmissionResult:
        """Submit answer for evaluation"""
        
        endpoint = "/submit"
        
        payload = {
            "task_id": task_id,
            "answer": str(answer).strip()
        }
        
        try:
            response = self._make_request('POST', endpoint, json=payload)
            data = response.json()
            
            result = SubmissionResult(
                task_id=task_id,
                submitted_answer=answer,
                success=True,
                score=data.get('score'),
                feedback=data.get('feedback'),
            )
            
            logger.info(f"✅ Submitted answer for {task_id}")
            if result.score is not None:
                logger.info(f"   Score: {result.score}")
            if result.feedback:
                logger.info(f"   Feedback: {result.feedback}")
                
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to submit answer for {task_id}: {e}")
            
            return SubmissionResult(
                task_id=task_id,
                submitted_answer=answer,
                success=False,
                error=str(e)
            )
    
    def validate_answer_format(self, answer: str, question: GAIAQuestion) -> bool:
        """Validate answer format before submission"""
        
        if not answer or not answer.strip():
            logger.warning("Empty answer provided")
            return False
            
        # Basic length validation
        if len(answer) > 1000:
            logger.warning("Answer is very long (>1000 chars)")
            
        # Remove common formatting issues
        cleaned_answer = answer.strip()
        
        # Log validation result
        logger.debug(f"Answer validation passed for {question.task_id}")
        return True
    
    def get_api_status(self) -> Dict[str, Any]:
        """Check API status and endpoints"""
        
        status = {
            "base_url": self.base_url,
            "requests_made": self.requests_made,
            "endpoints_tested": {}
        }
        
        # Test basic endpoints
        test_endpoints = [
            ("/questions", "GET"),
            ("/random-question", "GET"),
        ]
        
        for endpoint, method in test_endpoints:
            try:
                response = self._make_request(method, endpoint, timeout=5)
                status["endpoints_tested"][endpoint] = {
                    "status_code": response.status_code,
                    "success": True
                }
            except Exception as e:
                status["endpoints_tested"][endpoint] = {
                    "success": False,
                    "error": str(e)
                }
                
        return status
    
    def process_question_with_files(self, question: GAIAQuestion) -> GAIAQuestion:
        """Process question and download associated files if needed"""
        
        if question.file_name and question.task_id:
            logger.info(f"Downloading file for question {question.task_id}")
            file_path = self.download_file(question.task_id, question.file_name)
            
            if file_path:
                question.file_path = file_path
                logger.info(f"✅ File ready: {file_path}")
            else:
                logger.warning(f"❌ Failed to download file for {question.task_id}")
                
        return question

# Test functions
def test_api_connection():
    """Test basic API connectivity"""
    logger.info("🧪 Testing Unit 4 API connection...")
    
    client = Unit4APIClient()
    
    # Test API status
    status = client.get_api_status()
    logger.info("📊 API Status:")
    for endpoint, result in status["endpoints_tested"].items():
        status_str = "✅ PASS" if result["success"] else "❌ FAIL"
        logger.info(f"   {endpoint:20}: {status_str}")
        if not result["success"]:
            logger.info(f"      Error: {result.get('error', 'Unknown')}")
    
    return status

def test_question_fetching():
    """Test fetching questions from API"""
    logger.info("🧪 Testing question fetching...")
    
    client = Unit4APIClient()
    
    # Test random question
    question = client.get_random_question()
    if question:
        logger.info(f"✅ Random question fetched: {question.task_id}")
        logger.info(f"   Level: {question.level}")
        logger.info(f"   Question: {question.question[:100]}...")
        logger.info(f"   Has file: {question.file_name is not None}")
        
        # Test file download if available
        if question.file_name:
            question = client.process_question_with_files(question)
            
        return question
    else:
        logger.error("❌ Failed to fetch random question")
        return None

if __name__ == "__main__":
    # Run tests when script executed directly
    test_api_connection()
    test_question_fetching() 