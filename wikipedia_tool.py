#!/usr/bin/env python3
"""
Wikipedia Tool for GAIA Agent System
Handles Wikipedia searches, content extraction, and information retrieval
"""

import re
import logging
from typing import Dict, List, Optional, Any
import wikipediaapi  # Fixed import - using Wikipedia-API package
from urllib.parse import urlparse, unquote

from tools import BaseTool

logger = logging.getLogger(__name__)

class WikipediaSearchResult:
    """Container for Wikipedia search results"""
    
    def __init__(self, title: str, summary: str, url: str, content: str = ""):
        self.title = title
        self.summary = summary
        self.url = url
        self.content = content
        
    def to_dict(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "summary": self.summary,
            "url": self.url,
            "content": self.content[:1000] + "..." if len(self.content) > 1000 else self.content
        }

class WikipediaTool(BaseTool):
    """
    Wikipedia tool for searching and extracting information
    Handles disambiguation, missing pages, and content extraction
    """
    
    def __init__(self):
        super().__init__("wikipedia")
        
        # Initialize Wikipedia API client
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent='GAIA-Agent/1.0 (educational-purpose)'
        )
        
    def _execute_impl(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Execute Wikipedia operations based on input type
        
        Args:
            input_data: Can be:
                - str: Search query or Wikipedia URL
                - dict: {"query": str, "action": str, "limit": int}
        """
        
        if isinstance(input_data, str):
            # Handle both search queries and URLs
            if self._is_wikipedia_url(input_data):
                return self._extract_from_url(input_data)
            else:
                return self._get_page_info(input_data)
                
        elif isinstance(input_data, dict):
            query = input_data.get("query", "")
            action = input_data.get("action", "summary")
            
            if action == "summary":
                return self._get_summary(query)
            elif action == "content":
                return self._get_full_content(query)
            else:
                raise ValueError(f"Unknown action: {action}")
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _is_wikipedia_url(self, url: str) -> bool:
        """Check if URL is a Wikipedia URL"""
        return "wikipedia.org" in url.lower()
    
    def _extract_title_from_url(self, url: str) -> str:
        """Extract article title from Wikipedia URL"""
        try:
            parsed = urlparse(url)
            if "/wiki/" in parsed.path:
                title = parsed.path.split("/wiki/", 1)[1]
                return unquote(title).replace("_", " ")
            return ""
        except Exception:
            return ""
    
    def _extract_from_url(self, url: str) -> Dict[str, Any]:
        """Extract information from Wikipedia URL"""
        title = self._extract_title_from_url(url)
        if not title:
            raise ValueError(f"Could not extract title from URL: {url}")
            
        return self._get_full_content(title)
    
    def _get_page_info(self, query: str) -> Dict[str, Any]:
        """Get basic page information (summary-level)"""
        try:
            page = self.wiki.page(query)
            
            if not page.exists():
                return {
                    "query": query,
                    "found": False,
                    "message": f"Wikipedia page '{query}' does not exist",
                    "suggestions": self._get_suggestions(query)
                }
            
            # Get summary (first paragraph)
            summary = page.summary[:500] + "..." if len(page.summary) > 500 else page.summary
            
            result = WikipediaSearchResult(
                title=page.title,
                summary=summary,
                url=page.fullurl,
                content=""
            )
            
            return {
                "query": query,
                "found": True,
                "result": result.to_dict(),
                "message": "Successfully retrieved Wikipedia page info"
            }
            
        except Exception as e:
            raise Exception(f"Failed to get Wikipedia page info: {str(e)}")
    
    def _get_summary(self, title: str) -> Dict[str, Any]:
        """Get summary of a specific Wikipedia article"""
        try:
            page = self.wiki.page(title)
            
            if not page.exists():
                return {
                    "title": title,
                    "found": False,
                    "message": f"Wikipedia page '{title}' does not exist",
                    "suggestions": self._get_suggestions(title)
                }
            
            # Get summary (first few sentences)
            summary = page.summary[:800] + "..." if len(page.summary) > 800 else page.summary
            
            result = WikipediaSearchResult(
                title=page.title,
                summary=summary,
                url=page.fullurl
            )
            
            return {
                "title": title,
                "found": True,
                "result": result.to_dict(),
                "categories": list(page.categories.keys())[:5],  # First 5 categories
                "message": "Successfully retrieved Wikipedia summary"
            }
            
        except Exception as e:
            raise Exception(f"Failed to get Wikipedia summary: {str(e)}")
    
    def _get_full_content(self, title: str) -> Dict[str, Any]:
        """Get full content of a Wikipedia article"""
        try:
            page = self.wiki.page(title)
            
            if not page.exists():
                return {
                    "title": title,
                    "found": False,
                    "message": f"Wikipedia page '{title}' does not exist",
                    "suggestions": self._get_suggestions(title)
                }
            
            # Extract key sections
            content_sections = self._parse_content_sections(page.text)
            
            result = WikipediaSearchResult(
                title=page.title,
                summary=page.summary[:800] + "..." if len(page.summary) > 800 else page.summary,
                url=page.fullurl,
                content=page.text
            )
            
            # Get linked pages (limit to avoid overwhelming)
            links = []
            link_count = 0
            for link_title in page.links.keys():
                if link_count >= 20:  # Limit to first 20 links
                    break
                links.append(link_title)
                link_count += 1
            
            return {
                "title": title,
                "found": True,
                "result": result.to_dict(),
                "sections": content_sections,
                "links": links,
                "categories": list(page.categories.keys())[:10],  # First 10 categories
                "backlinks_count": len(page.backlinks),
                "message": "Successfully retrieved full Wikipedia content"
            }
            
        except Exception as e:
            raise Exception(f"Failed to get Wikipedia content: {str(e)}")
    
    def _parse_content_sections(self, content: str) -> Dict[str, str]:
        """Parse Wikipedia content into sections"""
        sections = {}
        current_section = "Introduction"
        current_content = []
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            # Check for section headers (== Section Name ==)
            if line.startswith('==') and line.endswith('==') and len(line) > 4:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line.strip('= ').strip()
                current_content = []
            else:
                if line:  # Skip empty lines
                    current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # Return only first few sections to avoid overwhelming output
        section_items = list(sections.items())[:5]
        return dict(section_items)
    
    def _get_suggestions(self, query: str) -> List[str]:
        """Get search suggestions for a query (simplified)"""
        # Wikipedia-API doesn't have direct search, so we'll provide basic suggestions
        # In a real implementation, you might use the Wikipedia search API
        common_suggestions = [
            query.lower(),
            query.title(),
            query.upper(),
            query.replace(' ', '_'),
        ]
        return list(set(common_suggestions))[:3]

def test_wikipedia_tool():
    """Test the Wikipedia tool with various queries"""
    tool = WikipediaTool()
    
    # Test cases
    test_cases = [
        "Albert Einstein",
        "https://en.wikipedia.org/wiki/Machine_learning",
        {"query": "Python (programming language)", "action": "summary"},
        {"query": "Artificial Intelligence", "action": "content"},
        "NonexistentPageTest12345"
    ]
    
    print("🧪 Testing Wikipedia Tool...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case} ---")
        try:
            result = tool.execute(test_case)
            
            if result.success:
                print(f"✅ Success: {result.result.get('message', 'No message')}")
                if result.result.get('found'):
                    if 'result' in result.result:
                        print(f"   Title: {result.result['result'].get('title', 'No title')}")
                        print(f"   Summary: {result.result['result'].get('summary', 'No summary')[:100]}...")
                else:
                    print(f"   Not found: {result.result.get('message', 'Unknown error')}")
            else:
                print(f"❌ Error: {result.error}")
            
            print(f"   Execution time: {result.execution_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Exception: {str(e)}")

if __name__ == "__main__":
    # Test when run directly
    test_wikipedia_tool() 