#!/usr/bin/env python3
"""
Web Search Tool for GAIA Agent System
Handles web searches using DuckDuckGo (primary), Tavily API (secondary), and Wikipedia (fallback)
"""

import re
import logging
import time
import os
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup

from tools import BaseTool

logger = logging.getLogger(__name__)

class WebSearchResult:
    """Container for web search results"""
    
    def __init__(self, title: str, url: str, snippet: str, content: str = "", source: str = ""):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.content = content
        self.source = source
        
    def to_dict(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "content": self.content[:1500] + "..." if len(self.content) > 1500 else self.content,
            "source": self.source
        }

class WebSearchTool(BaseTool):
    """
    Web search tool using DuckDuckGo (primary), Tavily API (secondary), and Wikipedia (fallback)
    Provides multiple search engine options for reliability
    """
    
    def __init__(self):
        super().__init__("web_search")
        
        # Configure requests session for web scraping
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.session.timeout = 10
        
        # Initialize search engines
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.use_tavily = self.tavily_api_key is not None
        
        # Try to import DuckDuckGo
        try:
            from duckduckgo_search import DDGS
            self.ddgs = DDGS()
            self.use_duckduckgo = True
            logger.info("✅ DuckDuckGo search initialized")
        except ImportError:
            logger.warning("⚠️ DuckDuckGo search not available - install duckduckgo-search package")
            self.use_duckduckgo = False
        
        # Try to import Wikipedia
        try:
            import wikipedia
            self.wikipedia = wikipedia
            self.use_wikipedia = True
            logger.info("✅ Wikipedia search initialized")
        except ImportError:
            logger.warning("⚠️ Wikipedia search not available - install wikipedia package")
            self.use_wikipedia = False
        
        if self.use_tavily:
            logger.info("✅ Tavily API key found - using as secondary search")
        
        # Search engine priority: DuckDuckGo -> Tavily -> Wikipedia
        search_engines = []
        if self.use_duckduckgo:
            search_engines.append("DuckDuckGo")
        if self.use_tavily:
            search_engines.append("Tavily")
        if self.use_wikipedia:
            search_engines.append("Wikipedia")
            
        logger.info(f"🔍 Available search engines: {', '.join(search_engines)}")
        
    def _execute_impl(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Execute web search operations based on input type
        
        Args:
            input_data: Can be:
                - str: Search query or URL to extract content from
                - dict: {"query": str, "action": str, "limit": int, "extract_content": bool}
        """
        
        if isinstance(input_data, str):
            # Handle both search queries and URLs
            if self._is_url(input_data):
                return self._extract_content_from_url(input_data)
            else:
                return self._search_web(input_data)
                
        elif isinstance(input_data, dict):
            query = input_data.get("query", "")
            action = input_data.get("action", "search")
            limit = input_data.get("limit", 5)
            extract_content = input_data.get("extract_content", False)
            
            if action == "search":
                return self._search_web(query, limit, extract_content)
            elif action == "extract":
                return self._extract_content_from_url(query)
            else:
                raise ValueError(f"Unknown action: {action}")
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _is_url(self, text: str) -> bool:
        """Check if text is a URL"""
        return bool(re.match(r'https?://', text))
    
    def _extract_search_terms(self, question: str, max_length: int = 200) -> str:
        """
        Extract intelligent search terms from a question
        Creates clean, focused queries that search engines can understand
        """
        import re
        
        # Handle backwards text questions - detect and reverse them
        if re.search(r'\.rewsna\b|etirw\b|dnatsrednu\b|ecnetnes\b', question.lower()):
            # This appears to be backwards text - reverse the entire question
            reversed_question = question[::-1]
            logger.info(f"🔄 Detected backwards text, reversed: '{reversed_question[:50]}...'")
            return self._extract_search_terms(reversed_question, max_length)
        
        # Clean the question first
        clean_question = question.strip()
        
        # Special handling for specific question types
        question_lower = clean_question.lower()
        
        # For YouTube video questions, extract the video ID and search for it
        youtube_match = re.search(r'youtube\.com/watch\?v=([a-zA-Z0-9_-]+)', question)
        if youtube_match:
            video_id = youtube_match.group(1)
            return f"youtube video {video_id}"
        
        # For file-based questions, don't search the web
        if any(phrase in question_lower for phrase in ['attached file', 'attached python', 'excel file contains', 'attached excel']):
            return "file processing data analysis"
        
        # Extract key entities using smart patterns
        search_terms = []
        
        # 1. Extract quoted phrases (highest priority)
        quoted_phrases = re.findall(r'"([^"]{3,})"', question)
        search_terms.extend(quoted_phrases[:2])  # Max 2 quoted phrases
        
        # 2. Extract proper nouns (names, places, organizations)
        # Look for capitalized sequences
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]*)*\b', question)
        # Filter out question starters and common words that should not be included
        excluded_words = {'How', 'What', 'Where', 'When', 'Who', 'Why', 'Which', 'The', 'This', 'That', 'If', 'Please', 'Hi', 'Could', 'Review', 'Provide', 'Give', 'On', 'In', 'At', 'To', 'For', 'Of', 'With', 'By', 'Examine', 'Given'}
        meaningful_nouns = []
        for noun in proper_nouns:
            if noun not in excluded_words and len(noun) > 2:
                meaningful_nouns.append(noun)
        search_terms.extend(meaningful_nouns[:4])  # Max 4 proper nouns
        
        # 3. Extract years (but avoid duplicates)
        years = list(set(re.findall(r'\b(19\d{2}|20\d{2})\b', question)))
        search_terms.extend(years[:2])  # Max 2 unique years
        
        # 4. Extract important domain-specific keywords
        domain_keywords = []
        
        # Music/entertainment
        if any(word in question_lower for word in ['album', 'song', 'artist', 'band', 'music']):
            domain_keywords.extend(['studio albums', 'discography'] if 'album' in question_lower else ['music'])
        
        # Wikipedia-specific
        if 'wikipedia' in question_lower:
            domain_keywords.extend(['wikipedia', 'featured article'] if 'featured' in question_lower else ['wikipedia'])
        
        # Sports/Olympics
        if any(word in question_lower for word in ['athlete', 'olympics', 'sport', 'team']):
            domain_keywords.append('olympics' if 'olympics' in question_lower else 'sports')
        
        # Competition/awards
        if any(word in question_lower for word in ['competition', 'winner', 'recipient', 'award']):
            domain_keywords.append('competition')
        
        # Add unique domain keywords
        for keyword in domain_keywords:
            if keyword not in [term.lower() for term in search_terms]:
                search_terms.append(keyword)
        
        # 5. Extract specific important terms from the question
        # Be more selective about stop words - keep important descriptive words
        words = re.findall(r'\b\w+\b', clean_question.lower())
        
        # Reduced skip words list - keep more meaningful terms
        skip_words = {
            'how', 'many', 'what', 'who', 'when', 'where', 'why', 'which', 'whose',
            'is', 'are', 'was', 'were', 'did', 'does', 'do', 'can', 'could', 'would', 'should',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'among', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our',
            'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their',
            'be', 'been', 'being', 'have', 'has', 'had', 'will', 'may', 'might', 'must',
            'please', 'tell', 'find', 'here', 'there', 'only', 'just', 'some', 'help', 'give', 'provide', 'review'
        }
        
        # Look for important content words - be more inclusive
        important_words = []
        for word in words:
            if (len(word) > 3 and 
                word not in skip_words and 
                word not in [term.lower() for term in search_terms] and
                not word.isdigit()):
                # Include important descriptive words
                important_words.append(word)
        
        # Add more important content words
        search_terms.extend(important_words[:4])  # Increased from 3 to 4
        
        # 6. Special inclusion of key terms that are often missed
        # Look for important terms that might have been filtered out
        key_terms_patterns = {
            'image': r'\b(image|picture|photo|visual)\b',
            'video': r'\b(video|clip|footage)\b', 
            'file': r'\b(file|document|attachment)\b',
            'chess': r'\b(chess|position|move|game)\b',
            'move': r'\b(move|next|correct|turn)\b',
            'dinosaur': r'\b(dinosaur|fossil|extinct)\b',
            'shopping': r'\b(shopping|grocery|list|market)\b',
            'list': r'\b(list|shopping|grocery)\b',
            'black': r'\b(black|white|color|turn)\b',
            'opposite': r'\b(opposite|reverse|contrary)\b',
            'nominated': r'\b(nominated|nominated|nomination)\b'
        }
        
        for key_term, pattern in key_terms_patterns.items():
            if re.search(pattern, question_lower) and key_term not in [term.lower() for term in search_terms]:
                search_terms.append(key_term)
        
        # 7. Build the final search query
        if search_terms:
            # Remove duplicates while preserving order
            unique_terms = []
            seen = set()
            for term in search_terms:
                term_lower = term.lower()
                if term_lower not in seen and len(term.strip()) > 0:
                    seen.add(term_lower)
                    unique_terms.append(term)
            
            search_query = ' '.join(unique_terms)
        else:
            # Fallback: extract the most important words from the question
            fallback_words = []
            for word in words:
                if len(word) > 3 and word not in skip_words:
                    fallback_words.append(word)
            search_query = ' '.join(fallback_words[:4])
        
        # Final cleanup
        search_query = ' '.join(search_query.split())  # Remove extra whitespace
        
        # Truncate at word boundary if too long
        if len(search_query) > max_length:
            search_query = search_query[:max_length].rsplit(' ', 1)[0]
        
        # Ensure we have something meaningful
        if not search_query.strip() or len(search_query.strip()) < 3:
            # Last resort: use the first few meaningful words from the original question
            words = question.split()
            meaningful_words = [w for w in words if len(w) > 2 and not w.lower() in skip_words]
            search_query = ' '.join(meaningful_words[:4])
        
        # Log for debugging
        logger.info(f"📝 Extracted search terms: '{search_query}' from question: '{question[:100]}...'")
        
        return search_query.strip()
    
    def _search_web(self, query: str, limit: int = 5, extract_content: bool = False) -> Dict[str, Any]:
        """
        Search the web using available search engines in priority order with improved search terms
        """
        
        # Extract clean search terms from the query
        search_query = self._extract_search_terms(query, max_length=200)
        
        # Try DuckDuckGo first (most comprehensive for general web search)
        if self.use_duckduckgo:
            try:
                ddg_result = self._search_with_duckduckgo(search_query, limit, extract_content)
                if ddg_result.get('success') and ddg_result.get('count', 0) > 0:
                    return {
                        'success': True,
                        'found': True,
                        'results': [r.to_dict() if hasattr(r, 'to_dict') else r for r in ddg_result['results']],
                        'query': query,
                        'source': 'DuckDuckGo',
                        'total_found': ddg_result['count']
                    }
            except Exception as e:
                logger.warning(f"DuckDuckGo search failed, trying Tavily: {e}")
        
        # Try Tavily if DuckDuckGo fails and API key is available
        if self.use_tavily:
            try:
                tavily_result = self._search_with_tavily(search_query, limit, extract_content)
                if tavily_result.get('success') and tavily_result.get('count', 0) > 0:
                    return {
                        'success': True,
                        'found': True,
                        'results': [r.to_dict() if hasattr(r, 'to_dict') else r for r in tavily_result['results']],
                        'query': query,
                        'source': 'Tavily',
                        'total_found': tavily_result['count']
                    }
            except Exception as e:
                logger.warning(f"Tavily search failed, trying Wikipedia: {e}")
        
        # Fallback to Wikipedia search
        if self.use_wikipedia:
            try:
                wiki_result = self._search_with_wikipedia(search_query, limit)
                if wiki_result.get('success') and wiki_result.get('count', 0) > 0:
                    return {
                        'success': True,
                        'found': True,
                        'results': [r.to_dict() if hasattr(r, 'to_dict') else r for r in wiki_result['results']],
                        'query': query,
                        'source': 'Wikipedia',
                        'total_found': wiki_result['count']
                    }
            except Exception as e:
                logger.warning(f"Wikipedia search failed: {e}")
        
        # No search engines available or all failed
        logger.warning("All search engines failed, returning empty results")
        return {
            "query": query,
            "found": False,
            "success": False,
            "message": "❌ All search engines failed or returned no results.",
            "results": [],
            "source": "none",
            "total_found": 0
        }
    
    def _search_with_duckduckgo(self, query: str, limit: int = 5, extract_content: bool = False) -> Dict[str, Any]:
        """
        Search using DuckDuckGo with robust rate limiting handling
        """
        try:
            logger.info(f"🦆 DuckDuckGo search for: {query}")
            
            # Add progressive delay to avoid rate limiting
            time.sleep(1.0)  # Increased base delay
            
            # Use DuckDuckGo text search with enhanced retry logic
            max_retries = 3  # Increased retries
            for attempt in range(max_retries):
                try:
                    # Create a fresh DDGS instance for each attempt to avoid session issues
                    from duckduckgo_search import DDGS
                    ddgs_instance = DDGS()
                    
                    ddg_results = list(ddgs_instance.text(query, max_results=min(limit, 8)))
                    
                    if ddg_results:
                        break
                    else:
                        logger.warning(f"DuckDuckGo returned no results on attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            time.sleep(2 * (attempt + 1))  # Progressive delay
                        
                except Exception as retry_error:
                    error_str = str(retry_error).lower()
                    if attempt < max_retries - 1:
                        # Increase delay for rate limiting
                        if "ratelimit" in error_str or "202" in error_str or "429" in error_str:
                            delay = 3 * (attempt + 1)  # 3s, 6s, 9s delays
                            logger.warning(f"DuckDuckGo rate limited on attempt {attempt + 1}, waiting {delay}s: {retry_error}")
                            time.sleep(delay)
                        else:
                            delay = 1 * (attempt + 1)  # Regular exponential backoff
                            logger.warning(f"DuckDuckGo error on attempt {attempt + 1}, retrying in {delay}s: {retry_error}")
                            time.sleep(delay)
                        continue
                    else:
                        logger.warning(f"DuckDuckGo failed after {max_retries} attempts: {retry_error}")
                        raise retry_error
            
            if not ddg_results:
                logger.warning("DuckDuckGo returned no results after all attempts")
                return self._search_with_fallback(query, limit)
            
            # Process DuckDuckGo results
            results = []
            for result in ddg_results:
                web_result = WebSearchResult(
                    title=result.get('title', 'No title'),
                    url=result.get('href', ''),
                    snippet=result.get('body', 'No description'),
                    source='DuckDuckGo'
                )
                results.append(web_result)
            
            logger.info(f"✅ DuckDuckGo found {len(results)} results")
            
            return {
                'success': True,
                'results': results,
                'source': 'DuckDuckGo',
                'query': query,
                'count': len(results)
            }
            
        except Exception as e:
            logger.warning(f"DuckDuckGo search completely failed: {str(e)}")
            # Add delay before fallback for severe rate limiting
            error_str = str(e).lower()
            if "ratelimit" in error_str or "429" in error_str or "202" in error_str:
                logger.warning("Severe rate limiting detected, adding 5s delay before fallback")
                time.sleep(5.0)
            return self._search_with_fallback(query, limit)
    
    def _search_with_fallback(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Enhanced fallback search when DuckDuckGo fails"""
        
        logger.info(f"🔄 Using fallback search engines for: {query}")
        
        # Try Tavily API first if available
        if hasattr(self, 'tavily') and self.tavily:
            try:
                logger.info("📡 Trying Tavily API search")
                tavily_result = self.tavily.search(query, max_results=limit)
                
                if tavily_result and 'results' in tavily_result:
                    results = []
                    for result in tavily_result['results'][:limit]:
                        web_result = WebSearchResult(
                            title=result.get('title', 'No title'),
                            url=result.get('url', ''),
                            snippet=result.get('content', 'No description'),
                            source='Tavily'
                        )
                        results.append(web_result)
                    
                    if results:
                        logger.info(f"✅ Tavily found {len(results)} results")
                        return {
                            'success': True,
                            'results': results,
                            'source': 'Tavily',
                            'query': query,
                            'count': len(results)
                        }
            except Exception as e:
                logger.warning(f"Tavily search failed: {str(e)}")
        
        # Fall back to Wikipedia search
        logger.info("📚 Wikipedia search for: " + query)
        try:
            wiki_results = self._search_with_wikipedia(query, limit)
            if wiki_results and wiki_results.get('success'):
                logger.info(f"✅ Wikipedia found {wiki_results.get('count', 0)} results")
                return wiki_results
        except Exception as e:
            logger.warning(f"Wikipedia fallback failed: {str(e)}")
        
        # Final fallback - return empty but successful result to allow processing to continue
        logger.warning("All search engines failed, returning empty results")
        return {
            'success': True,
            'results': [],
            'source': 'none',
            'query': query,
            'count': 0,
            'note': 'All search engines failed'
        }
    
    def _search_with_tavily(self, query: str, limit: int = 5, extract_content: bool = False) -> Dict[str, Any]:
        """
        Search using Tavily Search API - secondary search engine
        """
        try:
            logger.info(f"🔍 Tavily search for: {query}")
            
            # Prepare Tavily API request
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": False,
                "include_images": False,
                "include_raw_content": extract_content,
                "max_results": min(limit, 10)
            }
            
            # Make API request
            response = self.session.post(
                "https://api.tavily.com/search",
                json=payload,
                headers=headers,
                timeout=15
            )
            response.raise_for_status()
            
            tavily_data = response.json()
            
            # Process Tavily results
            results = []
            tavily_results = tavily_data.get('results', [])
            
            for result in tavily_results:
                web_result = WebSearchResult(
                    title=result.get('title', 'No title'),
                    url=result.get('url', ''),
                    snippet=result.get('content', 'No description'),
                    content=result.get('raw_content', '') if extract_content else ''
                )
                results.append(web_result)
            
            if results:
                logger.info(f"✅ Tavily found {len(results)} results")
                return {
                    'success': True,
                    'results': results,
                    'source': 'Tavily',
                    'query': query,
                    'count': len(results)
                }
            else:
                logger.warning("Tavily returned no results")
                # Fall back to Wikipedia
                if self.use_wikipedia:
                    return self._search_with_wikipedia(query, limit)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Tavily API request failed: {e}")
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
        
        # Fall back to Wikipedia if Tavily fails
        if self.use_wikipedia:
            return self._search_with_wikipedia(query, limit)
        
        return {
            'success': False,
            'results': [],
            'source': 'Tavily',
            'query': query,
            'count': 0,
            'note': 'Tavily search failed and no fallback available'
        }
    
    def _search_with_wikipedia(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Search using Wikipedia - fallback search engine for factual information
        """
        try:
            logger.info(f"📚 Wikipedia search for: {query}")
            
            self.wikipedia.set_lang("en")
            
            # Clean up query for Wikipedia search and ensure it's not too long
            search_terms = self._extract_search_terms(query, max_length=100)  # Wikipedia has stricter limits
            
            # Search Wikipedia pages
            wiki_results = self.wikipedia.search(search_terms, results=min(limit * 2, 10))
            
            if not wiki_results:
                return {
                    'success': False,
                    'results': [],
                    'source': 'Wikipedia',
                    'query': query,
                    'count': 0,
                    'note': 'No Wikipedia articles found for this query'
                }
            
            results = []
            processed = 0
            
            for page_title in wiki_results:
                if processed >= limit:
                    break
                    
                try:
                    page = self.wikipedia.page(page_title)
                    summary = page.summary[:300] + "..." if len(page.summary) > 300 else page.summary
                    
                    web_result = WebSearchResult(
                        title=f"{page_title} (Wikipedia)",
                        url=page.url,
                        snippet=summary,
                        content=page.summary[:1000] + "..." if len(page.summary) > 1000 else page.summary
                    )
                    results.append(web_result)
                    processed += 1
                    
                except self.wikipedia.exceptions.DisambiguationError as e:
                    # Try the first suggestion from disambiguation
                    try:
                        if e.options:
                            page = self.wikipedia.page(e.options[0])
                            summary = page.summary[:300] + "..." if len(page.summary) > 300 else page.summary
                            
                            web_result = WebSearchResult(
                                title=f"{e.options[0]} (Wikipedia)",
                                url=page.url,
                                snippet=summary,
                                content=page.summary[:1000] + "..." if len(page.summary) > 1000 else page.summary
                            )
                            results.append(web_result)
                            processed += 1
                    except:
                        continue
                        
                except self.wikipedia.exceptions.PageError:
                    # Page doesn't exist, skip
                    continue
                except Exception as e:
                    # Other Wikipedia errors, skip this page
                    logger.warning(f"Wikipedia page error for '{page_title}': {e}")
                    continue
            
            if results:
                logger.info(f"✅ Wikipedia found {len(results)} results")
                return {
                    'success': True,
                    'results': results,
                    'source': 'Wikipedia',
                    'query': query,
                    'count': len(results)
                }
            else:
                return {
                    'success': False,
                    'results': [],
                    'source': 'Wikipedia',
                    'query': query,
                    'count': 0,
                    'note': 'No accessible Wikipedia articles found for this query'
                }
                
        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
            return {
                'success': False,
                'results': [],
                'source': 'Wikipedia',
                'query': query,
                'count': 0,
                'note': f"Wikipedia search failed: {str(e)}"
            }
    
    def _extract_content_from_url(self, url: str) -> Dict[str, Any]:
        """
        Extract readable content from a web page
        """
        try:
            logger.info(f"Extracting content from: {url}")
            
            # Get page content
            response = self.session.get(url)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title"
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            # Extract metadata
            meta_description = ""
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                meta_description = meta_desc.get('content', '')
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True)[:10]:  # First 10 links
                link_url = urljoin(url, link['href'])
                link_text = link.get_text().strip()
                if link_text and len(link_text) > 5:  # Filter out short/empty links
                    links.append({"text": link_text, "url": link_url})
            
            return {
                "url": url,
                "found": True,
                "title": title_text,
                "content": content,
                "meta_description": meta_description,
                "links": links,
                "content_length": len(content),
                "message": "Successfully extracted content from URL"
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "url": url,
                "found": False,
                "message": f"Failed to fetch URL: {str(e)}",
                "error_type": "network_error"
            }
        except Exception as e:
            return {
                "url": url,
                "found": False,
                "message": f"Failed to extract content: {str(e)}",
                "error_type": "parsing_error"
            }
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract main content from HTML using various strategies
        """
        content_parts = []
        
        # Strategy 1: Look for article/main tags
        main_content = soup.find(['article', 'main'])
        if main_content:
            content_parts.append(main_content.get_text())
        
        # Strategy 2: Look for content in common div classes
        content_selectors = [
            'div.content',
            'div.article-content',
            'div.post-content',
            'div.entry-content',
            'div.main-content',
            'div#content',
            'div.text'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            for element in elements:
                content_parts.append(element.get_text())
        
        # Strategy 3: Look for paragraphs in body
        if not content_parts:
            paragraphs = soup.find_all('p')
            for p in paragraphs[:20]:  # First 20 paragraphs
                text = p.get_text().strip()
                if len(text) > 50:  # Filter out short paragraphs
                    content_parts.append(text)
        
        # Clean and combine content
        combined_content = '\n\n'.join(content_parts)
        
        # Clean up whitespace and formatting
        combined_content = re.sub(r'\n\s*\n', '\n\n', combined_content)  # Multiple newlines
        combined_content = re.sub(r' +', ' ', combined_content)  # Multiple spaces
        
        return combined_content.strip()[:5000]  # Limit to 5000 characters

def test_web_search_tool():
    """Test the web search tool with various queries"""
    tool = WebSearchTool()
    
    # Test cases
    test_cases = [
        "Python programming tutorial",
        "Mercedes Sosa studio albums 2000 2009",
        "artificial intelligence recent developments",
        "climate change latest research",
        "https://en.wikipedia.org/wiki/Machine_learning"
    ]
    
    print("🧪 Testing Web Search Tool...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case} ---")
        try:
            result = tool.execute(test_case)
            
            if result.success:
                print(f"✅ Success: {result.result.get('message', 'No message')}")
                search_engine = result.result.get('source', 'unknown')
                print(f"   Search engine: {search_engine}")
                
                if result.result.get('found'):
                    if 'results' in result.result:
                        print(f"   Found {len(result.result['results'])} results")
                        # Show first result details
                        if result.result['results']:
                            first_result = result.result['results'][0]
                            print(f"   First result: {first_result.get('title', 'No title')}")
                            print(f"   URL: {first_result.get('url', 'No URL')}")
                    elif 'content' in result.result:
                        print(f"   Extracted {len(result.result['content'])} characters")
                        print(f"   Title: {result.result.get('title', 'No title')}")
                else:
                    print(f"   Not found: {result.result.get('message', 'Unknown error')}")
            else:
                print(f"❌ Error: {result.error}")
            
            print(f"   Execution time: {result.execution_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Exception: {str(e)}")

if __name__ == "__main__":
    # Test when run directly
    test_web_search_tool() 