import os
import time
import html
import re
import requests
from bs4 import BeautifulSoup
from readability import Document
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from googleapiclient.discovery import build

# Default headers for web scraping
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; UltraGPT/1.0; +https://ultragpt.ai/bot)"
}

def allowed_by_robots(url, ua=HEADERS["User-Agent"]):
    """Check url against the site's robots.txt before scraping."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(ua, url)
    except Exception:
        return True  # fail-open if robots.txt is missing

def extract_text(html_doc):
    """Strip scripts, styles, and collapse whitespace."""
    try:  # readability works best for article pages
        html_doc = Document(html_doc).summary()
    except Exception:
        pass
    soup = BeautifulSoup(html_doc, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()  # removes the tag entirely
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", html.unescape(text))
    return text

def scrape_url(url, timeout=15, pause=1, max_length=5000):
    """Download url and return cleaned text."""
    if not allowed_by_robots(url):
        return None
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        text = extract_text(r.text)
        # Limit text length if specified
        if max_length and len(text) > max_length:
            text = text[:max_length] + "..."
        return text
    except requests.exceptions.RequestException:
        return None
    finally:
        time.sleep(pause)  # friendly crawl rate

def google_search(query, api_key, search_engine_id, num_results=10):
    """Perform Google Custom Search API search with comprehensive error handling"""
    try:
        if not api_key or not search_engine_id:
            return []
            
        service = build("customsearch", "v1", developerKey=api_key)
        response = (
            service.cse()
            .list(q=query, cx=search_engine_id, num=min(num_results, 10))  # Google API max is 10
            .execute()
        )
        return response.get("items", [])
        
    except Exception as e:
        # Silently fail and return empty results - errors will be logged by caller
        return []

#* Web search ---------------------------------------------------------------
def execute_tool(parameters):
    """Standard entry point for web search tool - takes AI-provided parameters directly"""
    try:
        query = parameters.get("query")
        url = parameters.get("url")
        num_results = parameters.get("num_results", 5)
        
        if url:
            # URL scraping mode
            try:
                content = scrape_url(url)
                if content:
                    return f"Content from {url}:\n{content}"
                else:
                    return f"Unable to scrape content from {url} (blocked or error)"
            except Exception as e:
                return f"Error scraping URL {url}: {str(e)}"
        elif query:
            # Web search mode - use direct Google search
            from ..config import config
            api_key = config.get("google_api_key")
            search_engine_id = config.get("search_engine_id")
            
            if not api_key or not search_engine_id:
                return "Google API credentials not configured. Please set google_api_key and search_engine_id in config."
            
            try:
                search_results = google_search(query, api_key, search_engine_id, num_results)
                if not search_results:
                    return f"No search results found for: {query}"
                
                formatted_results = []
                for result in search_results:
                    title = result.get("title", "")
                    url = result.get("link", "")
                    snippet = result.get("snippet", "")
                    formatted_results.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}")
                
                return f"Search results for '{query}':\n\n" + "\n---\n".join(formatted_results)
            except Exception as e:
                return f"Error searching for '{query}': {str(e)}"
        else:
            return "Please provide either a 'query' for web search or a 'url' for scraping."
    except Exception as e:
        return f"Web search tool error: {str(e)}"

def web_search(message, client, config, history=None):
    """Legacy function - now serves as fallback for direct calls"""
    return "Web search tool is now using native AI tool calling. Please use the UltraGPT chat interface to access web search functions."

def perform_web_search(queries, config):
    """Legacy function - now serves as fallback for old parameter format"""
    if isinstance(queries, list) and queries:
        # Convert old format to new format
        return execute_tool({"query": queries[0], "num_results": 5})
    return "Invalid query format for web search."