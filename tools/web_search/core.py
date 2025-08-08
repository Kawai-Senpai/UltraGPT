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
    debug_info = []
    try:
        if not api_key or not search_engine_id:
            debug_info.append("ERROR: Missing API credentials")
            return [], debug_info
        
        # Debug credential format (without exposing actual keys)
        api_key_debug = f"API key: {api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "API key: [short key]"
        engine_id_debug = f"Engine ID: {search_engine_id[:8]}...{search_engine_id[-4:]}" if len(search_engine_id) > 12 else f"Engine ID: {search_engine_id}"
        debug_info.append(f"Using credentials - {api_key_debug}, {engine_id_debug}")
        
        debug_info.append(f"Building Google Custom Search service...")
        service = build("customsearch", "v1", developerKey=api_key)
        debug_info.append(f"Service built successfully")
        
        debug_info.append(f"Executing search for query: '{query}' with {num_results} results")
        response = (
            service.cse()
            .list(q=query, cx=search_engine_id, num=min(num_results, 10))  # Google API max is 10
            .execute()
        )
        debug_info.append(f"API call completed successfully")
        
        items = response.get("items", [])
        total_results = response.get("searchInformation", {}).get("totalResults", "0")
        debug_info.append(f"Google API returned {len(items)} items out of {total_results} total results")
        
        # Debug the full response structure (first time only)
        if items:
            debug_info.append(f"Sample result keys: {list(items[0].keys())}")
        else:
            debug_info.append(f"Full response keys: {list(response.keys())}")
            if 'searchInformation' in response:
                search_info = response['searchInformation']
                debug_info.append(f"Search info: {search_info}")
        
        return items, debug_info
        
    except Exception as e:
        debug_info.append(f"ERROR in Google API call: {type(e).__name__}: {str(e)}")
        import traceback
        debug_info.append(f"Full traceback: {traceback.format_exc()}")
        return [], debug_info

#* Web search ---------------------------------------------------------------
def execute_tool(parameters):
    """Standard entry point for web search tool - takes AI-provided parameters directly"""
    try:
        query = parameters.get("query")
        url = parameters.get("url")
        num_results = parameters.get("num_results", 5)
        
        # Handle query parameter - it might come as a list or string
        if isinstance(query, list):
            query = query[0] if query else None
        
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
            # Web search mode - get credentials from thread-local context
            try:
                from .context import get_credentials
                api_key, search_engine_id = get_credentials()
                if not api_key or not search_engine_id:
                    # Fallback to environment variables
                    api_key = os.getenv('GOOGLE_API_KEY')
                    search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
            except ImportError:
                # Fallback to environment variables
                api_key = os.getenv('GOOGLE_API_KEY')
                search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
            
            # Debug credential status
            credential_status = f"API key: {'✓' if api_key else '✗'}, Search Engine ID: {'✓' if search_engine_id else '✗'}"
            
            if not api_key or not search_engine_id:
                return f"Google API credentials not configured. {credential_status}. Please provide google_api_key and search_engine_id to UltraGPT constructor or set environment variables GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID."
            
            try:
                search_results, debug_info = google_search(query, api_key, search_engine_id, num_results)
                
                # Include debug info in response for troubleshooting
                debug_section = "\n".join(debug_info)
                
                if not search_results:
                    return f"No search results found for query: '{query}'. {credential_status}.\n\nDEBUG INFO:\n{debug_section}\n\nThis may be due to: 1) Invalid API credentials, 2) Quota exceeded, 3) No matching results for this query, 4) API configuration issues."
                
                formatted_results = []
                for result in search_results:
                    title = result.get("title", "")
                    url = result.get("link", "")
                    snippet = result.get("snippet", "")
                    formatted_results.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}")
                
                return f"Search results for '{query}':\n\n" + "\n---\n".join(formatted_results) + f"\n\nDEBUG INFO:\n{debug_section}"
            except Exception as e:
                return f"Error searching for '{query}': {str(e)}. {credential_status}. Check your Google API credentials and quota."
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