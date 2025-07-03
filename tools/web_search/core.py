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
from .decision import query_finder

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
def web_search(message, client, config):
    """Perform web search using Google Custom Search API with optional scraping"""
    try:
        # Get required API credentials
        api_key = config.get("google_api_key")
        search_engine_id = config.get("search_engine_id")
        
        if not api_key or not search_engine_id:
            return ""
        
        # Get search queries
        queries = query_finder(message, client, config).get("query", [])
        if not queries:
            return ""
        
        # Configuration options
        max_results = config.get("max_results", 5)
        enable_scraping = config.get("enable_scraping", True)
        max_scrape_length = config.get("max_scrape_length", 5000)
        scrape_timeout = config.get("scrape_timeout", 15)
        scrape_pause = config.get("scrape_pause", 1)
        
        formatted_results = []
        total_results_collected = 0
        
        for query in queries:
            if total_results_collected >= max_results:
                break
                
            # Calculate how many results we still need
            results_needed = max_results - total_results_collected
            
            # Perform Google search with limited results
            search_results = google_search(query, api_key, search_engine_id, results_needed)
            
            if not search_results:
                continue
                
            for result in search_results:
                if total_results_collected >= max_results:
                    break
                    
                title = result.get("title", "")
                url = result.get("link", "")
                snippet = result.get("snippet", "")
                
                result_text = f"Title: {title}\nURL: {url}\nSnippet: {snippet}\n"
                
                # Optionally scrape the full content
                if enable_scraping and url:
                    try:
                        scraped_content = scrape_url(
                            url, 
                            timeout=scrape_timeout, 
                            pause=scrape_pause,
                            max_length=max_scrape_length
                        )
                        if scraped_content:
                            result_text += f"Full Content: {scraped_content}\n"
                        else:
                            result_text += f"Content: Unable to scrape (blocked or error)\n"
                    except Exception:
                        # Silently continue if scraping fails for individual URLs
                        result_text += f"Content: Unable to scrape (blocked or error)\n"
                
                formatted_results.append(result_text)
                total_results_collected += 1
        
        if not formatted_results:
            return ""
            
        return "\n---\n".join(formatted_results)
        
    except Exception:
        # Return empty string on any error - let the caller handle logging
        return ""