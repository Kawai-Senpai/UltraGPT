
#* Web Search Prompts ---------------------------------------------------------------
_info = "This allows you to search the web and scrape content from specific URLs"

_description = """This is a web search tool that allows you to:
1. Search the web using Google Custom Search API
2. Scrape and extract content from specific URLs
3. Get search results and website content for research purposes

Parameters:
- query: The search query string (required for web search)
- url: The URL to scrape content from (required for URL scraping)
- num_results: Number of search results to return (optional, default: 5)

Examples:
- "Search for Python tutorials" → query="Python tutorials"
- "Get content from https://example.com" → url="https://example.com"
- "Search for AI news with 10 results" → query="AI news", num_results=10"""