from pydantic import BaseModel
from typing import Optional

#! Web Search Tool Schema ---------------------------------------------------------------
class WebSearchQuery(BaseModel):
    """Schema for web search operations - NEVER include API keys in parameters"""
    query: Optional[str] = None  # Search query for Google search
    url: Optional[str] = None    # URL to scrape content from
    num_results: Optional[int] = 5  # Number of search results to return (max 10)