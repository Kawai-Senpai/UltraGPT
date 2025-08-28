"""
Context module for web search tool to access credentials securely
"""
import threading

# Thread-local storage for credentials
_context = threading.local()

def set_credentials(google_api_key: str, search_engine_id: str):
    """Set Google API credentials for the current thread"""
    _context.google_api_key = google_api_key
    _context.search_engine_id = search_engine_id

def get_credentials():
    """Get Google API credentials for the current thread"""
    return getattr(_context, 'google_api_key', None), getattr(_context, 'search_engine_id', None)

def clear_credentials():
    """Clear credentials for the current thread"""
    if hasattr(_context, 'google_api_key'):
        del _context.google_api_key
    if hasattr(_context, 'search_engine_id'):
        del _context.search_engine_id
