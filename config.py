# UltraGPT Configuration
# This file contains all configurable parameters for UltraGPT

# Default Models
DEFAULT_MODEL = "gpt-5"
DEFAULT_STEPS_MODEL = "gpt-5-nano"
DEFAULT_REASONING_MODEL = "gpt-5-nano"
DEFAULT_PARSE_MODEL = "gpt-5"
DEFAULT_TOOLS_MODEL = "gpt-5"

# History Configuration
MAX_CONTEXT_MESSAGES = 10

# Processing Configuration
DEFAULT_REASONING_ITERATIONS = 3
DEFAULT_TEMPERATURE = 0.7

# Token Configuration
DEFAULT_INPUT_TRUNCATION = "AUTO"  # Can be "AUTO", "OFF", or a specific token number

# Rate Limit Retry Configuration
RATE_LIMIT_RETRIES = 5  # Number of retries for rate limit errors
RATE_LIMIT_BASE_DELAY = 10  # Base delay in seconds (will use exponential backoff)
RATE_LIMIT_MAX_DELAY = 60  # Maximum delay in seconds
RATE_LIMIT_BACKOFF_MULTIPLIER = 2  # Multiplier for exponential backoff

# Tool Selection Configuration
TOOL_SELECTION_TEMPERATURE = 0.1

# Tool-specific Configuration
TOOLS_CONFIG = {
    "web-search": {
        "max_results": 5,
        "model": "gpt-5-nano",
        "enable_scraping": True,
        "max_scrape_length": 5000,
        "scrape_timeout": 15,
        "scrape_pause": 1,
        "max_history_items": 5
    },
    "calculator": {
        "model": "gpt-5-nano",
        "max_history_items": 5
    },
    "math-operations": {
        "model": "gpt-5-nano",
        "max_history_items": 5
    }
}

# Default Tools List
DEFAULT_TOOLS = []
