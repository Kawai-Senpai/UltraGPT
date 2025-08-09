# UltraGPT Configuration
# This file contains all configurable parameters for UltraGPT

# Default Models
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_STEPS_MODEL = "gpt-4.1-nano"
DEFAULT_REASONING_MODEL = "gpt-4.1-nano"
DEFAULT_PARSE_MODEL = "gpt-4.1"
DEFAULT_TOOLS_MODEL = "gpt-4.1"

# Token Configuration
MAX_TOKENS_DEFAULT = 4096

# History Configuration
MAX_CONTEXT_MESSAGES = 10

# Processing Configuration
DEFAULT_REASONING_ITERATIONS = 3
DEFAULT_TEMPERATURE = 0.7

# Tool Selection Configuration
TOOL_SELECTION_TEMPERATURE = 0.1

# Tool-specific Configuration
TOOLS_CONFIG = {
    "web-search": {
        "max_results": 5,
        "model": "gpt-4.1-nano",
        "enable_scraping": True,
        "max_scrape_length": 5000,
        "scrape_timeout": 15,
        "scrape_pause": 1,
        "max_history_items": 5
    },
    "calculator": {
        "model": "gpt-4.1-nano",
        "max_history_items": 5
    },
    "math-operations": {
        "model": "gpt-4.1-nano",
        "max_history_items": 5
    }
}

# Default Tools List
DEFAULT_TOOLS = []
