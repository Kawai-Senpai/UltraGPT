# 🤖 UltraGPT

**A powerful and modular library for advanced AI-based reasoning and step pipelines with multi-provider support**

[![Python](https://img.shields.io/badge/python-3.6+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.rst)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)](https://openai.com)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-purple.svg)](https://anthropic.com)

## 🌟 Features

- **🔄 Multi-Provider Support:** Use OpenAI and Anthropic Claude models seamlessly
- **📝 Steps Pipeline:** Break down complex tasks into manageable steps
- **🧠 Reasoning Pipeline:** Advanced multi-iteration reasoning capabilities
- **🛠️ Tool Integration:** Web search, calculator, math operations, and custom tools
- **🎯 Structured Output:** Get structured responses using Pydantic schemas
- **� Tool Calling:** Execute custom tools with validated parameters
- **📊 Token Management:** Comprehensive token tracking across providers

## 📦 Installation

```bash
pip install ultragpt

# For environment variable support (optional)
pip install python-dotenv
```

**Note:** Starting with version 4.0.0, Anthropic Claude support is included by default!

## 🚀 Quick Start

### Basic Usage (OpenAI)
```python
from ultragpt import UltraGPT

# Initialize with OpenAI (default)
ultragpt = UltraGPT(api_key="your-openai-api-key")

# Simple chat
response, tokens, details = ultragpt.chat([
    {"role": "user", "content": "Write a story about an elephant."}
])
print("Response:", response)
print("Tokens used:", tokens)
```

### Multi-Provider Support
```python
from ultragpt import UltraGPT

# OpenAI (default)
ultragpt_openai = UltraGPT(api_key="your-openai-api-key")

# Claude
ultragpt_claude = UltraGPT(
    api_key="your-anthropic-api-key", 
    provider="anthropic"
)

# Both work the same way!
response, tokens, details = ultragpt_claude.chat([
    {"role": "user", "content": "Hello Claude!"}
])
```

### Provider:Model Format
```python
# Use provider:model format for specific models
ultragpt = UltraGPT(
    api_key="your-openai-api-key",
    claude_api_key="your-anthropic-api-key"  # For Claude models
)

# OpenAI models
response = ultragpt.chat([
    {"role": "user", "content": "Hello!"}
], model="openai:gpt-4o")

# Claude models  
response = ultragpt.chat([
    {"role": "user", "content": "Hello!"}
], model="claude:claude-3-sonnet-20240229")
```

## 🌐 Web Search (Google) & Scraping

UltraGPT now supports **Google Custom Search API** for web search, with optional scraping of result pages for deeper context.

### 🔑 **Google API Setup**
1. Get a Google Custom Search API key from [Google Cloud Console](https://console.cloud.google.com/)
2. Create a Custom Search Engine at [cse.google.com](https://cse.google.com/)
3. Note your API key and Search Engine ID

### 🛠️ **Usage Example**

```python
from ultragpt import UltraGPT

ultra = UltraGPT(
    api_key="your-openai-api-key",
    google_api_key="your-google-api-key",
    search_engine_id="your-search-engine-id",
    verbose=True
)

# Configure web search tool (scraping enabled, max 3 results)
tools_config = {
    "web-search": {
        "max_results": 3,
        "enable_scraping": True,
        "max_scrape_length": 2000
    }
}

messages = [
    {"role": "user", "content": "What are the latest trends in AI?"}
]

response = ultra.chat(
    messages=messages,
    tools=["web-search"],
    tools_config=tools_config,
    steps_pipeline=False,
    reasoning_pipeline=False
)

print(response["output"])
```

#### 🔄 **Override Search Engine ID per Call**
You can override the search engine for a specific chat call:
```python
tools_config = {
    "web-search": {
        "search_engine_id": "another-engine-id",
        "max_results": 2
    }
}
```

### 🛡️ **Error Handling**
- All web search errors (API, quota, scraping) return an **empty string** to the AI (never error text)
- Errors are logged via `self.log` and shown in verbose mode, but never contaminate the AI's output
- Scraping failures are skipped silently

### 🕷️ **Scraping**
- Set `enable_scraping: True` to extract main content from result pages
- Control length with `max_scrape_length`
- Respects robots.txt and rate limits

## 📚 Advanced Usage

### Customizing Pipeline Settings

```python
ultragpt = UltraGPT(
    api_key="your-openai-api-key",
    model="gpt-4o",  # Specify model
    temperature=0.7,  # Adjust creativity
    reasoning_iterations=3,  # Set reasoning depth
    steps_pipeline=True,
    reasoning_pipeline=True,
    verbose=True
)
```

### Using Tools

```python
ultragpt = UltraGPT(
    api_key="your-openai-api-key",
    tools=["web-search", "calculator", "math-operations"],
    tools_config={
        "web-search": {
            "max_results": 1,
            "model": "gpt-4o"
        },
        "calculator": {
            "model": "gpt-4o"
        },
        "math-operations": {
            "model": "gpt-4o"
        }
    }
)
```

## 🆕 New Features

### Advanced Mathematical Operations Tool

UltraGPT now includes a powerful math operations tool that can handle complex mathematical queries and **multiple operations in a single request**:

```python
# Example: Multiple operations in one request
response = ultragpt.chat([{
    "role": "user", 
    "content": """
    Please perform these calculations:
    1. Check if [1, 5, 8] lie between 0 and 10
    2. Are 17, 23, 29 prime numbers?
    3. Get statistical summary of [1, 2, 3, 4, 5]
    4. Find outliers in [1, 2, 3, 100]
    """
}], tools=["math-operations"])

# Example: Multiple range checks
response = ultragpt.chat([{
    "role": "user", 
    "content": "Check if [1, 5, 8] lie between 0-10 and [15, 20, 25] lie between 10-30"
}], tools=["math-operations"])
```

**Available Operations:**
- **Range checking** (numbers between bounds) - supports multiple ranges
- **Outlier detection** (IQR and z-score methods) - multiple datasets
- **Proximity checking** (numbers close to target) - multiple proximity checks
- **Statistical summaries** (mean, median, std dev, etc.) - multiple datasets
- **Prime number checking** - multiple number sets
- **Factor analysis and prime factorization** - multiple numbers
- **Sequence analysis** (arithmetic/geometric) - multiple sequences
- **Percentage calculations and ratios** - multiple calculations

### Model Control for Pipelines

You can now use different models for different parts of the pipeline:

```python
ultragpt = UltraGPT(api_key="your-key")

response = ultragpt.chat(
    messages=[{"role": "user", "content": "Solve this complex problem"}],
    model="gpt-4o",  # Main model for final response
    steps_model="gpt-4o-mini",  # Cheaper model for step generation
    reasoning_model="gpt-4o-mini",  # Cheaper model for reasoning
    reasoning_iterations=3
)
```

This allows you to:
- **Save costs** by using cheaper models for intermediate steps
- **Optimize performance** by using faster models for simple operations
- **Maintain quality** by using premium models for final outputs

## 🔧 Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | Required | Your OpenAI API key |
| `model` | str | "gpt-4o" | Model to use |
| `temperature` | float | 0.7 | Output randomness |
| `reasoning_iterations` | int | 3 | Number of reasoning steps |
| `tools` | list | [] | Enabled tools |
| `verbose` | bool | False | Enable detailed logging |

## 🌐 Tool System

UltraGPT supports various tools to enhance its capabilities:

### Web Search
- Performs intelligent web searches
- Summarizes findings

### Calculator
- Advanced mathematical calculations
- Expression evaluation

### Math Operations
- Range checking and validation
- Statistical analysis and outlier detection
- Prime number checking and factorization
- Sequence analysis (arithmetic/geometric patterns)
- Percentage calculations and ratios
- Proximity checking with tolerance
- Integrates results into responses

### Calculator
- Handles mathematical operations
- Supports complex calculations
- Provides step-by-step solutions

### Math Operations
- Range checking (numbers between bounds)
- Outlier detection (IQR and z-score methods)
- Proximity checking (numbers close to target)
- Statistical summaries (mean, median, std dev, etc.)
- Prime number checking
- Factor analysis and prime factorization
- Sequence analysis (arithmetic/geometric)
- Percentage calculations and ratios

## 🔄 Pipeline System

### Steps Pipeline
1. Task Analysis
2. Step Generation
3. Step-by-Step Execution
4. Progress Verification
5. Final Compilation

### Reasoning Pipeline
1. Initial Analysis
2. Multi-iteration Thinking
3. Thought Development
4. Conclusion Formation

## 📋 Requirements

- Python 3.6+
- OpenAI API key
- Internet connection (for web tools)

## 🤝 Contributing

Contributions are always welcome! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make changes
4. Commit (`git commit -am 'Add new feature'`)
5. Push (`git push origin feature/improvement`)
6. Open a Pull Request

## 📝 License

This project is MIT licensed - see the [LICENSE](LICENSE) file for details.

## 👥 Author

**Ranit Bhowmick**
- Email: bhowmickranitking@duck.com
- GitHub: [@Kawai-Senpai](https://github.com/Kawai-Senpai)

---

<div align="center">
Made with ❤️ by Ranit Bhowmick
</div>
