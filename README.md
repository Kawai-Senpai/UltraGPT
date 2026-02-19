# 🤖 UltraGPT

<p align="center">
  <img src="assets/UltragptCover.jpg" alt="UltraGPT Cover" width="100%">
</p>

**The "Write Once, Run Everywhere" AI library that handles ALL the heavy lifting**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.rst)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-Universal%20Access-orange.svg)](https://openrouter.ai)

---

## 🎯 Why UltraGPT?

**This is NOT just another LangChain wrapper.** UltraGPT is a battle-tested, production-grade abstraction that solves the **real problems** developers face when building AI applications:

### The Problems We Solve

| Problem | What Others Do | What UltraGPT Does |
|---------|----------------|-------------------|
| **Message Format Hell** | Forces you to convert between formats | Auto-converts ANY format to LangChain, OpenAI, or provider-specific |
| **Tool Call Orphans** | Crashes when tool results are missing | Sanitizes history, removes orphans, validates pairing |
| **Token Limits** | Crashes on overflow | Smart truncation with atomic tool-call grouping |
| **Provider Quirks** | One provider = one codebase | True "write once, run everywhere" across ALL models |
| **Structured Output** | Breaks on Claude/Gemini | Universal schema support via tool-based fallback |
| **Reasoning Models** | Manual complexity | Auto-detects native thinking, preserves reasoning_details |
| **Rate Limits** | Crashes your app | Built-in exponential backoff with jitter |
| **Streaming Issues** | Connection pool leaks | Proper cleanup, mid-stream error detection |
| **Schema Validation** | 400 errors everywhere | Auto-sanitizes Pydantic → OpenAI strict mode |

---

## ✨ Key Features

### 🌐 **Universal Model Access via OpenRouter**
One API key, ALL models:
- **GPT-5** (400k context, reasoning tokens)
- **Claude Sonnet 4.5** (1M extended context!)
- **Claude Opus 4** (200k context)
- **Gemini 3 Pro/Flash** (1M context, reasoning)
- **Grok 4** (256k context, always-on reasoning)
- **Llama 3.3**, **DeepSeek v3.2**, **Mistral**, and more

### 🧠 **Native Thinking/Reasoning Support**
- Auto-detects models with native reasoning (Claude, o-series, GPT-5, Gemini 3)
- Preserves `reasoning_details` across tool call loops
- Falls back to simulated reasoning pipeline for non-reasoning models
- Full token breakdown: input, output, reasoning tokens

### 🛠️ **Production-Grade Tool Calling**
- Universal tool format that works across ALL providers
- Automatic schema sanitization for strict mode compliance
- Preserves reasoning context for multi-turn tool conversations
- Parallel and sequential tool call support

### 📊 **Structured Output That Actually Works**
- Pydantic schemas → provider-compatible JSON
- Tool-based fallback for providers without native support
- Handles `Optional` fields, nested objects, arrays
- No more 400 errors from schema validation

### 💾 **Intelligent Token Management**
- Auto-truncation with model-specific limits
- Atomic tool-call grouping (never orphan a tool result)
- Preserves system messages during truncation
- Configurable: `"AUTO"`, `"OFF"`, or specific token count

### 🔄 **Message History Sanitization**
- Removes orphaned tool results automatically
- Drops unresolved tool calls before API submission
- Consolidates multiple system messages safely
- Strips whitespace (Claude is strict about this!)

### 🔧 **LangChain Patches for OpenRouter**
- Preserves `reasoning_details`, `cache_control`, `thinking` fields
- Future-proof: unknown fields pass through automatically
- Works with streaming and non-streaming responses

---

## 📦 Installation

```bash
pip install ultragpt
```

### Requirements
- Python 3.9+
- OpenRouter API key (get one at [openrouter.ai/keys](https://openrouter.ai/keys))

---

## 🚀 Quick Start

### Basic Chat
```python
from ultragpt import UltraGPT

# Initialize with OpenRouter (universal access)
ultra = UltraGPT(openrouter_api_key="your-openrouter-key")

# Simple chat - works with ANY model
response, tokens, details = ultra.chat(
    messages=[{"role": "user", "content": "Explain quantum computing in 3 sentences."}],
    model="gpt-5"  # or "claude:sonnet", "gemini", etc.
)

print(response)
print(f"Tokens used: {tokens}")
```

### Model Selection (Friendly Names)
```python
# GPT models
ultra.chat(messages=[...], model="gpt-5")
ultra.chat(messages=[...], model="gpt-5-pro")
ultra.chat(messages=[...], model="gpt-4o")

# Claude models (extended 1M context for Sonnet!)
ultra.chat(messages=[...], model="claude:sonnet")  # Claude 3.7 Sonnet
ultra.chat(messages=[...], model="claude:opus")    # Claude Opus 4
ultra.chat(messages=[...], model="claude-sonnet-4.5")  # Latest Sonnet

# Gemini models
ultra.chat(messages=[...], model="gemini")  # Gemini 3 Pro
ultra.chat(messages=[...], model="gemini-3-flash")

# Other models
ultra.chat(messages=[...], model="grok")  # Grok 4
ultra.chat(messages=[...], model="deepseek")  # DeepSeek v3.2
ultra.chat(messages=[...], model="llama-3.3")
```

---

## 🧠 Native Thinking/Reasoning

UltraGPT automatically detects and uses native reasoning for supported models:

```python
# Native reasoning is auto-enabled for Claude, GPT-5, o-series, Gemini 3
response, tokens, details = ultra.chat(
    messages=[{"role": "user", "content": "Solve: If 3x + 7 = 22, find x"}],
    model="claude:sonnet",
    reasoning_pipeline=True,  # Triggers native thinking on supported models
)

# Access reasoning tokens and text
print(f"Reasoning tokens: {details.get('reasoning_tokens_api', 0)}")
print(f"Reasoning text: {details.get('reasoning_text')}")
print(f"Full details: {details.get('reasoning_details')}")
```

### Fake Reasoning Pipeline (for non-reasoning models)
```python
# For models without native reasoning (like GPT-4o), a simulated pipeline runs
response, tokens, details = ultra.chat(
    messages=[{"role": "user", "content": "Plan a trip to Japan"}],
    model="gpt-4o",
    reasoning_pipeline=True,
    reasoning_iterations=3,
)

# Get the thoughts generated
print(f"Reasoning thoughts: {details.get('reasoning')}")
```

---

## 📊 Structured Output

### Using Pydantic Schemas
```python
from pydantic import BaseModel, Field

class SentimentAnalysis(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="0.0 to 1.0")
    keywords: list[str] = Field(description="Key words from the text")

response, tokens, details = ultra.chat(
    messages=[{"role": "user", "content": "Analyze: 'I absolutely love this product!'"}],
    model="gpt-5",
    schema=SentimentAnalysis,
)

print(response)
# {'sentiment': 'positive', 'confidence': 0.95, 'keywords': ['love', 'absolutely', 'product']}
```

### Works Across ALL Providers
```python
# Same schema works with Claude (uses tool-based fallback automatically)
response, tokens, details = ultra.chat(
    messages=[{"role": "user", "content": "Analyze: 'This is terrible!'"}],
    model="claude:sonnet",
    schema=SentimentAnalysis,
)
# Still works! UltraGPT handles the differences automatically.
```

---

## 🛠️ Tool Calling

### Define Custom Tools
```python
from pydantic import BaseModel

class CalculatorParams(BaseModel):
    operation: str  # add, subtract, multiply, divide
    a: float
    b: float

calculator_tool = {
    "name": "calculator",
    "description": "Performs arithmetic calculations",
    "parameters_schema": CalculatorParams,
    "usage_guide": "Use for precise arithmetic calculations",
    "when_to_use": "When user needs numeric computation",
}

# Make a tool call
response, tokens, details = ultra.tool_call(
    messages=[{"role": "user", "content": "Calculate 25 * 8"}],
    user_tools=[calculator_tool],
    model="claude:sonnet",
)

print(response)
# [{'id': 'call_xxx', 'type': 'function', 'function': {'name': 'calculator', 'arguments': '{"operation": "multiply", "a": 25, "b": 8}'}}]
```

### Parallel Tool Calls
```python
# Allow multiple tools in single response
response, tokens, details = ultra.tool_call(
    messages=[{"role": "user", "content": "Add 10+5 and multiply 3*7"}],
    user_tools=[calculator_tool],
    allow_multiple=True,  # Returns array of tool calls
    model="gpt-5",
)
```

### Tool Calling with Native Reasoning
```python
# Reasoning models preserve context across tool loops
response, tokens, details = ultra.tool_call(
    messages=[{"role": "user", "content": "Use calculator to find 25 * 8"}],
    user_tools=[calculator_tool],
    model="claude:sonnet",
    reasoning_pipeline=True,  # Uses native thinking
)

# reasoning_details preserved for next turn
print(details.get("reasoning_details"))
```

### Tool Call Return Format

`tool_call()` returns `(response, tokens, details_dict)`. The `response` value depends on what the LLM produces:

| LLM Output | `allow_multiple=True` | `allow_multiple=False` |
|---|---|---|
| Tool calls only (most common) | `list` of tool call dicts | Single tool call `dict` |
| Tool calls + text content | `{"tool_calls": [...], "content": "text"}` | `{"tool_calls": [...], "content": "text"}` |
| Text only (no tool calls) | `{"content": "text"}` | `{"content": "text"}` |
| Empty response | `{"content": ""}` | `{"content": ""}` |

**Key guarantees:**
- `tool_call()` **never returns `None`** - always a list or dict
- LLM text content is **never dropped** - even when returned alongside tool calls
- Multi-modal content blocks (list-type) are automatically normalized to plain strings
- `details_dict["reasoning_details"]` is always available separately (3rd return value)

**Handling all cases:**
```python
response, tokens, details = ultra.tool_call(messages=msgs, user_tools=tools, allow_multiple=True)

if isinstance(response, list):
    # Tool calls only (backward-compatible array)
    for tc in response:
        name = tc["function"]["name"]
        args = tc["function"]["arguments"]

elif isinstance(response, dict):
    if "tool_calls" in response:
        # Mixed: tool calls + accompanying text from LLM
        tool_calls = response["tool_calls"]
        llm_text = response.get("content", "")  # "Let me check that..."
    else:
        # Text only (no tool calls)
        text = response.get("content", "")
```

---

## 📝 Pipelines

### Steps Pipeline
Break complex tasks into manageable steps:

```python
response, tokens, details = ultra.chat(
    messages=[{"role": "user", "content": "Plan a 2-week trip to Japan"}],
    model="gpt-5",
    steps_pipeline=True,
    steps_model="gpt-5-nano",  # Use cheaper model for planning
)

print(f"Steps: {details.get('steps')}")
print(f"Conclusion: {response}")
```

### Reasoning Pipeline
Multi-iteration deep thinking:

```python
response, tokens, details = ultra.chat(
    messages=[{"role": "user", "content": "What are the long-term implications of AI on employment?"}],
    model="gpt-4o",
    reasoning_pipeline=True,
    reasoning_iterations=5,
    reasoning_model="gpt-4o-mini",  # Use cheaper model for iterations
)

print(f"Thoughts generated: {len(details.get('reasoning', []))}")
```

---

## 💾 Token Management

### Automatic Truncation
```python
ultra = UltraGPT(
    openrouter_api_key="...",
    input_truncation="AUTO",  # Automatically fits model's context limit
)

# Or specify exact limit
ultra = UltraGPT(
    openrouter_api_key="...",
    input_truncation=50000,  # Max 50k tokens
)

# Or disable
ultra = UltraGPT(
    openrouter_api_key="...",
    input_truncation="OFF",
)
```

### How Truncation Works
1. Groups tool calls with their results (never orphans)
2. Preserves system messages
3. Removes oldest messages first (keeps newest)
4. Ensures at least one HumanMessage remains

---

## 🌐 Web Search (Built-in Tool)

```python
ultra = UltraGPT(
    openrouter_api_key="...",
    google_api_key="your-google-api-key",
    search_engine_id="your-search-engine-id",
)

response, tokens, details = ultra.chat(
    messages=[{"role": "user", "content": "What are the latest AI trends in 2026?"}],
    model="gpt-5",
    tools=["web-search"],
    tools_config={
        "web-search": {
            "max_results": 3,
            "enable_scraping": True,
            "max_scrape_length": 5000,
        }
    },
)
```

---

## 🔧 Advanced Configuration

### Full Initialization Options
```python
ultra = UltraGPT(
    # API Keys
    openrouter_api_key="...",  # Required: Universal access to all models
    google_api_key="...",      # Optional: For web search
    search_engine_id="...",    # Optional: For web search
    
    # Token Management
    max_tokens=4096,           # Max output tokens
    input_truncation="AUTO",   # "AUTO", "OFF", or int
    
    # Logging
    verbose=True,              # Show detailed logs
    logger_name="ultragpt",
    log_to_file=False,
    log_to_console=True,
    log_level="DEBUG",
)
```

### Chat Method Full Signature
```python
response, tokens, details = ultra.chat(
    messages=[...],
    
    # Model Selection
    model="gpt-5",              # Model to use
    temperature=0.7,            # Creativity (0-1)
    max_tokens=4096,            # Max output tokens
    
    # Structured Output
    schema=MyPydanticSchema,    # Optional: Force structured response
    
    # Pipelines
    steps_pipeline=False,       # Enable step-by-step planning
    reasoning_pipeline=False,   # Enable multi-iteration reasoning
    steps_model="gpt-5-nano",   # Model for steps
    reasoning_model="gpt-5-nano", # Model for reasoning
    reasoning_iterations=3,     # Reasoning depth
    
    # Tools
    tools=["web-search"],       # Built-in tools
    tools_config={...},         # Tool configuration
    
    # Token Management
    input_truncation="AUTO",    # Override instance setting
)
```

---

## 📊 Response Details

Every call returns `(response, tokens, details)`:

```python
response, tokens, details = ultra.chat(...)

# Token breakdown
print(f"Input tokens: {details.get('input_tokens')}")
print(f"Output tokens: {details.get('output_tokens')}")
print(f"Total tokens: {details.get('total_tokens')}")
print(f"Reasoning tokens: {details.get('reasoning_tokens_api')}")

# Pipeline metrics (if used)
print(f"Reasoning pipeline tokens: {details.get('reasoning_pipeline_total_tokens')}")
print(f"Steps pipeline tokens: {details.get('steps_pipeline_total_tokens')}")

# Reasoning content (for reasoning models)
print(f"Reasoning text: {details.get('reasoning_text')}")
print(f"Reasoning details: {details.get('reasoning_details')}")

# Tools used
print(f"Tools called: {details.get('tools_used')}")
```

---

## 🛡️ Production Features

### Rate Limit Handling
```python
# Built-in exponential backoff with jitter
# Configurable in config/config.py:
# - RATE_LIMIT_RETRIES = 5
# - RATE_LIMIT_BASE_DELAY = 10
# - RATE_LIMIT_MAX_DELAY = 60
# - RATE_LIMIT_BACKOFF_MULTIPLIER = 2
```

### Stream Timeout Protection
```python
# Streams have wall-clock deadlines (default 1 hour)
# Prevents infinite hanging on stalled connections
# Proper cleanup prevents connection pool leaks
```

### Message Sanitization
```python
# Before each API call, UltraGPT:
# 1. Removes orphaned tool results
# 2. Drops unresolved tool calls
# 3. Consolidates system messages
# 4. Strips trailing whitespace (Claude requirement)
# 5. Validates tool call pairing
```

### Schema Sanitization
```python
# Pydantic schemas are automatically transformed:
# 1. anyOf/Optional patterns → direct types
# 2. additionalProperties: false added
# 3. required arrays completed
# 4. "default" keywords stripped (causes 400s)
# 5. Nested objects recursively fixed
```

---

## 🔌 Using the LLM Directly

Need the raw LangChain `ChatOpenAI` instance?

```python
# Get the underlying LLM for custom operations
llm = ultra.provider_manager.get_provider("openrouter").build_llm(
    model="gpt-5",
    temperature=0.7,
    max_tokens=4096,
)

# Use directly with LangChain
response = llm.invoke([...])
```

---

## 📁 Project Structure

```
ultragpt/
├── core/
│   ├── core.py          # Main UltraGPT orchestrator
│   ├── chat_flow.py     # Chat operations
│   └── pipelines.py     # Reasoning & Steps pipelines
├── providers/
│   ├── providers.py     # OpenRouter provider
│   └── _langchain_patches.py  # Field preservation patches
├── messaging/
│   ├── message_ops.py   # Message consolidation
│   ├── history_utils.py # Orphan removal, validation
│   ├── token_manager.py # Message normalization
│   └── token_limits/
│       └── langchain_limiter.py  # Smart truncation
├── schemas/
│   ├── schema_utils.py  # Pydantic → OpenAI conversion
│   └── tool_schemas.py  # Tool/ExpertTool definitions
├── tooling/
│   └── tools_manager.py # Tool loading & execution
├── tools/
│   ├── web_search/      # Google search & scraping
│   ├── calculator/      # Basic calculator
│   └── math_operations/ # Advanced math
├── prompts/
│   └── prompts.py       # Pipeline prompts
└── config/
    └── config.py        # Default settings
```

---

## � Running Tests

UltraGPT doesn't include tests in the package, but here are essential verification scripts you should run:

### Basic Functionality Test
```python
from ultragpt import UltraGPT
import os

# Initialize
ultra = UltraGPT(openrouter_api_key=os.getenv("OPENROUTER_API_KEY"))

# Test 1: Basic chat
response, tokens, details = ultra.chat(
    messages=[{"role": "user", "content": "What is 2+2?"}],
    model="gpt-5"
)
print(f"✓ Basic chat: {response} (tokens: {tokens})")

# Test 2: Structured output
from pydantic import BaseModel
class Answer(BaseModel):
    result: int
    explanation: str

response, tokens, details = ultra.chat(
    messages=[{"role": "user", "content": "What is 5*8? Explain."}],
    model="gpt-5",
    schema=Answer,
)
print(f"✓ Structured output: {response}")

# Test 3: Tool calling
calculator = {
    "name": "calculator",
    "description": "Performs arithmetic",
    "parameters_schema": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["add", "multiply"]},
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["operation", "a", "b"]
    },
    "usage_guide": "Use for calculations",
    "when_to_use": "When user needs math",
}

response, tokens, details = ultra.tool_call(
    messages=[{"role": "user", "content": "Calculate 25 * 8"}],
    user_tools=[calculator],
    model="gpt-5",
)
print(f"✓ Tool calling: {response}")
```

### Native Thinking Test
```python
# Test with reasoning model
response, tokens, details = ultra.chat(
    messages=[{"role": "user", "content": "Solve step by step: If 3x + 7 = 22, find x"}],
    model="claude:sonnet",
    reasoning_pipeline=True,  # Auto-detects native thinking
)

print(f"Response: {response}")
print(f"Reasoning tokens: {details.get('reasoning_tokens_api', 0)}")
print(f"Has reasoning: {'reasoning_text' in details}")
```

### Multi-Provider Test
```python
# Test different providers with same code
for model in ["gpt-5", "claude:sonnet", "gemini"]:
    response, tokens, details = ultra.chat(
        messages=[{"role": "user", "content": "Say hello"}],
        model=model,
    )
    print(f"✓ {model}: {response} ({tokens} tokens)")
```

### Message Sanitization Test
```python
# Test orphan removal (shouldn't crash)
from langchain_core.messages import AIMessage, ToolMessage

messages = [
    {"role": "user", "content": "Hello"},
    AIMessage(content="", tool_calls=[{"id": "call_123", "name": "test", "args": {}}]),
    # Missing tool result - should be sanitized automatically
]

response, tokens, details = ultra.chat(
    messages=messages,
    model="gpt-5",
)
print("✓ Orphan tool calls handled gracefully")
```

---

## �🤝 Contributing

Contributions welcome! Please ensure:
1. All tests pass
2. Code follows existing patterns
3. Documentation updated for new features

---

## 📄 License

MIT License - see [LICENSE.rst](LICENSE.rst)

---

## 🙏 Acknowledgments

Built on top of [LangChain](https://github.com/langchain-ai/langchain) with patches for OpenRouter compatibility.

Powered by [OpenRouter](https://openrouter.ai) for universal model access.

---

**UltraGPT: Stop fighting with AI providers. Start building.**
