# UltraGPT Source Package

This is the source package for UltraGPT. For full documentation, see the [main README](../../README.md).

## Module Overview

| Module | Description |
|--------|-------------|
| `core/` | Main orchestrator, chat flow, and pipelines |
| `providers/` | OpenRouter provider with LangChain patches |
| `messaging/` | Message normalization, history utils, token limiting |
| `schemas/` | Pydantic schema sanitization for OpenAI |
| `tooling/` | Tool manager for internal and user tools |
| `tools/` | Built-in tools (web-search, calculator, math) |
| `prompts/` | Pipeline prompts for reasoning and steps |
| `config/` | Default configuration settings |

## Quick Import

```python
from ultragpt import UltraGPT

ultra = UltraGPT(openrouter_api_key="...")
response, tokens, details = ultra.chat(messages=[...], model="gpt-5")
```
