#!/usr/bin/env python3
"""
Example 1: Basic Chat with OpenRouter
Demonstrates simple chat across different models with one API key
"""

import os
from dotenv import load_dotenv
from ultragpt import UltraGPT

# Load API key from .env
load_dotenv()

# Initialize UltraGPT with OpenRouter (universal access to all models)
ultra = UltraGPT(
    openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
    verbose=True
)

print("=" * 70)
print("BASIC CHAT EXAMPLE")
print("=" * 70)

# Example 1: Simple chat with GPT-5
print("\n1. GPT-5 Chat:")
response, tokens, details = ultra.chat(
    messages=[{"role": "user", "content": "What is the capital of France? Be brief."}],
    model="gpt-5"
)
print(f"Response: {response}")
print(f"Tokens: {tokens}")

# Example 2: Same question with Claude (1M context!)
print("\n2. Claude Sonnet 4.5 Chat (1M context):")
response, tokens, details = ultra.chat(
    messages=[{"role": "user", "content": "What is the capital of France? Be brief."}],
    model="claude-sonnet-4.5"  # or "claude:sonnet"
)
print(f"Response: {response}")
print(f"Tokens: {tokens}")

# Example 3: Gemini 3 Pro
print("\n3. Gemini 3 Pro Chat:")
response, tokens, details = ultra.chat(
    messages=[{"role": "user", "content": "What is the capital of France? Be brief."}],
    model="gemini"  # or "gemini-3-pro"
)
print(f"Response: {response}")
print(f"Tokens: {tokens}")

# Example 4: Multi-turn conversation
print("\n4. Multi-Turn Conversation:")
messages = [
    {"role": "user", "content": "What is 15 * 24?"},
    {"role": "assistant", "content": "15 * 24 = 360"},
    {"role": "user", "content": "Now divide that by 5"}
]

response, tokens, details = ultra.chat(
    messages=messages,
    model="gpt-5"
)
print(f"Response: {response}")
print(f"Total tokens: {tokens}")

print("\n✓ Example complete! Same code works with ALL models.")
