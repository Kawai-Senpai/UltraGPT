#!/usr/bin/env python3
"""
Example 4: Native Thinking / Reasoning
Demonstrates native reasoning with Claude, GPT-5, Gemini 3
"""

import os
from dotenv import load_dotenv
from ultragpt import UltraGPT

# Load API key
load_dotenv()

# Initialize
ultra = UltraGPT(
    openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
    verbose=True
)

print("=" * 70)
print("NATIVE THINKING / REASONING EXAMPLE")
print("=" * 70)

# Example 1: Claude with native thinking
print("\n1. Claude Sonnet with Native Reasoning:")
response, tokens, details = ultra.chat(
    messages=[{
        "role": "user",
        "content": "Solve step by step: If 3x + 7 = 22, find x"
    }],
    model="claude:sonnet",
    reasoning_pipeline=True  # Triggers native thinking
)
print(f"Response: {response}")
print(f"\nToken Breakdown:")
print(f"  Input tokens: {details.get('input_tokens', 0)}")
print(f"  Output tokens: {details.get('output_tokens', 0)}")
print(f"  Reasoning tokens (API): {details.get('reasoning_tokens_api', 0)}")
print(f"  Total tokens: {tokens}")

if details.get('reasoning_text'):
    print(f"\nReasoning text sample: {details['reasoning_text'][:100]}...")

# Example 2: GPT-4o WITHOUT native thinking (uses fake pipeline)
print("\n2. GPT-4o with Fake Reasoning Pipeline:")
response, tokens, details = ultra.chat(
    messages=[{
        "role": "user",
        "content": "What are the long-term implications of AI on employment?"
    }],
    model="gpt-4o",
    reasoning_pipeline=True,  # Uses simulated pipeline
    reasoning_iterations=3
)
print(f"Response: {response[:200]}...")
print(f"\nReasoning Pipeline:")
print(f"  Thoughts generated: {len(details.get('reasoning', []))}")
if details.get('reasoning'):
    print("  Sample thoughts:")
    for i, thought in enumerate(details['reasoning'][:3], 1):
        print(f"    {i}. {thought[:60]}...")
print(f"\nTotal tokens: {tokens}")

# Example 3: Comparison - with vs without thinking
print("\n3. Comparison - With vs Without Thinking:")

# Without thinking
response_basic, tokens_basic, details_basic = ultra.chat(
    messages=[{"role": "user", "content": "What is 15% of 240?"}],
    model="claude:sonnet",
    reasoning_pipeline=False
)
print(f"Without thinking:")
print(f"  Response: {response_basic}")
print(f"  Tokens: {tokens_basic}")

# With thinking
response_thinking, tokens_thinking, details_thinking = ultra.chat(
    messages=[{"role": "user", "content": "What is 15% of 240?"}],
    model="claude:sonnet",
    reasoning_pipeline=True
)
print(f"\nWith thinking:")
print(f"  Response: {response_thinking}")
print(f"  Tokens: {tokens_thinking}")
print(f"  Reasoning tokens: {details_thinking.get('reasoning_tokens_api', 0)}")
print(f"  Reasoning enabled: {details_thinking.get('reasoning_tokens_api', 0) > 0}")

# Example 4: Gemini 3 Pro with reasoning
print("\n4. Gemini 3 Pro with Native Reasoning:")
response, tokens, details = ultra.chat(
    messages=[{
        "role": "user",
        "content": "Explain why the sky is blue in simple terms"
    }],
    model="gemini-3-pro",
    reasoning_pipeline=True
)
print(f"Response: {response[:200]}...")
print(f"Reasoning tokens: {details.get('reasoning_tokens_api', 0)}")

print("\n✓ Native thinking/reasoning works seamlessly!")
print("Models with native support: Claude (all), GPT-5, o-series, Gemini 3")
print("Models without: Use simulated reasoning pipeline automatically")
