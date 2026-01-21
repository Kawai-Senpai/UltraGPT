#!/usr/bin/env python3
"""
Example 5: Pipelines (Steps & Reasoning)
Demonstrates multi-step planning and deep reasoning pipelines
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
print("PIPELINES EXAMPLE")
print("=" * 70)

# Example 1: Steps Pipeline
print("\n1. Steps Pipeline - Task Planning:")
response, tokens, details = ultra.chat(
    messages=[{
        "role": "user",
        "content": "Plan a 2-week trip to Japan covering Tokyo, Kyoto, and Osaka"
    }],
    model="gpt-5",
    steps_pipeline=True,
    steps_model="gpt-5-nano"  # Use cheaper model for planning
)
print(f"Final Plan: {response[:300]}...")
print(f"\nPipeline Breakdown:")
if 'steps' in details:
    print(f"  Number of steps: {len(details['steps'])}")
    print("  Steps executed:")
    for i, step in enumerate(details['steps'][:3], 1):
        print(f"    {i}. {step['step'][:60]}...")
print(f"  Total tokens: {tokens}")
print(f"  Steps pipeline tokens: {details.get('steps_pipeline_total_tokens', 0)}")

# Example 2: Reasoning Pipeline (for non-reasoning models)
print("\n2. Reasoning Pipeline - Multi-Iteration Thinking:")
response, tokens, details = ultra.chat(
    messages=[{
        "role": "user",
        "content": "What are the ethical implications of self-driving cars?"
    }],
    model="gpt-4o",  # Non-reasoning model
    reasoning_pipeline=True,
    reasoning_iterations=5,  # Generate 5 iterations of thoughts
    reasoning_model="gpt-4o-mini"  # Use cheaper model
)
print(f"Response: {response[:300]}...")
print(f"\nReasoning Pipeline:")
if 'reasoning' in details:
    print(f"  Thoughts generated: {len(details['reasoning'])}")
    print("  Sample thoughts:")
    for i, thought in enumerate(details['reasoning'][:4], 1):
        print(f"    {i}. {thought[:70]}...")
print(f"  Total tokens: {tokens}")
print(f"  Reasoning pipeline tokens: {details.get('reasoning_pipeline_total_tokens', 0)}")

# Example 3: Combined Pipelines
print("\n3. Combined Steps + Reasoning:")
response, tokens, details = ultra.chat(
    messages=[{
        "role": "user",
        "content": "Develop a marketing strategy for a new AI product"
    }],
    model="gpt-4o",
    steps_pipeline=True,
    reasoning_pipeline=True,
    reasoning_iterations=3,
    steps_model="gpt-4o-mini",
    reasoning_model="gpt-4o-mini"
)
print(f"Final Strategy: {response[:250]}...")
print(f"\nCombined Pipeline Metrics:")
print(f"  Total tokens: {tokens}")
print(f"  Steps pipeline tokens: {details.get('steps_pipeline_total_tokens', 0)}")
print(f"  Reasoning pipeline tokens: {details.get('reasoning_pipeline_total_tokens', 0)}")

# Example 4: Native Reasoning vs Fake Pipeline
print("\n4. Native Reasoning (Claude) vs Fake Pipeline (GPT-4o):")

# Claude with native reasoning
response_claude, tokens_claude, details_claude = ultra.chat(
    messages=[{"role": "user", "content": "Explain quantum computing simply"}],
    model="claude:sonnet",
    reasoning_pipeline=True  # Uses native thinking
)
print(f"\nClaude (Native):")
print(f"  Response length: {len(response_claude)} chars")
print(f"  Reasoning tokens (API): {details_claude.get('reasoning_tokens_api', 0)}")
print(f"  Native reasoning: {'Yes' if details_claude.get('reasoning_tokens_api', 0) > 0 else 'No'}")

# GPT-4o with fake pipeline
response_gpt, tokens_gpt, details_gpt = ultra.chat(
    messages=[{"role": "user", "content": "Explain quantum computing simply"}],
    model="gpt-4o",
    reasoning_pipeline=True  # Uses fake pipeline
)
print(f"\nGPT-4o (Fake Pipeline):")
print(f"  Response length: {len(response_gpt)} chars")
print(f"  Thoughts generated: {len(details_gpt.get('reasoning', []))}")
print(f"  Fake pipeline used: {'Yes' if details_gpt.get('reasoning') else 'No'}")

print("\n✓ Pipelines provide structured thinking for complex tasks!")
print("\nTips:")
print("  - Use steps_pipeline for task planning and organization")
print("  - Use reasoning_pipeline for deep analysis (auto-detects native thinking)")
print("  - Combine both for maximum cognitive depth")
print("  - Use cheaper models for pipeline iterations to save costs")
