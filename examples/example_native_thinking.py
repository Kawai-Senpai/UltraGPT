#!/usr/bin/env python3
"""Example 4: Native Thinking / Reasoning.

Compares native reasoning models with simulated reasoning pipeline behavior.
"""

import os

from dotenv import load_dotenv

from ultragpt import UltraGPT


def require_openrouter_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key
    raise RuntimeError("Missing OPENROUTER_API_KEY in environment or .env")


def print_reasoning_metrics(tokens: int, details: dict) -> None:
    print(f"  Input tokens: {details.get('input_tokens', 0)}")
    print(f"  Output tokens: {details.get('output_tokens', 0)}")
    print(f"  Reasoning tokens (API): {details.get('reasoning_tokens_api', 0)}")
    print(f"  Total tokens: {tokens}")
    if details.get("selected_model"):
        print(f"  Selected model: {details.get('selected_model')}")


def main() -> None:
    load_dotenv()
    ultra = UltraGPT(
        openrouter_api_key=require_openrouter_key(),
        fallback_models=["openai/gpt-4.1"],
        verbose=True,
    )

    print("=" * 70)
    print("NATIVE THINKING / REASONING EXAMPLE")
    print("=" * 70)

    print("\n1. Claude Sonnet with Native Reasoning")
    response, tokens, details = ultra.chat(
        messages=[{"role": "user", "content": "Solve step by step: If 3x + 7 = 22, find x."}],
        model="claude:sonnet",
        reasoning_pipeline=True,
    )
    print(f"Response: {response}")
    print_reasoning_metrics(tokens, details)

    print("\n2. GPT-4o with Simulated Reasoning Pipeline")
    response, tokens, details = ultra.chat(
        messages=[{"role": "user", "content": "What are long-term AI employment implications?"}],
        model="gpt-4o",
        reasoning_pipeline=True,
        reasoning_iterations=3,
    )
    print(f"Response: {response[:220]}...")
    thoughts = details.get("reasoning", [])
    print(f"Thoughts generated: {len(thoughts)}")
    for i, thought in enumerate(thoughts[:3], start=1):
        print(f"  {i}. {thought[:80]}...")
    print_reasoning_metrics(tokens, details)

    print("\n3. Comparison - Same Prompt with/without reasoning_pipeline")
    prompt = {"role": "user", "content": "What is 15% of 240?"}

    basic_response, basic_tokens, basic_details = ultra.chat(
        messages=[prompt],
        model="claude:sonnet",
        reasoning_pipeline=False,
    )
    thinking_response, thinking_tokens, thinking_details = ultra.chat(
        messages=[prompt],
        model="claude:sonnet",
        reasoning_pipeline=True,
    )

    print("Without reasoning_pipeline:")
    print(f"  Response: {basic_response}")
    print(f"  Tokens: {basic_tokens}")
    print(f"  Reasoning tokens: {basic_details.get('reasoning_tokens_api', 0)}")

    print("With reasoning_pipeline:")
    print(f"  Response: {thinking_response}")
    print(f"  Tokens: {thinking_tokens}")
    print(f"  Reasoning tokens: {thinking_details.get('reasoning_tokens_api', 0)}")

    print("\n4. Gemini with Native Reasoning")
    response, tokens, details = ultra.chat(
        messages=[{"role": "user", "content": "Explain why the sky is blue in simple terms."}],
        model="gemini",
        reasoning_pipeline=True,
    )
    print(f"Response: {response[:220]}...")
    print_reasoning_metrics(tokens, details)

    print("\n✓ Native thinking and simulated reasoning now share one consistent API surface.")


if __name__ == "__main__":
    main()
