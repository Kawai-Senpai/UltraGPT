#!/usr/bin/env python3
"""Example 5: Pipelines (Steps + Reasoning).

Shows practical pipeline usage and token diagnostics for planning-heavy tasks.
"""

import os

from dotenv import load_dotenv

from ultragpt import UltraGPT


def require_openrouter_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key
    raise RuntimeError("Missing OPENROUTER_API_KEY in environment or .env")


def print_pipeline_metrics(tokens: int, details: dict) -> None:
    print(f"  Total tokens: {tokens}")
    print(f"  Steps pipeline tokens: {details.get('steps_pipeline_total_tokens', 0)}")
    print(f"  Reasoning pipeline tokens: {details.get('reasoning_pipeline_total_tokens', 0)}")
    print(f"  Final model selected: {details.get('selected_model')}")


def main() -> None:
    load_dotenv()

    ultra = UltraGPT(
        openrouter_api_key=require_openrouter_key(),
        fallback_models=["openai/gpt-4.1"],
        verbose=True,
    )

    print("=" * 70)
    print("PIPELINES EXAMPLE")
    print("=" * 70)

    print("\n1. Steps Pipeline - Task Planning")
    response, tokens, details = ultra.chat(
        messages=[{"role": "user", "content": "Plan a 2-week trip to Japan (Tokyo, Kyoto, Osaka)."}],
        model="gpt-5",
        steps_pipeline=True,
        steps_model="gpt-5.4-nano",
    )
    print(f"Final plan excerpt: {response[:300]}...")
    steps = details.get("steps", [])
    print(f"Steps generated: {len(steps)}")
    for i, step in enumerate(steps[:3], start=1):
        if isinstance(step, dict):
            print(f"  {i}. {step.get('step', '')[:80]}...")
    print_pipeline_metrics(tokens, details)

    print("\n2. Reasoning Pipeline - Multi-Iteration Thinking")
    response, tokens, details = ultra.chat(
        messages=[{"role": "user", "content": "What are ethical implications of self-driving cars?"}],
        model="gpt-4o",
        reasoning_pipeline=True,
        reasoning_iterations=5,
        reasoning_model="gpt-4o-mini",
    )
    print(f"Response excerpt: {response[:300]}...")
    thoughts = details.get("reasoning", [])
    print(f"Thoughts generated: {len(thoughts)}")
    for i, thought in enumerate(thoughts[:4], start=1):
        print(f"  {i}. {thought[:90]}...")
    print_pipeline_metrics(tokens, details)

    print("\n3. Combined Steps + Reasoning")
    response, tokens, details = ultra.chat(
        messages=[{"role": "user", "content": "Develop a marketing strategy for a new AI product."}],
        model="gpt-4o",
        steps_pipeline=True,
        reasoning_pipeline=True,
        reasoning_iterations=3,
        steps_model="gpt-4o-mini",
        reasoning_model="gpt-4o-mini",
    )
    print(f"Final strategy excerpt: {response[:260]}...")
    print_pipeline_metrics(tokens, details)

    print("\n4. Disable fallback for deterministic benchmarking")
    response, tokens, details = ultra.chat(
        messages=[{"role": "user", "content": "Explain quantum computing simply."}],
        model="claude:sonnet",
        reasoning_pipeline=True,
        fallback_models=[],
    )
    print(f"Response length: {len(response)} chars")
    print(f"Fallback used: {details.get('fallback_used')}")
    print(f"Reasoning tokens (API): {details.get('reasoning_tokens_api', 0)}")

    print("\n✓ Pipelines provide structured cognition with stable operational controls.")


if __name__ == "__main__":
    main()
