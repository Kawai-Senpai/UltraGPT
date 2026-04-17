#!/usr/bin/env python3
"""Example 1: Basic Chat with OpenRouter.

This script demonstrates:
1) one client calling multiple model families,
2) optional fallback chains,
3) model-selection metadata in the details payload.
"""

import os

from dotenv import load_dotenv

from ultragpt import UltraGPT


def require_openrouter_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key
    raise RuntimeError(
        "Missing OPENROUTER_API_KEY. Add it to your environment or .env file before running this example."
    )


def run_prompt(ultra: UltraGPT, *, title: str, model: str, prompt: str) -> None:
    print(f"\n{title}")
    response, tokens, details = ultra.chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
    )
    print(f"Model: {model}")
    print(f"Response: {response}")
    print(f"Tokens: {tokens}")
    print(f"Selected model: {details.get('selected_model')}")
    print(f"Fallback used: {details.get('fallback_used')}")


def main() -> None:
    load_dotenv()

    ultra = UltraGPT(
        openrouter_api_key=require_openrouter_key(),
        fallback_models=["openai/gpt-4.1"],
        verbose=True,
    )

    print("=" * 70)
    print("BASIC CHAT EXAMPLE")
    print("=" * 70)

    question = "What is the capital of France? Be brief."
    run_prompt(ultra, title="1. GPT-5 Chat", model="gpt-5", prompt=question)
    run_prompt(ultra, title="2. Claude Sonnet Chat", model="claude:sonnet", prompt=question)
    run_prompt(ultra, title="3. Gemini Chat", model="gemini", prompt=question)

    print("\n4. Multi-Turn Conversation")
    messages = [
        {"role": "user", "content": "What is 15 * 24?"},
        {"role": "assistant", "content": "15 * 24 = 360"},
        {"role": "user", "content": "Now divide that by 5."},
    ]
    response, tokens, details = ultra.chat(
        messages=messages,
        model="gpt-5",
    )
    print(f"Response: {response}")
    print(f"Total tokens: {tokens}")
    print(f"Attempted models: {details.get('attempted_models')}")

    print("\n5. Disable fallback for one call")
    response, tokens, details = ultra.chat(
        messages=[{"role": "user", "content": "Say hello in exactly two words."}],
        model="gpt-5",
        fallback_models=[],
    )
    print(f"Response: {response}")
    print(f"Fallback used: {details.get('fallback_used')}")

    print("\n✓ Example complete! Same API, multiple model families, optional fallback controls.")


if __name__ == "__main__":
    main()
