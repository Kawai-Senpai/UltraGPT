#!/usr/bin/env python3
"""Example 2: Structured Output with Pydantic.

This script shows robust schema-based calls with consistent printing,
fallback metadata visibility, and explicit error handling.
"""

import os
from typing import List, Type

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from ultragpt import UltraGPT


class SentimentAnalysis(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="Confidence score 0.0 to 1.0")
    keywords: List[str] = Field(description="Key emotional words from the text")
    explanation: str = Field(description="Brief explanation of the sentiment")


class MathSolution(BaseModel):
    problem: str = Field(description="The original problem")
    steps: List[str] = Field(description="Solution steps in order")
    final_answer: str = Field(description="The final answer")
    verification: str = Field(description="How to verify the answer is correct")


def require_openrouter_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key
    raise RuntimeError("Missing OPENROUTER_API_KEY in environment or .env")


def run_structured(
    ultra: UltraGPT,
    *,
    title: str,
    model: str,
    prompt: str,
    schema: Type[BaseModel],
) -> None:
    print(f"\n{title}")
    response, tokens, details = ultra.chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        schema=schema,
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
    print("STRUCTURED OUTPUT EXAMPLE")
    print("=" * 70)

    run_structured(
        ultra,
        title="1. Sentiment Analysis (GPT-5)",
        model="gpt-5",
        prompt="Analyze: 'I absolutely love this product! Best purchase ever!'",
        schema=SentimentAnalysis,
    )

    run_structured(
        ultra,
        title="2. Sentiment Analysis (Claude)",
        model="claude:sonnet",
        prompt="Analyze: 'This is terrible! I hate it!'",
        schema=SentimentAnalysis,
    )

    print("\n3. Math Solution (GPT-5)")
    response, tokens, details = ultra.chat(
        messages=[{"role": "user", "content": "Solve: If 3x + 7 = 22, find x. Show all steps."}],
        model="gpt-5",
        schema=MathSolution,
    )
    print(f"Problem: {response['problem']}")
    print("Steps:")
    for i, step in enumerate(response["steps"], start=1):
        print(f"  {i}. {step}")
    print(f"Answer: {response['final_answer']}")
    print(f"Verification: {response['verification']}")
    print(f"Tokens: {tokens}")

    print("\n4. Explicit Error Handling Example")
    try:
        ultra.chat(
            messages=[{"role": "user", "content": "Return malformed output"}],
            model="claude:sonnet",
            schema=SentimentAnalysis,
            fallback_models=[],
        )
        print("Call succeeded (model returned valid schema).")
    except Exception as exc:  # noqa: BLE001
        print(f"Structured parse failed cleanly: {type(exc).__name__}: {exc}")

    print("\n✓ Structured output works consistently with explicit schema safety.")


if __name__ == "__main__":
    main()
