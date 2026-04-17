#!/usr/bin/env python3
"""Example 3: Tool Calling.

Demonstrates robust handling of all `tool_call()` response shapes:
- tool calls only,
- tool calls + assistant text,
- text-only fallback.
"""

import json
import os
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from ultragpt import UltraGPT


class CalculatorParams(BaseModel):
    operation: str = Field(description="Operation: add, subtract, multiply, divide")
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


class WeatherParams(BaseModel):
    location: str = Field(description="City name or location")
    units: str = Field(description="Units: celsius or fahrenheit", default="celsius")


calculator_tool = {
    "name": "calculator",
    "description": "Performs basic arithmetic operations on two numbers",
    "parameters_schema": CalculatorParams,
    "usage_guide": "Use for precise arithmetic calculations",
    "when_to_use": "When user needs numeric computation results",
}

weather_tool = {
    "name": "get_weather",
    "description": "Gets current weather information for a location",
    "parameters_schema": WeatherParams,
    "usage_guide": "Use to fetch weather data for any city or location",
    "when_to_use": "When user asks about weather, temperature, or climate",
}


def require_openrouter_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key
    raise RuntimeError("Missing OPENROUTER_API_KEY in environment or .env")


def normalize_tool_response(payload: Any) -> Tuple[List[Dict[str, Any]], str]:
    if isinstance(payload, list):
        return payload, ""
    if isinstance(payload, dict):
        tool_calls = payload.get("tool_calls", [])
        content = payload.get("content", "") or ""
        if isinstance(tool_calls, list):
            return tool_calls, content
        return [], content
    return [], str(payload)


def print_tool_calls(payload: Any) -> None:
    tool_calls, content = normalize_tool_response(payload)
    if content.strip():
        print(f"Assistant text: {content}")

    if not tool_calls:
        print("No tool calls returned.")
        return

    print(f"Tool calls returned: {len(tool_calls)}")
    for i, call in enumerate(tool_calls, start=1):
        fn = call.get("function", {})
        fn_name = fn.get("name", "<unknown>")
        raw_args = fn.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:  # noqa: BLE001
            args = raw_args
        print(f"  {i}. {fn_name} -> {args}")


def main() -> None:
    load_dotenv()

    ultra = UltraGPT(
        openrouter_api_key=require_openrouter_key(),
        fallback_models=["openai/gpt-4.1"],
        verbose=True,
    )

    print("=" * 70)
    print("TOOL CALLING EXAMPLE")
    print("=" * 70)

    print("\n1. Single Tool Call (Calculator)")
    response, tokens, details = ultra.tool_call(
        messages=[{"role": "user", "content": "Calculate 25 * 8"}],
        user_tools=[calculator_tool],
        model="gpt-5",
        allow_multiple=False,
    )
    print_tool_calls(response)
    print(f"Tokens: {tokens}")
    print(f"Selected model: {details.get('selected_model')}")

    print("\n2. Multiple Tool Calls")
    response, tokens, details = ultra.tool_call(
        messages=[{"role": "user", "content": "Add 10 and 5, then multiply 3 by 7"}],
        user_tools=[calculator_tool],
        model="gpt-5",
        allow_multiple=True,
    )
    print_tool_calls(response)
    print(f"Tokens: {tokens}")

    print("\n3. Tool Selection (Weather vs Calculator)")
    response, tokens, details = ultra.tool_call(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        user_tools=[calculator_tool, weather_tool],
        model="gpt-5",
    )
    print_tool_calls(response)
    print(f"Tokens: {tokens}")

    print("\n4. Tool Calling + Native Reasoning")
    response, tokens, details = ultra.tool_call(
        messages=[{"role": "user", "content": "Use calculator to find 15% of 240"}],
        user_tools=[calculator_tool],
        model="claude:sonnet",
        reasoning_pipeline=True,
    )
    print_tool_calls(response)
    print(f"Reasoning tokens (API): {details.get('reasoning_tokens_api', 0)}")
    print(f"Has reasoning_details: {'reasoning_details' in details}")

    print("\n✓ Tool calling works uniformly across all providers and response shapes.")


if __name__ == "__main__":
    main()
