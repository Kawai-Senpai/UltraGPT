#!/usr/bin/env python3
"""
Example 3: Tool Calling
Demonstrates native tool calling with custom tools
"""

import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from ultragpt import UltraGPT

# Load API key
load_dotenv()

# Define tool parameter schemas
class CalculatorParams(BaseModel):
    """Calculator parameters"""
    operation: str = Field(description="Operation: add, subtract, multiply, divide")
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

class WeatherParams(BaseModel):
    """Weather lookup parameters"""
    location: str = Field(description="City name or location")
    units: str = Field(description="Units: celsius or fahrenheit", default="celsius")

# Define tools
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

# Initialize
ultra = UltraGPT(
    openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
    verbose=True
)

print("=" * 70)
print("TOOL CALLING EXAMPLE")
print("=" * 70)

# Example 1: Single tool call
print("\n1. Single Tool Call (Calculator):")
response, tokens, details = ultra.tool_call(
    messages=[{"role": "user", "content": "Calculate 25 * 8"}],
    user_tools=[calculator_tool],
    model="gpt-5",
    allow_multiple=False  # Only one tool call
)
print(f"Tool called: {response['function']['name']}")
args = json.loads(response['function']['arguments'])
print(f"Arguments: {json.dumps(args, indent=2)}")
print(f"Tokens: {tokens}")

# Example 2: Multiple tool calls
print("\n2. Multiple Tool Calls:")
response, tokens, details = ultra.tool_call(
    messages=[{"role": "user", "content": "Add 10 and 5, then multiply 3 by 7"}],
    user_tools=[calculator_tool],
    model="gpt-5",
    allow_multiple=True  # Returns array of tool calls
)
print(f"Number of tool calls: {len(response)}")
for i, call in enumerate(response, 1):
    print(f"\nCall {i}:")
    print(f"  Tool: {call['function']['name']}")
    args = json.loads(call['function']['arguments'])
    print(f"  Arguments: {json.dumps(args, indent=2)}")

# Example 3: Tool selection
print("\n3. Tool Selection (Weather vs Calculator):")
response, tokens, details = ultra.tool_call(
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    user_tools=[calculator_tool, weather_tool],
    model="gpt-5"
)
tool_name = response[0]['function']['name'] if isinstance(response, list) else response['function']['name']
print(f"Model correctly selected: {tool_name}")
print(f"Tokens: {tokens}")

# Example 4: Tool calling with Claude
print("\n4. Tool Calling with Claude:")
response, tokens, details = ultra.tool_call(
    messages=[{"role": "user", "content": "Divide 100 by 4"}],
    user_tools=[calculator_tool],
    model="claude:sonnet"
)
print(f"Tool: {response['function']['name']}")
args = json.loads(response['function']['arguments'])
print(f"Arguments: {json.dumps(args, indent=2)}")
print(f"Tokens: {tokens}")

# Example 5: Tool calling with native reasoning
print("\n5. Tool Calling + Native Reasoning:")
response, tokens, details = ultra.tool_call(
    messages=[{"role": "user", "content": "Use calculator to find 15% of 240"}],
    user_tools=[calculator_tool],
    model="claude:sonnet",
    reasoning_pipeline=True  # Auto-detects native thinking
)
print(f"Tool: {response['function']['name']}")
print(f"Reasoning tokens (API): {details.get('reasoning_tokens_api', 0)}")
print(f"Has reasoning_details: {'reasoning_details' in details}")

print("\n✓ Tool calling works uniformly across all providers!")
