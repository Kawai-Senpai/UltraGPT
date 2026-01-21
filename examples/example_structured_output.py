#!/usr/bin/env python3
"""
Example 2: Structured Output with Pydantic
Demonstrates schema-based responses across providers
"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from ultragpt import UltraGPT

# Load API key
load_dotenv()

# Define Pydantic schemas
class SentimentAnalysis(BaseModel):
    """Sentiment analysis result"""
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="Confidence score 0.0 to 1.0")
    keywords: List[str] = Field(description="Key emotional words from the text")
    explanation: str = Field(description="Brief explanation of the sentiment")

class MathSolution(BaseModel):
    """Step-by-step math solution"""
    problem: str = Field(description="The original problem")
    steps: List[str] = Field(description="Solution steps in order")
    final_answer: str = Field(description="The final answer")
    verification: str = Field(description="How to verify the answer is correct")

# Initialize
ultra = UltraGPT(
    openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
    verbose=True
)

print("=" * 70)
print("STRUCTURED OUTPUT EXAMPLE")
print("=" * 70)

# Example 1: Sentiment Analysis with GPT-5
print("\n1. Sentiment Analysis (GPT-5):")
response, tokens, details = ultra.chat(
    messages=[{
        "role": "user",
        "content": "Analyze: 'I absolutely love this product! Best purchase ever!'"
    }],
    model="gpt-5",
    schema=SentimentAnalysis
)
print(f"Sentiment: {response['sentiment']}")
print(f"Confidence: {response['confidence']}")
print(f"Keywords: {', '.join(response['keywords'])}")
print(f"Explanation: {response['explanation']}")
print(f"Tokens: {tokens}")

# Example 2: Same with Claude (uses tool-based fallback automatically)
print("\n2. Sentiment Analysis (Claude - auto fallback):")
response, tokens, details = ultra.chat(
    messages=[{
        "role": "user",
        "content": "Analyze: 'This is terrible! I hate it!'"
    }],
    model="claude:sonnet",
    schema=SentimentAnalysis
)
print(f"Sentiment: {response['sentiment']}")
print(f"Confidence: {response['confidence']}")
print(f"Keywords: {', '.join(response['keywords'])}")
print(f"Tokens: {tokens}")

# Example 3: Math Solution with Steps
print("\n3. Math Solution (GPT-5):")
response, tokens, details = ultra.chat(
    messages=[{
        "role": "user",
        "content": "Solve: If 3x + 7 = 22, find x. Show all steps."
    }],
    model="gpt-5",
    schema=MathSolution
)
print(f"Problem: {response['problem']}")
print("Steps:")
for i, step in enumerate(response['steps'], 1):
    print(f"  {i}. {step}")
print(f"Answer: {response['final_answer']}")
print(f"Verification: {response['verification']}")
print(f"Tokens: {tokens}")

print("\n✓ Structured output works seamlessly across all providers!")
