"""
Provider abstraction layer for different AI providers (OpenAI, Claude, etc.)
"""
from openai import OpenAI
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import json


class BaseProvider:
    """Base class for AI providers"""
    
    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        
    def chat_completion(self, messages: List[Dict], model: str, temperature: float, max_tokens: int = 4096) -> tuple:
        """
        Standard chat completion
        Returns: (content: str, tokens: int)
        """
        raise NotImplementedError
        
    def chat_completion_with_schema(self, messages: List[Dict], schema: BaseModel, model: str, temperature: float, max_tokens: int = 4096) -> tuple:
        """
        Chat completion with structured output
        Returns: (parsed_content: dict, tokens: int)
        """
        raise NotImplementedError
        
    def convert_messages(self, messages: List[Dict]) -> tuple:
        """
        Convert OpenAI format messages to provider-specific format
        Returns: (converted_messages, system_prompt)
        """
        return messages, None


class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation"""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.client = OpenAI(api_key=api_key)
        
    def chat_completion(self, messages: List[Dict], model: str, temperature: float, max_tokens: int = 4096) -> tuple:
        """Standard OpenAI chat completion"""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens
        )
        content = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens
        return content, tokens
        
    def chat_completion_with_schema(self, messages: List[Dict], schema: BaseModel, model: str, temperature: float, max_tokens: int = 4096) -> tuple:
        """OpenAI structured output with schema"""
        response = self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=schema,
            temperature=temperature,
            max_tokens=max_tokens
        )
        content = response.choices[0].message.parsed
        if isinstance(content, BaseModel):
            content = content.model_dump(by_alias=True)
        tokens = response.usage.total_tokens
        return content, tokens


class ClaudeProvider(BaseProvider):
    """Anthropic Claude provider implementation"""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        if Anthropic is None:
            raise ImportError("anthropic package is required for Claude support. Install with: pip install anthropic")
        self.client = Anthropic(api_key=api_key)
        
    def convert_messages(self, messages: List[Dict]) -> tuple:
        """
        Convert OpenAI format to Claude format
        - Extract system messages to separate system prompt
        - Ensure alternating user/assistant pattern
        - Convert content format if needed
        """
        system_parts = []
        converted_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role in ["system", "developer"]:
                system_parts.append(content)
            elif role == "user":
                converted_messages.append({
                    "role": "user",
                    "content": content
                })
            elif role == "assistant":
                converted_messages.append({
                    "role": "assistant", 
                    "content": content
                })
            # Skip tool/function messages for now - would need special handling
            
        # Ensure we end with a user message (Claude requirement)
        if converted_messages and converted_messages[-1]["role"] != "user":
            # If last message is assistant, we're good for prefill scenario
            # Otherwise, we might need to add a dummy user message
            pass
            
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return converted_messages, system_prompt
        
    def chat_completion(self, messages: List[Dict], model: str, temperature: float, max_tokens: int = 4096) -> tuple:
        """Claude chat completion"""
        converted_messages, system_prompt = self.convert_messages(messages)
        
        kwargs = {
            "model": model,
            "messages": converted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
            
        response = self.client.messages.create(**kwargs)
        content = response.content[0].text.strip()
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return content, tokens
        
    def chat_completion_with_schema(self, messages: List[Dict], schema: BaseModel, model: str, temperature: float, max_tokens: int = 4096) -> tuple:
        """
        Claude structured output with schema using tool-based approach
        
        Note: Claude doesn't support OpenAI's response_format parameter.
        Instead, we use Claude's tool calling feature to enforce structured output:
        1. Convert the Pydantic schema to a tool definition
        2. Force Claude to use that tool with tool_choice
        3. Extract the validated data from the tool call
        """
        converted_messages, system_prompt = self.convert_messages(messages)
        
        # Create a tool definition from the Pydantic schema
        tool_name = f"return_{schema.__name__.lower()}"
        tool = {
            "name": tool_name,
            "description": f"Return the response as structured data matching the {schema.__name__} schema",
            "input_schema": schema.model_json_schema()
        }
        
        # Add instruction to use the tool in system prompt
        tool_instruction = f"You must use the {tool_name} tool to provide your response in the required format."
        if system_prompt:
            system_prompt = f"{system_prompt}\n\n{tool_instruction}"
        else:
            system_prompt = tool_instruction
        
        kwargs = {
            "model": model,
            "messages": converted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": tool_name}  # Force using the tool
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
            
        response = self.client.messages.create(**kwargs)
        
        # Extract the tool call result
        tool_use_block = None
        for block in response.content:
            if hasattr(block, 'type') and block.type == "tool_use":
                tool_use_block = block
                break
        
        if not tool_use_block:
            # Fallback: try to parse as JSON from text content
            text_content = response.content[0].text.strip()
            try:
                content = schema.model_validate_json(text_content)
                if isinstance(content, BaseModel):
                    content = content.model_dump(by_alias=True)
            except Exception:
                # If all else fails, return empty structure
                content = schema().model_dump(by_alias=True)
        else:
            # Use the validated tool input
            content = tool_use_block.input
            
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return content, tokens


class ProviderManager:
    """Manages different AI providers"""
    
    def __init__(self):
        self.providers = {}
        
    def add_provider(self, name: str, provider: BaseProvider):
        """Add a provider instance"""
        self.providers[name] = provider
        
    def get_provider(self, name: str) -> BaseProvider:
        """Get provider by name"""
        if name not in self.providers:
            raise ValueError(f"Provider '{name}' not found. Available providers: {list(self.providers.keys())}")
        return self.providers[name]
        
    def parse_model_string(self, model: str) -> tuple:
        """
        Parse model string to extract provider and model name
        Format: "provider:model" or just "model" (defaults to openai)
        Returns: (provider_name, model_name)
        """
        if ":" in model:
            provider_name, model_name = model.split(":", 1)
            return provider_name, model_name
        else:
            # Default to openai if no provider specified
            return "openai", model
            
    def chat_completion(self, model: str, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 4096) -> tuple:
        """Route chat completion to appropriate provider"""
        provider_name, model_name = self.parse_model_string(model)
        provider = self.get_provider(provider_name)
        return provider.chat_completion(messages, model_name, temperature, max_tokens)
        
    def chat_completion_with_schema(self, model: str, messages: List[Dict], schema: BaseModel, temperature: float = 0.7, max_tokens: int = 4096) -> tuple:
        """Route structured chat completion to appropriate provider"""
        provider_name, model_name = self.parse_model_string(model)
        provider = self.get_provider(provider_name)
        return provider.chat_completion_with_schema(messages, schema, model_name, temperature, max_tokens)
