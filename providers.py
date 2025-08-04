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
        
    def chat_completion(self, messages: List[Dict], model: str, temperature: float, max_tokens: Optional[int] = 4096) -> tuple:
        """
        Standard chat completion
        Returns: (content: str, tokens: int)
        """
        raise NotImplementedError
        
    def chat_completion_with_schema(self, messages: List[Dict], schema: BaseModel, model: str, temperature: float, max_tokens: Optional[int] = 4096) -> tuple:
        """
        Chat completion with structured output
        Returns: (parsed_content: dict, tokens: int)
        """
        raise NotImplementedError
        
    def chat_completion_with_tools(self, messages: List[Dict], tools: List[Dict], model: str, temperature: float, max_tokens: Optional[int] = 4096, parallel_tool_calls: Optional[bool] = None) -> tuple:
        """
        Chat completion with native tool calling
        Returns: (response_message: dict, tokens: int)
        Response message contains either content or tool_calls
        Args:
            parallel_tool_calls: For OpenAI - whether to allow parallel tool calls (None = default behavior)
                                 For Claude - converted to disable_parallel_tool_use internally
        Note: AI will always be required to choose at least one tool from the provided tools
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
        
    def chat_completion(self, messages: List[Dict], model: str, temperature: float, max_tokens: Optional[int] = 4096) -> tuple:
        """Standard OpenAI chat completion"""
        kwargs = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": temperature
        }
        
        # Only add max_tokens if it's not None
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
            
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens
        return content, tokens
        
    def chat_completion_with_schema(self, messages: List[Dict], schema: BaseModel, model: str, temperature: float, max_tokens: Optional[int] = 4096) -> tuple:
        """OpenAI structured output with schema"""
        kwargs = {
            "model": model,
            "messages": messages,
            "response_format": schema,
            "temperature": temperature
        }
        
        # Only add max_tokens if it's not None
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
            
        response = self.client.beta.chat.completions.parse(**kwargs)
        content = response.choices[0].message.parsed
        if isinstance(content, BaseModel):
            content = content.model_dump(by_alias=True)
        tokens = response.usage.total_tokens
        return content, tokens
        
    def chat_completion_with_tools(self, messages: List[Dict], tools: List[Dict], model: str, temperature: float, max_tokens: Optional[int] = 4096, parallel_tool_calls: Optional[bool] = None) -> tuple:
        """OpenAI native tool calling - always requires at least one tool to be called"""
        kwargs = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "tool_choice": "required"  # Always require the AI to choose at least one tool
        }
        
        # Only add max_tokens if it's not None
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
            
        # Add parallel_tool_calls if specified (OpenAI specific)
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
            
        response = self.client.chat.completions.create(**kwargs)
        response_message = response.choices[0].message
        
        # Convert to dict format for consistency
        message_dict = {
            "role": "assistant",
            "content": response_message.content
        }
        
        # Add tool_calls if present
        if response_message.tool_calls:
            message_dict["tool_calls"] = []
            for tool_call in response_message.tool_calls:
                message_dict["tool_calls"].append({
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })
        
        tokens = response.usage.total_tokens
        return message_dict, tokens


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
        - Handle tool messages properly
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
                assistant_msg = {
                    "role": "assistant", 
                    "content": content or ""
                }
                # Handle tool calls in assistant messages
                if "tool_calls" in msg and msg["tool_calls"]:
                    assistant_msg["content"] = []
                    if content:
                        assistant_msg["content"].append({
                            "type": "text",
                            "text": content
                        })
                    
                    for tool_call in msg["tool_calls"]:
                        assistant_msg["content"].append({
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": json.loads(tool_call["function"]["arguments"])
                        })
                
                converted_messages.append(assistant_msg)
            elif role == "tool":
                # Convert tool response to Claude format
                tool_result_msg = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id"),
                            "content": content
                        }
                    ]
                }
                converted_messages.append(tool_result_msg)
            
        # Ensure we end with a user message (Claude requirement)
        if converted_messages and converted_messages[-1]["role"] != "user":
            # If last message is assistant, we're good for prefill scenario
            # Otherwise, we might need to add a dummy user message
            pass
            
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return converted_messages, system_prompt
        
    def chat_completion(self, messages: List[Dict], model: str, temperature: float, max_tokens: Optional[int] = 4096) -> tuple:
        """Claude chat completion"""
        converted_messages, system_prompt = self.convert_messages(messages)
        
        kwargs = {
            "model": model,
            "messages": converted_messages,
            "temperature": temperature
        }
        
        # Only add max_tokens if it's not None
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        
        if system_prompt:
            kwargs["system"] = system_prompt
            
        response = self.client.messages.create(**kwargs)
        content = response.content[0].text.strip()
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return content, tokens
        
    def chat_completion_with_schema(self, messages: List[Dict], schema: BaseModel, model: str, temperature: float, max_tokens: Optional[int] = 4096) -> tuple:
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
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": tool_name}  # Force using the tool
        }
        
        # Only add max_tokens if it's not None
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        
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
        
    def chat_completion_with_tools(self, messages: List[Dict], tools: List[Dict], model: str, temperature: float, max_tokens: Optional[int] = 4096, parallel_tool_calls: Optional[bool] = None) -> tuple:
        """Claude native tool calling - always requires at least one tool to be called"""
        converted_messages, system_prompt = self.convert_messages(messages)
        
        # Convert OpenAI-format tools to Claude format
        claude_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                # Convert from OpenAI format to Claude format
                function_def = tool["function"]
                claude_tool = {
                    "name": function_def["name"],
                    "description": function_def["description"],
                    "input_schema": function_def["parameters"]
                }
                claude_tools.append(claude_tool)
            else:
                # Already in Claude format or unknown format
                claude_tools.append(tool)
        
        kwargs = {
            "model": model,
            "messages": converted_messages,
            "temperature": temperature,
            "tools": claude_tools,  # Use converted Claude tools
            "tool_choice": {"type": "any"}  # Always require the AI to choose at least one tool
        }
        
        # Only add max_tokens if it's not None
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        
        # Handle parallel tool calls (Claude uses disable_parallel_tool_use)
        if parallel_tool_calls is not None:
            kwargs["tool_choice"]["disable_parallel_tool_use"] = not parallel_tool_calls
        
        if system_prompt:
            kwargs["system"] = system_prompt
            
        response = self.client.messages.create(**kwargs)
        
        # Convert Claude response to standardized format
        message_dict = {
            "role": "assistant",
            "content": None
        }
        
        # Extract text content and tool uses
        text_content = ""
        tool_uses = []
        
        for block in response.content:
            if hasattr(block, 'type'):
                if block.type == "text":
                    text_content += block.text
                elif block.type == "tool_use":
                    tool_uses.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input)
                        }
                    })
        
        # Set content (Claude can return text even with tool calls)
        if text_content.strip():
            message_dict["content"] = text_content.strip()
        
        # Add tool_calls if present
        if tool_uses:
            message_dict["tool_calls"] = tool_uses
        
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return message_dict, tokens


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
            
    def chat_completion(self, model: str, messages: List[Dict], temperature: float = 0.7, max_tokens: Optional[int] = 4096) -> tuple:
        """Route chat completion to appropriate provider"""
        provider_name, model_name = self.parse_model_string(model)
        provider = self.get_provider(provider_name)
        return provider.chat_completion(messages, model_name, temperature, max_tokens)
        
    def chat_completion_with_schema(self, model: str, messages: List[Dict], schema: BaseModel, temperature: float = 0.7, max_tokens: Optional[int] = 4096) -> tuple:
        """Route structured chat completion to appropriate provider"""
        provider_name, model_name = self.parse_model_string(model)
        provider = self.get_provider(provider_name)
        return provider.chat_completion_with_schema(messages, schema, model_name, temperature, max_tokens)
        
    def chat_completion_with_tools(self, model: str, messages: List[Dict], tools: List[Dict], temperature: float = 0.7, max_tokens: Optional[int] = 4096, parallel_tool_calls: Optional[bool] = None) -> tuple:
        """Route tool calling to appropriate provider - always requires at least one tool to be called"""
        provider_name, model_name = self.parse_model_string(model)
        provider = self.get_provider(provider_name)
        return provider.chat_completion_with_tools(messages, tools, model_name, temperature, max_tokens, parallel_tool_calls)
