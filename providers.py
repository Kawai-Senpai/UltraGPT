"""
Provider abstraction layer for different AI providers (OpenAI, Claude, etc.)
"""
import json
import time
import random
import logging
import importlib
import threading
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from openai import OpenAI
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union, Tuple
from . import config

logger = logging.getLogger(__name__)

_openai_modules_warmed = False
_openai_warm_lock = threading.Lock()

def _warm_openai_modules() -> None:
    """Preload OpenAI resource modules to avoid ModuleLock deadlocks under threading."""
    global _openai_modules_warmed
    if _openai_modules_warmed:
        return

    with _openai_warm_lock:
        if _openai_modules_warmed:
            return

        module_names = [
            "openai.resources.chat",
            "openai.resources.responses",
            "openai.resources.responses.responses",
        ]

        for module_name in module_names:
            try:
                importlib.import_module(module_name)
            except Exception as warm_exc:
                logger.debug("Optional OpenAI warmup import failed for %s: %s", module_name, warm_exc)

        _openai_modules_warmed = True

def _parse_retry_after(retry_after_value: Optional[str]) -> Optional[float]:
    """Convert Retry-After header value to seconds."""
    if not retry_after_value:
        return None

    try:
        return float(retry_after_value)
    except (TypeError, ValueError):
        pass

    try:
        retry_datetime = parsedate_to_datetime(retry_after_value)
    except (TypeError, ValueError):
        return None

    if retry_datetime is None:
        return None

    if retry_datetime.tzinfo is None:
        retry_datetime = retry_datetime.replace(tzinfo=timezone.utc)

    delay_seconds = (retry_datetime - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, delay_seconds)

def is_rate_limit_error(error) -> bool:
    """Check if an error is a rate limit error for any provider"""
    error_str = str(error).lower()
    error_code = getattr(error, 'status_code', None) or getattr(error, 'code', None)
    
    # Check for HTTP 429 status code
    if error_code == 429:
        return True
        
    # Check for rate limit keywords in error message
    rate_limit_keywords = [
        'rate limit', 'rate_limit', 'too many requests', 'quota exceeded',
        'request limit', 'usage limit', 'throttle', 'rate-limit'
    ]
    
    return any(keyword in error_str for keyword in rate_limit_keywords)

def retry_on_rate_limit(func):
    """Decorator to retry API calls on rate limit errors with exponential backoff"""
    def wrapper(*args, **kwargs):
        max_retries = config.RATE_LIMIT_RETRIES
        base_delay = config.RATE_LIMIT_BASE_DELAY
        max_delay = config.RATE_LIMIT_MAX_DELAY
        multiplier = config.RATE_LIMIT_BACKOFF_MULTIPLIER
        
        for attempt in range(max_retries + 1):  # +1 for the initial attempt
            try:
                return func(*args, **kwargs)
            except Exception as e:
                attempt_number = attempt + 1
                total_attempts = max_retries + 1
                logger.error(
                    "Provider call %s.%s failed on attempt %d/%d: %s",
                    func.__module__,
                    func.__name__,
                    attempt_number,
                    total_attempts,
                    e,
                    exc_info=True
                )

                if not is_rate_limit_error(e) or attempt == max_retries:
                    # Not a rate limit error or we've exhausted retries.
                    raise

                response = getattr(e, "response", None)
                headers = getattr(response, "headers", {}) or {}
                retry_after_header = None
                for header_key, header_value in headers.items():
                    if header_key.lower() == "retry-after":
                        retry_after_header = header_value
                        break

                header_delay = _parse_retry_after(retry_after_header)

                if header_delay is not None:
                    total_delay = header_delay
                    delay_source = "retry-after header"
                else:
                    delay = min(base_delay * (multiplier ** attempt), max_delay)
                    jitter = random.uniform(0.1, 0.3) * delay  # Add 10-30% jitter
                    total_delay = delay + jitter
                    delay_source = "exponential backoff"

                logger.info(
                    "Rate limit hit for %s.%s, retrying in %.2f seconds via %s (attempt %d/%d)",
                    func.__module__,
                    func.__name__,
                    total_delay,
                    delay_source,
                    attempt_number,
                    total_attempts
                )
                time.sleep(total_delay)
        
        # This shouldn't be reached, but just in case
        raise Exception("Maximum retries exceeded for rate limit")
    
    return wrapper

class BaseProvider:
    """Base class for AI providers"""
    
    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        
    def chat_completion(self, messages: List[Dict], model: str, temperature: float, max_tokens: Optional[int] = None, deepthink: Optional[bool] = None) -> tuple:
        """
        Standard chat completion
        Returns: (content: str, tokens: int)
        Args:
            deepthink: Enable deep thinking mode for supported models (future functionality)
        """
        raise NotImplementedError
        
    def chat_completion_with_schema(self, messages: List[Dict], schema: BaseModel, model: str, temperature: float, max_tokens: Optional[int] = None, deepthink: Optional[bool] = None) -> tuple:
        """
        Chat completion with structured output
        Returns: (parsed_content: dict, tokens: int)
        Args:
            deepthink: Enable deep thinking mode for supported models (future functionality)
        """
        raise NotImplementedError
        
    def chat_completion_with_tools(self, messages: List[Dict], tools: List[Dict], model: str, temperature: float, max_tokens: Optional[int] = None, parallel_tool_calls: Optional[bool] = None, deepthink: Optional[bool] = None) -> tuple:
        """
        Chat completion with native tool calling
        Returns: (response_message: dict, tokens: int)
        Response message contains either content or tool_calls
        Args:
            parallel_tool_calls: For OpenAI - whether to allow parallel tool calls (None = default behavior)
                                    For Claude - converted to disable_parallel_tool_use internally
            deepthink: Enable deep thinking mode for supported models (future functionality)
        Note: AI will always be required to choose at least one tool from the provided tools
        """
        raise NotImplementedError
        
    def get_model_input_tokens(self, model: str) -> Optional[int]:
        """
        Get model-specific max input tokens from provider limits.
        Default implementation returns None. Providers should override this.
        
        Args:
            model (str): Model name to get input limits for
            
        Returns:
            Optional[int]: Maximum input tokens for the model, or None if not found
        """
        return None
        
    def convert_messages(self, messages: List[Dict]) -> tuple:
        """
        Convert OpenAI format messages to provider-specific format
        Returns: (converted_messages, system_prompt)
        """
        return messages, None

class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation"""
    
    # Model name prefixes that don't support temperature parameter
    NO_TEMPERATURE_MODELS = ["o1", "o2", "o3", "o4", "gpt-5"]
    NO_MAX_TOKENS_MODELS = ["o1", "o2", "o3", "o4", "gpt-5"]
    LIMITS = {
        "gpt-5": {"max_input_tokens": 400000, "max_output_tokens": 128000},
        "gpt-5-pro": {"max_input_tokens": 400000, "max_output_tokens": 128000},
        "gpt-5-mini": {"max_input_tokens": 400000, "max_output_tokens": 128000},
        "gpt-5-nano": {"max_input_tokens": 400000, "max_output_tokens": 128000},
        "gpt-5-chat-latest": {"max_input_tokens": 128000, "max_output_tokens": 16384},

        "gpt-4.1": {"max_input_tokens": 1000000, "max_output_tokens": 32768},
        "gpt-4.1-mini": {"max_input_tokens": 1000000, "max_output_tokens": 32768},
        "gpt-4.1-nano": {"max_input_tokens": 1000000, "max_output_tokens": 32768},

        "gpt-4o": {"max_input_tokens": 128000, "max_output_tokens": 16384},
        "gpt-4o-mini": {"max_input_tokens": 128000, "max_output_tokens": 16384},
        "gpt-4o-realtime-preview": {"max_input_tokens": 128000, "max_output_tokens": 4096},
        "gpt-4o-mini-realtime-preview": {"max_input_tokens": 128000, "max_output_tokens": 4096},
        "gpt-4o-audio-preview": {"max_input_tokens": 128000, "max_output_tokens": 16384},
        "gpt-4o-mini-transcribe": {"max_input_tokens": 16000, "max_output_tokens": 2000},

        "o3": {"max_input_tokens": 200000, "max_output_tokens": 100000},
        "o3-pro": {"max_input_tokens": 200000, "max_output_tokens": 100000},
        "o3-deep-research": {"max_input_tokens": 200000, "max_output_tokens": 100000},
        "o1": {"max_input_tokens": 200000, "max_output_tokens": 100000},

        "gpt-4-turbo": {"max_input_tokens": 128000, "max_output_tokens": 4096},
        "gpt-3.5-turbo": {"max_input_tokens": 16385, "max_output_tokens": 4096},
        
        # Default fallback for unknown OpenAI models
        "default": {"max_input_tokens": 128000, "max_output_tokens": 4096}
    }

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        _warm_openai_modules()
        self.client = OpenAI(api_key=api_key)
        
        # Pre-sort model keys by length (longest first) for efficient substring matching
        self._sorted_model_keys = sorted([k for k in self.LIMITS.keys() if k != "default"], key=len, reverse=True)

        # Track which response.id produced each tool call_id for follow-up chaining
        self._callid_to_responseid: Dict[str, str] = {}

        self._warm_client_resources()
        
    def _should_include_temperature(self, model: str) -> bool:
        """Check if model supports temperature parameter"""
        return not any(prefix in model for prefix in self.NO_TEMPERATURE_MODELS)

    def _should_include_max_tokens(self, model: str) -> bool:
        """Check if model supports max_tokens parameter"""
        return not any(prefix in model for prefix in self.NO_MAX_TOKENS_MODELS)

    def _warm_client_resources(self) -> None:
        """Trigger lazy client imports to avoid contention under concurrent execution."""
        try:
            _ = self.client.chat.completions
        except Exception as warm_exc:
            logger.debug("OpenAI chat warmup failed: %s", warm_exc)

        try:
            _ = self.client.responses
        except Exception as warm_exc:
            logger.debug("OpenAI responses warmup failed: %s", warm_exc)
    
    def convert_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Translate Chat Completions-style payloads to Responses API format.

        Handles message normalization, instruction extraction, tool flattening,
        and previous response chaining in a single pass (mirroring the Claude
        provider approach for simplicity).
        """
        items: List[Dict[str, Any]] = []
        function_outputs: List[Dict[str, Any]] = []
        instruction_parts: List[str] = []
        tool_result_ids: List[str] = []

        for message in messages:
            role = message.get("role")

            if role in ("system", "developer"):
                # Track instruction text for follow-up responses while still
                # passing the content through to the Responses API payload.
                content_value = message.get("content")
                if isinstance(content_value, str) and content_value.strip():
                    instruction_parts.append(content_value.strip())

            if role in ("tool", "function"):
                call_id = (
                    message.get("tool_call_id")
                    or message.get("call_id")
                    or message.get("id")
                    or message.get("name")
                )
                if call_id:
                    tool_result_ids.append(call_id)

                output = message.get("content")
                if not isinstance(output, str):
                    output = json.dumps(output or "")

                function_payload = {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }
                items.append(function_payload)
                function_outputs.append(function_payload)
                continue

            if role in ("system", "user", "assistant", "developer"):
                content = message.get("content", "")
                if isinstance(content, str):
                    content_payload = {"type": "input_text", "text": content}
                else:
                    content_payload = {
                        "type": "input_text",
                        "text": json.dumps(content),
                    }

                items.append({
                    "role": role,
                    "content": [content_payload],
                })
                continue

            # Ignore any other roles to preserve current caller behavior.

        previous_response_id = None
        for call_id in tool_result_ids:
            mapped_id = self._callid_to_responseid.get(call_id)
            if mapped_id:
                previous_response_id = mapped_id
                break

        normalized_tools: Optional[List[Dict[str, Any]]] = None
        if tools:
            normalized_tools = []
            for tool in tools:
                if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                    fn = tool["function"]
                    flattened = {
                        "type": "function",
                        "name": fn.get("name"),
                        "description": fn.get("description") or tool.get("description"),
                        "parameters": fn.get("parameters")
                        or fn.get("json_schema")
                        or tool.get("parameters"),
                    }
                    if "strict" in fn:
                        flattened["strict"] = fn["strict"]
                    elif "strict" in tool:
                        flattened["strict"] = tool["strict"]
                    normalized_tools.append(flattened)
                else:
                    normalized_tools.append(dict(tool))

            for idx, tool in enumerate(normalized_tools):
                if tool.get("type") == "function":
                    if not tool.get("name"):
                        raise ValueError(f"tools[{idx}].name is required for Responses API")
                    if "parameters" not in tool or tool["parameters"] is None:
                        tool["parameters"] = {"type": "object", "properties": {}}

        metadata = {
            "previous_response_id": previous_response_id,
            "instructions": "\n\n".join(instruction_parts) if instruction_parts else None,
            "tools": normalized_tools,
            "function_call_outputs": function_outputs,
        }

        return items, metadata

    def _get_model_max_output_tokens(self, model: str) -> Optional[int]:
        """Get model-specific max output tokens from LIMITS, returns None if not found"""
        # Use pre-sorted keys for efficient longest-first matching
        for model_key in self._sorted_model_keys:
            if model_key in model:
                return self.LIMITS[model_key].get("max_output_tokens")
        
        # Use default if no specific model match found
        if "default" in self.LIMITS:
            return self.LIMITS["default"].get("max_output_tokens")
        
        return None
        
    def get_model_input_tokens(self, model: str) -> Optional[int]:
        """Get model-specific max input tokens from LIMITS using pre-sorted keys"""
        # Use pre-sorted keys for efficient longest-first matching
        for model_key in self._sorted_model_keys:
            if model_key in model:
                return self.LIMITS[model_key].get("max_input_tokens")
        
        # Use default if no specific model match found
        if "default" in self.LIMITS:
            return self.LIMITS["default"].get("max_input_tokens")
        
        return None
        
    @retry_on_rate_limit
    def chat_completion(self, messages: List[Dict], model: str, temperature: float, max_tokens: Optional[int] = None, deepthink: Optional[bool] = None) -> tuple:
        """Standard OpenAI chat completion using Responses API."""
        items, metadata = self.convert_messages(messages)

        kwargs: Dict[str, Any] = {"model": model, "input": items}

        prev_id = metadata.get("previous_response_id")
        if prev_id:
            fn_outputs = metadata.get("function_call_outputs", [])
            if not fn_outputs:
                raise RuntimeError("Tool output detected but no function_call_output items were formed.")
            kwargs["input"] = fn_outputs
            kwargs["previous_response_id"] = prev_id

            instructions = metadata.get("instructions")
            if instructions:
                kwargs["instructions"] = instructions

        if self._should_include_temperature(model):
            kwargs["temperature"] = temperature

        if max_tokens is not None and self._should_include_max_tokens(model):
            kwargs["max_output_tokens"] = max_tokens
        elif max_tokens is None and self._should_include_max_tokens(model):
            model_limit = self._get_model_max_output_tokens(model)
            if model_limit is not None:
                kwargs["max_output_tokens"] = model_limit

        response = self.client.responses.create(**kwargs)
        content = (response.output_text or "").strip()
        tokens = response.usage.total_tokens
        return content, tokens
        
    @retry_on_rate_limit
    def chat_completion_with_schema(self, messages: List[Dict], schema: BaseModel, model: str, temperature: float, max_tokens: Optional[int] = None, deepthink: Optional[bool] = None) -> tuple:
        """OpenAI structured output with schema using Responses API."""
        items, metadata = self.convert_messages(messages)

        kwargs: Dict[str, Any] = {"model": model, "input": items, "text_format": schema}

        prev_id = metadata.get("previous_response_id")
        if prev_id:
            fn_outputs = metadata.get("function_call_outputs", [])
            if not fn_outputs:
                raise RuntimeError("Tool output detected but no function_call_output items were formed.")
            kwargs["input"] = fn_outputs
            kwargs["previous_response_id"] = prev_id
            instructions = metadata.get("instructions")
            if instructions:
                kwargs["instructions"] = instructions

        if self._should_include_temperature(model):
            kwargs["temperature"] = temperature

        if max_tokens is not None and self._should_include_max_tokens(model):
            kwargs["max_output_tokens"] = max_tokens
        elif max_tokens is None and self._should_include_max_tokens(model):
            model_limit = self._get_model_max_output_tokens(model)
            if model_limit is not None:
                kwargs["max_output_tokens"] = model_limit

        response = self.client.responses.parse(**kwargs)
        content = response.output_parsed
        if isinstance(content, BaseModel):
            content = content.model_dump(by_alias=True)
        tokens = response.usage.total_tokens
        return content, tokens
        
    @retry_on_rate_limit
    def chat_completion_with_tools(self, messages: List[Dict], tools: List[Dict], model: str, temperature: float, max_tokens: Optional[int] = None, parallel_tool_calls: Optional[bool] = None, deepthink: Optional[bool] = None) -> tuple:
        """OpenAI native tool calling via Responses API - tool use is required."""
        items, metadata = self.convert_messages(messages, tools)

        normalized_tools = metadata.get("tools") or []
        
        # Check if this is a follow-up with tool results
        prev_id = metadata.get("previous_response_id")
        fn_outputs = metadata.get("function_call_outputs", [])
        is_follow_up = prev_id and fn_outputs

        kwargs: Dict[str, Any] = {
            "model": model,
            "input": items,
        }

        # For follow-up requests with tool outputs
        if is_follow_up:
            # Only send the function_call_output items and link to previous response
            kwargs["input"] = fn_outputs
            kwargs["previous_response_id"] = prev_id
            instructions = metadata.get("instructions")
            if instructions:
                kwargs["instructions"] = instructions
            # Do NOT send tools or tool_choice in follow-up requests
        else:
            # Initial tool request - must have tools
            if not normalized_tools:
                raise ValueError("tool_choice='required' but tools list is empty")
            kwargs["tools"] = normalized_tools
            kwargs["tool_choice"] = "required"

        if parallel_tool_calls is not None and not is_follow_up:
            kwargs["parallel_tool_calls"] = parallel_tool_calls

        if self._should_include_temperature(model):
            kwargs["temperature"] = temperature

        if max_tokens is not None and self._should_include_max_tokens(model):
            kwargs["max_output_tokens"] = max_tokens
        elif max_tokens is None and self._should_include_max_tokens(model):
            model_limit = self._get_model_max_output_tokens(model)
            if model_limit is not None:
                kwargs["max_output_tokens"] = model_limit

        response = self.client.responses.create(**kwargs)

        message_dict: Dict[str, Any] = {
            "role": "assistant",
            "content": None,
        }

        text_content = (response.output_text or "").strip()
        if text_content:
            message_dict["content"] = text_content

        tool_calls_list: List[Dict[str, Any]] = []

        # Normalize Responses API tool calls to the existing format
        for item in getattr(response, "output", []) or []:
            item_type = getattr(item, "type", None)

            if item_type in {"tool_call", "function_call", "tool_use"}:
                call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
                func_name = getattr(item, "name", None) or getattr(item, "tool_name", None)
                arguments = getattr(item, "arguments", None) or getattr(item, "input", None)
                if not isinstance(arguments, str):
                    try:
                        arguments = json.dumps(arguments)
                    except Exception:
                        arguments = str(arguments)
                tool_calls_list.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": arguments,
                    }
                })

            elif item_type == "message":
                for content_item in getattr(item, "content", []) or []:
                    content_type = getattr(content_item, "type", None)
                    if content_type in {"tool_call", "function_call", "tool_use"}:
                        call_id = getattr(content_item, "tool_call_id", None) or getattr(content_item, "id", None)
                        func_name = getattr(content_item, "name", None) or getattr(content_item, "tool_name", None)
                        arguments = getattr(content_item, "arguments", None) or getattr(content_item, "input", None)
                        if not isinstance(arguments, str):
                            try:
                                arguments = json.dumps(arguments)
                            except Exception:
                                arguments = str(arguments)
                        tool_calls_list.append({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": arguments,
                            }
                        })
                    elif content_type in {"text", "output_text"}:
                        segment = getattr(content_item, "text", "").strip()
                        if segment:
                            message_dict["content"] = (
                                f"{message_dict['content']}\n{segment}".strip()
                                if message_dict["content"]
                                else segment
                            )

        # Remember which response produced each call_id for follow-up chaining
        resp_id = response.id
        for tc in tool_calls_list:
            cid = tc.get("id")
            if cid:
                self._callid_to_responseid[cid] = resp_id

        if tool_calls_list:
            message_dict["tool_calls"] = tool_calls_list

        tokens = response.usage.total_tokens
        return message_dict, tokens

class ClaudeProvider(BaseProvider):
    """Anthropic Claude provider implementation"""
    
    # Model name prefixes that don't support temperature parameter
    NO_TEMPERATURE_MODELS = []
    NO_MAX_TOKENS_MODELS = []
    LIMITS = {
        "claude-opus-4-1": {"max_input_tokens": 200000, "max_output_tokens": 32000},
        "claude-opus-4": {"max_input_tokens": 200000, "max_output_tokens": 32000},
        "claude-sonnet-4": {"max_input_tokens": 200000, "max_output_tokens": 64000},
        "claude-sonnet-4-5": {"max_input_tokens": 200000, "max_output_tokens": 64000},
        "claude-3-7-sonnet": {"max_input_tokens": 200000, "max_output_tokens": 64000},
        "claude-3-5-haiku": {"max_input_tokens": 200000, "max_output_tokens": 8192},
        "claude-3-haiku": {"max_input_tokens": 200000, "max_output_tokens": 4096},
        
        # Default fallback for unknown Claude models
        "default": {"max_input_tokens": 200000, "max_output_tokens": 8192}
    } #! Always check if these substrings are in the model name, as full model names can have suffixes like -2024-10-18 etc.

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        if Anthropic is None:
            raise ImportError("anthropic package is required for Claude support. Install with: pip install anthropic")
        self.client = Anthropic(api_key=api_key)
        
        # Pre-sort model keys by length (longest first) for efficient substring matching
        self._sorted_model_keys = sorted([k for k in self.LIMITS.keys() if k != "default"], key=len, reverse=True)

    def _should_include_temperature(self, model: str) -> bool:
        """Check if model supports temperature parameter"""
        return not any(prefix in model for prefix in self.NO_TEMPERATURE_MODELS)
    
    def _should_include_max_tokens(self, model: str) -> bool:
        """Check if model supports max_tokens parameter"""
        return not any(prefix in model for prefix in self.NO_MAX_TOKENS_MODELS)
    
    def _get_model_max_output_tokens(self, model: str) -> Optional[int]:
        """Get model-specific max output tokens from LIMITS, returns None if not found"""
        # Use pre-sorted keys for efficient longest-first matching
        for model_key in self._sorted_model_keys:
            if model_key in model:
                return self.LIMITS[model_key].get("max_output_tokens")
        
        # Use default if no specific model match found
        if "default" in self.LIMITS:
            return self.LIMITS["default"].get("max_output_tokens")
        
        return None
        
    def get_model_input_tokens(self, model: str) -> Optional[int]:
        """Get model-specific max input tokens from LIMITS using pre-sorted keys"""
        # Use pre-sorted keys for efficient longest-first matching
        for model_key in self._sorted_model_keys:
            if model_key in model:
                return self.LIMITS[model_key].get("max_input_tokens")
        
        # Use default if no specific model match found
        if "default" in self.LIMITS:
            return self.LIMITS["default"].get("max_input_tokens")
        
        return None
        
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
        
    @retry_on_rate_limit
    def chat_completion(self, messages: List[Dict], model: str, temperature: float, max_tokens: Optional[int] = None, deepthink: Optional[bool] = None) -> tuple:
        """Claude chat completion with streaming to avoid timeout issues"""
        converted_messages, system_prompt = self.convert_messages(messages)
        
        kwargs = {
            "model": model,
            "messages": converted_messages
        }

        # Only add temperature if model supports it
        if self._should_include_temperature(model):
            kwargs["temperature"] = temperature 
        
        # Only add max_tokens if it's not None and model supports it
        if max_tokens is not None and self._should_include_max_tokens(model):
            kwargs["max_tokens"] = max_tokens
        elif max_tokens is None and self._should_include_max_tokens(model):
            # Use model-specific output limit if max_tokens is None
            model_limit = self._get_model_max_output_tokens(model)
            if model_limit is not None:
                kwargs["max_tokens"] = model_limit
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        # Use streaming to avoid 10-minute timeout issues
        with self.client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()
            
        content = response.content[0].text.strip()
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return content, tokens
        
    @retry_on_rate_limit
    def chat_completion_with_schema(self, messages: List[Dict], schema: BaseModel, model: str, temperature: float, max_tokens: Optional[int] = None, deepthink: Optional[bool] = None) -> tuple:
        """
        Claude structured output with schema using tool-based approach with streaming
        
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
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": tool_name}  # Force using the tool
        }
        
        # Only add temperature if model supports it
        if self._should_include_temperature(model):
            kwargs["temperature"] = temperature

        # Only add max_tokens if it's not None and model supports it
        if max_tokens is not None and self._should_include_max_tokens(model):
            kwargs["max_tokens"] = max_tokens
        elif max_tokens is None and self._should_include_max_tokens(model):
            # Use model-specific output limit if max_tokens is None
            model_limit = self._get_model_max_output_tokens(model)
            if model_limit is not None:
                kwargs["max_tokens"] = model_limit
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        # Use streaming to avoid 10-minute timeout issues
        with self.client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()
        
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
        
    @retry_on_rate_limit
    def chat_completion_with_tools(self, messages: List[Dict], tools: List[Dict], model: str, temperature: float, max_tokens: Optional[int] = None, parallel_tool_calls: Optional[bool] = None, deepthink: Optional[bool] = None) -> tuple:
        """Claude native tool calling with streaming - always requires at least one tool to be called"""
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
            "tools": claude_tools,  # Use converted Claude tools
            "tool_choice": {"type": "any"}  # Always require the AI to choose at least one tool
        }

        # Only add temperature if model supports it
        if self._should_include_temperature(model):
            kwargs["temperature"] = temperature
        
        # Only add max_tokens if it's not None and model supports it
        if max_tokens is not None and self._should_include_max_tokens(model):
            kwargs["max_tokens"] = max_tokens
        elif max_tokens is None and self._should_include_max_tokens(model):
            # Use model-specific output limit if max_tokens is None
            model_limit = self._get_model_max_output_tokens(model)
            if model_limit is not None:
                kwargs["max_tokens"] = model_limit
        
        # Handle parallel tool calls (Claude uses disable_parallel_tool_use)
        if parallel_tool_calls is not None:
            kwargs["tool_choice"]["disable_parallel_tool_use"] = not parallel_tool_calls
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        # Use streaming to avoid 10-minute timeout issues
        with self.client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()
        
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
            
    def chat_completion(self, model: str, messages: List[Dict], temperature: float = 0.7, max_tokens: Optional[int] = None, deepthink: Optional[bool] = None) -> tuple:
        """Route chat completion to appropriate provider"""
        provider_name, model_name = self.parse_model_string(model)
        provider = self.get_provider(provider_name)
        return provider.chat_completion(messages, model_name, temperature, max_tokens, deepthink)
        
    def chat_completion_with_schema(self, model: str, messages: List[Dict], schema: BaseModel, temperature: float = 0.7, max_tokens: Optional[int] = None, deepthink: Optional[bool] = None) -> tuple:
        """Route structured chat completion to appropriate provider"""
        provider_name, model_name = self.parse_model_string(model)
        provider = self.get_provider(provider_name)
        return provider.chat_completion_with_schema(messages, schema, model_name, temperature, max_tokens, deepthink)
        
    def chat_completion_with_tools(self, model: str, messages: List[Dict], tools: List[Dict], temperature: float = 0.7, max_tokens: Optional[int] = None, parallel_tool_calls: Optional[bool] = None, deepthink: Optional[bool] = None) -> tuple:
        """Route tool calling to appropriate provider - always requires at least one tool to be called"""
        provider_name, model_name = self.parse_model_string(model)
        provider = self.get_provider(provider_name)
        return provider.chat_completion_with_tools(messages, tools, model_name, temperature, max_tokens, parallel_tool_calls, deepthink)
