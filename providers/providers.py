"""LangChain powered provider abstraction for UltraGPT.

This module wraps ChatOpenAI and ChatAnthropic and exposes a stable interface
that matches what callers already use:
    - chat_completion
    - chat_completion_with_schema
    - chat_completion_with_tools

All three paths stream results so long responses do not block. Structured
output also streams and still returns parsed JSON plus usage metadata.
"""

from __future__ import annotations

import importlib
import json
import logging
import random
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from .. import config
from ..schemas import prepare_schema_for_openai, sanitize_tool_parameters_schema
from ..messaging import (
    LangChainTokenLimiter,
    consolidate_system_messages_safe,
    ensure_langchain_messages,
    remove_orphaned_tool_results_lc,
    drop_unresolved_tool_calls_lc,
)

logger = logging.getLogger(__name__)

_openai_modules_warmed = False
_openai_warm_lock = threading.Lock()

def _warm_openai_modules() -> None:
    """Preload OpenAI modules to avoid ModuleLock contention under threading."""
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
            except Exception as exc:  # noqa: BLE001
                logger.debug("Optional OpenAI warmup import failed for %s: %s", module_name, exc)

        _openai_modules_warmed = True

def _parse_retry_after(retry_after_value: Optional[str]) -> Optional[float]:
    """Convert a Retry-After header to seconds."""
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

def is_rate_limit_error(error: Exception) -> bool:
    """Return True if the error looks like a rate limit response."""
    error_str = str(error).lower()
    error_code = getattr(error, "status_code", None) or getattr(error, "code", None)

    if error_code == 429:
        return True

    keywords = [
        "rate limit",
        "rate_limit",
        "too many requests",
        "quota exceeded",
        "request limit",
        "usage limit",
        "throttle",
        "rate-limit",
    ]
    return any(keyword in error_str for keyword in keywords)

def retry_on_rate_limit(func):
    """Retry decorated function on rate limit errors using exponential backoff."""

    def wrapper(*args, **kwargs):
        max_retries = config.RATE_LIMIT_RETRIES
        base_delay = config.RATE_LIMIT_BASE_DELAY
        max_delay = config.RATE_LIMIT_MAX_DELAY
        multiplier = config.RATE_LIMIT_BACKOFF_MULTIPLIER

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as err:  # noqa: BLE001
                current_attempt = attempt + 1
                total_attempts = max_retries + 1
                logger.error(
                    "Provider call %s.%s failed on attempt %d/%d: %s",
                    func.__module__,
                    func.__name__,
                    current_attempt,
                    total_attempts,
                    err,
                    exc_info=True,
                )

                if not is_rate_limit_error(err) or attempt == max_retries:
                    raise

                response = getattr(err, "response", None)
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
                    delay = min(base_delay * (multiplier**attempt), max_delay)
                    jitter = random.uniform(0.1, 0.3) * delay
                    total_delay = delay + jitter
                    delay_source = "exponential backoff"

                logger.info(
                    "Rate limit hit for %s.%s, retrying in %.2f seconds via %s (attempt %d/%d)",
                    func.__module__,
                    func.__name__,
                    total_delay,
                    delay_source,
                    current_attempt,
                    total_attempts,
                )
                time.sleep(total_delay)

        raise Exception("Maximum retries exceeded for rate limit")

    return wrapper

class BaseProvider:
    """Abstract provider contract."""

    def __init__(self, api_key: str, **_: Any):
        self.api_key = api_key

    def chat_completion(
        self,
        messages: List[BaseMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        deepthink: Optional[bool] = None,
    ) -> Tuple[str, int, Dict[str, Any]]:
        raise NotImplementedError

    def chat_completion_with_schema(
        self,
        messages: List[BaseMessage],
        schema: BaseModel,
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        deepthink: Optional[bool] = None,
    ) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
        raise NotImplementedError

    def chat_completion_with_tools(
        self,
        messages: List[BaseMessage],
        tools: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        deepthink: Optional[bool] = None,
        tool_choice: str = "required",
    ) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
        raise NotImplementedError

    def get_model_input_tokens(self, model: str) -> Optional[int]:
        return None

    def build_llm(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ):
        raise NotImplementedError

class BaseOpenAICompatibleProvider(BaseProvider):
    """Abstract base for any OpenAI-compatible API endpoint.
    
    This base class provides all the streaming, tool calling, and structured output
    logic that works with any OpenAI-compatible endpoint (OpenAI, OpenRouter, Together, etc.).
    
    Subclasses only need to customize:
    - base_url
    - default_headers
    - model limits (LIMITS dict)
    - model name transformations
    - temperature/max_tokens exclusions (provider-specific)
    """
    
    # Subclasses MUST override this
    LIMITS: Dict[str, Dict[str, int]] = {
        "default": {"max_input_tokens": 128000, "max_output_tokens": 4096},
    }

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ):
        super().__init__(api_key, **kwargs)
        self.base_url = base_url
        self.default_headers = default_headers or {}
        _warm_openai_modules()
        self._sorted_model_keys = sorted([k for k in self.LIMITS if k != "default"], key=len, reverse=True)

    def _should_include_temperature(self, model: str) -> bool:
        """Override in subclass for provider-specific exclusions.
        
        By default, always include temperature (most models support it).
        OpenAI reasoning models (o1, o3, etc.) should override this.
        """
        return True

    def _should_include_max_tokens(self, model: str) -> bool:
        """Override in subclass for provider-specific exclusions.
        
        By default, always include max_tokens (most models support it).
        OpenAI reasoning models (o1, o3, etc.) should override this.
        """
        return True

    def _transform_model_name(self, model: str) -> str:
        """Transform friendly model names to provider-specific format.
        
        Override in subclass for custom transformations (e.g., claude-sonnet-4 → anthropic/claude-sonnet-4).
        """
        return model

    def _guess_max_output_tokens(self, model: str) -> Optional[int]:
        for model_key in self._sorted_model_keys:
            if model_key in model:
                return self.LIMITS[model_key].get("max_output_tokens")
        return self.LIMITS.get("default", {}).get("max_output_tokens")

    def get_model_input_tokens(self, model: str) -> Optional[int]:
        for model_key in self._sorted_model_keys:
            if model_key in model:
                return self.LIMITS[model_key].get("max_input_tokens")
        return self.LIMITS.get("default", {}).get("max_input_tokens")

    def _build_llm(
        self,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        *,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ChatOpenAI:
        """Build ChatOpenAI instance with provider-specific settings."""
        transformed_model = self._transform_model_name(model)
        
        kwargs: Dict[str, Any] = {
            "model": transformed_model,
            "api_key": self.api_key,
        }

        # Add base_url if specified
        if self.base_url:
            kwargs["base_url"] = self.base_url

        # Add default headers
        if self.default_headers:
            kwargs["default_headers"] = self.default_headers

        # Temperature (skip for reasoning models)
        if self._should_include_temperature(model):
            kwargs["temperature"] = temperature

        # Max tokens (skip for reasoning models)
        if self._should_include_max_tokens(model):
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            else:
                guess = self._guess_max_output_tokens(model) or config.DEFAULT_MAX_OUTPUT_TOKENS
                kwargs["max_tokens"] = guess

        # Stream usage for token tracking
        kwargs["stream_usage"] = True

        # Merge extra kwargs
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        return ChatOpenAI(**kwargs)

    def build_llm(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> ChatOpenAI:
        return self._build_llm(model=model, temperature=temperature, max_tokens=max_tokens)

    def _build_extra_body(
        self,
        model: str,
        max_tokens: Optional[int],
        deepthink: Optional[bool],
    ) -> Optional[Dict[str, Any]]:
        """Construct provider-specific extra_body payload."""
        if deepthink is True and hasattr(self, "_does_support_thinking") and self._does_support_thinking(model):
            return {"reasoning": {"effort": "high"}}
        return None

    @staticmethod
    def _accumulate_stream(stream_iter) -> AIMessageChunk:
        """Merge AIMessageChunk pieces yielded by llm.stream."""
        first: Optional[AIMessageChunk] = None
        try:
            first = next(stream_iter)
        except StopIteration:
            return AIMessageChunk(content="")
        aggregate = first
        for chunk in stream_iter:
            aggregate += chunk
        return aggregate

    @staticmethod
    def _accumulate_structured_stream(stream_iter) -> Any:
        """Collect and merge all chunks from structured_llm.stream."""
        result: Dict[str, Any] = {}
        for item in stream_iter:
            if isinstance(item, dict):
                logger.debug(f"Structured stream chunk keys: {list(item.keys())}")
                result.update(item)
                if "parsed" in item:
                    logger.debug(f"Parsed data type: {type(item.get('parsed'))}")
                if "raw" in item:
                    raw_msg = item.get("raw")
                    usage = getattr(raw_msg, "usage_metadata", None)
                    logger.debug(f"Raw message usage_metadata: {usage}")
        return result if result else None

    @staticmethod
    def _usage_total_tokens_from_message(msg: Union[AIMessage, AIMessageChunk, None]) -> int:
        """Extract total token usage from AIMessage or chunk."""
        if msg is None:
            return 0

        # Check usage_metadata first (LangChain standard)
        usage_meta = getattr(msg, "usage_metadata", None)
        if isinstance(usage_meta, dict):
            try:
                return int(usage_meta.get("input_tokens", 0)) + int(usage_meta.get("output_tokens", 0))
            except Exception:  # noqa: BLE001
                pass

        # Fallback to response_metadata
        response_meta = getattr(msg, "response_metadata", None)
        if isinstance(response_meta, dict):
            usage = response_meta.get("usage", {}) or {}
            try:
                return int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
            except Exception:  # noqa: BLE001
                pass
        return 0

    @staticmethod
    def _extract_usage_details(msg: Union[AIMessage, AIMessageChunk, None]) -> Dict[str, Any]:
        """Extract detailed token usage and reasoning information from AIMessage or chunk."""
        if msg is None:
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "reasoning_text": None,
            }

        details = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
            "reasoning_text": None,
        }

        # Check usage_metadata first (LangChain standard)
        usage_meta = getattr(msg, "usage_metadata", None)
        if isinstance(usage_meta, dict):
            details["input_tokens"] = int(usage_meta.get("input_tokens", 0))
            details["output_tokens"] = int(usage_meta.get("output_tokens", 0))
            details["total_tokens"] = details["input_tokens"] + details["output_tokens"]

            output_details = usage_meta.get("output_token_details", {}) or {}
            reasoning_from_usage = (
                output_details.get("reasoning")
                if isinstance(output_details, dict) else None
            )
            if reasoning_from_usage is None and isinstance(output_details, dict):
                reasoning_from_usage = output_details.get("reasoning_tokens")
            if reasoning_from_usage is not None:
                try:
                    details["reasoning_tokens"] = int(reasoning_from_usage)
                except (TypeError, ValueError):
                    logger.debug("Failed to parse reasoning token count from usage_metadata: %s", reasoning_from_usage)

        # Fallback to response_metadata
        response_meta = getattr(msg, "response_metadata", None)
        if isinstance(response_meta, dict):
            usage = response_meta.get("usage", {}) or {}
            token_usage = response_meta.get("token_usage", {}) or {}
            
            # Prefer token_usage if available (OpenRouter format)
            if token_usage:
                details["input_tokens"] = int(token_usage.get("prompt_tokens", 0))
                details["output_tokens"] = int(token_usage.get("completion_tokens", 0))
                details["total_tokens"] = int(token_usage.get("total_tokens", 0))
                
                # Check for reasoning tokens in completion_tokens_details
                completion_details = token_usage.get("completion_tokens_details", {}) or {}
                details["reasoning_tokens"] = int(completion_details.get("reasoning_tokens", 0))
            elif usage:
                # Standard OpenAI format
                details["input_tokens"] = int(usage.get("prompt_tokens", 0))
                details["output_tokens"] = int(usage.get("completion_tokens", 0))
                details["total_tokens"] = int(usage.get("total_tokens", 0))

        # Try to extract reasoning text if available (from message content or metadata)
        # Some models may return reasoning in special format
        if hasattr(msg, "additional_kwargs"):
            additional = getattr(msg, "additional_kwargs", {}) or {}
            # Check if there's reasoning content
            if "reasoning" in additional:
                details["reasoning_text"] = additional.get("reasoning")

        return details

    def _finalize_structured_result(self, final_obj: Any) -> Tuple[Dict[str, Any], int]:
        """Normalize structured stream output into parsed dict and usage."""
        if isinstance(final_obj, dict) and "parsed" in final_obj:
            parsed_part = final_obj.get("parsed")
            raw_msg = final_obj.get("raw")
            tokens_used = self._usage_total_tokens_from_message(raw_msg)

            if isinstance(parsed_part, BaseModel):
                parsed_dict = parsed_part.model_dump(by_alias=True)
            elif isinstance(parsed_part, dict):
                parsed_dict = parsed_part
            else:
                try:
                    parsed_dict = json.loads(str(parsed_part))
                except Exception:  # noqa: BLE001
                    parsed_dict = {"result": str(parsed_part)}

            return parsed_dict, tokens_used

        tokens_used = 0
        if isinstance(final_obj, BaseModel):
            return final_obj.model_dump(by_alias=True), tokens_used
        if isinstance(final_obj, dict):
            return final_obj, tokens_used

        try:
            parsed_attempt = json.loads(str(final_obj))
            if isinstance(parsed_attempt, dict):
                return parsed_attempt, tokens_used
        except Exception:  # noqa: BLE001
            pass

        return {"result": str(final_obj)}, tokens_used

    @retry_on_rate_limit
    def chat_completion(
        self,
        messages: List[BaseMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        deepthink: Optional[bool] = None,
    ) -> Tuple[str, int, Dict[str, Any]]:
        extra_body = self._build_extra_body(model, max_tokens, deepthink)
        extra_kwargs = {"extra_body": extra_body} if extra_body else {}
        
        llm = self._build_llm(model=model, temperature=temperature, max_tokens=max_tokens, extra_kwargs=extra_kwargs)
        stream = llm.stream(messages)
        final_chunk = self._accumulate_stream(stream)
        content = getattr(final_chunk, "content", "") or ""
        tokens_used = self._usage_total_tokens_from_message(final_chunk)
        usage_details = self._extract_usage_details(final_chunk)
        return str(content).strip(), tokens_used, usage_details

    @retry_on_rate_limit
    def chat_completion_with_schema(
        self,
        messages: List[BaseMessage],
        schema: BaseModel,
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        deepthink: Optional[bool] = None,
    ) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
        extra_body = self._build_extra_body(model, max_tokens, deepthink)
        extra_kwargs = {"extra_body": extra_body} if extra_body else {}
        
        llm = self._build_llm(model=model, temperature=temperature, max_tokens=max_tokens, extra_kwargs=extra_kwargs)
        schema_dict = prepare_schema_for_openai(schema.model_json_schema())
        
        structured_llm = llm.with_structured_output(
            schema_dict,
            include_raw=True,
            method="json_schema",
        )
        stream = structured_llm.stream(messages)
        final_obj = self._accumulate_structured_stream(stream)
        parsed_dict, tokens_used = self._finalize_structured_result(final_obj)
        
        # Extract usage details from raw message if available
        usage_details = {}
        if isinstance(final_obj, dict) and "raw" in final_obj:
            usage_details = self._extract_usage_details(final_obj.get("raw"))
        
        return parsed_dict, tokens_used, usage_details

    def chat_completion_with_schema_via_tools(
        self,
        messages: List[BaseMessage],
        schema: BaseModel,
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        deepthink: Optional[bool] = None,
    ) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
        """
        Tool-based structured output workaround for providers that don't support native structured output.
        
        Converts schema to a tool definition, forces model to call that tool,
        then extracts and validates the result from the tool call.
        
        This approach works for Claude and other models via OpenRouter that don't support
        LangChain's native with_structured_output() method.
        """
        tool_name = f"return_{schema.__name__.lower()}"
        schema_dict = prepare_schema_for_openai(schema.model_json_schema())
        
        # Create tool from schema
        tool = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": f"Return the response as structured data matching the {schema.__name__} schema",
                "parameters": schema_dict
            }
        }
        
        # Add instruction to system message
        tool_instruction = f"You must use the {tool_name} tool to provide your response in the required format."
        updated_messages = list(messages)
        
        # Find or create system message
        system_idx = None
        for idx, msg in enumerate(updated_messages):
            if hasattr(msg, 'type') and msg.type == 'system':
                system_idx = idx
                break
        
        if system_idx is not None:
            # Append to existing system message
            from langchain_core.messages import SystemMessage
            existing_content = updated_messages[system_idx].content
            updated_messages[system_idx] = SystemMessage(content=f"{existing_content}\n\n{tool_instruction}")
        else:
            # Prepend new system message
            from langchain_core.messages import SystemMessage
            updated_messages.insert(0, SystemMessage(content=tool_instruction))
        
        # Use our own chat_completion_with_tools method to force tool usage
        response_message, tokens_used, usage_details = self.chat_completion_with_tools(
            messages=updated_messages,
            tools=[tool],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            parallel_tool_calls=False,
            deepthink=deepthink,
            tool_choice="required"
        )
        
        # Extract tool call result
        tool_calls = response_message.get("tool_calls", [])
        
        if tool_calls:
            # Get the first tool call's arguments
            function_args = tool_calls[0].get("function", {}).get("arguments", "{}")
            try:
                import json
                parsed_dict = json.loads(function_args)
            except Exception:
                # If parsing fails, return empty structure
                parsed_dict = schema().model_dump(by_alias=True)
        else:
            # Fallback: try to parse content as JSON
            content = response_message.get("content", "") or ""
            try:
                import json
                parsed_dict = json.loads(content)
            except Exception:
                # If all fails, return empty structure
                parsed_dict = schema().model_dump(by_alias=True)
        
        return parsed_dict, tokens_used, usage_details

    @retry_on_rate_limit
    def chat_completion_with_tools(
        self,
        messages: List[BaseMessage],
        tools: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        deepthink: Optional[bool] = None,
        tool_choice: str = "required",
    ) -> Tuple[Dict[str, Any], int]:
        extra_body = self._build_extra_body(model, max_tokens, deepthink)
        extra_kwargs = {"extra_body": extra_body} if extra_body else {}
        
        llm = self._build_llm(model=model, temperature=temperature, max_tokens=max_tokens, extra_kwargs=extra_kwargs)
        
        # Normalize and sanitize tools
        normalized_tools = []
        for tool in tools or []:
            if isinstance(tool, dict):
                if "function" in tool:
                    fn = tool["function"]
                    cleaned_params = sanitize_tool_parameters_schema(fn.get("parameters", {}) or {})
                    normalized_tools.append({
                        "type": "function",
                        "function": {
                            "name": fn["name"],
                            "description": fn.get("description", ""),
                            "parameters": cleaned_params,
                        }
                    })
                elif "name" in tool and "type" in tool:
                    cleaned_params = sanitize_tool_parameters_schema(tool.get("parameters", {}) or {})
                    normalized_tools.append({
                        "type": tool.get("type", "function"),
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": cleaned_params,
                        }
                    })
                else:
                    normalized_tools.append(tool)
            else:
                normalized_tools.append(tool)
        
        # Bind tools
        llm_with_tools = llm.bind_tools(normalized_tools, strict=False)

        invoke_kwargs: Dict[str, Any] = {}
        # Anthropic requires tools to be present if tool_choice is specified
        # Only set tool_choice if tools array is not empty
        if normalized_tools and tool_choice and tool_choice != "auto":
            # LangChain maps "required" to Anthropic's "any" automatically
            invoke_kwargs["tool_choice"] = tool_choice

        if parallel_tool_calls is not None:
            invoke_kwargs["parallel_tool_calls"] = parallel_tool_calls

        stream = llm_with_tools.stream(messages, **invoke_kwargs)
        final_chunk = self._accumulate_stream(stream)
        tokens_used = self._usage_total_tokens_from_message(final_chunk)

        content = getattr(final_chunk, "content", None)
        normalized_calls: List[Dict[str, Any]] = []
        for call in getattr(final_chunk, "tool_calls", []) or []:
            call_id = call.get("id")
            call_name = call.get("name")
            call_args = call.get("args", {})
            try:
                arguments = json.dumps(call_args)
            except Exception:  # noqa: BLE001
                arguments = str(call_args)
            normalized_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call_name,
                        "arguments": arguments,
                    },
                }
            )

        response_message: Dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if normalized_calls:
            response_message["tool_calls"] = normalized_calls

        usage_details = self._extract_usage_details(final_chunk)
        return response_message, tokens_used, usage_details


class OpenRouterProvider(BaseOpenAICompatibleProvider):
    """Universal OpenRouter provider supporting ALL models via OpenAI-compatible API.
    
    Routes to any model through OpenRouter's unified endpoint:
    - GPT models (gpt-4o, gpt-4, gpt-3.5-turbo, etc.)
    - Claude models (claude-sonnet-4.5, claude-opus-4, etc.) with 1M context
    - Llama models (llama-3.3, llama-3.1, etc.)
    - Mistral, Gemini, and more
    
    Inherits all streaming, tool calling, and structured output logic from BaseOpenAICompatibleProvider.
    """

    # Systematic model behavior maps (OpenRouter-specific)
    NO_TEMPERATURE_MODELS = ["openai/o", "openai/gpt-5"]  # OpenAI reasoning models routed via OpenRouter
    NO_MAX_TOKENS_MODELS = ["openai/o", "openai/gpt-5"]  # OpenAI reasoning models routed via OpenRouter
    REASONING_MODELS = [
        "openai/o",  # o1, o3, etc.
        "openai/gpt-5",  # gpt-5, gpt-5-pro, etc.
        "anthropic/claude-sonnet",  # Claude Sonnet family supports reasoning tokens
        "anthropic/claude-3.7-sonnet",  # Explicit prefix for 3.7 Sonnet slug
        "anthropic/claude-opus",  # Claude Opus family supports reasoning tokens
        "deepseek/deepseek-chat",  # DeepSeek v3.1 exposes reasoning traces via effort flag
        "x-ai/grok-4",  # Grok 4 always returns reasoning traces
        "google/gemini-3-pro-preview",  # Gemini 3 Pro Preview exposes reasoning tokens via OpenRouter
    ]
    
    # Models that DON'T support native structured output via LangChain's with_structured_output
    # Need tool-based workaround (convert schema to tool)
    NO_NATIVE_STRUCTURED_OUTPUT = [
        "anthropic/claude",  # All Claude models via OpenRouter don't support native structured output
    ]

    # Models that expect max_output_tokens instead of max_tokens in the payload
    MAX_OUTPUT_TOKEN_MODELS = [
        "google/gemini",
        "x-ai/grok-4",
    ]

    # Model slug mappings for friendly names → OpenRouter format
    _MODEL_SLUGS = {
        # Claude models (extended 1M context for Sonnet 4/4.5)
        "claude-sonnet-4.5": "anthropic/claude-sonnet-4.5",
        "claude-sonnet-4-5": "anthropic/claude-sonnet-4.5",
        "claude-sonnet-4": "anthropic/claude-sonnet-4",
        # Opus models (200k context)
        "claude-opus-4.5": "anthropic/claude-opus-4.5",
        "claude-opus-4-5": "anthropic/claude-opus-4.5",
        "claude-opus-4.1": "anthropic/claude-opus-4.1",
        "claude-opus-4-1": "anthropic/claude-opus-4.1",
        "claude-opus-4": "anthropic/claude-opus-4",
        "claude-3.7-sonnet": "anthropic/claude-3.7-sonnet",
        "claude-3-7-sonnet": "anthropic/claude-3.7-sonnet",
        "claude-3.5-haiku": "anthropic/claude-3.5-haiku",
        "claude-3-5-haiku": "anthropic/claude-3.5-haiku",
        "claude-3-haiku": "anthropic/claude-3-haiku",
        # GPT models (explicit GPT-5.1 mappings)
        "gpt-5.1": "openai/gpt-5.1",
        "gpt-5.1-chat": "openai/gpt-5.1-chat",
        "gpt5.1": "openai/gpt-5.1",
        "gpt5.1-chat": "openai/gpt-5.1-chat",
        # GPT models pass through as-is
        # Gemini models
        "gemini-2.5-pro": "google/gemini-2.5-pro",
        "gemini-pro-2.5": "google/gemini-2.5-pro",
        "gemini-pro": "google/gemini-2.5-pro",
        "gemini25pro": "google/gemini-2.5-pro",
        "gemini-3-pro-preview": "google/gemini-3-pro-preview",
        "gemini-3-pro": "google/gemini-3-pro-preview",
        "gemini3-pro-preview": "google/gemini-3-pro-preview",
        "gemini3pro": "google/gemini-3-pro-preview",
        # Grok models
        "grok-4": "x-ai/grok-4",
        "grok-4.1-fast": "x-ai/grok-4.1-fast",
        "grok-4-1-fast": "x-ai/grok-4.1-fast",
        # DeepSeek models
        "deepseek-chat-v3.1": "deepseek/deepseek-chat-v3.1",
        "deepseek-v3.1": "deepseek/deepseek-chat-v3.1",
        # Llama, Mistral, etc. pass through as-is
    }

    LIMITS = {
        # Extended 1M context for Claude Sonnet 4/4.5 via OpenRouter
        "claude-sonnet-4.5": {"max_input_tokens": 1_000_000, "max_output_tokens": 64000},
        "claude-sonnet-4": {"max_input_tokens": 1_000_000, "max_output_tokens": 64000},

        # 200k context for other Claude models
        "claude-opus-4.5": {"max_input_tokens": 200_000, "max_output_tokens": 32000},
        "claude-opus-4.1": {"max_input_tokens": 200_000, "max_output_tokens": 32000},
        "claude-opus-4": {"max_input_tokens": 200_000, "max_output_tokens": 32000},
        "claude-3.7-sonnet": {"max_input_tokens": 200_000, "max_output_tokens": 64000},
        "claude-3-7-sonnet": {"max_input_tokens": 200_000, "max_output_tokens": 64000},
        "claude-3.5-haiku": {"max_input_tokens": 200_000, "max_output_tokens": 8192},
        "claude-3-5-haiku": {"max_input_tokens": 200_000, "max_output_tokens": 8192},
        "claude-3-haiku": {"max_input_tokens": 200_000, "max_output_tokens": 4096},

        # GPT models (OpenRouter routes to OpenAI)
        "gpt-5": {"max_input_tokens": 400000, "max_output_tokens": 128000},
        "gpt-5-pro": {"max_input_tokens": 400000, "max_output_tokens": 128000},
        "gpt-5-mini": {"max_input_tokens": 400000, "max_output_tokens": 128000},
        "gpt-5-nano": {"max_input_tokens": 400000, "max_output_tokens": 128000},
        "gpt-5.1": {"max_input_tokens": 400000, "max_output_tokens": 128000},
        "openai/gpt-5.1": {"max_input_tokens": 400000, "max_output_tokens": 128000},
        "gpt-5.1-chat": {"max_input_tokens": 400000, "max_output_tokens": 128000},
        "openai/gpt-5.1-chat": {"max_input_tokens": 400000, "max_output_tokens": 128000},
        "gpt-5-chat-latest": {"max_input_tokens": 128000, "max_output_tokens": 16384},
        "gpt-4.1": {"max_input_tokens": 1_000_000, "max_output_tokens": 32768},
        "gpt-4.1-mini": {"max_input_tokens": 1_000_000, "max_output_tokens": 32768},
        "gpt-4.1-nano": {"max_input_tokens": 1_000_000, "max_output_tokens": 32768},
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

        # Gemini models
        "gemini-2.5-pro": {"max_input_tokens": 1_048_576, "max_output_tokens": 65535},
        "google/gemini-2.5-pro": {"max_input_tokens": 1_048_576, "max_output_tokens": 65535},
        "gemini-3-pro-preview": {"max_input_tokens": 1_048_576, "max_output_tokens": 65536},
        "google/gemini-3-pro-preview": {"max_input_tokens": 1_048_576, "max_output_tokens": 65536},

        # Grok model
        "grok-4": {"max_input_tokens": 256_000, "max_output_tokens": 32768},
        "x-ai/grok-4": {"max_input_tokens": 256_000, "max_output_tokens": 32768},
        # Grok 4.1 Fast (2M context, ~30k output as per xAI/OpenRouter docs)
        "grok-4.1-fast": {"max_input_tokens": 2_000_000, "max_output_tokens": 30_000},
        "x-ai/grok-4.1-fast": {"max_input_tokens": 2_000_000, "max_output_tokens": 30_000},

        # DeepSeek model
        "deepseek-chat-v3.1": {"max_input_tokens": 163_840, "max_output_tokens": 8192},
        "deepseek/deepseek-chat-v3.1": {"max_input_tokens": 163_840, "max_output_tokens": 8192},

        # Llama models
        "llama-3.3": {"max_input_tokens": 128000, "max_output_tokens": 8192},
        "llama-3.1": {"max_input_tokens": 128000, "max_output_tokens": 8192},
        "default": {"max_input_tokens": 200_000, "max_output_tokens": 8192},
    }

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
        **kwargs: Any
    ):
        # Build default headers for OpenRouter (optional attribution)
        default_headers = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if app_name:
            default_headers["X-Title"] = app_name
        
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers if default_headers else None,
            **kwargs
        )

    def _transform_model_name(self, model: str) -> str:
        """Transform friendly model names to OpenRouter slugs."""
        # Check if model is already in OpenRouter format (contains /)
        if "/" in model:
            return model

        short_names = {
            "haiku": "anthropic/claude-3-haiku",
            "sonnet": "anthropic/claude-3.7-sonnet",
            "opus": "anthropic/claude-opus-4",
            "gemini": "google/gemini-3-pro-preview",
            "grok": "x-ai/grok-4",
            "deepseek": "deepseek/deepseek-chat-v3.1",
        }

        name_key = model.lower().strip()
        if name_key in short_names:
            return short_names[name_key]
        
        # Try to find a mapping
        lower_model = name_key
        for key, slug in self._MODEL_SLUGS.items():
            if key in lower_model:
                return slug

        # Pass through if no mapping found (already in correct format)
        return model

    def _does_support_thinking(self, model: str) -> bool:
        """Check if model supports reasoning/thinking capabilities."""
        transformed = self._transform_model_name(model)
        return any(transformed.startswith(prefix) for prefix in self.REASONING_MODELS)

    def _should_include_temperature(self, model: str) -> bool:
        """Check if temperature parameter should be included for this model."""
        transformed = self._transform_model_name(model)
        return not any(transformed.startswith(prefix) for prefix in self.NO_TEMPERATURE_MODELS)

    def _should_include_max_tokens(self, model: str) -> bool:
        """Check if max_tokens parameter should be included for this model."""
        transformed = self._transform_model_name(model)
        if any(transformed.startswith(prefix) for prefix in self.NO_MAX_TOKENS_MODELS):
            return False
        if any(transformed.startswith(prefix) for prefix in self.MAX_OUTPUT_TOKEN_MODELS):
            return False
        return True

    def _supports_native_structured_output(self, model: str) -> bool:
        """Check if model supports native structured output via LangChain's with_structured_output."""
        transformed = self._transform_model_name(model)
        return not any(transformed.startswith(prefix) for prefix in self.NO_NATIVE_STRUCTURED_OUTPUT)

    def _build_extra_body(
        self,
        model: str,
        max_tokens: Optional[int],
        deepthink: Optional[bool],
    ) -> Optional[Dict[str, Any]]:
        base_extra = super()._build_extra_body(model, max_tokens, deepthink) or {}
        extra_body: Dict[str, Any] = dict(base_extra)

        transformed = self._transform_model_name(model)
        needs_max_output_tokens = any(
            transformed.startswith(prefix) for prefix in self.MAX_OUTPUT_TOKEN_MODELS
        )
        if needs_max_output_tokens:
            desired_tokens = max_tokens
            if desired_tokens is None:
                desired_tokens = self._guess_max_output_tokens(model) or config.DEFAULT_MAX_OUTPUT_TOKENS
            extra_body["max_output_tokens"] = desired_tokens

        return extra_body or None

    @retry_on_rate_limit
    def chat_completion_with_schema(
        self,
        messages: List[BaseMessage],
        schema: BaseModel,
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        deepthink: Optional[bool] = None,
    ) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
        """
        Structured output with schema support.
        
        For models that support native structured output (e.g., OpenAI):
        - Uses LangChain's with_structured_output()
        
        For models that DON'T support it (e.g., Claude):
        - Uses shared tool-based workaround via chat_completion_with_schema_via_tools()
        """
        if self._supports_native_structured_output(model):
            # Use native structured output (OpenAI, etc.)
            return super().chat_completion_with_schema(messages, schema, model, temperature, max_tokens, deepthink)
        
        # Use tool-based workaround for Claude and other models without native support
        return self.chat_completion_with_schema_via_tools(messages, schema, model, temperature, max_tokens, deepthink)


class ProviderManager:
    """Registry facade for providers with centralized truncation and cleanup."""

    def __init__(
        self,
        token_limiter: Optional[LangChainTokenLimiter] = None,
        *,
        default_input_truncation: Optional[Union[str, int]] = config.DEFAULT_INPUT_TRUNCATION,
        logger: Optional[logging.Logger] = None,
        verbose: bool = False,
        reserve_ratio: float = config.DEFAULT_RESERVE_RATIO,
    ) -> None:
        self.providers: Dict[str, BaseProvider] = {}
        self._token_limiter = token_limiter or LangChainTokenLimiter()
        self._default_input_truncation = default_input_truncation
        self._log = logger
        self._verbose = verbose
        self._reserve_ratio = reserve_ratio

    def add_provider(self, name: str, provider: BaseProvider) -> None:
        self.providers[name] = provider

    def get_provider(self, name: str) -> BaseProvider:
        if name not in self.providers:
            raise ValueError(f"Provider '{name}' not found. Available: {list(self.providers.keys())}")
        return self.providers[name]

    def parse_model_string(self, model: str) -> Tuple[str, str]:
        cleaned = (model or "").strip()
        if "/" in cleaned:
            return "openrouter", cleaned
        return "openrouter", cleaned

    def does_support_thinking(self, model: str) -> bool:
        """Check if the model supports native reasoning/thinking capabilities.
        
        Returns True if the provider and model support native reasoning.
        This allows the caller to skip fake deepthink pipeline and use native reasoning instead.
        """
        try:
            provider_name, model_name = self.parse_model_string(model)
            provider = self.get_provider(provider_name)
            
            # Check if provider has _does_support_thinking method
            if hasattr(provider, '_does_support_thinking'):
                return provider._does_support_thinking(model_name)
            
            return False
        except Exception:  # noqa: BLE001
            return False

    def _resolve_limit(
        self,
        provider_name: str,
        model_name: str,
        setting: Optional[Union[str, int]],
    ) -> Optional[int]:
        if setting == "OFF":
            return None

        provider = self.get_provider(provider_name)

        if setting in {None, "AUTO"}:
            return provider.get_model_input_tokens(model_name) or config.DEFAULT_AUTO_INPUT_LIMIT

        if isinstance(setting, int) and setting > 0:
            return setting

        if self._verbose and self._log:
            self._log.warning("Invalid input_truncation setting %s; skipping truncation", setting)
        return None

    def _prepare_messages(
        self,
        provider_name: str,
        model_name: str,
        messages: List[Any],
        *,
        input_truncation: Optional[Union[str, int]],
        keep_newest: bool,
    ) -> List[BaseMessage]:
        lc_messages = ensure_langchain_messages(messages)
        cleaned = remove_orphaned_tool_results_lc(lc_messages, verbose=self._verbose)
        # Also drop assistant messages that contain unresolved tool_calls to satisfy strict providers
        cleaned = drop_unresolved_tool_calls_lc(cleaned, verbose=self._verbose)
        
        # CONSOLIDATE system messages BEFORE truncation (important!)
        # This ensures we truncate with a single system message, not multiple scattered ones
        # Also strips whitespace from all message content (moved inside consolidate function)
        cleaned = consolidate_system_messages_safe(cleaned)

        setting = input_truncation if input_truncation is not None else self._default_input_truncation
        max_tokens = self._resolve_limit(provider_name, model_name, setting)
        if max_tokens is None:
            return cleaned

        effective_limit = max(1, int(max_tokens * self._reserve_ratio))

        try:
            provider = self.get_provider(provider_name)
            llm = provider.build_llm(model=model_name, temperature=0.0)
            truncated = self._token_limiter.apply_input_truncation(
                llm=llm,
                message_groups=[cleaned],
                max_tokens=effective_limit,
                keep_newest=keep_newest,
                preserve_system=True,
                verbose=self._verbose,
            )
            # No need to re-sanitize after truncation - grouping prevents orphans by design
            return truncated
        except Exception as exc:  # noqa: BLE001
            if self._log:
                self._log.error(
                    "Input truncation failed for %s:%s - %s",
                    provider_name,
                    model_name,
                    exc,
                )
            return cleaned

    def chat_completion(
        self,
        model: str,
        messages: List[Any],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        deepthink: Optional[bool] = None,
        input_truncation: Optional[Union[str, int]] = None,
        keep_newest: bool = True,
    ) -> Tuple[str, int, Dict[str, Any]]:
        provider_name, model_name = self.parse_model_string(model)
        provider = self.get_provider(provider_name)
        prepared = self._prepare_messages(
            provider_name,
            model_name,
            messages,
            input_truncation=input_truncation,
            keep_newest=keep_newest,
        )
        return provider.chat_completion(prepared, model_name, temperature, max_tokens, deepthink)

    def chat_completion_with_schema(
        self,
        model: str,
        messages: List[Any],
        schema: BaseModel,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        deepthink: Optional[bool] = None,
        input_truncation: Optional[Union[str, int]] = None,
        keep_newest: bool = True,
    ) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
        provider_name, model_name = self.parse_model_string(model)
        provider = self.get_provider(provider_name)
        prepared = self._prepare_messages(
            provider_name,
            model_name,
            messages,
            input_truncation=input_truncation,
            keep_newest=keep_newest,
        )
        return provider.chat_completion_with_schema(prepared, schema, model_name, temperature, max_tokens, deepthink)

    def chat_completion_with_tools(
        self,
        model: str,
        messages: List[Any],
        tools: List[Dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        deepthink: Optional[bool] = None,
        tool_choice: str = "required",
        input_truncation: Optional[Union[str, int]] = None,
        keep_newest: bool = True,
    ) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
        provider_name, model_name = self.parse_model_string(model)
        provider = self.get_provider(provider_name)
        prepared = self._prepare_messages(
            provider_name,
            model_name,
            messages,
            input_truncation=input_truncation,
            keep_newest=keep_newest,
        )
        return provider.chat_completion_with_tools(
            prepared,
            tools,
            model_name,
            temperature,
            max_tokens,
            parallel_tool_calls,
            deepthink,
            tool_choice,
        )

__all__ = [
    "BaseProvider",
    "BaseOpenAICompatibleProvider",
    "OpenRouterProvider",
    "ProviderManager",
]
