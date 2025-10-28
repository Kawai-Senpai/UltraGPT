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
from ..schemas import prepare_schema_for_openai
from ..messaging import (
    LangChainTokenLimiter,
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
    ) -> Tuple[str, int]:
        raise NotImplementedError

    def chat_completion_with_schema(
        self,
        messages: List[BaseMessage],
        schema: BaseModel,
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        deepthink: Optional[bool] = None,
    ) -> Tuple[Dict[str, Any], int]:
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
    ) -> Tuple[Dict[str, Any], int]:
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



class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation built on LangChain ChatOpenAI."""

    NO_TEMPERATURE_MODELS = ["o1", "o2", "o3", "o4", "gpt-5"]
    NO_MAX_TOKENS_MODELS = ["o1", "o2", "o3", "o4", "gpt-5"]
    LIMITS = {
        "gpt-5": {"max_input_tokens": 400000, "max_output_tokens": 128000},
        "gpt-5-pro": {"max_input_tokens": 400000, "max_output_tokens": 128000},
        "gpt-5-mini": {"max_input_tokens": 400000, "max_output_tokens": 128000},
        "gpt-5-nano": {"max_input_tokens": 400000, "max_output_tokens": 128000},
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
        "default": {"max_input_tokens": 128000, "max_output_tokens": 4096},
    }

    def __init__(self, api_key: str, **kwargs: Any):
        super().__init__(api_key, **kwargs)
        _warm_openai_modules()
        self._sorted_model_keys = sorted([key for key in self.LIMITS if key != "default"], key=len, reverse=True)

    def _should_include_temperature(self, model: str) -> bool:
        return not any(prefix in model for prefix in self.NO_TEMPERATURE_MODELS)

    def _should_include_max_tokens(self, model: str) -> bool:
        return not any(prefix in model for prefix in self.NO_MAX_TOKENS_MODELS)

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
        kwargs: Dict[str, Any] = {
            "model": model,
            "api_key": self.api_key,
        }

        if self._should_include_temperature(model):
            kwargs["temperature"] = temperature

        if self._should_include_max_tokens(model):
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            else:
                # Use model's limit or fallback to config default (no reserve ratio for output)
                guess = self._guess_max_output_tokens(model) or config.DEFAULT_MAX_OUTPUT_TOKENS
                kwargs["max_tokens"] = guess

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
        """Collect and merge all chunks from structured_llm.stream.
        
        With include_raw=True, the stream yields multiple dicts with single keys:
        First chunk: {"raw": AIMessage}
        Second chunk: {"parsed": data}
        Third chunk: {"parsing_error": err}
        We need to merge all chunks into a single result dict.
        """
        result: Dict[str, Any] = {}
        for item in stream_iter:
            # Debug logging to see what's being streamed
            if isinstance(item, dict):
                logger.debug(f"Structured stream chunk keys: {list(item.keys())}")
                result.update(item)  # Merge all chunks
                if "parsed" in item:
                    logger.debug(f"Parsed data type: {type(item.get('parsed'))}")
                if "raw" in item:
                    raw_msg = item.get("raw")
                    usage = getattr(raw_msg, "usage_metadata", None)
                    logger.debug(f"Raw message usage_metadata: {usage}")
        return result if result else None

    @staticmethod
    def _usage_total_tokens_from_message(msg: Union[AIMessage, AIMessageChunk, None]) -> int:
        """Extract total token usage from a streamed AIMessage or chunk."""
        if msg is None:
            return 0

        usage_meta = getattr(msg, "usage_metadata", None)
        if isinstance(usage_meta, dict):
            total = usage_meta.get("total_tokens")
            if total is not None:
                try:
                    return int(total)
                except Exception:  # noqa: BLE001
                    return 0
            prompt_tokens = usage_meta.get("prompt_tokens", 0)
            completion_tokens = usage_meta.get("completion_tokens", 0)
            try:
                return int(prompt_tokens) + int(completion_tokens)
            except Exception:  # noqa: BLE001
                return 0

        response_meta = getattr(msg, "response_metadata", None)
        if isinstance(response_meta, dict):
            token_usage = response_meta.get("token_usage", {}) or {}
            total = token_usage.get("total_tokens")
            if total is not None:
                try:
                    return int(total)
                except Exception:  # noqa: BLE001
                    return 0
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            try:
                return int(prompt_tokens) + int(completion_tokens)
            except Exception:  # noqa: BLE001
                return 0

        return 0

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
    ) -> Tuple[str, int]:
        llm = self._build_llm(model=model, temperature=temperature, max_tokens=max_tokens)
        stream = llm.stream(messages)
        final_chunk = self._accumulate_stream(stream)
        content = getattr(final_chunk, "content", "") or ""
        
        # Handle cases where content is a list (structured content)
        if isinstance(content, list):
            # Extract text from structured content
            text_parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            content = " ".join(text_parts)
        
        tokens_used = self._usage_total_tokens_from_message(final_chunk)
        return content.strip(), tokens_used

    @retry_on_rate_limit
    def chat_completion_with_schema(
        self,
        messages: List[BaseMessage],
        schema: BaseModel,
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        deepthink: Optional[bool] = None,
    ) -> Tuple[Dict[str, Any], int]:
        llm = self._build_llm(model=model, temperature=temperature, max_tokens=max_tokens)
        
        # Prepare schema for OpenAI strict mode compliance
        schema_dict = prepare_schema_for_openai(schema.model_json_schema())
        
        structured_llm = llm.with_structured_output(
            schema_dict,
            include_raw=True,
            method="json_schema",
        )
        stream = structured_llm.stream(messages)
        final_obj = self._accumulate_structured_stream(stream)
        parsed_dict, tokens_used = self._finalize_structured_result(final_obj)
        return parsed_dict, tokens_used

    # Removed dead Pydantic tool conversion utilities; we bind raw JSON tool specs directly now.

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
        llm = self._build_llm(model=model, temperature=temperature, max_tokens=max_tokens)
        
        # Normalize tool format: ensure tools[0].function exists (OpenAI format)
        normalized_tools = []
        for tool in tools or []:
            if isinstance(tool, dict):
                if "function" in tool:
                    # Already in correct format
                    normalized_tools.append(tool)
                elif "name" in tool and "type" in tool:
                    # Old format: convert to new format
                    normalized_tools.append({
                        "type": tool.get("type", "function"),
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        }
                    })
                else:
                    # Pass through unknown format
                    normalized_tools.append(tool)
            else:
                normalized_tools.append(tool)
        
        # Bind tools in non-strict mode to mimic classic OpenAI function calling behavior
        llm_with_tools = llm.bind_tools(normalized_tools, strict=False)

        invoke_kwargs: Dict[str, Any] = {}
        if tool_choice and tool_choice != "auto":
            # Only set when caller overrides default; otherwise let model decide
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

        return response_message, tokens_used


class ClaudeProvider(BaseProvider):
    """Anthropic Claude provider through LangChain ChatAnthropic.

    Streaming is enabled for chat, structured, and tool calls so long outputs
    do not block and token usage metadata is always available.
    """

    LIMITS = {
        "claude-opus-4-1": {"max_input_tokens": 200000, "max_output_tokens": 32000},
        "claude-opus-4": {"max_input_tokens": 200000, "max_output_tokens": 32000},
        "claude-sonnet-4": {"max_input_tokens": 200000, "max_output_tokens": 64000},
        "claude-sonnet-4-5": {"max_input_tokens": 200000, "max_output_tokens": 64000},
        "claude-3-7-sonnet": {"max_input_tokens": 200000, "max_output_tokens": 64000},
        "claude-3-5-haiku": {"max_input_tokens": 200000, "max_output_tokens": 8192},
        "claude-3-haiku": {"max_input_tokens": 200000, "max_output_tokens": 4096},
        "default": {"max_input_tokens": 200000, "max_output_tokens": 8192},
    }

    def __init__(self, api_key: str, **kwargs: Any):
        super().__init__(api_key, **kwargs)
        self._sorted_model_keys = sorted([key for key in self.LIMITS if key != "default"], key=len, reverse=True)

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
    ) -> ChatAnthropic:
        kwargs: Dict[str, Any] = {
            "model": model,
            "api_key": self.api_key,
            "stream_usage": True,
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._guess_max_output_tokens(model) or config.DEFAULT_MAX_OUTPUT_TOKENS,
        }
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return ChatAnthropic(**kwargs)

    def build_llm(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> ChatAnthropic:
        return self._build_llm(model=model, temperature=temperature, max_tokens=max_tokens)

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
        """Collect and merge all chunks from structured_llm.stream.
        
        With include_raw=True, the stream yields multiple dicts with single keys:
        First chunk: {"raw": AIMessage}
        Second chunk: {"parsed": data}
        Third chunk: {"parsing_error": err}
        We need to merge all chunks into a single result dict.
        """
        result: Dict[str, Any] = {}
        for item in stream_iter:
            # Debug logging to see what's being streamed
            if isinstance(item, dict):
                logger.debug(f"Claude structured stream chunk keys: {list(item.keys())}")
                result.update(item)  # Merge all chunks
                if "parsed" in item:
                    logger.debug(f"Claude parsed data type: {type(item.get('parsed'))}")
                if "raw" in item:
                    raw_msg = item.get("raw")
                    usage = getattr(raw_msg, "usage_metadata", None)
                    logger.debug(f"Claude raw message usage_metadata: {usage}")
        return result if result else None

    @staticmethod
    def _usage_total_tokens_from_message(msg: Union[AIMessage, AIMessageChunk, None]) -> int:
        """Extract total token usage from a Claude AIMessage or chunk."""
        if msg is None:
            return 0

        usage_meta = getattr(msg, "usage_metadata", None)
        if isinstance(usage_meta, dict):
            try:
                return int(usage_meta.get("input_tokens", 0)) + int(
                    usage_meta.get("output_tokens", 0)
                )
            except Exception:  # noqa: BLE001
                return 0

        response_meta = getattr(msg, "response_metadata", None)
        if isinstance(response_meta, dict):
            usage = response_meta.get("usage", {}) or {}
            try:
                return int(usage.get("input_tokens", 0)) + int(
                    usage.get("output_tokens", 0)
                )
            except Exception:  # noqa: BLE001
                return 0
        return 0

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

    # Removed dead Pydantic tool conversion utilities from Claude provider as well.

    @retry_on_rate_limit
    def chat_completion(
        self,
        messages: List[BaseMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        deepthink: Optional[bool] = None,
    ) -> Tuple[str, int]:
        llm = self._build_llm(model=model, temperature=temperature, max_tokens=max_tokens)
        stream = llm.stream(messages)
        final_chunk = self._accumulate_stream(stream)
        content = getattr(final_chunk, "content", "") or ""
        
        # Handle cases where content is a list (structured content)
        if isinstance(content, list):
            # Extract text from structured content
            text_parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            content = " ".join(text_parts)
        
        tokens_used = self._usage_total_tokens_from_message(final_chunk)
        return content.strip(), tokens_used

    @retry_on_rate_limit
    def chat_completion_with_schema(
        self,
        messages: List[BaseMessage],
        schema: BaseModel,
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        deepthink: Optional[bool] = None,
    ) -> Tuple[Dict[str, Any], int]:
        llm = self._build_llm(model=model, temperature=temperature, max_tokens=max_tokens)
        
        # Prepare schema for strict mode compliance (Claude also benefits from clean schemas)
        schema_dict = prepare_schema_for_openai(schema.model_json_schema())
        
        structured_llm = llm.with_structured_output(
            schema_dict,
            include_raw=True,
            method="json_schema",
        )
        stream = structured_llm.stream(messages)
        final_obj = self._accumulate_structured_stream(stream)
        parsed_dict, tokens_used = self._finalize_structured_result(final_obj)
        return parsed_dict, tokens_used

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
        llm = self._build_llm(model=model, temperature=temperature, max_tokens=max_tokens)
        # Pass raw JSON tool dicts directly; no Pydantic conversion
        
        # Map "required" to "any" for Claude (Anthropic uses "any")
        claude_tool_choice = "any" if tool_choice == "required" else tool_choice
        
        bound_llm = llm.bind_tools(
            tools,
            tool_choice=claude_tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        stream = bound_llm.stream(messages)
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

        return response_message, tokens_used


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
        if ":" in model:
            provider_name, model_name = model.split(":", 1)
            return provider_name, model_name
        return "openai", model

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
            # Re-sanitize after truncation in case pairing got broken
            truncated = remove_orphaned_tool_results_lc(truncated, verbose=self._verbose)
            truncated = drop_unresolved_tool_calls_lc(truncated, verbose=self._verbose)
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
    ) -> Tuple[str, int]:
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
    ) -> Tuple[Dict[str, Any], int]:
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
    ) -> Tuple[Dict[str, Any], int]:
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
    "OpenAIProvider",
    "ClaudeProvider",
    "ProviderManager",
]
