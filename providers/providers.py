"""LangChain-powered provider abstraction for UltraGPT."""

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

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, create_model

from .. import config
from ..messaging import (
    LangChainTokenLimiter,
    ensure_langchain_messages,
    remove_orphaned_tool_results_lc,
)

logger = logging.getLogger(__name__)

DEFAULT_AUTO_INPUT_LIMIT = 128_000
DEFAULT_RESERVE_RATIO = 0.8

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
            "use_responses_api": True,
            "output_version": "responses/v1",
            "stream_options": {"include_usage": True},
        }

        if self._should_include_temperature(model):
            kwargs["temperature"] = temperature

        if self._should_include_max_tokens(model):
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            else:
                guess = self._guess_max_output_tokens(model)
                if guess is not None:
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
    def _usage_total_tokens_from_chunk(chunk: AIMessageChunk) -> int:
        usage_meta = getattr(chunk, "usage_metadata", None)
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

        response_meta = getattr(chunk, "response_metadata", None)
        if isinstance(response_meta, dict):
            token_usage = response_meta.get("token_usage", {})
            if isinstance(token_usage, dict):
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
        tokens_used = self._usage_total_tokens_from_chunk(final_chunk)
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
        structured_llm = llm.with_structured_output(schema)
        structured_resp = structured_llm.invoke(messages)
        if isinstance(structured_resp, BaseModel):
            parsed = structured_resp.model_dump(by_alias=True)
        elif isinstance(structured_resp, dict):
            parsed = structured_resp
        else:
            try:
                parsed = json.loads(str(structured_resp))
            except Exception:  # noqa: BLE001
                parsed = {"result": str(structured_resp)}
        return parsed, 0

    @staticmethod
    def _json_schema_type_to_pytype(prop_schema: Dict[str, Any]) -> Any:
        schema_type = prop_schema.get("type")
        if schema_type == "string":
            return str
        if schema_type == "number":
            return float
        if schema_type == "integer":
            return int
        if schema_type == "boolean":
            return bool
        if schema_type == "object":
            return dict
        if schema_type == "array":
            return list
        return Any

    def _build_pydantic_tools_from_specs(self, tools: List[Dict[str, Any]]) -> Tuple[List[Any], bool]:
        pydantic_models: List[Any] = []
        strict_any = False

        for tool in tools or []:
            if tool.get("type") != "function":
                pydantic_models.append(tool)
                continue

            fn_block = tool.get("function", {}) or {}
            tool_name = fn_block.get("name") or "Tool"
            description = fn_block.get("description") or ""
            params_schema = fn_block.get("parameters") or {}
            required_fields = params_schema.get("required", []) or []
            properties = params_schema.get("properties", {}) or {}

            if fn_block.get("strict") or tool.get("strict"):
                strict_any = True

            field_defs: Dict[str, Tuple[Any, Any]] = {}
            for arg_name, arg_schema in properties.items():
                py_type = self._json_schema_type_to_pytype(arg_schema)
                arg_description = arg_schema.get("description", "") or ""
                if arg_name in required_fields:
                    field_defs[arg_name] = (py_type, Field(..., description=arg_description))
                else:
                    field_defs[arg_name] = (Optional[py_type], Field(default=None, description=arg_description))

            dynamic_model = create_model(tool_name, **field_defs)  # type: ignore[arg-type]
            dynamic_model.__doc__ = description or f"Tool {tool_name}"
            pydantic_models.append(dynamic_model)

        return pydantic_models, strict_any

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
        pydantic_tools, strict_any = self._build_pydantic_tools_from_specs(tools)
        llm_with_tools = llm.bind_tools(pydantic_tools, strict=strict_any)

        invoke_kwargs: Dict[str, Any] = {}
        if tool_choice == "required":
            invoke_kwargs["tool_choice"] = "any"
        elif tool_choice != "auto":
            invoke_kwargs["tool_choice"] = tool_choice

        if parallel_tool_calls is not None:
            invoke_kwargs["parallel_tool_calls"] = parallel_tool_calls

        stream = llm_with_tools.stream(messages, **invoke_kwargs)
        final_chunk = self._accumulate_stream(stream)
        tokens_used = self._usage_total_tokens_from_chunk(final_chunk)

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
    """Anthropic Claude provider implemented via LangChain ChatAnthropic."""

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
            "max_tokens": max_tokens if max_tokens is not None else self._guess_max_output_tokens(model) or 1024,
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
    def _usage_total_tokens_from_chunk(chunk: AIMessageChunk) -> int:
        usage_meta = getattr(chunk, "usage_metadata", None)
        if isinstance(usage_meta, dict):
            try:
                return int(usage_meta.get("input_tokens", 0)) + int(usage_meta.get("output_tokens", 0))
            except Exception:  # noqa: BLE001
                return 0

        response_meta = getattr(chunk, "response_metadata", None)
        if isinstance(response_meta, dict):
            usage = response_meta.get("usage", {})
            if isinstance(usage, dict):
                try:
                    return int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
                except Exception:  # noqa: BLE001
                    return 0
        return 0

    def _json_schema_type_to_pytype(self, prop_schema: Dict[str, Any]) -> Any:
        schema_type = prop_schema.get("type")
        if schema_type == "string":
            return str
        if schema_type == "number":
            return float
        if schema_type == "integer":
            return int
        if schema_type == "boolean":
            return bool
        if schema_type == "object":
            return dict
        if schema_type == "array":
            return list
        return Any

    def _build_pydantic_tools_from_specs(self, tools: List[Dict[str, Any]]) -> Tuple[List[Any], bool]:
        pydantic_models: List[Any] = []
        strict_any = False
        for tool in tools or []:
            if tool.get("type") != "function":
                pydantic_models.append(tool)
                continue

            fn_block = tool.get("function", {}) or {}
            tool_name = fn_block.get("name") or "Tool"
            description = fn_block.get("description") or ""
            params_schema = fn_block.get("parameters") or {}
            required_fields = params_schema.get("required", []) or []
            properties = params_schema.get("properties", {}) or {}

            if fn_block.get("strict") or tool.get("strict"):
                strict_any = True

            field_defs: Dict[str, Tuple[Any, Any]] = {}
            for arg_name, arg_schema in properties.items():
                py_type = self._json_schema_type_to_pytype(arg_schema)
                description_text = arg_schema.get("description", "") or ""
                if arg_name in required_fields:
                    field_defs[arg_name] = (py_type, Field(..., description=description_text))
                else:
                    field_defs[arg_name] = (Optional[py_type], Field(default=None, description=description_text))

            dynamic_model = create_model(tool_name, **field_defs)  # type: ignore[arg-type]
            dynamic_model.__doc__ = description or f"Tool {tool_name}"
            pydantic_models.append(dynamic_model)

        return pydantic_models, strict_any

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
        tokens_used = self._usage_total_tokens_from_chunk(final_chunk)
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
        structured_llm = llm.with_structured_output(schema)
        structured_resp = structured_llm.invoke(messages)
        if isinstance(structured_resp, BaseModel):
            parsed = structured_resp.model_dump(by_alias=True)
        elif isinstance(structured_resp, dict):
            parsed = structured_resp
        else:
            try:
                parsed = json.loads(str(structured_resp))
            except Exception:  # noqa: BLE001
                parsed = {"result": str(structured_resp)}
        return parsed, 0

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
        pydantic_tools, strict_any = self._build_pydantic_tools_from_specs(tools)
        llm_with_tools = llm.bind_tools(pydantic_tools, strict=strict_any)
        stream = llm_with_tools.stream(messages)
        final_chunk = self._accumulate_stream(stream)
        tokens_used = self._usage_total_tokens_from_chunk(final_chunk)

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
        reserve_ratio: float = DEFAULT_RESERVE_RATIO,
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
            return provider.get_model_input_tokens(model_name) or DEFAULT_AUTO_INPUT_LIMIT

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
            return remove_orphaned_tool_results_lc(truncated, verbose=self._verbose)
        except Exception as exc:  # noqa: BLE001
            if self._log:
                self._log.error(
                    "Input truncation failed for %s:%s — %s",
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
