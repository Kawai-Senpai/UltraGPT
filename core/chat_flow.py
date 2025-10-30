"""Core chat flows orchestrating messaging, tools, and providers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.messages import BaseMessage

from ..messaging import (
    add_message_before_system,
    append_message_to_system,
    ensure_langchain_messages,
    integrate_tool_call_prompt,
    turnoff_system_message,
)
from ..providers import ProviderManager
from ..tooling import ToolManager


class ChatFlow:
    """Encapsulates chat-related operations previously in ``UltraGPT``."""

    def __init__(
        self,
        provider_manager: "ProviderManager",
        tool_manager: "ToolManager",
        *,
        log,
        verbose: bool,
        max_tokens: Optional[int],
    ) -> None:
        self._providers = provider_manager
        self._tools = tool_manager
        self._log = log
        self._verbose = verbose
        self._max_tokens = max_tokens

    def _ensure_lc_messages(self, messages: List[Any]) -> List[BaseMessage]:
        return ensure_langchain_messages(messages)

    # ------------------------------------------------------------------
    # Core chat flows
    # ------------------------------------------------------------------

    def chat_with_ai_sync(
        self,
        messages: List[Any],
        model: str,
        temperature: float,
        tools: List[Any],
        tools_config: Dict[str, Any],
        *,
        max_tokens: Optional[int] = None,
        input_truncation: Optional[Union[str, int]] = None,
        deepthink: Optional[bool] = None,
    ) -> Tuple[str, int, Dict[str, Any]]:
        """Synchronously call the provider and return text output."""

        lc_messages = self._ensure_lc_messages(messages)

        self._log.debug("Sending request to AI provider (msgs: %d)", len(lc_messages))
        if self._verbose:
            provider_name, model_name = self._providers.parse_model_string(model)
            self._log.debug(
                "AI Request → Provider: %s, Model: %s, Messages: %d",
                provider_name,
                model_name,
                len(lc_messages),
            )

        tool_response, tool_usage_details = self._tools.execute_tools(lc_messages, tools, tools_config)
        if tool_response:
            if self._verbose:
                self._log.debug("Appending tool responses to message")
            lc_messages = append_message_to_system(lc_messages, f"Tool Responses:\n{tool_response}")
        elif self._verbose:
            self._log.debug("No tool responses needed")

        content, tokens = self._providers.chat_completion(
            model=model,
            messages=lc_messages,
            temperature=temperature,
            max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
            input_truncation=input_truncation,
            deepthink=deepthink,
        )

        details_dict = {"tools_used": tool_usage_details}
        self._log.debug("Response received (tokens: %d)", tokens)
        return content, tokens, details_dict

    def chat_with_model_parse(
        self,
        messages: List[Any],
        schema,
        *,
        model: Optional[str],
        temperature: float,
        tools: List[Any],
        tools_config: Dict[str, Any],
        max_tokens: Optional[int],
        input_truncation: Optional[Union[str, int]],
        deepthink: Optional[bool],
    ) -> Tuple[Any, int, Dict[str, Any]]:
        model = model or "gpt-4o"
        lc_messages = self._ensure_lc_messages(messages)

        self._log.debug("Sending parse request with schema: %s", schema)

        tool_response, tool_usage_details = self._tools.execute_tools(lc_messages, tools, tools_config)
        if tool_response:
            lc_messages = append_message_to_system(lc_messages, f"Tool Responses:\n{tool_response}")

        content, tokens = self._providers.chat_completion_with_schema(
            model=model,
            messages=lc_messages,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
            input_truncation=input_truncation,
            deepthink=deepthink,
        )

        details_dict = {"tools_used": tool_usage_details}
        self._log.debug("Parse response received (tokens: %d)", tokens)
        return content, tokens, details_dict

    def chat_with_model_tools(
        self,
        messages: List[Any],
        user_tools: List[Any],
        *,
        model: str,
        temperature: float,
        tools: List[Any],
        tools_config: Dict[str, Any],
        max_tokens: Optional[int],
        input_truncation: Optional[Union[str, int]],
        parallel_tool_calls: Optional[bool],
        deepthink: Optional[bool],
    ) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
        lc_messages = self._ensure_lc_messages(messages)

        self._log.debug("Sending native tool calling request")

        tool_response, tool_usage_details = self._tools.execute_tools(lc_messages, tools, tools_config)
        if tool_response:
            lc_messages = append_message_to_system(lc_messages, f"Helpful Details:\n{tool_response}")

        native_tools = self._tools.convert_user_tools_to_native_format(user_tools)

        response_message, tokens = self._providers.chat_completion_with_tools(
            model=model,
            messages=lc_messages,
            tools=native_tools,
            temperature=temperature,
            max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
            parallel_tool_calls=parallel_tool_calls,
            input_truncation=input_truncation,
            deepthink=deepthink,
        )

        details_dict = {"tools_used": tool_usage_details}
        self._log.debug("Native tool calling response received (tokens: %d)", tokens)
        return response_message, tokens, details_dict

    # ------------------------------------------------------------------
    # Message helpers re-exported for convenience
    # ------------------------------------------------------------------

    @staticmethod
    def turnoff_system_message(messages: List[Any]) -> List[BaseMessage]:
        return turnoff_system_message(messages)

    @staticmethod
    def add_message_before_system(messages: List[Any], new_message: Union[Dict[str, Any], BaseMessage]) -> List[BaseMessage]:
        return add_message_before_system(messages, new_message)

    @staticmethod
    def append_message_to_system(messages: List[Any], new_message: str) -> List[BaseMessage]:
        return append_message_to_system(messages, new_message)

    @staticmethod
    def integrate_tool_call_prompt(messages: List[Any], tool_prompt: str) -> List[BaseMessage]:
        return integrate_tool_call_prompt(messages, tool_prompt)


__all__ = ["ChatFlow"]
