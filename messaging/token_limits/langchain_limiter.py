"""LangChain token limiting utilities."""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, List, Optional, Tuple, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from ..history_utils import remove_orphaned_tool_results_lc


def _serialize_message_for_token_count(message: BaseMessage) -> str:
    role = "assistant"
    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, ToolMessage):
        role = "tool"

    base_text = f"{role}: {getattr(message, 'content', '')}"

    if isinstance(message, AIMessage):
        tool_calls = getattr(message, "tool_calls", []) or []
        if tool_calls:
            try:
                serialized_calls = json.dumps(tool_calls)
            except Exception:  # noqa: BLE001
                serialized_calls = str(tool_calls)
            base_text += f"\nTOOL_CALLS: {serialized_calls}"

    return base_text


class LangChainTokenLimiter:
    """Token utilities that rely on LangChain models instead of manual tiktoken math."""

    def __init__(self, log_handler: Optional[logging.Logger] = None) -> None:
        self._log = log_handler or logging.getLogger(__name__)

    def _estimate_tokens(self, llm: Any, messages: Iterable[BaseMessage]) -> int:
        if hasattr(llm, "get_num_tokens_from_messages"):
            try:
                return int(llm.get_num_tokens_from_messages(list(messages)))
            except Exception:  # noqa: BLE001
                pass

        if hasattr(llm, "get_num_tokens"):
            total = 0
            for message in messages:
                serialized = _serialize_message_for_token_count(message)
                try:
                    total += int(llm.get_num_tokens(serialized))
                except Exception:  # noqa: BLE001
                    total += max(1, len(serialized) // 4)
            return total

        approx_chars = sum(len(_serialize_message_for_token_count(message)) for message in messages)
        return max(1, approx_chars // 4)

    def count_tokens(self, llm: Any, content: Union[str, List[BaseMessage]]) -> int:
        try:
            if isinstance(content, str):
                if hasattr(llm, "get_num_tokens"):
                    return int(llm.get_num_tokens(content))
                return max(1, len(content) // 4)

            if isinstance(content, list):
                return self._estimate_tokens(llm, content)

            raise ValueError("Content must be a string or list of LangChain messages")
        except Exception as exc:  # noqa: BLE001
            self._log.error("Token counting failed: %s", exc)
            if isinstance(content, str):
                return max(1, len(content) // 4)
            if isinstance(content, list):
                approx_chars = sum(len(_serialize_message_for_token_count(message)) for message in content)
                return max(1, approx_chars // 4)
            return 0

    def limit_tokens(
        self,
        llm: Any,
        messages: List[BaseMessage],
        max_tokens: int,
        *,
        keep_newest: bool = True,
        preserve_system: bool = True,
    ) -> List[BaseMessage]:
        if not messages:
            return []

        system_entries: List[Tuple[int, BaseMessage]] = []
        conversation_entries: List[Tuple[int, BaseMessage]] = []

        for index, message in enumerate(messages):
            if preserve_system and isinstance(message, SystemMessage):
                system_entries.append((index, message))
            else:
                conversation_entries.append((index, message))

        system_messages = [message for _, message in system_entries]
        system_tokens = self.count_tokens(llm, system_messages) if system_messages else 0

        budget = max_tokens - system_tokens
        if budget <= 0:
            return [message for _, message in sorted(system_entries, key=lambda item: item[0])]

        chosen_conversation: List[Tuple[int, BaseMessage]] = []
        running_total = 0
        ordered_entries = list(reversed(conversation_entries)) if keep_newest else list(conversation_entries)

        for original_index, message in ordered_entries:
            message_tokens = self.count_tokens(llm, [message])
            if running_total + message_tokens <= budget:
                chosen_conversation.append((original_index, message))
                running_total += message_tokens
            else:
                break

        if keep_newest:
            chosen_conversation.reverse()

        merged = system_entries + chosen_conversation
        merged.sort(key=lambda item: item[0])
        return [message for _, message in merged]

    def apply_input_truncation(
        self,
        llm: Any,
        message_groups: List[List[BaseMessage]],
        *,
        max_tokens: int,
        keep_newest: bool = True,
        preserve_system: bool = True,
        verbose: bool = False,
    ) -> List[BaseMessage]:
        merged: List[BaseMessage] = []
        for group in message_groups:
            if group:
                merged.extend(group)

        cleaned = remove_orphaned_tool_results_lc(merged, verbose=verbose)

        trimmed = self.limit_tokens(
            llm=llm,
            messages=cleaned,
            max_tokens=max_tokens,
            keep_newest=keep_newest,
            preserve_system=preserve_system,
        )

        if verbose:
            token_usage = self.count_tokens(llm, trimmed)
            self._log.info(
                "Truncated messages to %d entries (~%d tokens of %d budget)",
                len(trimmed),
                token_usage,
                max_tokens,
            )

        return trimmed

__all__ = ["LangChainTokenLimiter"]
