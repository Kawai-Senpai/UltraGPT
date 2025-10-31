"""LangChain token limiting utilities."""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, List, Optional, Tuple, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from ..history_utils import remove_orphaned_tool_results_lc, group_tool_call_pairs_lc


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
        """Limit messages to max_tokens budget by removing groups atomically.
        
        Groups tool call pairs (AIMessage + ToolMessages) for atomic removal,
        preventing orphaned tool results and avoiding Claude count_tokens errors.
        """
        if not messages:
            return []

        # Separate system messages if preserving
        system_entries: List[Tuple[int, BaseMessage]] = []
        conversation_messages: List[BaseMessage] = []

        for index, message in enumerate(messages):
            if preserve_system and isinstance(message, SystemMessage):
                system_entries.append((index, message))
            else:
                conversation_messages.append(message)

        # Count system message tokens
        system_messages = [message for _, message in system_entries]
        system_tokens = self.count_tokens(llm, system_messages) if system_messages else 0

        budget = max_tokens - system_tokens
        if budget <= 0:
            return [message for _, message in sorted(system_entries, key=lambda item: item[0])]

        # Group conversation messages (tool calls + results stay together)
        message_groups = group_tool_call_pairs_lc(conversation_messages)
        
        # Track original indices for each group
        current_index = 0
        indexed_groups: List[Tuple[int, List[BaseMessage]]] = []
        for group in message_groups:
            indexed_groups.append((current_index, group))
            current_index += len(group)

        # Order groups based on keep_newest
        ordered_groups = list(reversed(indexed_groups)) if keep_newest else list(indexed_groups)

        # Select groups that fit within budget
        chosen_groups: List[Tuple[int, List[BaseMessage]]] = []
        running_total = 0

        for original_index, group in ordered_groups:
            # Count tokens for entire group
            group_tokens = self.count_tokens(llm, group)
            if running_total + group_tokens <= budget:
                chosen_groups.append((original_index, group))
                running_total += group_tokens
            else:
                # Group doesn't fit, skip it entirely
                continue

        # EDGE CASE: Ensure at least one non-system message remains
        # Claude and other providers require at least one conversation message
        # AND it should be a HumanMessage for valid conversation structure
        if not chosen_groups and ordered_groups:
            # Find the most recent HumanMessage group
            human_group = None
            if keep_newest:
                # Search from newest (start of ordered_groups after reversal)
                for idx, group in enumerate(ordered_groups):
                    if any(isinstance(msg, HumanMessage) for msg in group):
                        human_group = ordered_groups[idx]
                        break
            else:
                # Search from oldest (end of ordered_groups)
                for idx in range(len(ordered_groups) - 1, -1, -1):
                    if any(isinstance(msg, HumanMessage) for msg in ordered_groups[idx]):
                        human_group = ordered_groups[idx]
                        break
            
            # If no HumanMessage found, take any non-empty group as fallback
            chosen_groups = [human_group] if human_group else [ordered_groups[0]] if keep_newest else [ordered_groups[-1]]
            
            if self._log:
                self._log.warning(
                    "Token budget too small - kept at least one message group to satisfy provider requirements"
                )

        # Restore chronological order if we reversed
        if keep_newest:
            chosen_groups.reverse()

        # Flatten groups back to messages
        chosen_messages: List[Tuple[int, BaseMessage]] = []
        for group_index, group in chosen_groups:
            for offset, message in enumerate(group):
                chosen_messages.append((group_index + offset, message))

        # Merge with system messages and restore original order
        all_messages = system_entries + chosen_messages
        all_messages.sort(key=lambda item: item[0])
        
        return [message for _, message in all_messages]

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
        """Apply token truncation with tool-call-aware grouping.
        
        Cleans orphans BEFORE truncation, then uses atomic group removal.
        No need to clean orphans AFTER since grouping prevents them.
        """
        merged: List[BaseMessage] = []
        for group in message_groups:
            if group:
                merged.extend(group)

        # Clean orphans BEFORE truncation (in case input is already malformed)
        cleaned = remove_orphaned_tool_results_lc(merged, verbose=verbose)

        # Truncate using grouped logic (atomic removal of tool call pairs)
        trimmed = self.limit_tokens(
            llm=llm,
            messages=cleaned,
            max_tokens=max_tokens,
            keep_newest=keep_newest,
            preserve_system=preserve_system,
        )

        # No need to clean orphans AFTER - grouping prevents them by design

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
