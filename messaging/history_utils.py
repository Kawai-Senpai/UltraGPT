"""History utilities implemented with LangChain message objects."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core LangChain-aware operations
# ---------------------------------------------------------------------------

def remove_orphaned_tool_results_lc(messages: List[BaseMessage], verbose: bool = False) -> List[BaseMessage]:
    """Remove tool results that do not match any AI tool call id."""

    if not messages:
        return messages

    valid_ids = set()
    for message in messages:
        if isinstance(message, AIMessage):
            for call in getattr(message, "tool_calls", []) or []:
                call_id = call.get("id")
                if call_id:
                    valid_ids.add(call_id)

    cleaned: List[BaseMessage] = []
    orphaned_count = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            call_id = getattr(message, "tool_call_id", None)
            if call_id not in valid_ids:
                orphaned_count += 1
                if verbose:
                    log.warning("Removed orphaned tool result with call_id=%s", call_id)
                continue
        cleaned.append(message)

    if orphaned_count and verbose:
        log.info("Filtered out %d orphaned tool results", orphaned_count)

    return cleaned


def validate_tool_call_pairing_lc(messages: List[BaseMessage]) -> Dict[str, Any]:
    """Return diagnostics describing any tool call/result mismatches."""

    expected = set()
    actual = set()

    for message in messages:
        if isinstance(message, AIMessage):
            for call in getattr(message, "tool_calls", []) or []:
                call_id = call.get("id")
                if call_id:
                    expected.add(call_id)
        elif isinstance(message, ToolMessage):
            call_id = getattr(message, "tool_call_id", None)
            if call_id:
                actual.add(call_id)

    orphaned = list(actual - expected)
    missing = list(expected - actual)
    valid = not orphaned and not missing

    summary_bits = []
    if valid:
        summary_bits.append("All tool calls are paired with results")
    else:
        if orphaned:
            summary_bits.append(f"{len(orphaned)} orphaned tool results")
        if missing:
            summary_bits.append(f"{len(missing)} tool calls missing results")

    return {
        "valid": valid,
        "orphaned_tool_results": orphaned,
        "missing_tool_results": missing,
        "summary": " | ".join(summary_bits) if summary_bits else "",
    }


def concat_messages_safe_lc(*message_lists: List[BaseMessage], verbose: bool = False) -> List[BaseMessage]:
    """Concatenate LangChain messages and drop orphaned tool results."""

    merged: List[BaseMessage] = []
    for batch in message_lists:
        if batch:
            merged.extend(batch)
    return remove_orphaned_tool_results_lc(merged, verbose=verbose)


def filter_messages_safe_lc(messages: List[BaseMessage], filter_func: Callable[[BaseMessage], bool], verbose: bool = False) -> List[BaseMessage]:
    """Filter LangChain messages and drop orphaned tool results."""

    filtered = [message for message in messages if filter_func(message)]
    return remove_orphaned_tool_results_lc(filtered, verbose=verbose)

__all__ = [
    "remove_orphaned_tool_results_lc",
    "validate_tool_call_pairing_lc",
    "concat_messages_safe_lc",
    "filter_messages_safe_lc",
]
