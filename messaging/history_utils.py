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


def drop_unresolved_tool_calls_lc(messages: List[BaseMessage], verbose: bool = False) -> List[BaseMessage]:
    """Remove assistant messages that contain tool_calls without matching tool outputs.

    OpenAI Responses API will reject inputs that include an assistant tool call
    without a corresponding tool output message. This sanitizer drops any
    assistant messages whose tool_calls reference call ids that have not yet
    been answered by a ToolMessage in the history.
    """

    if not messages:
        return messages

    diag = validate_tool_call_pairing_lc(messages)
    missing_ids = set(diag.get("missing_tool_results", []) or [])
    if not missing_ids:
        return messages

    cleaned: List[BaseMessage] = []
    dropped = 0
    for message in messages:
        if isinstance(message, AIMessage):
            has_missing = False
            for call in getattr(message, "tool_calls", []) or []:
                call_id = call.get("id")
                if call_id and call_id in missing_ids:
                    has_missing = True
                    break
            if has_missing:
                dropped += 1
                if verbose:
                    log.warning("Dropping assistant message with unresolved tool_calls: %s", message)
                continue
        cleaned.append(message)

    if dropped and verbose:
        log.info("Dropped %d assistant messages with unresolved tool_calls", dropped)

    return cleaned


def group_tool_call_pairs_lc(messages: List[BaseMessage]) -> List[List[BaseMessage]]:
    """Group AIMessage with tool_calls and their corresponding ToolMessage results.
    
    Returns a list of groups where each group is:
    - [AIMessage with tool_calls, ToolMessage, ToolMessage, ...] for tool call sequences
    - [single message] for all other messages
    
    This grouping ensures atomic removal during truncation - if we need to drop
    a tool call, we drop the entire group (call + all results), maintaining
    message integrity for providers like Claude that validate tool pairing.
    """
    if not messages:
        return []
    
    groups: List[List[BaseMessage]] = []
    i = 0
    
    while i < len(messages):
        message = messages[i]
        
        # Check if this is an AIMessage with tool_calls
        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", []) or []
            if tool_calls:
                # Start a new group with this AI message
                group = [message]
                call_ids = {call.get("id") for call in tool_calls if call.get("id")}
                
                # Look ahead for matching ToolMessages
                j = i + 1
                while j < len(messages) and call_ids:
                    next_msg = messages[j]
                    if isinstance(next_msg, ToolMessage):
                        tool_call_id = getattr(next_msg, "tool_call_id", None)
                        if tool_call_id in call_ids:
                            group.append(next_msg)
                            call_ids.discard(tool_call_id)
                            j += 1
                        else:
                            # Tool message doesn't match this group
                            break
                    else:
                        # Non-tool message, stop grouping
                        break
                
                groups.append(group)
                i = j
                continue
        
        # Single message group
        groups.append([message])
        i += 1
    
    return groups


__all__ = [
    "remove_orphaned_tool_results_lc",
    "validate_tool_call_pairing_lc",
    "concat_messages_safe_lc",
    "filter_messages_safe_lc",
    "drop_unresolved_tool_calls_lc",
    "group_tool_call_pairs_lc",
]
