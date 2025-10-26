"""
History Utilities for UltraGPT

Provides helper functions for cleaning and validating conversation history,
particularly for OpenAI-compatible message formats with tool calling.
"""

from typing import List, Dict, Any, Set
import logging

log = logging.getLogger(__name__)


def remove_orphaned_tool_results(messages: List[Dict[str, Any]], verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Remove orphaned tool result messages (tool messages without matching assistant tool_calls).
    
    OpenAI API requires that every tool message has a matching assistant message with tool_calls
    that contains the same tool_call_id. When concatenating, filtering, or manipulating history,
    tool results can become "orphaned" if their matching assistant tool_call is removed.
    
    This causes OpenAI API 400 errors: "No tool call found for function call output with call_id XXX"
    
    PERFORMANCE: Fast early-exit if no tool messages present. Does not reorder or tamper with messages.
    
    Args:
        messages: List of message dicts in OpenAI format
        verbose: If True, log info about removed orphans
        
    Returns:
        Filtered list with orphaned tool messages removed (or original list if no tool messages)
        
    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hi"},
        ...     {"role": "assistant", "content": "Hello", "tool_calls": [{"id": "call_1", ...}]},
        ...     {"role": "tool", "tool_call_id": "call_1", "name": "search", "content": "..."},
        ...     {"role": "tool", "tool_call_id": "call_2", "name": "calc", "content": "..."}  # ORPHANED!
        ... ]
        >>> clean = remove_orphaned_tool_results(messages)
        >>> # call_2 tool result removed since no matching assistant tool_call exists
    """
    if not messages:
        return messages
    
    # FAST PATH: Early exit if no tool messages present
    # This avoids expensive set building for most calls
    has_tool_messages = any(msg.get("role") == "tool" for msg in messages)
    if not has_tool_messages:
        return messages  # No tool messages = nothing to clean
    
    # Build set of valid tool_call_ids from all assistant messages
    valid_tool_call_ids: Set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg.get("tool_calls", []):
                call_id = tc.get("id")
                if call_id:
                    valid_tool_call_ids.add(call_id)
    
    # Filter out orphaned tool messages (preserves order, does not reorder)
    filtered_messages = []
    orphaned_count = 0
    
    for msg in messages:
        if msg.get("role") == "tool":
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id not in valid_tool_call_ids:
                # Orphaned tool result - exclude it
                orphaned_count += 1
                if verbose:
                    log.warning(
                        f"🗑️  Removed orphaned tool result '{msg.get('name')}' "
                        f"(call_id={tool_call_id}) - no matching assistant tool_call"
                    )
                continue
        
        filtered_messages.append(msg)
    
    if orphaned_count > 0 and verbose:
        log.info(f"✅ Filtered out {orphaned_count} orphaned tool results")
    
    return filtered_messages


def validate_tool_call_pairing(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate that all tool_calls have matching tool results and vice versa.
    
    Returns diagnostic information about missing pairs without modifying the messages.
    
    Args:
        messages: List of message dicts in OpenAI format
        
    Returns:
        Dict with validation results:
        {
            "valid": bool,  # True if all pairs are matched
            "orphaned_tool_results": List[str],  # tool_call_ids without matching tool_call
            "missing_tool_results": List[str],  # tool_call_ids without matching tool result
            "summary": str  # Human-readable summary
        }
    """
    # Collect all tool_call_ids from assistant messages
    expected_tool_call_ids: Set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg.get("tool_calls", []):
                call_id = tc.get("id")
                if call_id:
                    expected_tool_call_ids.add(call_id)
    
    # Collect all tool_call_ids from tool messages
    actual_tool_call_ids: Set[str] = set()
    for msg in messages:
        if msg.get("role") == "tool":
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id:
                actual_tool_call_ids.add(tool_call_id)
    
    # Find mismatches
    orphaned_tool_results = list(actual_tool_call_ids - expected_tool_call_ids)
    missing_tool_results = list(expected_tool_call_ids - actual_tool_call_ids)
    
    valid = len(orphaned_tool_results) == 0 and len(missing_tool_results) == 0
    
    summary_parts = []
    if valid:
        summary_parts.append("✅ All tool calls properly paired with results")
    else:
        if orphaned_tool_results:
            summary_parts.append(f"⚠️  {len(orphaned_tool_results)} orphaned tool results (no matching tool_call)")
        if missing_tool_results:
            summary_parts.append(f"⚠️  {len(missing_tool_results)} tool calls missing results")
    
    return {
        "valid": valid,
        "orphaned_tool_results": orphaned_tool_results,
        "missing_tool_results": missing_tool_results,
        "summary": " | ".join(summary_parts)
    }


def concat_messages_safe(*message_lists: List[Dict[str, Any]], verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Safely concatenate multiple message lists and remove orphaned tool results.
    
    When combining conversation histories from different sources, tool call/result
    pairs can become separated. This function concatenates the lists and cleans up
    any orphaned tool results to ensure OpenAI API compatibility.
    
    Args:
        *message_lists: Variable number of message list arguments
        verbose: If True, log detailed cleaning information
        
    Returns:
        Concatenated and cleaned message list
        
    Example:
        >>> hist1 = [{"role": "user", "content": "Hi"}]
        >>> hist2 = [{"role": "assistant", "content": "Hello"}]
        >>> combined = concat_messages_safe(hist1, hist2)
    """
    # Concatenate all lists
    concatenated = []
    for msg_list in message_lists:
        if msg_list:
            concatenated.extend(msg_list)
    
    # Remove orphaned tool results
    cleaned = remove_orphaned_tool_results(concatenated, verbose=verbose)
    
    return cleaned


def filter_messages_safe(messages: List[Dict[str, Any]], 
                         filter_func: callable, 
                         verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Apply a filter function to messages and clean up orphaned tool results.
    
    When filtering messages (e.g., removing old messages, filtering by role),
    tool call/result pairs can become separated. This function applies the filter
    and cleans up any orphaned tool results.
    
    Args:
        messages: List of message dicts
        filter_func: Function that takes a message dict and returns True to keep it
        verbose: If True, log detailed cleaning information
        
    Returns:
        Filtered and cleaned message list
        
    Example:
        >>> # Remove all user messages older than index 10
        >>> filtered = filter_messages_safe(
        ...     messages,
        ...     lambda msg: msg.get("role") != "user" or messages.index(msg) >= 10
        ... )
    """
    # Apply filter
    filtered = [msg for msg in messages if filter_func(msg)]
    
    # Remove orphaned tool results
    cleaned = remove_orphaned_tool_results(filtered, verbose=verbose)
    
    return cleaned
