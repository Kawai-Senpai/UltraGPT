"""Utilities for manipulating conversation messages using LangChain objects.
OPTIMIZED: Just prepend system messages - consolidation happens once at provider level.
"""

from __future__ import annotations

from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


# ---------- consolidation (call once at provider entry) ----------

def consolidate_system_messages_safe(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Consolidate all system messages into one at a safe position.
    Call this ONCE at provider entry point, not in every function.
    
    Finds the safest position that doesn't break tool call/result adjacency.
    Prefers position 0, but can insert elsewhere if needed.
    """
    if not messages:
        return []
    
    system_contents: List[str] = []
    non_system_messages: List[BaseMessage] = []
    
    # Single pass to separate and collect
    for msg in messages:
        if isinstance(msg, SystemMessage) and msg.content:
            system_contents.append(msg.content)
        elif not isinstance(msg, SystemMessage):
            non_system_messages.append(msg)
    
    if not system_contents:
        return non_system_messages
    
    # Find safe insertion position
    insertion_index = _find_safe_system_insert_index(non_system_messages)
    
    consolidated_content = "\n\n".join(system_contents).strip()
    consolidated_system = SystemMessage(content=consolidated_content)
    
    # Insert at safe position
    result = [
        *non_system_messages[:insertion_index],
        consolidated_system,
        *non_system_messages[insertion_index:]
    ]
    
    return result


def _find_safe_system_insert_index(messages: List[BaseMessage]) -> int:
    """Find safe insertion index that preserves tool call/result adjacency.
    
    Returns the best position working backwards from the end:
    - Prefers position 0 if safe
    - Avoids positions immediately after AIMessage with tool_calls
    - Falls back to earliest safe position
    """
    # Build set of disallowed positions (right after tool calls)
    disallowed: set[int] = set()
    for idx, message in enumerate(messages):
        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                # Can't insert right after a tool call
                disallowed.add(idx + 1)
    
    # Try position 0 first (best for OpenAI)
    if 0 not in disallowed:
        return 0
    
    # Work backwards to find safe position
    for candidate in range(len(messages), -1, -1):
        if candidate not in disallowed:
            return candidate
    
    # Fallback to 0 if nothing else works
    return 0


# ---------- public api (fast prepend operations) ----------

def turnoff_system_message(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Convert all system messages into human messages, preserving order."""
    out: List[BaseMessage] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            out.append(
                HumanMessage(content=message.content, additional_kwargs=getattr(message, "additional_kwargs", {}))
            )
        else:
            out.append(message)
    return out


def add_message_before_system(
    messages: List[BaseMessage],
    new_message: BaseMessage,
) -> List[BaseMessage]:
    """Just prepend the new message - no loops, no complex logic.
    Consolidation happens at provider level.
    """
    if isinstance(new_message, SystemMessage):
        return [new_message] + list(messages)
    else:
        return list(messages) + [new_message]


def append_message_to_system(messages: List[BaseMessage], new_message: str) -> List[BaseMessage]:
    """Just prepend a system message - no loops, no searching.
    Consolidation happens at provider level.
    """
    if not new_message or not new_message.strip():
        return list(messages)
    
    new_system = SystemMessage(content=new_message.strip())
    return [new_system] + list(messages)


def integrate_tool_call_prompt(messages: List[BaseMessage], tool_prompt: str) -> List[BaseMessage]:
    """Just prepend the tool prompt as a system message - no loops.
    Consolidation happens at provider level.
    """
    if not tool_prompt or not tool_prompt.strip():
        return list(messages)
    
    tool_system = SystemMessage(content=tool_prompt.strip())
    return [tool_system] + list(messages)


__all__ = [
    "turnoff_system_message",
    "add_message_before_system",
    "append_message_to_system",
    "integrate_tool_call_prompt",
    "consolidate_system_messages_safe",
]
