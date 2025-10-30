"""Utilities for manipulating conversation messages using LangChain objects.
Policy - keep exactly one SystemMessage and place it as the second last message.
"""

from __future__ import annotations
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


# ---------- helpers ----------

def _split_non_system(messages: List[BaseMessage]) -> tuple[List[BaseMessage], List[SystemMessage]]:
    """Return (non_system_messages_preserving_order, system_messages_preserving_order)."""
    others: List[BaseMessage] = []
    systems: List[SystemMessage] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            systems.append(m)
        else:
            others.append(m)
    return others, systems


def _merge_system_contents(systems: List[SystemMessage]) -> str:
    """Join all system contents with a blank line."""
    parts = [s.content for s in systems if s.content]
    return "\n\n".join(parts).strip()


def _insert_system_at_penultimate(others: List[BaseMessage], system_content: str) -> List[BaseMessage]:
    """Place a single SystemMessage as the second last element, preserving order of others."""
    sys_msg = SystemMessage(content=system_content)
    n = len(others)
    if n == 0:
        # no other messages - just the system
        return [sys_msg]
    # place system right before the last message
    return [*others[:-1], sys_msg, others[-1]]


# ---------- public api ----------

def turnoff_system_message(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Convert all system messages into human messages, preserving order."""
    out: List[BaseMessage] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append(HumanMessage(content=m.content, additional_kwargs=getattr(m, "additional_kwargs", {})))
        else:
            out.append(m)
    return out


def add_message_before_system(
    messages: List[BaseMessage],
    new_message: BaseMessage,
) -> List[BaseMessage]:
    """Insert new_message immediately before the system message.
    - Ensures a single system message ends up second last.
    - If no system exists, just appends new_message to the end.
    - If new_message is a SystemMessage, its content is merged into the system instead of creating a second system.
    """
    others, systems = _split_non_system(messages)
    base = _merge_system_contents(systems)

    # No system present - keep it simple
    if not base:
        return [*others, new_message]

    # Have a system - keep it penultimate and place new_message right before it
    if isinstance(new_message, SystemMessage):
        merged = (base.rstrip() + "\n" + new_message.content) if new_message.content else base
        return _insert_system_at_penultimate(others, merged.strip())

    # Non-system insert right before the penultimate system spot
    # Do it by first adding the new_message to others, then penultimate-insert the system
    others_with_insert = [*others[:-1], new_message, others[-1]] if len(others) >= 1 else [new_message]
    return _insert_system_at_penultimate(others_with_insert, base)


def append_message_to_system(messages: List[BaseMessage], new_message: str) -> List[BaseMessage]:
    """Append text to the system message.
    - Merges all existing system messages into one.
    - Keeps the single system second last.
    - If no system exists, creates one.
    """
    if not new_message or not new_message.strip():
        return list(messages)

    others, systems = _split_non_system(messages)
    base = _merge_system_contents(systems)
    combined = ((base.rstrip() + "\n") if base else "") + new_message
    return _insert_system_at_penultimate(others, combined.strip())


def integrate_tool_call_prompt(messages: List[BaseMessage], tool_prompt: str) -> List[BaseMessage]:
    """Integrate tool instructions into the system message.
    - Merges all system messages + tool_prompt.
    - Keeps the single system second last.
    - If no system exists, creates one with tool_prompt.
    """
    if not tool_prompt or not tool_prompt.strip():
        # nothing to integrate - also normalize to a single system if there are multiple
        others, systems = _split_non_system(messages)
        base = _merge_system_contents(systems)
        return _insert_system_at_penultimate(others, base) if base else list(messages)

    others, systems = _split_non_system(messages)
    base = _merge_system_contents(systems)
    merged = ("\n\n".join([p for p in [base, tool_prompt] if p])).strip()
    return _insert_system_at_penultimate(others, merged)


__all__ = [
    "turnoff_system_message",
    "add_message_before_system",
    "append_message_to_system",
    "integrate_tool_call_prompt",
]
