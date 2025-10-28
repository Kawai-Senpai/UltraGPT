"""Utilities for manipulating conversation messages using LangChain objects."""

from __future__ import annotations

from typing import List, Union

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


def turnoff_system_message(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Convert all system messages into human messages."""

    lc_messages = list(messages)
    processed: List[BaseMessage] = []

    for message in lc_messages:
        if isinstance(message, SystemMessage):
            processed.append(HumanMessage(content=message.content))
        else:
            processed.append(message)

    return processed


def add_message_before_system(
    messages: List[BaseMessage],
    new_message: BaseMessage,
) -> List[BaseMessage]:
    """Insert ``new_message`` immediately before the first system message."""

    lc_messages = list(messages)
    insert_message = new_message

    processed: List[BaseMessage] = []
    inserted = False

    for message in lc_messages:
        if isinstance(message, SystemMessage) and not inserted:
            processed.append(insert_message)
            inserted = True
        processed.append(message)

    if not inserted:
        processed.append(insert_message)

    return processed


def append_message_to_system(messages: List[BaseMessage], new_message: str) -> List[BaseMessage]:
    """Append ``new_message`` to the existing system prompt or create one."""

    lc_messages = list(messages)
    processed: List[BaseMessage] = []
    has_system = False

    for message in lc_messages:
        if isinstance(message, SystemMessage):
            combined = f"{message.content}\n{new_message}".strip()
            processed.append(SystemMessage(content=combined))
            has_system = True
        else:
            processed.append(message)

    if not has_system:
        processed.insert(0, SystemMessage(content=new_message))

    return processed


def integrate_tool_call_prompt(messages: List[BaseMessage], tool_prompt: str) -> List[BaseMessage]:
    """Integrate tool instructions while preserving original ordering."""

    lc_messages = list(messages)
    conversation_messages: List[BaseMessage] = []
    system_messages: List[SystemMessage] = []

    for message in lc_messages:
        if isinstance(message, SystemMessage):
            system_messages.append(message)
        else:
            conversation_messages.append(message)

    merged: List[BaseMessage] = []
    merged.extend(conversation_messages)

    if system_messages:
        combined = "\n\n".join(msg.content for msg in system_messages)
        final_content = f"{combined}\n\n{tool_prompt}".strip()
        merged.append(SystemMessage(content=final_content))
    else:
        merged.append(SystemMessage(content=tool_prompt))

    return merged


__all__ = [
    "turnoff_system_message",
    "add_message_before_system",
    "append_message_to_system",
    "integrate_tool_call_prompt",
]
