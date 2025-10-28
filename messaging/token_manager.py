"""Message normalization utilities for LangChain-compatible workflows."""

from __future__ import annotations

import json
from typing import Any, Iterable, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


def _extract_text(content_val: Any) -> str:
    """Convert mixed content payloads into plain text."""

    if isinstance(content_val, str):
        return content_val

    if isinstance(content_val, Iterable) and not isinstance(content_val, (bytes, bytearray)):
        parts: List[str] = []
        for segment in content_val:
            if isinstance(segment, dict):
                seg_type = segment.get("type")
                if seg_type in {"text", "output_text"}:
                    text_value = segment.get("text", "")
                    if text_value:
                        parts.append(text_value)
            elif isinstance(segment, str):
                parts.append(segment)
        filtered = [part for part in parts if part]
        if filtered:
            return "\n".join(filtered)

    try:
        return json.dumps(content_val)
    except Exception:  # noqa: BLE001
        return str(content_val)


def _build_ai_message(content: Any, message: dict) -> AIMessage:
    text_content = _extract_text(content)
    tool_calls_payload: List[dict] = []

    for call in message.get("tool_calls") or []:
        call_id = call.get("id") or call.get("tool_call_id")
        fn_block = call.get("function", {}) or {}
        fn_name = fn_block.get("name")
        raw_args = fn_block.get("arguments")

        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args)
            except Exception:  # noqa: BLE001
                parsed_args = {"__raw": raw_args}
        else:
            parsed_args = raw_args

        if fn_name:
            tool_calls_payload.append(
                {
                    "id": call_id,
                    "name": fn_name,
                    "args": parsed_args or {},
                    "type": "tool_call",
                }
            )

    if tool_calls_payload:
        return AIMessage(content=text_content or "", tool_calls=tool_calls_payload)

    return AIMessage(content=text_content or "")


def _build_tool_message(content: Any, message: dict) -> ToolMessage:
    call_id = (
        message.get("tool_call_id")
        or message.get("call_id")
        or message.get("id")
        or message.get("name")
    )
    tool_name = message.get("name") or message.get("tool_name") or "tool"
    payload = content
    if not isinstance(payload, str):
        try:
            payload = json.dumps(payload)
        except Exception:  # noqa: BLE001
            payload = str(payload)

    return ToolMessage(content=payload or "", tool_call_id=call_id, name=tool_name)


def ensure_langchain_messages(messages: List[Any]) -> List[BaseMessage]:
    """Return a list of LangChain messages regardless of input format."""

    if not messages:
        return []

    if all(isinstance(message, BaseMessage) for message in messages):
        return [message for message in messages]

    normalized: List[BaseMessage] = []

    for message in messages:
        if isinstance(message, BaseMessage):
            normalized.append(message)
            continue

        role = message.get("role")
        content = message.get("content", "")

        if role in {"system", "developer"}:
            normalized.append(SystemMessage(content=_extract_text(content)))
            continue

        if role == "user":
            normalized.append(HumanMessage(content=_extract_text(content)))
            continue

        if role == "assistant":
            normalized.append(_build_ai_message(content, message))
            continue

        if role in {"tool", "function"}:
            normalized.append(_build_tool_message(content, message))
            continue

        normalized.append(HumanMessage(content=_extract_text(content)))

    return normalized


__all__ = ["ensure_langchain_messages"]
