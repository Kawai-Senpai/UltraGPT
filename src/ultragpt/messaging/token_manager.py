"""Message normalization utilities for LangChain-compatible workflows."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


_TEXT_BLOCK_TYPES = {"text", "input_text", "output_text"}
_MULTIMODAL_BLOCK_TYPES = {
    "image",
    "image_url",
    "input_image",
    "audio",
    "input_audio",
    "video",
    "file",
}


def _extract_text_from_block(segment: Any) -> str:
    if isinstance(segment, str):
        return segment

    if not isinstance(segment, dict):
        return ""

    seg_type = segment.get("type")
    if seg_type in _TEXT_BLOCK_TYPES:
        for key in ("text", "content"):
            value = segment.get(key)
            if value not in (None, ""):
                return str(value)
        return ""

    if "text" in segment and segment.get("text") not in (None, ""):
        return str(segment["text"])

    content = segment.get("content")
    if isinstance(content, str):
        return content

    return ""


def _extract_text(content_val: Any) -> str:
    """Convert mixed content payloads into plain text."""

    if isinstance(content_val, str):
        return content_val

    if isinstance(content_val, dict):
        text = _extract_text_from_block(content_val)
        if text:
            return text
        try:
            return json.dumps(content_val, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            return str(content_val)

    if isinstance(content_val, Iterable) and not isinstance(content_val, (bytes, bytearray)):
        parts = [_extract_text_from_block(segment) for segment in content_val]
        filtered = [part for part in parts if part]
        if filtered:
            return "\n".join(filtered)

    try:
        return json.dumps(content_val, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(content_val)


def _normalize_text_block(segment: Any) -> dict | None:
    text_value = _extract_text_from_block(segment)
    if text_value:
        return {"type": "text", "text": text_value}
    return None


def _normalize_multimodal_block(segment: dict) -> dict | None:
    seg_type = segment.get("type")

    if seg_type in _MULTIMODAL_BLOCK_TYPES:
        return dict(segment)

    if "image_url" in segment:
        normalized = dict(segment)
        normalized.setdefault("type", "image_url")
        return normalized

    if any(key in segment for key in ("url", "base64", "file_id")):
        normalized = dict(segment)
        normalized.setdefault("type", "image")
        return normalized

    return None


def _normalize_user_content(content_val: Any) -> Any:
    """Preserve supported multimodal user blocks while normalizing text-only payloads."""
    if not isinstance(content_val, list):
        return _extract_text(content_val)

    normalized: List[Any] = []
    has_multimodal = False
    for segment in content_val:
        if isinstance(segment, dict):
            seg_type = segment.get("type")
            if seg_type in _TEXT_BLOCK_TYPES:
                text_block = _normalize_text_block(segment)
                if text_block:
                    normalized.append(text_block)
                continue

            media_block = _normalize_multimodal_block(segment)
            if media_block is not None:
                normalized.append(media_block)
                has_multimodal = True
                continue

            text_block = _normalize_text_block(segment)
            if text_block:
                normalized.append(text_block)
        elif isinstance(segment, str) and segment:
            normalized.append({"type": "text", "text": segment})

    if has_multimodal:
        return normalized
    return _extract_text(content_val)


def _build_ai_message(content: Any, message: dict) -> AIMessage:
    text_content = _extract_text(content)
    tool_calls_payload: List[dict] = []
    
    # Build additional_kwargs for extra fields like reasoning_details
    additional_kwargs = {}
    if message.get("reasoning_details"):
        additional_kwargs["reasoning_details"] = message["reasoning_details"]
    if message.get("reasoning"):
        additional_kwargs["reasoning"] = message["reasoning"]
    if message.get("refusal"):
        additional_kwargs["refusal"] = message["refusal"]

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
        return AIMessage(
            content=text_content or "", 
            tool_calls=tool_calls_payload,
            additional_kwargs=additional_kwargs if additional_kwargs else {}
        )

    return AIMessage(
        content=text_content or "",
        additional_kwargs=additional_kwargs if additional_kwargs else {}
    )


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
            normalized.append(HumanMessage(content=_normalize_user_content(content)))
            continue

        if role == "assistant":
            normalized.append(_build_ai_message(content, message))
            continue

        if role in {"tool", "function"}:
            normalized.append(_build_tool_message(content, message))
            continue

        normalized.append(HumanMessage(content=_normalize_user_content(content)))

    return normalized


__all__ = ["ensure_langchain_messages"]
