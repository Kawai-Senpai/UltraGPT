"""
LangChain OpenAI patches for OpenRouter compatibility.

This module patches LangChain's OpenAI message conversion functions to preserve
extra fields that OpenRouter and other providers use:

- reasoning_details: Reasoning tokens from thinking models (Gemini 3, Claude reasoning, GPT-5, etc.)
- cache_control: Anthropic/Gemini prompt caching hints
- reasoning: Response reasoning field (alternative to reasoning_details)
- refusal: OpenAI refusal field

The patches are applied at import time and are designed to be:
1. Forward-compatible: Unknown fields in additional_kwargs are passed through
2. Safe: Original functions are called first, then extra fields are added
3. Version-tolerant: Works with different LangChain versions

Usage:
    # Just import this module to apply patches
    import ultragpt.providers._langchain_patches
    
    # Or explicitly apply patches
    from ultragpt.providers._langchain_patches import apply_patches
    apply_patches()

Author: UltraGPT
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, List, Mapping, Set

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration: Fields to preserve across message conversions
# ============================================================================

# Fields to preserve from AIMessage.additional_kwargs when serializing TO API
# These are passed in the assistant message dict
ASSISTANT_EXTRA_FIELDS_TO_SERIALIZE: Set[str] = {
    # OpenRouter reasoning tokens (must be preserved for tool call continuity)
    "reasoning_details",
    # Alternative reasoning field name (some models use this)
    "reasoning",
    # OpenAI refusal field
    "refusal",
    # Provider-specific metadata
    "provider_metadata",
    # Thinking/reasoning content (alternative formats)
    "thinking",
    "thinking_content",
    # Extra content for provider-specific data (e.g., Google's thought_signature wrapper)
    "extra_content",
    # Audio content (for multimodal)
    "audio",
}

# Fields to capture from API response dict INTO AIMessage.additional_kwargs
# These come from the API response and need to be stored
ASSISTANT_FIELDS_TO_CAPTURE: Set[str] = {
    "reasoning_details",
    "reasoning",
    "refusal",
    "thinking",
    "thinking_content",
    "extra_content",
    "audio",
}

# Fields to preserve in message content blocks (for cache_control support)
# These are part of the content array, not top-level message fields
CONTENT_BLOCK_EXTRA_FIELDS: Set[str] = {
    "cache_control",
}

# ============================================================================
# Patch tracking
# ============================================================================

_patches_applied = False


def _sanitize_reasoning_details_for_api(value: Any) -> Any:
    """Prepare reasoning_details for sending back to the API.

    Some reasoning models (GPT-5/o-series, Gemini reasoning) return reasoning_details
    items that include both:
      - an ephemeral server-side `id` (e.g., rs_...)
      - encrypted portable `data`

    If we send `id` back while `store=false`, the server may attempt to look up the
    item and fail with:
      "Item with id 'rs_...' not found. Items are not persisted when `store` is set to false."

    Fix: when `data` is present, strip `id` so the payload is fully portable.
    """

    if value is None:
        return None

    def strip_id_if_data(item: Any) -> Any:
        if not isinstance(item, Mapping):
            return item
        if item.get("data") is None:
            return item
        if "id" not in item:
            return item

        # Only strip truly ephemeral reasoning ids (commonly rs_*).
        # Gemini tool-call continuity may rely on non-rs ids that link encrypted
        # thought_signature blocks to specific tool calls.
        try:
            item_id = item.get("id")
            if isinstance(item_id, str) and not item_id.startswith("rs_"):
                return item
        except Exception:
            # If id is weird/unexpected, keep it rather than risk breaking continuity.
            return item

        copied = dict(item)
        copied.pop("id", None)
        return copied

    if isinstance(value, list):
        return [strip_id_if_data(item) for item in value]

    # Be tolerant: some providers may use a single dict instead of list
    return strip_id_if_data(value)


def _is_patched() -> bool:
    """Check if patches have been applied."""
    return _patches_applied


# ============================================================================
# Patch: _convert_message_to_dict (REQUEST serialization)
# ============================================================================

def _create_patched_convert_message_to_dict(original_func):
    """Create a patched version of _convert_message_to_dict.
    
    This patch adds extra fields from AIMessage.additional_kwargs to the
    serialized message dict, enabling reasoning_details preservation for
    tool calling continuity with reasoning models.
    """
    @wraps(original_func)
    def patched_convert_message_to_dict(message, api="chat/completions"):
        # Call original function first
        result = original_func(message, api)
        
        # For assistant messages, copy extra fields from additional_kwargs
        if result.get("role") == "assistant":
            additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
            
            for field in ASSISTANT_EXTRA_FIELDS_TO_SERIALIZE:
                if field in additional_kwargs and additional_kwargs[field] is not None:
                    # Don't overwrite if already present
                    if field not in result:
                        value = additional_kwargs[field]
                        if field == "reasoning_details":
                            value = _sanitize_reasoning_details_for_api(value)
                        result[field] = value

            # Defensive: if a newer LangChain version already serialized reasoning_details,
            # still sanitize to avoid store=false replay failures.
            if "reasoning_details" in result:
                result["reasoning_details"] = _sanitize_reasoning_details_for_api(result["reasoning_details"])
        
        return result
    
    return patched_convert_message_to_dict


# ============================================================================
# Patch: _convert_dict_to_message (RESPONSE parsing - non-streaming)
# ============================================================================

def _create_patched_convert_dict_to_message(original_func):
    """Create a patched version of _convert_dict_to_message.
    
    This patch captures extra fields from the API response dict and stores
    them in AIMessage.additional_kwargs for later use.
    """
    @wraps(original_func)
    def patched_convert_dict_to_message(_dict: Mapping[str, Any]):
        # Call original function first
        message = original_func(_dict)
        
        # For assistant messages, capture extra fields into additional_kwargs
        role = _dict.get("role")
        if role == "assistant":
            # Get existing additional_kwargs (may already have some fields)
            additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
            modified = False
            
            for field in ASSISTANT_FIELDS_TO_CAPTURE:
                if field in _dict and _dict[field] is not None:
                    # Be tolerant across LangChain versions/providers:
                    # - Some versions may already set these fields (possibly partially)
                    # - We always prefer the actual API response value
                    if additional_kwargs.get(field) != _dict[field]:
                        additional_kwargs[field] = _dict[field]
                        modified = True
            
            # If we added fields, update the message
            if modified:
                # AIMessage.additional_kwargs is mutable, just update it
                if hasattr(message, "additional_kwargs"):
                    message.additional_kwargs.update(additional_kwargs)
        
        return message
    
    return patched_convert_dict_to_message


# ============================================================================
# Patch: _convert_delta_to_message_chunk (RESPONSE parsing - streaming)
# ============================================================================

def _create_patched_convert_delta_to_message_chunk(original_func):
    """Create a patched version of _convert_delta_to_message_chunk.
    
    This patch captures extra fields from streaming delta chunks and stores
    them in AIMessageChunk.additional_kwargs for later accumulation.
    """
    @wraps(original_func)
    def patched_convert_delta_to_message_chunk(_dict: Mapping[str, Any], default_class):
        # Call original function first
        message_chunk = original_func(_dict, default_class)
        
        # For assistant chunks, capture extra fields
        role = _dict.get("role")
        # Also check if it's an AI chunk by class name (role may be None in deltas)
        is_assistant = role == "assistant" or "AIMessage" in message_chunk.__class__.__name__
        
        if is_assistant:
            additional_kwargs = getattr(message_chunk, "additional_kwargs", {}) or {}
            modified = False
            
            for field in ASSISTANT_FIELDS_TO_CAPTURE:
                if field in _dict and _dict[field] is not None:
                    if additional_kwargs.get(field) != _dict[field]:
                        additional_kwargs[field] = _dict[field]
                        modified = True
            
            if modified and hasattr(message_chunk, "additional_kwargs"):
                message_chunk.additional_kwargs.update(additional_kwargs)
        
        return message_chunk
    
    return patched_convert_delta_to_message_chunk


# ============================================================================
# Apply patches
# ============================================================================

def apply_patches() -> bool:
    """Apply all LangChain patches for OpenRouter compatibility.
    
    Returns:
        True if patches were applied successfully, False if already applied
        or if patching failed.
    """
    global _patches_applied
    
    if _patches_applied:
        logger.debug("LangChain patches already applied, skipping")
        return False
    
    try:
        from langchain_openai.chat_models import base as lc_base
        
        # Patch _convert_message_to_dict (request serialization)
        if hasattr(lc_base, "_convert_message_to_dict"):
            original_to_dict = lc_base._convert_message_to_dict
            lc_base._convert_message_to_dict = _create_patched_convert_message_to_dict(original_to_dict)
            logger.debug("Patched _convert_message_to_dict")
        else:
            logger.warning("_convert_message_to_dict not found in langchain_openai.chat_models.base")
        
        # Patch _convert_dict_to_message (response parsing - non-streaming)
        if hasattr(lc_base, "_convert_dict_to_message"):
            original_from_dict = lc_base._convert_dict_to_message
            lc_base._convert_dict_to_message = _create_patched_convert_dict_to_message(original_from_dict)
            logger.debug("Patched _convert_dict_to_message")
        else:
            logger.warning("_convert_dict_to_message not found in langchain_openai.chat_models.base")
        
        # Patch _convert_delta_to_message_chunk (response parsing - streaming)
        if hasattr(lc_base, "_convert_delta_to_message_chunk"):
            original_delta = lc_base._convert_delta_to_message_chunk
            lc_base._convert_delta_to_message_chunk = _create_patched_convert_delta_to_message_chunk(original_delta)
            logger.debug("Patched _convert_delta_to_message_chunk")
        else:
            logger.warning("_convert_delta_to_message_chunk not found in langchain_openai.chat_models.base")
        
        _patches_applied = True
        logger.info("LangChain OpenRouter compatibility patches applied successfully")
        return True
        
    except ImportError as e:
        logger.warning("Could not import langchain_openai.chat_models.base: %s", e)
        return False
    except Exception as e:
        logger.error("Failed to apply LangChain patches: %s", e, exc_info=True)
        return False


def add_extra_field_to_serialize(field_name: str) -> None:
    """Add a new field to be serialized from additional_kwargs to API dict.
    
    Use this to extend the list of fields that are preserved when sending
    messages to the API.
    
    Args:
        field_name: The field name to add (e.g., "custom_metadata")
    """
    ASSISTANT_EXTRA_FIELDS_TO_SERIALIZE.add(field_name)


def add_extra_field_to_capture(field_name: str) -> None:
    """Add a new field to be captured from API response to additional_kwargs.
    
    Use this to extend the list of fields that are captured from API responses
    into LangChain message objects.
    
    Args:
        field_name: The field name to add (e.g., "custom_response_field")
    """
    ASSISTANT_FIELDS_TO_CAPTURE.add(field_name)


# ============================================================================
# Auto-apply patches on import
# ============================================================================

apply_patches()


__all__ = [
    "apply_patches",
    "add_extra_field_to_serialize",
    "add_extra_field_to_capture",
    "ASSISTANT_EXTRA_FIELDS_TO_SERIALIZE",
    "ASSISTANT_FIELDS_TO_CAPTURE",
    "CONTENT_BLOCK_EXTRA_FIELDS",
]
