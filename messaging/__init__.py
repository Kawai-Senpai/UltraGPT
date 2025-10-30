"""Messaging utilities and helpers for UltraGPT."""

from .history_utils import (
    concat_messages_safe_lc,
    filter_messages_safe_lc,
    remove_orphaned_tool_results_lc,
    validate_tool_call_pairing_lc,
    drop_unresolved_tool_calls_lc,
)
from .message_ops import (
    add_message_before_system,
    append_message_to_system,
    consolidate_system_messages_safe,
    integrate_tool_call_prompt,
    turnoff_system_message,
)
from .token_manager import ensure_langchain_messages
from .token_limits import LangChainTokenLimiter

remove_orphaned_tool_results = remove_orphaned_tool_results_lc
validate_tool_call_pairing = validate_tool_call_pairing_lc
concat_messages_safe = concat_messages_safe_lc
filter_messages_safe = filter_messages_safe_lc

__all__ = [
    "ensure_langchain_messages",
    "remove_orphaned_tool_results_lc",
    "validate_tool_call_pairing_lc",
    "concat_messages_safe_lc",
    "filter_messages_safe_lc",
    "drop_unresolved_tool_calls_lc",
    "remove_orphaned_tool_results",
    "validate_tool_call_pairing",
    "concat_messages_safe",
    "filter_messages_safe",
    "add_message_before_system",
    "append_message_to_system",
    "integrate_tool_call_prompt",
    "turnoff_system_message",
    "consolidate_system_messages_safe",
    "LangChainTokenLimiter",
]
