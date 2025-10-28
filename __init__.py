"""UltraGPT public API exports."""

from .core import ChatFlow, PipelineRunner, UltraGPT
from .messaging import (
    LangChainTokenLimiter,
    add_message_before_system,
    append_message_to_system,
    concat_messages_safe_lc,
    ensure_langchain_messages,
    filter_messages_safe_lc,
    integrate_tool_call_prompt,
    remove_orphaned_tool_results_lc,
    turnoff_system_message,
    validate_tool_call_pairing_lc,
)
from .prompts import (
    combine_all_pipeline_prompts,
    each_step_prompt,
    generate_conclusion_prompt,
    generate_multiple_tool_call_prompt,
    generate_reasoning_prompt,
    generate_single_tool_call_prompt,
    generate_steps_prompt,
    generate_tool_call_prompt,
)
from .schemas import (
    ExpertTool,
    Reasoning,
    Steps,
    UserTool,
    ensure_openai_strict_compliance,
    normalize_pydantic_optional_fields,
    prepare_schema_for_openai,
)
from .providers import (
    BaseProvider,
    ClaudeProvider,
    OpenAIProvider,
    ProviderManager,
    is_rate_limit_error,
)
from .tooling import ToolManager

# Backwards compatible aliases for legacy helper names
remove_orphaned_tool_results = remove_orphaned_tool_results_lc
validate_tool_call_pairing = validate_tool_call_pairing_lc
concat_messages_safe = concat_messages_safe_lc
filter_messages_safe = filter_messages_safe_lc

__all__ = [
    "UltraGPT",
    "ChatFlow",
    "PipelineRunner",
    "ToolManager",
    "BaseProvider",
    "OpenAIProvider",
    "ClaudeProvider",
    "ProviderManager",
    "is_rate_limit_error",
    "LangChainTokenLimiter",
    "ensure_langchain_messages",
    "add_message_before_system",
    "append_message_to_system",
    "integrate_tool_call_prompt",
    "turnoff_system_message",
    "remove_orphaned_tool_results",
    "validate_tool_call_pairing",
    "concat_messages_safe",
    "filter_messages_safe",
    "combine_all_pipeline_prompts",
    "generate_multiple_tool_call_prompt",
    "generate_single_tool_call_prompt",
    "generate_tool_call_prompt",
    "generate_steps_prompt",
    "each_step_prompt",
    "generate_conclusion_prompt",
    "generate_reasoning_prompt",
    "Reasoning",
    "Steps",
    "UserTool",
    "ExpertTool",
    # Schema utilities
    "normalize_pydantic_optional_fields",
    "ensure_openai_strict_compliance",
    "prepare_schema_for_openai",
]
