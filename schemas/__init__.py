"""Schema utilities and tool schemas for UltraGPT."""

from .schema_utils import (
    ensure_openai_strict_compliance,
    normalize_pydantic_optional_fields,
    prepare_schema_for_openai,
)
from .tool_schemas import (
    ExpertTool,
    FunctionCall,
    NativeToolCall,
    NativeToolCallMessage,
    Reasoning,
    Steps,
    ToolAnalysisSchema,
    ToolResult,
    UserTool,
)

__all__ = [
    # Schema normalization utilities
    "normalize_pydantic_optional_fields",
    "ensure_openai_strict_compliance",
    "prepare_schema_for_openai",
    # Tool schemas
    "ExpertTool",
    "FunctionCall",
    "NativeToolCall",
    "NativeToolCallMessage",
    "Reasoning",
    "Steps",
    "ToolAnalysisSchema",
    "ToolResult",
    "UserTool",
]
