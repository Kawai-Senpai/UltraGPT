"""Prompt builders for UltraGPT."""

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

# Import schemas from their new location for backward compatibility
from ..schemas import ExpertTool, Reasoning, Steps, UserTool

__all__ = [
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
]
