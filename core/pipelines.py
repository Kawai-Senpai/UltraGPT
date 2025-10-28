"""Reasoning and steps pipelines coordinating agent workflows."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import BaseMessage, SystemMessage

from ..messaging import turnoff_system_message
from ..prompts import (
    Reasoning,
    Steps,
    combine_all_pipeline_prompts,
    each_step_prompt,
    generate_conclusion_prompt,
    generate_reasoning_prompt,
    generate_steps_prompt,
)
from .chat_flow import ChatFlow


class PipelineRunner:
    """Pipeline operations such as reasoning and task steps."""

    def __init__(
        self,
        chat_flow: ChatFlow,
        *,
        log,
        verbose: bool,
    ) -> None:
        self._chat = chat_flow
        self._log = log
        self._verbose = verbose

    def run_steps_pipeline(
        self,
        messages: List[BaseMessage],
        model: str,
        temperature: float,
        tools: List[Any],
        tools_config: Dict[str, Any],
        *,
        steps_model: Optional[str],
        max_tokens: Optional[int],
        input_truncation: Optional[int],
        deepthink: Optional[bool],
    ) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
        active_model = steps_model if steps_model else model

        if self._verbose:
            self._log.debug("➤ Starting Steps Pipeline")
            if steps_model:
                self._log.debug("Using steps model: %s", steps_model)
        else:
            self._log.info("Starting steps pipeline")

        total_tokens = 0
        all_tools_used: List[Dict[str, Any]] = []

        messages_no_system = turnoff_system_message(messages)
        steps_generator_message = messages_no_system + [SystemMessage(content=generate_steps_prompt())]

        steps_json, tokens, steps_details = self._chat.chat_with_model_parse(
            steps_generator_message,
            schema=Steps,
            model=active_model,
            temperature=temperature,
            tools=tools,
            tools_config=tools_config,
            max_tokens=max_tokens,
            input_truncation=input_truncation,
            deepthink=deepthink,
        )
        total_tokens += tokens
        all_tools_used.extend(steps_details.get("tools_used", []))

        steps = steps_json.get("steps", [])
        if self._verbose:
            self._log.debug("Generated %d steps", len(steps))
            for idx, step in enumerate(steps, 1):
                self._log.debug("  %d. %s", idx, step)

        memory: List[Dict[str, Any]] = []

        for idx, step in enumerate(steps, 1):
            if self._verbose:
                self._log.debug("Processing step %d/%d", idx, len(steps))
            step_prompt = each_step_prompt(memory, step)
            step_message = messages_no_system + [SystemMessage(content=step_prompt)]
            step_response, tokens, step_details = self._chat.chat_with_ai_sync(
                step_message,
                model=active_model,
                temperature=temperature,
                tools=tools,
                tools_config=tools_config,
                max_tokens=max_tokens,
                input_truncation=input_truncation,
                deepthink=deepthink,
            )
            if self._verbose:
                self._log.debug("Step %d response preview: %s", idx, step_response[:100])
            total_tokens += tokens
            all_tools_used.extend(step_details.get("tools_used", []))
            memory.append({"step": step, "answer": step_response})

        conclusion_prompt = generate_conclusion_prompt(memory)
        conclusion_message = messages_no_system + [SystemMessage(content=conclusion_prompt)]
        conclusion, tokens, conclusion_details = self._chat.chat_with_ai_sync(
            conclusion_message,
            model=active_model,
            temperature=temperature,
            tools=tools,
            tools_config=tools_config,
            max_tokens=max_tokens,
            input_truncation=input_truncation,
            deepthink=deepthink,
        )
        total_tokens += tokens
        all_tools_used.extend(conclusion_details.get("tools_used", []))

        if self._verbose:
            self._log.debug("✓ Steps pipeline completed")

        return {"steps": memory, "conclusion": conclusion}, total_tokens, {"tools_used": all_tools_used}

    def run_reasoning_pipeline(
        self,
        messages: List[BaseMessage],
        model: str,
        temperature: float,
        reasoning_iterations: int,
        tools: List[Any],
        tools_config: Dict[str, Any],
        *,
        reasoning_model: Optional[str],
        max_tokens: Optional[int],
        input_truncation: Optional[int],
        deepthink: Optional[bool],
    ) -> Tuple[List[str], int, Dict[str, Any]]:
        active_model = reasoning_model if reasoning_model else model

        if self._verbose:
            self._log.debug(
                "➤ Starting Reasoning Pipeline (%d iterations)",
                reasoning_iterations,
            )
            if reasoning_model:
                self._log.debug("Using reasoning model: %s", reasoning_model)
        else:
            self._log.info("Starting reasoning pipeline (%d iterations)", reasoning_iterations)

        total_tokens = 0
        all_thoughts: List[str] = []
        all_tools_used: List[Dict[str, Any]] = []
        messages_no_system = turnoff_system_message(messages)

        for iteration in range(reasoning_iterations):
            if self._verbose:
                self._log.debug("Iteration %d/%d", iteration + 1, reasoning_iterations)
            reasoning_message = messages_no_system + [
                SystemMessage(content=generate_reasoning_prompt(all_thoughts))
            ]

            reasoning_json, tokens, iteration_details = self._chat.chat_with_model_parse(
                reasoning_message,
                schema=Reasoning,
                model=active_model,
                temperature=temperature,
                tools=tools,
                tools_config=tools_config,
                max_tokens=max_tokens,
                input_truncation=input_truncation,
                deepthink=deepthink,
            )
            total_tokens += tokens
            all_tools_used.extend(iteration_details.get("tools_used", []))

            new_thoughts = reasoning_json.get("thoughts", [])
            all_thoughts.extend(new_thoughts)

            if self._verbose:
                self._log.debug("Generated %d new thoughts", len(new_thoughts))

        return all_thoughts, total_tokens, {"tools_used": all_tools_used}

    def combine_pipeline_outputs(
        self,
        reasoning_output: List[str],
        steps_output: Dict[str, Any],
    ) -> Optional[str]:
        conclusion = steps_output.get("conclusion", "")
        if not reasoning_output and not conclusion:
            return None
        return combine_all_pipeline_prompts(reasoning_output, conclusion)


__all__ = ["PipelineRunner"]
