"""Reasoning and steps pipelines coordinating agent workflows."""

from __future__ import annotations

import re

from typing import Any, Dict, List, Optional, Set, Tuple

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


def _canonicalize_thought(text: Any) -> Tuple[str, str]:
    """Return canonical key and cleaned text for deduping reasoning pipeline thoughts."""

    raw = str(text).strip()
    if not raw:
        return "", ""

    # Remove common numbering prefixes like "Thought 1:" or "1." while keeping intent.
    stripped = re.sub(r"^thought\s*\d+\s*[:.\-\)\]]\s*", "", raw, flags=re.IGNORECASE)
    stripped = re.sub(r"^[\-\*\(\)\[\]\d\.]+\s*", "", stripped)
    condensed = re.sub(r"\s+", " ", stripped).strip()
    if not condensed:
        return "", ""

    canonical = re.sub(r"[\s.:;\-]+$", "", condensed.lower())
    return canonical, condensed


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
        aggregate_input_tokens = 0
        aggregate_output_tokens = 0
        aggregate_reasoning_tokens_api = 0
        reasoning_text_snapshots: List[str] = []
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
        step_input_tokens = int(steps_details.get("input_tokens", 0))
        step_output_tokens = int(steps_details.get("output_tokens", 0))
        step_total_tokens = int(steps_details.get("total_tokens", 0))

        if step_total_tokens <= 0:
            step_total_tokens = step_input_tokens + step_output_tokens
        if step_total_tokens <= 0:
            step_total_tokens = int(tokens)

        aggregate_input_tokens += step_input_tokens
        aggregate_output_tokens += step_output_tokens
        aggregate_reasoning_tokens_api += int(steps_details.get("reasoning_tokens", 0))
        if steps_details.get("reasoning_text"):
            reasoning_text_snapshots.append(str(steps_details.get("reasoning_text")))

        total_tokens += step_total_tokens
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
            step_iter_input = int(step_details.get("input_tokens", 0))
            step_iter_output = int(step_details.get("output_tokens", 0))
            step_iter_total = int(step_details.get("total_tokens", 0))

            if step_iter_total <= 0:
                step_iter_total = step_iter_input + step_iter_output
            if step_iter_total <= 0:
                step_iter_total = int(tokens)

            aggregate_input_tokens += step_iter_input
            aggregate_output_tokens += step_iter_output
            aggregate_reasoning_tokens_api += int(step_details.get("reasoning_tokens", 0))
            if step_details.get("reasoning_text"):
                reasoning_text_snapshots.append(str(step_details.get("reasoning_text")))

            total_tokens += step_iter_total
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
        conclusion_input = int(conclusion_details.get("input_tokens", 0))
        conclusion_output = int(conclusion_details.get("output_tokens", 0))
        conclusion_total = int(conclusion_details.get("total_tokens", 0))

        if conclusion_total <= 0:
            conclusion_total = conclusion_input + conclusion_output
        if conclusion_total <= 0:
            conclusion_total = int(tokens)

        aggregate_input_tokens += conclusion_input
        aggregate_output_tokens += conclusion_output
        aggregate_reasoning_tokens_api += int(conclusion_details.get("reasoning_tokens", 0))
        if conclusion_details.get("reasoning_text"):
            reasoning_text_snapshots.append(str(conclusion_details.get("reasoning_text")))

        total_tokens += conclusion_total
        all_tools_used.extend(conclusion_details.get("tools_used", []))

        if self._verbose:
            self._log.debug("✓ Steps pipeline completed")

        usage_summary: Dict[str, Any] = {
            "input_tokens": aggregate_input_tokens,
            "output_tokens": aggregate_output_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens_api": aggregate_reasoning_tokens_api,
        }

        if reasoning_text_snapshots:
            usage_summary["reasoning_text"] = reasoning_text_snapshots[-1]
            usage_summary["reasoning_texts"] = reasoning_text_snapshots

        return {"steps": memory, "conclusion": conclusion}, total_tokens, {
            "tools_used": all_tools_used,
            "usage": usage_summary,
        }

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
        seen_thoughts: Set[str] = set()
        all_tools_used: List[Dict[str, Any]] = []
        aggregate_input_tokens = 0
        aggregate_output_tokens = 0
        aggregate_reasoning_tokens_api = 0
        reasoning_text_snapshots: List[str] = []
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
            iteration_input_tokens = int(iteration_details.get("input_tokens", 0))
            iteration_output_tokens = int(iteration_details.get("output_tokens", 0))
            iteration_total_tokens = int(iteration_details.get("total_tokens", 0))

            if iteration_total_tokens <= 0:
                iteration_total_tokens = iteration_input_tokens + iteration_output_tokens
            if iteration_total_tokens <= 0:
                iteration_total_tokens = int(tokens)

            aggregate_input_tokens += iteration_input_tokens
            aggregate_output_tokens += iteration_output_tokens
            aggregate_reasoning_tokens_api += int(iteration_details.get("reasoning_tokens", 0))
            if iteration_details.get("reasoning_text"):
                reasoning_text_snapshots.append(str(iteration_details.get("reasoning_text")))

            total_tokens += iteration_total_tokens
            all_tools_used.extend(iteration_details.get("tools_used", []))

            raw_thoughts = reasoning_json.get("thoughts", []) or []
            sanitized_payload: List[str] = []
            base_index = len(all_thoughts)

            for thought in raw_thoughts:
                canonical_key, cleaned_text = _canonicalize_thought(thought)
                if not canonical_key or canonical_key in seen_thoughts:
                    continue
                seen_thoughts.add(canonical_key)
                sanitized_payload.append(cleaned_text)

            new_thoughts_count = 0
            if sanitized_payload:
                numbered_thoughts = [
                    f"Thought {base_index + idx + 1}: {text}"
                    for idx, text in enumerate(sanitized_payload)
                ]
                new_thoughts_count = len(numbered_thoughts)
                all_thoughts.extend(numbered_thoughts)

            if self._verbose:
                self._log.debug("Generated %d new thoughts", new_thoughts_count)

        usage_summary: Dict[str, Any] = {
            "input_tokens": aggregate_input_tokens,
            "output_tokens": aggregate_output_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens_api": aggregate_reasoning_tokens_api,
        }

        if reasoning_text_snapshots:
            usage_summary["reasoning_text"] = reasoning_text_snapshots[-1]
            usage_summary["reasoning_texts"] = reasoning_text_snapshots

        return all_thoughts, total_tokens, {
            "tools_used": all_tools_used,
            "usage": usage_summary,
        }

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
