"""Primary UltraGPT orchestrator built on modular helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.messages import BaseMessage, HumanMessage
from ultraprint.logging import logger

from .. import config
from ..messaging import (
    add_message_before_system,
    append_message_to_system,
    ensure_langchain_messages,
    integrate_tool_call_prompt,
    LangChainTokenLimiter,
)
from ..prompts import (
    combine_all_pipeline_prompts,
    generate_multiple_tool_call_prompt,
    generate_single_tool_call_prompt,
)
from ..providers import ClaudeProvider, OpenAIProvider, ProviderManager
from ..tooling import ToolManager
from ..tools.web_search.core import google_search, scrape_url
from .chat_flow import ChatFlow
from .pipelines import PipelineRunner


class UltraGPT:
    """High-level façade coordinating providers, tools, and pipelines."""

    def __init__(
        self,
        api_key: str = None,
        openai_api_key: str = None,
        claude_api_key: str = None,
    provider: str = None,
        google_api_key: str = None,
        search_engine_id: str = None,
        max_tokens: Optional[int] = None,
        input_truncation: Union[str, int] = None,
        verbose: bool = False,
        logger_name: str = "ultragpt",
        logger_filename: str = "debug/ultragpt.log",
        log_extra_info: bool = False,
        log_to_file: bool = False,
        log_to_console: bool = False,
        log_level: str = "DEBUG",
    ) -> None:
        if openai_api_key and api_key and openai_api_key != api_key:
            raise ValueError("Provide either api_key or openai_api_key, not both")

        # Backwards compatibility: some callers used provider="openai"/"anthropic"
        if provider and provider not in {"openai", "anthropic", "claude"}:
            raise ValueError("provider must be 'openai' or 'anthropic'")

        primary_api_key = openai_api_key or api_key

        # If provider is explicitly set, only enable that provider
        if provider == "openai":
            claude_api_key = None
        elif provider in {"anthropic", "claude"}:
            primary_api_key = None

        self.verbose = verbose
        self.google_api_key = google_api_key
        self.search_engine_id = search_engine_id
        self.max_tokens = max_tokens
        self.input_truncation = input_truncation if input_truncation is not None else config.DEFAULT_INPUT_TRUNCATION

        self.log = logger(
            name=logger_name,
            filename=logger_filename,
            include_extra_info=log_extra_info,
            write_to_file=log_to_file,
            log_level=log_level,
            log_to_console=True if verbose else log_to_console,
        )

        self.log.info("Initializing UltraGPT")
        if self.verbose:
            self.log.debug("=" * 50)
            self.log.debug("Initializing UltraGPT")
            self.log.debug("=" * 50)

        self.token_limiter = LangChainTokenLimiter(log_handler=self.log)
        self.provider_manager = ProviderManager(
            token_limiter=self.token_limiter,
            default_input_truncation=self.input_truncation,
            logger=self.log,
            verbose=self.verbose,
        )

        if primary_api_key:
            openai_provider = OpenAIProvider(api_key=primary_api_key)
            self.provider_manager.add_provider("openai", openai_provider)

        if claude_api_key:
            try:
                claude_provider = ClaudeProvider(api_key=claude_api_key)
                self.provider_manager.add_provider("claude", claude_provider)
            except ImportError as exc:
                if self.verbose:
                    self.log.warning("Claude provider not available: %s", exc)

        if not self.provider_manager.providers:
            raise ValueError("At least one API key must be provided for the selected provider(s)")

        self.tool_manager = ToolManager(self)
        self.chat_flow = ChatFlow(
            provider_manager=self.provider_manager,
            tool_manager=self.tool_manager,
            log=self.log,
            verbose=self.verbose,
            max_tokens=self.max_tokens,
        )
        self.pipeline_runner = PipelineRunner(
            chat_flow=self.chat_flow,
            log=self.log,
            verbose=self.verbose,
        )


    @staticmethod
    def _ensure_lc_messages(messages: List[Any]) -> List[BaseMessage]:
        return ensure_langchain_messages(messages)

    # ------------------------------------------------------------------
    # Core chat entry points
    # ------------------------------------------------------------------

    def chat_with_ai_sync(
        self,
        messages: list,
        model: str,
        temperature: float,
        tools: list,
        tools_config: dict,
        max_tokens: Optional[int] = None,
        input_truncation: Optional[Union[str, int]] = None,
        deepthink: Optional[bool] = None,
    ) -> Tuple[str, int, Dict[str, Any]]:
        tools = tools or []
        tools_config = tools_config or {}
        return self.chat_flow.chat_with_ai_sync(
            messages,
            model=model,
            temperature=temperature,
            tools=tools,
            tools_config=tools_config,
            max_tokens=max_tokens,
            input_truncation=input_truncation,
            deepthink=deepthink,
        )

    def chat_with_model_parse(
        self,
        messages: list,
        schema=None,
        model: str = None,
        temperature: float = None,
        tools: list = None,
        tools_config: dict = None,
        max_tokens: Optional[int] = None,
        input_truncation: Optional[Union[str, int]] = None,
        deepthink: Optional[bool] = None,
    ) -> Tuple[Any, int, Dict[str, Any]]:
        model = model or config.DEFAULT_PARSE_MODEL
        temperature = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
        tools = tools or []
        tools_config = tools_config or {}
        return self.chat_flow.chat_with_model_parse(
            messages,
            schema=schema,
            model=model,
            temperature=temperature,
            tools=tools,
            tools_config=tools_config,
            max_tokens=max_tokens,
            input_truncation=input_truncation,
            deepthink=deepthink,
        )

    def chat_with_model_tools(
        self,
        messages: list,
        user_tools: list,
        model: str = None,
        temperature: float = None,
        tools: list = None,
        tools_config: dict = None,
        max_tokens: Optional[int] = None,
        input_truncation: Optional[Union[str, int]] = None,
        parallel_tool_calls: Optional[bool] = None,
        deepthink: Optional[bool] = None,
    ) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
        model = model or config.DEFAULT_MODEL
        temperature = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
        tools = tools or []
        tools_config = tools_config or {}

        validated_tools = self.tool_manager.validate_user_tools(user_tools)
        instruction_prompt = self._build_tool_instruction_prompt(validated_tools)
        prepared_messages = integrate_tool_call_prompt(messages, instruction_prompt) if instruction_prompt else messages

        return self.chat_flow.chat_with_model_tools(
            prepared_messages,
            validated_tools,
            model=model,
            temperature=temperature,
            tools=tools,
            tools_config=tools_config,
            max_tokens=max_tokens,
            input_truncation=input_truncation,
            parallel_tool_calls=parallel_tool_calls,
            deepthink=deepthink,
        )

    def _build_tool_instruction_prompt(self, tools: List[Dict[str, Any]]) -> Optional[str]:
        if not tools:
            return None

        bullet_lines = [f"- {tool.get('name', 'unknown')}: {tool.get('description', 'No description')}" for tool in tools]
        bullet_block = "\n".join(bullet_lines)

        instructions = (
            "Available tools:\n"
            f"{bullet_block}\n\n"
            "IMPORTANT TOOL USAGE GUIDELINES:\n"
            "- Every tool call MUST include 'reasoning': explain to the user why this tool helps their request.\n"
            "- Every tool call MUST include 'stop_after_tool_call': true when the task is done or user input is needed, false when you plan additional tool calls.\n"
            "- Think step by step and combine tools strategically.\n"
            "- Stop after tool execution when a review is required or the task is complete.\n"
            "- Continue with more tools only when further automated steps are necessary."
        )
        return instructions

    # ------------------------------------------------------------------
    # Pipelines
    # ------------------------------------------------------------------

    def run_steps_pipeline(
        self,
        messages: list,
        model: str,
        temperature: float,
        tools: list,
        tools_config: dict,
        steps_model: str = None,
        max_tokens: Optional[int] = None,
        input_truncation: Optional[Union[str, int]] = None,
        deepthink: Optional[bool] = None,
    ):
        base_messages = self._ensure_lc_messages(messages)
        return self.pipeline_runner.run_steps_pipeline(
            base_messages,
            model,
            temperature,
            tools,
            tools_config,
            steps_model=steps_model,
            max_tokens=max_tokens,
            input_truncation=input_truncation,
            deepthink=deepthink,
        )

    def run_reasoning_pipeline(
        self,
        messages: list,
        model: str,
        temperature: float,
        reasoning_iterations: int,
        tools: list,
        tools_config: dict,
        reasoning_model: str = None,
        max_tokens: Optional[int] = None,
        input_truncation: Optional[Union[str, int]] = None,
        deepthink: Optional[bool] = None,
    ):
        base_messages = self._ensure_lc_messages(messages)
        return self.pipeline_runner.run_reasoning_pipeline(
            base_messages,
            model,
            temperature,
            reasoning_iterations,
            tools,
            tools_config,
            reasoning_model=reasoning_model,
            max_tokens=max_tokens,
            input_truncation=input_truncation,
            deepthink=deepthink,
        )

    # ------------------------------------------------------------------
    # High level chat orchestration
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list,
        schema=None,
        model: str = None,
        temperature: float = None,
        max_tokens: Optional[int] = None,
        input_truncation: Optional[Union[str, int]] = None,
        reasoning_iterations: int = None,
        steps_pipeline: bool = False,
        reasoning_pipeline: bool = False,
        steps_model: str = None,
        reasoning_model: str = None,
        tools: list = None,
        tools_config: dict = None,
    ) -> Tuple[Any, int, Dict[str, Any]]:
        
        model = model or config.DEFAULT_MODEL
        temperature = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
        reasoning_iterations = reasoning_iterations or config.DEFAULT_REASONING_ITERATIONS
        steps_model = steps_model or config.DEFAULT_STEPS_MODEL
        reasoning_model = reasoning_model or config.DEFAULT_REASONING_MODEL
        tools = tools if tools is not None else config.DEFAULT_TOOLS
        tools_config = tools_config if tools_config is not None else config.TOOLS_CONFIG.copy()

        base_messages = self._ensure_lc_messages(messages)

        if self.verbose:
            self.log.debug("=" * 50)
            self.log.debug("Starting Chat Session")
            self.log.debug("Messages: %d", len(base_messages))
            self.log.debug("Schema: %s", schema)
            self.log.debug("Model: %s", model)
            self.log.debug("Tools: %s", ", ".join(tools) if tools else "None")
            self.log.debug("=" * 50)
        else:
            self.log.info("Starting chat session")

        reasoning_output: List[str] = []
        reasoning_tokens = 0
        reasoning_tools_used: List[Dict[str, Any]] = []
        steps_output: Dict[str, Any] = {"steps": [], "conclusion": ""}
        steps_tokens = 0
        steps_tools_used: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures: List[Tuple[str, Any]] = []
            if reasoning_pipeline:
                futures.append(
                    (
                        "reasoning",
                        executor.submit(
                            self.pipeline_runner.run_reasoning_pipeline,
                            base_messages,
                            model,
                            temperature,
                            reasoning_iterations,
                            tools,
                            tools_config,
                            reasoning_model=reasoning_model,
                            max_tokens=max_tokens,
                            input_truncation=input_truncation,
                            deepthink=False,
                        ),
                    )
                )
            if steps_pipeline:
                futures.append(
                    (
                        "steps",
                        executor.submit(
                            self.pipeline_runner.run_steps_pipeline,
                            base_messages,
                            model,
                            temperature,
                            tools,
                            tools_config,
                            steps_model=steps_model,
                            max_tokens=max_tokens,
                            input_truncation=input_truncation,
                            deepthink=False,
                        ),
                    )
                )

            for label, future in futures:
                result, tokens, details = future.result()
                if label == "reasoning":
                    reasoning_output = result
                    reasoning_tokens = tokens
                    reasoning_tools_used = details.get("tools_used", [])
                else:
                    steps_output = result
                    steps_tokens = tokens
                    steps_tools_used = details.get("tools_used", [])

        conclusion = steps_output.get("conclusion", "")
        steps_list = steps_output.get("steps", [])

        combined_prompt = None
        if reasoning_pipeline or steps_pipeline:
            combined_prompt = self.pipeline_runner.combine_pipeline_outputs(reasoning_output, steps_output)

        final_messages = base_messages
        if combined_prompt:
            final_messages = add_message_before_system(final_messages, HumanMessage(content=combined_prompt))

        final_deepthink = bool(reasoning_pipeline)
        if schema:
            final_output, tokens, final_details = self.chat_with_model_parse(
                final_messages,
                schema=schema,
                model=model,
                temperature=temperature,
                tools=tools,
                tools_config=tools_config,
                max_tokens=max_tokens,
                input_truncation=input_truncation,
                deepthink=final_deepthink,
            )
        else:
            final_output, tokens, final_details = self.chat_with_ai_sync(
                final_messages,
                model=model,
                temperature=temperature,
                tools=tools,
                tools_config=tools_config,
                max_tokens=max_tokens,
                input_truncation=input_truncation,
                deepthink=final_deepthink,
            )

        if steps_list:
            steps_list.append(conclusion)

        all_tools_used = reasoning_tools_used + steps_tools_used + final_details.get("tools_used", [])
        details_dict = {
            "reasoning": reasoning_output,
            "steps": steps_list,
            "reasoning_tokens": reasoning_tokens,
            "steps_tokens": steps_tokens,
            "final_tokens": tokens,
            "tools_used": all_tools_used,
        }
        total_tokens = reasoning_tokens + steps_tokens + tokens

        if self.verbose:
            self.log.debug("=" * 50)
            self.log.debug("✓ Chat Session Completed")
            self.log.debug("Tokens Used:")
            self.log.debug("  - Reasoning: %d", reasoning_tokens)
            self.log.debug("  - Steps: %d", steps_tokens)
            self.log.debug("  - Final: %d", tokens)
            self.log.debug("  - Total: %d", total_tokens)
            self.log.debug("=" * 50)
        else:
            self.log.info("Chat completed (total tokens: %d)", total_tokens)

        return final_output, total_tokens, details_dict

    # ------------------------------------------------------------------
    # Tool execution helpers
    # ------------------------------------------------------------------

    def execute_tools(self, history: list, tools: list, tools_config: dict) -> tuple:
        history_lc = self._ensure_lc_messages(history)
        return self.tool_manager.execute_tools(history_lc, tools, tools_config)

    def tool_call(
        self,
        messages: list,
        user_tools: list,
        allow_multiple: bool = True,
        model: str = None,
        temperature: float = None,
        input_truncation: Optional[Union[str, int]] = None,
        reasoning_iterations: int = None,
        steps_pipeline: bool = False,
        reasoning_pipeline: bool = False,
        steps_model: str = None,
        reasoning_model: str = None,
        tools: list = None,
        tools_config: dict = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[Any, int, Dict[str, Any]]:
        model = model or config.DEFAULT_MODEL
        temperature = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
        reasoning_iterations = reasoning_iterations if reasoning_iterations is not None else config.DEFAULT_REASONING_ITERATIONS
        steps_model = steps_model or config.DEFAULT_STEPS_MODEL
        reasoning_model = reasoning_model or config.DEFAULT_REASONING_MODEL
        tools = tools or config.DEFAULT_TOOLS
        tools_config = tools_config or config.TOOLS_CONFIG

        validated_tools = self.tool_manager.validate_user_tools(user_tools)
        tool_prompt = (
            generate_multiple_tool_call_prompt(validated_tools)
            if allow_multiple
            else generate_single_tool_call_prompt(validated_tools)
        )

        base_messages = self._ensure_lc_messages(messages)
        tool_call_messages = integrate_tool_call_prompt(base_messages, tool_prompt)

        reasoning_output: List[str] = []
        reasoning_tokens = 0
        reasoning_tools_used: List[Dict[str, Any]] = []
        steps_output: Dict[str, Any] = {"steps": [], "conclusion": ""}
        steps_tokens = 0
        steps_tools_used: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures: List[Tuple[str, Any]] = []
            if reasoning_pipeline:
                futures.append(
                    (
                        "reasoning",
                        executor.submit(
                            self.pipeline_runner.run_reasoning_pipeline,
                            tool_call_messages,
                            model,
                            temperature,
                            reasoning_iterations,
                            tools,
                            tools_config,
                            reasoning_model=reasoning_model,
                            max_tokens=max_tokens,
                            input_truncation=input_truncation,
                            deepthink=False,
                        ),
                    )
                )
            if steps_pipeline:
                futures.append(
                    (
                        "steps",
                        executor.submit(
                            self.pipeline_runner.run_steps_pipeline,
                            tool_call_messages,
                            model,
                            temperature,
                            tools,
                            tools_config,
                            steps_model=steps_model,
                            max_tokens=max_tokens,
                            input_truncation=input_truncation,
                            deepthink=False,
                        ),
                    )
                )

            for label, future in futures:
                result, tokens, details = future.result()
                if label == "reasoning":
                    reasoning_output = result
                    reasoning_tokens = tokens
                    reasoning_tools_used = details.get("tools_used", [])
                else:
                    steps_output = result
                    steps_tokens = tokens
                    steps_tools_used = details.get("tools_used", [])

        conclusion = steps_output.get("conclusion", "")
        combined_prompt = None
        if reasoning_pipeline or steps_pipeline:
            combined_prompt = combine_all_pipeline_prompts(reasoning_output, conclusion)

        enhanced_messages = tool_call_messages
        if combined_prompt:
            enhanced_messages = append_message_to_system(enhanced_messages, combined_prompt)

        parallel_calls = allow_multiple
        response_message, tokens, final_details = self.chat_flow.chat_with_model_tools(
            enhanced_messages,
            validated_tools,
            model=model,
            temperature=temperature,
            tools=tools,
            tools_config=tools_config,
            max_tokens=max_tokens,
            input_truncation=input_truncation,
            parallel_tool_calls=parallel_calls,
            deepthink=bool(reasoning_pipeline),
        )

        total_tokens = reasoning_tokens + steps_tokens + tokens
        all_tools_used = reasoning_tools_used + steps_tools_used + final_details.get("tools_used", [])

        details_dict = {
            "reasoning": reasoning_output,
            "steps": steps_output.get("steps", []),
            "conclusion": steps_output.get("conclusion", ""),
            "reasoning_tokens": reasoning_tokens,
            "steps_tokens": steps_tokens,
            "final_tokens": tokens,
            "tools_used": all_tools_used,
        }

        simplified_response: Any
        if response_message.get("tool_calls"):
            simplified_response = response_message.get("tool_calls")
            if not allow_multiple and simplified_response:
                simplified_response = simplified_response[0]
        else:
            content = response_message.get("content")
            simplified_response = {"content": content} if content and str(content).strip() else None

        return simplified_response, total_tokens, details_dict

    # ------------------------------------------------------------------
    # Web search utilities
    # ------------------------------------------------------------------

    def web_search(
        self,
        query: Optional[str] = None,
        url: Optional[str] = None,
        num_results: int = 5,
        enable_scraping: bool = True,
        max_scrape_length: int = 5000,
        scrape_timeout: int = 15,
        return_debug_info: bool = False,
    ) -> Union[List[Dict], Dict]:
        if not query and not url:
            raise ValueError("Either 'query' for web search or 'url' for scraping must be provided")

        if self.verbose:
            self.log.debug("=" * 50)
            self.log.debug("Starting web search operation")
            if query:
                self.log.debug("Search query: %s", query)
            if url:
                self.log.debug("Scraping URL: %s", url)

        if url:
            if self.verbose:
                self.log.debug("Scraping content from: %s", url)
            try:
                content = scrape_url(url, timeout=scrape_timeout, max_length=max_scrape_length)
                result = {
                    "type": "url_scraping",
                    "url": url,
                    "success": content is not None,
                    "content": content or "Unable to scrape content (blocked by robots.txt or error)",
                    "content_length": len(content) if content else 0,
                }
                return result
            except Exception as exc:  # noqa: BLE001
                return {
                    "type": "url_scraping",
                    "url": url,
                    "success": False,
                    "content": "",
                    "error": f"Error scraping URL {url}: {exc}",
                }

        api_key = self.google_api_key or __import__("os").getenv("GOOGLE_API_KEY")
        search_engine_id = self.search_engine_id or __import__("os").getenv("GOOGLE_SEARCH_ENGINE_ID")
        if not api_key or not search_engine_id:
            raise ValueError(
                "Google API credentials not configured. Provide google_api_key/search_engine_id or set environment variables."
            )

        search_results, debug_info = google_search(query, api_key, search_engine_id, num_results)
        if not search_results:
            result = {
                "type": "web_search",
                "query": query,
                "results": [],
                "total_results": 0,
            }
            if return_debug_info:
                result["debug_info"] = debug_info
            return result

        processed_results: List[Dict[str, Any]] = []
        for idx, item in enumerate(search_results, 1):
            processed_results.append(
                {
                    "rank": idx,
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "scraped_content": None,
                    "scraping_success": False,
                }
            )

        if enable_scraping:
            from concurrent.futures import ThreadPoolExecutor as ScrapeExecutor, as_completed

            def scrape_single(result: Dict[str, Any]) -> Tuple[int, Optional[str], bool]:
                link = result["url"]
                if not link:
                    return result["rank"], None, False
                try:
                    scraped_content = scrape_url(link, timeout=scrape_timeout, max_length=max_scrape_length)
                    return result["rank"], scraped_content, bool(scraped_content)
                except Exception:  # noqa: BLE001
                    return result["rank"], None, False

            with ScrapeExecutor(max_workers=min(5, len(processed_results))) as scrape_pool:
                future_to_rank = {
                    scrape_pool.submit(scrape_single, result): result["rank"] for result in processed_results
                }
                for future in as_completed(future_to_rank):
                    rank, scraped_content, success = future.result()
                    for result in processed_results:
                        if result["rank"] == rank:
                            result["scraped_content"] = scraped_content
                            result["scraping_success"] = success
                            break

        final_result = {
            "type": "web_search",
            "query": query,
            "results": processed_results,
            "total_results": len(processed_results),
        }
        if return_debug_info:
            final_result["debug_info"] = debug_info
        return final_result

__all__ = ["UltraGPT"]
