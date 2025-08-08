from openai import OpenAI 
from .prompts import (
generate_steps_prompt, 
each_step_prompt, generate_reasoning_prompt, 
generate_conclusion_prompt, combine_all_pipeline_prompts,
make_tool_analysis_prompt, generate_tool_call_prompt,
generate_single_tool_call_prompt, generate_multiple_tool_call_prompt
)
from pydantic import BaseModel
from .schemas import Steps, Reasoning, ToolAnalysisSchema, UserTool, ExpertTool
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultraprint.logging import logger
from .providers import ProviderManager, OpenAIProvider, ClaudeProvider
from typing import Optional
import os
import importlib
import inspect
from . import config

from itertools import islice

class UltraGPT:
    def __init__(
        self, 
        api_key: str = None,
        openai_api_key: str = None,
        claude_api_key: str = None,
        google_api_key: str = None,
        search_engine_id: str = None,
        max_tokens: Optional[int] = None,
        verbose: bool = False,
        logger_name: str = 'ultragpt',
        logger_filename: str = 'debug/ultragpt.log',
        log_extra_info: bool = False,
        log_to_file: bool = False,
        log_to_console: bool = False,
        log_level: str = 'DEBUG',
    ):
        """
        Initialize the UltraGPT class with multi-provider support.
        Args:
            api_key (str, optional): The API key for accessing the OpenAI service.
            claude_api_key (str, optional): The API key for accessing Claude/Anthropic service.
            google_api_key (str, optional): Google Custom Search API key for web search tool.
            search_engine_id (str, optional): Google Custom Search Engine ID for web search tool.
            max_tokens (int, optional): Maximum number of tokens to generate. Set to None to use provider defaults. Defaults to 4096.
            verbose (bool, optional): Whether to enable verbose logging. Defaults to False.
            logger_name (str, optional): The name of the logger. Defaults to 'ultragpt'.
            logger_filename (str, optional): The filename for the logger. Defaults to 'debug/ultragpt.log'.
            log_extra_info (bool, optional): Whether to include extra info in logs. Defaults to False.
            log_to_file (bool, optional): Whether to log to a file. Defaults to False.
            log_to_console (bool, optional): Whether to log to console. Defaults to True.
            log_level (str, optional): The logging level. Defaults to 'DEBUG'.
        Raises:
            ValueError: If no API keys are provided or if an invalid tool is provided.
        """

        # Initialize provider manager
        self.provider_manager = ProviderManager()
        
        # Add providers based on available API keys
        if api_key or openai_api_key:
            openai_provider = OpenAIProvider(api_key=api_key or openai_api_key)
            self.provider_manager.add_provider("openai", openai_provider)
            
        if claude_api_key:
            try:
                claude_provider = ClaudeProvider(api_key=claude_api_key)
                self.provider_manager.add_provider("claude", claude_provider)
            except ImportError as e:
                if verbose:
                    print(f"Warning: Claude provider not available: {e}")
        
        # Ensure at least one provider is available
        if not self.provider_manager.providers:
            raise ValueError("At least one API key (api_key or claude_api_key) must be provided")
        
        # Keep backward compatibility
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
        
        # Store Google Search credentials
        self.google_api_key = google_api_key
        self.search_engine_id = search_engine_id
        
        # Store max_tokens setting
        self.max_tokens = max_tokens if max_tokens is not None else config.MAX_TOKENS_DEFAULT
        
        self.verbose = verbose
        self.log = logger(
            name=logger_name,
            filename=logger_filename,
            include_extra_info=log_extra_info,
            write_to_file=log_to_file,
            log_level=log_level,
            log_to_console = True if verbose else log_to_console
        )
        
        self.log.info("Initializing UltraGPT")
        if self.verbose:
            self.log.debug("=" * 50)
            self.log.debug("Initializing UltraGPT")
            self.log.debug("=" * 50)

    def chat_with_ai_sync(
        self,
        messages: list,
        model: str,
        temperature: float,
        tools: list,
        tools_config: dict,
        tool_batch_size: int,
        tool_max_workers: int,
        max_tokens: Optional[int] = None
    ):
        """
        Sends a synchronous chat request to the specified AI provider and processes the response.
        Args:
            messages (list): A list of message dictionaries to be sent to the AI provider.
            model (str): The model to use (format: "provider:model" or just "model" for OpenAI).
            temperature (float): The temperature for the model's output.
            tools (list): The list of tools to enable.
            tools_config (dict): The configuration for the tools.
            tool_batch_size (int): The batch size for tool processing.
            tool_max_workers (int): The maximum number of workers for tool processing.
        Returns:
            tuple: A tuple containing the response content (str) and the total number of tokens used (int).
        Raises:
            Exception: If the request to the AI provider fails.
        Logs:
            Debug: Logs the number of messages sent, the number of tokens in the response, and any errors encountered.
            Verbose: Optionally logs detailed steps of the request and response process.
        """
        try:
            self.log.debug("Sending request to AI provider (msgs: " + str(len(messages)) + ")")
            if self.verbose:
                provider_name, model_name = self.provider_manager.parse_model_string(model)
                self.log.debug(f"AI Request → Provider: {provider_name}, Model: {model_name}, Messages: " + str(len(messages)))
                self.log.debug("Checking for tool needs...")
            
            tool_response = self.execute_tools(message=messages[-1]["content"], history=messages, tools=tools, tools_config=tools_config, tool_batch_size=tool_batch_size, tool_max_workers=tool_max_workers)
            if tool_response:
                if self.verbose:
                    self.log.debug("Appending tool responses to message")
                tool_response = "Tool Responses:\n" + tool_response
                messages = self.append_message_to_system(messages, tool_response)
            elif self.verbose:
                self.log.debug("No tool responses needed")
            
            content, tokens = self.provider_manager.chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens
            )
            
            self.log.debug("Response received (tokens: " + str(tokens) + ")")
            if self.verbose:
                self.log.debug("✓ Response received (" + str(tokens) + " tokens)")
            return content, tokens
        except Exception as e:
            self.log.error("AI sync request failed: " + str(e))
            if self.verbose:
                self.log.debug("✗ AI request failed: " + str(e))
            raise e

    def chat_with_model_parse(
        self,
        messages: list,
        schema=None,
        model: str = None,
        temperature: float = None,
        tools: list = [],
        tools_config: dict = {},
        tool_batch_size: int = None,
        tool_max_workers: int = None,
        max_tokens: Optional[int] = None
    ):
        """
        Sends a chat message to the model for parsing and returns the parsed response.
        Args:
            messages (list): A list of message dictionaries to be sent to the model.
            schema (optional): The schema to be used for parsing the response. Defaults to None.
            model (str): The model to use (format: "provider:model" or just "model" for OpenAI).
            temperature (float): The temperature for the model's output.
            tools (list): The list of tools to enable.
            tools_config (dict): The configuration for the tools.
            tool_batch_size (int): The batch size for tool processing.
            tool_max_workers (int): The maximum number of workers for tool processing.
        Returns:
            tuple: A tuple containing the parsed content and the total number of tokens used.
        Raises:
            Exception: If the parse request fails.
        """
        # Use config defaults if not provided
        model = model or config.DEFAULT_PARSE_MODEL
        temperature = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
        tool_batch_size = tool_batch_size or config.DEFAULT_TOOL_BATCH_SIZE
        tool_max_workers = tool_max_workers or config.DEFAULT_TOOL_MAX_WORKERS
        try:
            self.log.debug("Sending parse request with schema: %s", schema)
            
            tool_response = self.execute_tools(message=messages[-1]["content"], history=messages, tools=tools, tools_config=tools_config, tool_batch_size=tool_batch_size, tool_max_workers=tool_max_workers)
            if tool_response:
                tool_response = "Tool Responses:\n" + tool_response
            messages = self.append_message_to_system(messages, tool_response)

            content, tokens = self.provider_manager.chat_completion_with_schema(
                model=model,
                messages=messages,
                schema=schema,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens
            )
            
            self.log.debug("Parse response received (tokens: " + str(tokens) + ")")
            return content, tokens
        except Exception as e:
            self.log.error("Parse request failed: " + str(e))
            raise e

    def chat_with_model_tools(
        self,
        messages: list,
        user_tools: list,
        model: str = None,
        temperature: float = None,
        tools: list = [],
        tools_config: dict = {},
        tool_batch_size: int = None,
        tool_max_workers: int = None,
        max_tokens: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None
    ):
        """
        Sends a chat message to the model with native tool calling support.
        AI will always be required to choose at least one tool from the provided tools.
        """
        try:
            self.log.debug("Sending native tool calling request")
            
            # Execute UltraGPT tools if any
            tool_response = self.execute_tools(
                message=messages[-1]["content"], 
                history=messages, 
                tools=tools, 
                tools_config=tools_config, 
                tool_batch_size=tool_batch_size, 
                tool_max_workers=tool_max_workers
            )
            if tool_response:
                tool_response = "Tool Responses:\n" + tool_response
                messages = self.append_message_to_system(messages, tool_response)
            
            # Convert user tools to native tool format
            native_tools = self._convert_user_tools_to_native_format(user_tools)
            
            # Add tool usage instructions to the messages if tools are available
            if user_tools and len(user_tools) > 0:
                # Generate tool prompts for available tools to help the model understand them better
                tool_prompts = []
                for tool in user_tools:
                    if isinstance(tool, dict):
                        name = tool.get("name", "unknown")
                        description = tool.get("description", "No description")
                    elif hasattr(tool, 'model_dump'):
                        tool_dict = tool.model_dump()
                        name = tool_dict.get("name", "unknown")
                        description = tool_dict.get("description", "No description")
                    else:
                        name = str(tool)
                        description = "Tool"
                    
                    tool_prompts.append(f"- {name}: {description}")
                
                # Add tool usage instructions to the message
                tool_instructions = f"""
Available tools:
{chr(10).join(tool_prompts)}

IMPORTANT TOOL USAGE GUIDELINES:
- Every tool call MUST include 'reasoning' parameter: Provide detailed reasoning for why this specific tool was chosen and how it will help solve the user's request
- Every tool call MUST include 'stop_after_tool_call' parameter: Set to true if the task will be complete after this tool call OR if user input is needed, false if you plan to call more tools afterward
- Always think step by step and use tools strategically to solve the user's request
- When using tools, provide meaningful reasoning that explains your decision-making process
- Use stop_after_tool_call=true when: task is complete, you need user feedback, or the result requires user review
- Use stop_after_tool_call=false when: you plan to use the tool result for additional tool calls to complete the task

"""
                
                # Make a copy of messages to avoid modifying the original
                messages = messages.copy()
                if isinstance(messages[-1], dict) and messages[-1]["role"] == "user":
                    messages[-1] = messages[-1].copy()
                    messages[-1]["content"] = tool_instructions + messages[-1]["content"]
                else:
                    messages.append({"role": "system", "content": tool_instructions})
            
            # Make native tool call - AI will always choose at least one tool
            response_message, tokens = self.provider_manager.chat_completion_with_tools(
                model=model,
                messages=messages,
                tools=native_tools,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                parallel_tool_calls=parallel_tool_calls
            )
            
            self.log.debug("Native tool calling response received (tokens: " + str(tokens) + ")")
            return response_message, tokens
        except Exception as e:
            self.log.error("Native tool calling request failed: " + str(e))
            raise e

    def _convert_user_tools_to_native_format(self, user_tools: list) -> list:
        """Convert UserTool objects to native AI provider tool format"""
        native_tools = []
        
        for tool in user_tools:
            if isinstance(tool, dict):
                tool_dict = tool
            elif hasattr(tool, 'model_dump'):
                tool_dict = tool.model_dump()
            else:
                self.log.warning("Invalid tool format: " + str(type(tool)))
                continue
            
            # Get parameters schema and ensure it has additionalProperties: false for OpenAI strict mode
            parameters_schema = tool_dict["parameters_schema"].copy()
            
            # Surgically add reasoning and stop_after_tool_call parameters to the schema
            if "properties" not in parameters_schema:
                parameters_schema["properties"] = {}
            
            # Add reasoning parameter
            parameters_schema["properties"]["reasoning"] = {
                "type": "string",
                "description": "Detailed reasoning for why this tool was chosen and how it will help solve the user's request"
            }
            
            # Add stop_after_tool_call parameter  
            parameters_schema["properties"]["stop_after_tool_call"] = {
                "type": "boolean",
                "description": "Whether to stop execution after this tool call (true if task is complete or user input needed, false to continue with more tools)"
            }
            
            # Ensure additionalProperties is false and required includes all properties for OpenAI strict mode
            def ensure_openai_strict_compliance(schema):
                if isinstance(schema, dict):
                    if schema.get("type") == "object":
                        schema["additionalProperties"] = False
                        # For OpenAI strict mode, required must include ALL properties if any are specified
                        if "properties" in schema:
                            all_properties = list(schema["properties"].keys())
                            schema["required"] = all_properties
                    
                    for key, value in schema.items():
                        if key == "properties" and isinstance(value, dict):
                            for prop_value in value.values():
                                ensure_openai_strict_compliance(prop_value)
                        elif isinstance(value, dict):
                            ensure_openai_strict_compliance(value)
                        elif isinstance(value, list):
                            for item in value:
                                ensure_openai_strict_compliance(item)
            
            ensure_openai_strict_compliance(parameters_schema)
            
            # Convert to OpenAI function calling format (Claude will handle conversion)
            native_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict["description"],
                    "parameters": parameters_schema,
                    "strict": True
                }
            }
            native_tools.append(native_tool)
        
        return native_tools

    #! Message Alteration ---------------------------------------------------
    def turnoff_system_message(self, messages: list):
        # set system message to user message
        processed = []
        for message in messages:
            if message["role"] == "system" or message["role"] == "developer":
                message["role"] = "user"
            processed.append(message)
        return processed
    
    def add_message_before_system(self, messages: list, new_message: dict):
        # add message before system message
        processed = []
        for message in messages:
            if message["role"] == "system" or message["role"] == "developer":
                processed.append(new_message)
            processed.append(message)
        return processed

    def append_message_to_system(self, messages: list, new_message: dict):
        # add message after system message
        processed = []
        for message in messages:
            if message["role"] == "system" or message["role"] == "developer":
                processed.append({
                    "role": message["role"],
                    "content": f"{message['content']}\n{new_message}"
                })
            else:
                processed.append(message)
        return processed
    
    def integrate_tool_call_prompt(self, messages: list, tool_prompt: str) -> list:
        """
        Properly integrate tool call prompt with existing system/developer messages
        Places all conversation messages first, then the system message at the end
        """
        processed = []
        system_messages = []
        other_messages = []
        
        # Separate system/developer messages from other messages
        for message in messages:
            if message["role"] in ["system", "developer"]:
                system_messages.append(message)
            else:
                other_messages.append(message)
        
        # Add all conversation messages first
        processed.extend(other_messages)
        
        # Then add the system message at the end
        if system_messages:
            # Combine all existing system messages
            combined_content = "\n\n".join([msg["content"] for msg in system_messages])
            # Add our tool call prompt
            final_content = f"{combined_content}\n\n{tool_prompt}"
            
            # Use the role of the first system message
            processed.append({
                "role": system_messages[0]["role"],
                "content": final_content
            })
        else:
            # No existing system messages, just add our tool prompt at the end
            processed.append({
                "role": "system",
                "content": tool_prompt
            })
        
        return processed
    
    #! Pipelines -----------------------------------------------------------
    def run_steps_pipeline(
        self,
        messages: list,
        model: str,
        temperature: float,
        tools: list,
        tools_config: dict,
        tool_batch_size: int,
        tool_max_workers: int,
        steps_model: str = None,
        max_tokens: Optional[int] = None
    ):
        # Use steps_model if provided, otherwise use main model
        active_model = steps_model if steps_model else model
        
        if self.verbose:
            self.log.debug("➤ Starting Steps Pipeline")
            if steps_model:
                self.log.debug("Using steps model: " + steps_model)
        else:
            self.log.info("Starting steps pipeline")
        total_tokens = 0

        messages = self.turnoff_system_message(messages)
        steps_generator_message = messages + [{"role": "system", "content": generate_steps_prompt()}]

        steps_json, tokens = self.chat_with_model_parse(steps_generator_message, schema=Steps, model=active_model, temperature=temperature, tools=tools, tools_config=tools_config, tool_batch_size=tool_batch_size, tool_max_workers=tool_max_workers, max_tokens=max_tokens)
        total_tokens += tokens
        steps = steps_json.get("steps", [])
        if self.verbose:
            self.log.debug("Generated " + str(len(steps)) + " steps:")
            for idx, step in enumerate(steps, 1):
                self.log.debug("  " + str(idx) + ". " + step)
        else:
            self.log.debug("Generated " + str(len(steps)) + " steps")

        memory = []

        for idx, step in enumerate(steps, 1):
            if self.verbose:
                self.log.debug("Processing step " + str(idx) + "/" + str(len(steps)))
            self.log.debug("Processing step " + str(idx) + "/" + str(len(steps)))
            step_prompt = each_step_prompt(memory, step)
            step_message = messages + [{"role": "system", "content": step_prompt}]
            step_response, tokens = self.chat_with_ai_sync(step_message, model=active_model, temperature=temperature, tools=tools, tools_config=tools_config, tool_batch_size=tool_batch_size, tool_max_workers=tool_max_workers, max_tokens=max_tokens)
            self.log.debug("Step " + str(idx) + " response: " + step_response[:100] + "...")
            total_tokens += tokens
            memory.append(
                {
                    "step": step,
                    "answer": step_response
                }
            )

        # Generate final conclusion
        conclusion_prompt = generate_conclusion_prompt(memory)
        conclusion_message = messages + [{"role": "system", "content": conclusion_prompt}]
        conclusion, tokens = self.chat_with_ai_sync(conclusion_message, model=active_model, temperature=temperature, tools=tools, tools_config=tools_config, tool_batch_size=tool_batch_size, tool_max_workers=tool_max_workers, max_tokens=max_tokens)
        total_tokens += tokens

        if self.verbose:
            self.log.debug("✓ Steps pipeline completed")
        
        return {
            "steps": memory,
            "conclusion": conclusion
        }, total_tokens

    def run_reasoning_pipeline(
        self,
        messages: list,
        model: str,
        temperature: float,
        reasoning_iterations: int,
        tools: list,
        tools_config: dict,
        tool_batch_size: int,
        tool_max_workers: int,
        reasoning_model: str = None,
        max_tokens: Optional[int] = None
    ):
        # Use reasoning_model if provided, otherwise use main model
        active_model = reasoning_model if reasoning_model else model
        
        if self.verbose:
            self.log.debug("➤ Starting Reasoning Pipeline (" + str(reasoning_iterations) + " iterations)")
            if reasoning_model:
                self.log.debug("Using reasoning model: " + reasoning_model)
        else:
            self.log.info("Starting reasoning pipeline (" + str(reasoning_iterations) + " iterations)")
        total_tokens = 0
        all_thoughts = []
        messages = self.turnoff_system_message(messages)

        for iteration in range(reasoning_iterations):
            if self.verbose:
                self.log.debug("Iteration " + str(iteration + 1) + "/" + str(reasoning_iterations))
            self.log.debug("Iteration " + str(iteration + 1) + "/" + str(reasoning_iterations))
            # Generate new thoughts based on all previous thoughts
            reasoning_message = messages + [
                {"role": "system", "content": generate_reasoning_prompt(all_thoughts)}
            ]
            
            reasoning_json, tokens = self.chat_with_model_parse(
                reasoning_message, 
                schema=Reasoning,
                model=active_model,
                temperature=temperature,
                tools=tools,
                tools_config=tools_config,
                tool_batch_size=tool_batch_size,
                tool_max_workers=tool_max_workers,
                max_tokens=max_tokens
            )
            total_tokens += tokens
            
            new_thoughts = reasoning_json.get("thoughts", [])
            all_thoughts.extend(new_thoughts)
            
            if self.verbose:
                self.log.debug("Generated " + str(len(new_thoughts)) + " thoughts:")
                for idx, thought in enumerate(new_thoughts, 1):
                    self.log.debug("  " + str(idx) + ". " + thought)
            else:
                self.log.debug("Generated " + str(len(new_thoughts)) + " new thoughts")

        return all_thoughts, total_tokens
    
    #! Main Chat Function ---------------------------------------------------
    def chat(
        self,
        messages: list,
        schema=None,
        model: str = None,
        temperature: float = None,
        max_tokens: Optional[int] = None,  # Override instance default if provided
        reasoning_iterations: int = None,
        steps_pipeline: bool = True,
        reasoning_pipeline: bool = True,
        steps_model: str = None,
        reasoning_model: str = None,
        tools: list = None,
        tools_config: dict = None,
        tool_batch_size: int = None,
        tool_max_workers: int = None,
    ):
        """
        Initiates a chat session with the given messages and optional schema.
        Args:
            messages (list): A list of message dictionaries to be processed.
            schema (optional): A schema to parse the final output, defaults to None.
            model (str, optional): The model to use. Format: "provider:model" (e.g., "claude:claude-3-sonnet-20240229") or just "model" (defaults to OpenAI). Defaults to "gpt-4o".
            temperature (float, optional): The temperature for the model's output. Defaults to 0.7.
            max_tokens (int, optional): Maximum tokens to generate. Overrides instance default if provided. None uses provider defaults.
            reasoning_iterations (int, optional): The number of reasoning iterations. Defaults to 3.
            steps_pipeline (bool, optional): Whether to use steps pipeline. Defaults to True.
            reasoning_pipeline (bool, optional): Whether to use reasoning pipeline. Defaults to True.
            steps_model (str, optional): Specific model for steps pipeline. Format: "provider:model" or just "model". Uses main model if None.
            reasoning_model (str, optional): Specific model for reasoning pipeline. Format: "provider:model" or just "model". Uses main model if None.
            tools (list, optional): The list of tools to enable. Defaults to ["web-search", "calculator", "math-operations"].
            tools_config (dict, optional): The configuration for the tools. Each tool's "model" field supports "provider:model" format. Defaults to predefined configurations.
            tool_batch_size (int, optional): The batch size for tool processing. Defaults to 3.
            tool_max_workers (int, optional): The maximum number of workers for tool processing. Defaults to 10.
        Returns:
            tuple: A tuple containing the final output, total tokens used, and a details dictionary.
                - final_output: The final response from the chat model.
                - total_tokens (int): The total number of tokens used during the session.
                - details_dict (dict): A dictionary with detailed information about the session.
                
        Model Format Examples:
            - "gpt-4o" or "openai:gpt-4o" → OpenAI GPT-4o
            - "claude:claude-3-sonnet-20240229" → Anthropic Claude 3 Sonnet
            - "claude:claude-3-haiku-20240307" → Anthropic Claude 3 Haiku
        """
        # Use config defaults if not provided
        model = model or config.DEFAULT_MODEL
        temperature = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
        reasoning_iterations = reasoning_iterations or config.DEFAULT_REASONING_ITERATIONS
        steps_model = steps_model or config.DEFAULT_STEPS_MODEL
        reasoning_model = reasoning_model or config.DEFAULT_REASONING_MODEL
        tools = tools if tools is not None else config.DEFAULT_TOOLS
        tools_config = tools_config if tools_config is not None else config.TOOLS_CONFIG.copy()
        tool_batch_size = tool_batch_size or config.DEFAULT_TOOL_BATCH_SIZE
        tool_max_workers = tool_max_workers or config.DEFAULT_TOOL_MAX_WORKERS
        
        if self.verbose:
            self.log.debug("=" * 50)
            self.log.debug("Starting Chat Session")
            self.log.debug("Messages: " + str(len(messages)))
            self.log.debug("Schema: " + str(schema))
            self.log.debug("Model: " + model)
            self.log.debug("Tools: " + (', '.join(tools) if tools else 'None'))
            self.log.debug("=" * 50)
        else:
            self.log.info("Starting chat session")

        reasoning_output = []
        reasoning_tokens = 0
        steps_output = {"steps": [], "conclusion": ""}
        steps_tokens = 0

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            
            if reasoning_pipeline:
                futures.append({
                    "type": "reasoning",
                    "future": executor.submit(self.run_reasoning_pipeline, messages, model, temperature, reasoning_iterations, tools, tools_config, tool_batch_size, tool_max_workers, reasoning_model, max_tokens)
                })
            
            if steps_pipeline:
                futures.append({
                    "type": "steps",
                    "future": executor.submit(self.run_steps_pipeline, messages, model, temperature, tools, tools_config, tool_batch_size, tool_max_workers, steps_model, max_tokens)
                })

            for future in futures:
                if future["type"] == "reasoning":
                    reasoning_output, reasoning_tokens = future["future"].result()
                elif future["type"] == "steps":
                    steps_output, steps_tokens = future["future"].result()

        conclusion = steps_output.get("conclusion", "")
        steps = steps_output.get("steps", [])

        if reasoning_pipeline or steps_pipeline:
            prompt = combine_all_pipeline_prompts(reasoning_output, conclusion)
            messages = self.add_message_before_system(messages, {"role": "user", "content": prompt})

        if schema:
            final_output, tokens = self.chat_with_model_parse(messages, schema=schema, model=model, temperature=temperature, tools=tools, tools_config=tools_config, tool_batch_size=tool_batch_size, tool_max_workers=tool_max_workers, max_tokens=max_tokens)
        else:
            final_output, tokens = self.chat_with_ai_sync(messages, model=model, temperature=temperature, tools=tools, tools_config=tools_config, tool_batch_size=tool_batch_size, tool_max_workers=tool_max_workers, max_tokens=max_tokens)

        if steps:
            steps.append(conclusion)
            
        details_dict = {
            "reasoning": reasoning_output,
            "steps": steps,
            "reasoning_tokens": reasoning_tokens,
            "steps_tokens": steps_tokens,
            "final_tokens": tokens
        }
        total_tokens = reasoning_tokens + steps_tokens + tokens
        if self.verbose:
            self.log.debug("=" * 50)
            self.log.debug("✓ Chat Session Completed")
            self.log.debug("Tokens Used:")
            self.log.debug("  - Reasoning: " + str(reasoning_tokens))
            self.log.debug("  - Steps: " + str(steps_tokens))
            self.log.debug("  - Final: " + str(tokens))
            self.log.debug("  - Total: " + str(total_tokens))
            self.log.debug("=" * 50)
        else:
            self.log.info("Chat completed (total tokens: " + str(total_tokens) + ")")
        
        #! Return as tuple for consistent API 
        #! DO NOT CHANGE THIS RETURN FORMAT
        return final_output, total_tokens, details_dict

    #! Tools ----------------------------------------------------------------
    def _load_internal_tools(self, tools: list) -> dict:
        """Dynamically load internal tool configurations from tool directories"""
        tools_base_path = os.path.join(os.path.dirname(__file__), 'tools')
        loaded_tools = {}
        
        for tool_name in tools:
            tool_name_normalized = tool_name.replace('-', '_')  # Convert web-search to web_search
            tool_path = os.path.join(tools_base_path, tool_name_normalized)
            
            if not os.path.exists(tool_path):
                self.log.warning(f"Tool directory not found: {tool_path}")
                continue
                
            try:
                # Load prompts module to get _info and _description
                prompts_module = importlib.import_module(f'.tools.{tool_name_normalized}.prompts', package='ultragpt')
                info = getattr(prompts_module, '_info', f"Tool: {tool_name}")
                description = getattr(prompts_module, '_description', info)
                
                # Load schemas module to get schema classes
                schemas_module = importlib.import_module(f'.tools.{tool_name_normalized}.schemas', package='ultragpt')
                
                # Find the main schema class (usually ends with 'Query')
                schema_class = None
                for name, obj in inspect.getmembers(schemas_module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseModel) and 
                        name.endswith('Query')):
                        schema_class = obj
                        break
                
                if not schema_class:
                    self.log.warning(f"No Query schema found for tool: {tool_name}")
                    continue
                
                # Convert schema to tool format for native calling
                schema_dict = schema_class.model_json_schema()
                
                loaded_tools[tool_name_normalized] = {  # Use normalized name as key
                    'name': tool_name_normalized,
                    'display_name': tool_name,
                    'description': description,
                    'schema': schema_class,
                    'native_schema': {
                        'type': 'function',
                        'function': {
                            'name': tool_name_normalized,
                            'description': description,
                            'parameters': schema_dict
                        }
                    }
                }
                
                self.log.debug(f"✓ Loaded tool: {tool_name} (as {tool_name_normalized})")
                
            except Exception as e:
                self.log.warning(f"Failed to load tool {tool_name}: {str(e)}")
                continue
        
        return loaded_tools

    def _convert_internal_tools_to_native_format(self, loaded_tools: dict) -> list:
        """Convert loaded internal tools to native AI provider tool format"""
        native_tools = []
        
        for tool_name, tool_config in loaded_tools.items():
            # Use the native_schema that was already created in _load_internal_tools
            if 'native_schema' in tool_config:
                native_tools.append(tool_config['native_schema'])
            else:
                # Fallback: create schema from tool config
                native_tool = {
                    "type": "function",
                    "function": {
                        "name": tool_config['name'],
                        "description": tool_config['description'],
                        "parameters": {}
                    }
                }
                native_tools.append(native_tool)
        
        return native_tools
        
        return native_tools

    def execute_tools(
        self,
        message: str,
        history: list,
        tools: list,
        tools_config: dict,
        tool_batch_size: int,
        tool_max_workers: int
    ) -> str:
        """Execute tools using native AI tool calling - completely refactored approach"""
        if not tools:
            return ""
        
        try:
            self.log.info(f"Loading and executing {len(tools)} tools using native AI tool calling")
            if self.verbose:
                self.log.debug(f"➤ Loading {len(tools)} tools for native AI tool calling")
                self.log.debug(f"Query: {message[:100] + '...' if len(message) > 100 else message}")
                self.log.debug("-" * 40)
            
            # Load internal tools dynamically
            loaded_tools = self._load_internal_tools(tools)
            if not loaded_tools:
                self.log.warning("No tools could be loaded")
                return ""
            
            # Convert to native tool format
            native_tools = self._convert_internal_tools_to_native_format(loaded_tools)
            
            # Create tool selection prompt
            tool_descriptions = []
            for tool_name, tool_config in loaded_tools.items():
                tool_descriptions.append(f"- {tool_config['display_name']}: {tool_config['description']}")
            
            tool_selection_prompt = f"""
Available internal tools:
{chr(10).join(tool_descriptions)}

Analyze the user's message and select the appropriate tools with their parameters. Each tool should be called with the specific parameters needed to help answer the user's question.

User message: "{message}"

IMPORTANT: 
- Only call tools that are actually needed to help answer the user's question
- Use the tool's schema to provide the correct parameters
- If no tools are needed, don't call any tools
- You can call multiple tools if needed
"""
            
            # Prepare messages for tool calling
            tool_messages = [
                {"role": "system", "content": tool_selection_prompt},
                {"role": "user", "content": message}
            ]
            
            # Add conversation history context (limited)
            if history:
                context_messages = history[-config.MAX_CONTEXT_MESSAGES:]  # Last N messages for context
                for msg in context_messages:
                    if msg.get("content") and msg.get("role") in ["user", "assistant"]:
                        tool_messages.insert(-1, {
                            "role": msg["role"], 
                            "content": msg["content"]
                        })
            
            # Get model from tools_config or use default
            model = config.DEFAULT_TOOLS_MODEL
            for tool_config in tools_config.values():
                if 'model' in tool_config:
                    model = tool_config['model']
                    break
            
            # Make native tool call
            try:
                response_message, tokens = self.provider_manager.chat_completion_with_tools(
                    model=model,
                    messages=tool_messages,
                    tools=native_tools,
                    temperature=config.TOOL_SELECTION_TEMPERATURE,  # Low temperature for tool selection
                    max_tokens=self.max_tokens
                )
                
                if self.verbose:
                    self.log.debug(f"AI tool selection completed (tokens: {tokens})")
                
            except Exception as e:
                # If native tool calling fails, fall back to no tools
                self.log.warning(f"Native tool calling failed, proceeding without tools: {str(e)}")
                return ""
            
            # Process tool calls if any were made
            if not response_message.get('tool_calls'):
                if self.verbose:
                    self.log.debug("AI decided no tools are needed")
                return ""
            
            # Execute the selected tools
            tool_results = []
            for tool_call in response_message.get('tool_calls', []):
                function_name = tool_call.get('function', {}).get('name')
                function_args = tool_call.get('function', {}).get('arguments', {})
                
                # Parse arguments if they're in string format
                if isinstance(function_args, str):
                    try:
                        import json
                        function_args = json.loads(function_args)
                    except json.JSONDecodeError:
                        self.log.error(f"Failed to parse tool arguments: {function_args}")
                        continue
                
                if function_name in loaded_tools:
                    tool_config = loaded_tools[function_name]
                    
                    try:
                        if self.verbose:
                            self.log.debug(f"Executing tool: {function_name}")
                            self.log.debug(f"Parameters: {function_args}")
                        
                        # Execute the tool with the AI-selected parameters
                        tool_result = self._execute_internal_tool_with_params(
                            tool_config, function_args, message, history, tools_config
                        )
                        
                        tool_results.append({
                            "tool": tool_config['display_name'],
                            "response": tool_result
                        })
                        
                        if self.verbose:
                            self.log.debug(f"✓ {function_name} completed")
                            self.log.debug("-" * 40)
                            self.log.debug(tool_result if tool_result else "(empty result)")
                            self.log.debug("-" * 40)
                            
                    except Exception as e:
                        self.log.error(f"Tool {function_name} execution failed: {str(e)}")
                        tool_results.append({
                            "tool": tool_config['display_name'],
                            "response": f"Tool execution failed: {str(e)}"
                        })
                else:
                    self.log.warning(f"Unknown tool called: {function_name}")
            
            # Format results
            if not tool_results:
                return ""
                
            formatted_responses = []
            for result in tool_results:
                tool_name = result['tool'].upper()
                response = result['response'].strip() if result['response'] else ""
                if response:
                    formatted = f"[{tool_name}]\n{response}"
                    formatted_responses.append(formatted)
            
            success_count = len([r for r in tool_results if r['response'] and not r['response'].startswith('Tool execution failed')])
            self.log.info(f"Tools execution completed ({success_count}/{len(tool_results)} successful)")
            
            if self.verbose:
                self.log.debug(f"✓ Tools execution completed ({success_count}/{len(tool_results)} successful)")
                
            return "\n\n".join(formatted_responses)
                
        except Exception as e:
            self.log.error(f"Tool execution failed: {str(e)}")
            if self.verbose:
                self.log.debug(f"✗ Tool execution failed: {str(e)}")
            return ""

    def _execute_internal_tool_with_params(
        self, 
        tool_config: dict, 
        parameters: dict, 
        message: str, 
        history: list, 
        tools_config: dict
    ) -> str:
        """Execute an internal tool directly with AI-provided parameters"""
        try:
            # Get tool-specific config
            tool_name_normalized = tool_config['name']  # Use normalized name
            config = tools_config.get(tool_name_normalized, {})
            
            # Import and execute the tool's execute_tool function
            try:
                core_module = importlib.import_module(f'.tools.{tool_name_normalized}.core', package='ultragpt')
                execute_function = getattr(core_module, 'execute_tool', None)
                
                if execute_function:
                    # Pass additional config for web search if needed
                    if tool_name_normalized == "web_search":
                        parameters['google_api_key'] = self.google_api_key
                        parameters['search_engine_id'] = self.search_engine_id
                    
                    result = execute_function(parameters)
                    return result if result else ""
                else:
                    return f"No execute_tool function found for {tool_name_normalized}"
                    
            except Exception as e:
                return f"Error executing tool {tool_name_normalized}: {str(e)}"
                
        except Exception as e:
            return f"Tool execution error: {str(e)}"

    #! Tool Call Functionality --------------------------------------------
    def tool_call(
        self,
        messages: list,
        user_tools: list,
        allow_multiple: bool = True,
        model: str = None,  # Format: "provider:model" or just "model" (defaults to OpenAI)
        temperature: float = None,
        reasoning_iterations: int = None,
        steps_pipeline: bool = True,
        reasoning_pipeline: bool = True,
        steps_model: str = None,  # Format: "provider:model" or just "model" (defaults to OpenAI)  
        reasoning_model: str = None,  # Format: "provider:model" or just "model" (defaults to OpenAI)
        tools: list = None,
        tools_config: dict = None,
        tool_batch_size: int = None,
        tool_max_workers: int = None,
        max_tokens: Optional[int] = None
    ):
        """
        Tool call functionality that uses UltraGPT's execution layer to determine 
        which user-defined tools to call and with what parameters.
        
        Args:
            messages (list): A list of message dictionaries to be processed.
            user_tools (list): List of user-defined tools with schemas and prompts.
            allow_multiple (bool, optional): Whether to allow multiple tool calls. Defaults to True.
            model (str, optional): The model to use. Format: "provider:model" or just "model" (defaults to OpenAI). Defaults to "gpt-4o".
            temperature (float, optional): The temperature for the model's output. Defaults to 0.7.
            reasoning_iterations (int, optional): The number of reasoning iterations. Defaults to 3.
            steps_pipeline (bool, optional): Whether to use steps pipeline. Defaults to True.
            reasoning_pipeline (bool, optional): Whether to use reasoning pipeline. Defaults to True.
            steps_model (str, optional): Specific model for steps pipeline. Format: "provider:model" or just "model". Uses main model if None.
            reasoning_model (str, optional): Specific model for reasoning pipeline. Format: "provider:model" or just "model". Uses main model if None.
            tools (list, optional): The list of internal tools to enable. Defaults to ["web-search", "calculator", "math-operations"].
            tools_config (dict, optional): The configuration for internal tools.
            tool_batch_size (int, optional): The batch size for tool processing. Defaults to 3.
            tool_max_workers (int, optional): The maximum number of workers for tool processing. Defaults to 10.
            max_tokens (Optional[int], optional): Maximum number of tokens to generate. If None, uses instance default or provider default. Defaults to None.
        
        Returns:
            tuple: A tuple containing the tool call response, total tokens used, and details dictionary.
                - tool_call_response: The tool calls with parameters and reasoning.
                - total_tokens (int): The total number of tokens used during the session.
                - details_dict (dict): Dictionary containing reasoning output, steps, token breakdown, and tool call response.
        """
        if self.verbose:
            self.log.debug("=" * 50)
            self.log.debug("Starting UltraGPT Tool Call Mode")
            self.log.debug("=" * 50)
            tool_names = []
            for tool in user_tools:
                if isinstance(tool, dict):
                    tool_names.append(tool.get('name', 'Unknown'))
                else:
                    tool_names.append(getattr(tool, 'name', 'Unknown'))
            self.log.debug("User Tools: " + str(tool_names))
            self.log.debug("Allow Multiple: " + str(allow_multiple))
        else:
            self.log.info("Starting tool call mode with " + str(len(user_tools)) + " user tools")
        
        # Apply config defaults for model parameters
        model = model or config.DEFAULT_MODEL
        steps_model = steps_model or config.DEFAULT_STEPS_MODEL
        reasoning_model = reasoning_model or config.DEFAULT_REASONING_MODEL
        tools_config = tools_config or config.TOOLS_CONFIG
        
        # Apply config defaults for processing parameters
        temperature = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
        reasoning_iterations = reasoning_iterations if reasoning_iterations is not None else config.DEFAULT_REASONING_ITERATIONS
        tools = tools or config.DEFAULT_TOOLS
        tool_batch_size = tool_batch_size if tool_batch_size is not None else config.DEFAULT_TOOL_BATCH_SIZE
        tool_max_workers = tool_max_workers if tool_max_workers is not None else config.DEFAULT_TOOL_MAX_WORKERS
        
        # Validate user tools
        validated_tools = self._validate_user_tools(user_tools)
        
        # Create tool call prompt
        if allow_multiple:
            tool_prompt = generate_multiple_tool_call_prompt(validated_tools)
        else:
            tool_prompt = generate_single_tool_call_prompt(validated_tools)
        
        # Properly integrate tool call prompt with existing system messages
        tool_call_messages = self.integrate_tool_call_prompt(messages, tool_prompt)
        
        # Use UltraGPT's execution layer to analyze and determine tool calls
        reasoning_output = []
        reasoning_tokens = 0
        steps_output = {"steps": [], "conclusion": ""}
        steps_tokens = 0

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            
            if reasoning_pipeline:
                future = executor.submit(
                    self.run_reasoning_pipeline,
                    tool_call_messages, model, temperature, reasoning_iterations,
                    tools, tools_config, tool_batch_size, tool_max_workers, reasoning_model, max_tokens
                )
                futures.append(("reasoning", future))
            
            if steps_pipeline:
                future = executor.submit(
                    self.run_steps_pipeline,
                    tool_call_messages, model, temperature,
                    tools, tools_config, tool_batch_size, tool_max_workers, steps_model, max_tokens
                )
                futures.append(("steps", future))
            
            for name, future in futures:
                try:
                    result, tokens = future.result()
                    if name == "reasoning":
                        reasoning_output = result
                        reasoning_tokens = tokens
                    elif name == "steps":
                        steps_output = result
                        steps_tokens = tokens
                except Exception as e:
                    self.log.error("Pipeline " + name + " failed: " + str(e))
                    if self.verbose:
                        self.log.debug("✗ " + name.title() + " pipeline failed: " + str(e))

        # Combine pipeline outputs for enhanced tool decision making
        conclusion = steps_output.get("conclusion", "")

        if reasoning_pipeline or steps_pipeline:
            combined_prompt = combine_all_pipeline_prompts(reasoning_output, conclusion)
            enhanced_messages = self.append_message_to_system(tool_call_messages, combined_prompt)
        else:
            enhanced_messages = tool_call_messages

        # Generate tool call response using native tool calling
        # AI will always choose at least one tool - parallel_tool_calls controls how many
        parallel_calls = allow_multiple  # Simple: allow_multiple directly controls parallel calls
        
        tool_call_response, tokens = self.chat_with_model_tools(
            enhanced_messages, 
            user_tools=validated_tools,
            model=model, 
            temperature=temperature,
            tools=tools,
            tools_config=tools_config,
            tool_batch_size=tool_batch_size,
            tool_max_workers=tool_max_workers,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_calls
        )

        total_tokens = reasoning_tokens + steps_tokens + tokens
        
        # Create details_dict similar to chat method for consistency
        details_dict = {
            "reasoning": reasoning_output,
            "steps": steps_output.get("steps", []),
            "conclusion": steps_output.get("conclusion", ""),
            "reasoning_tokens": reasoning_tokens,
            "steps_tokens": steps_tokens,
            "final_tokens": tokens
        }
        
        if self.verbose:
            self.log.debug("✓ Tool call analysis completed")
            # Handle native tool calling response format
            if tool_call_response.get('tool_calls'):
                self.log.debug("Generated " + str(len(tool_call_response.get('tool_calls', []))) + " tool calls")
                for i, tool_call in enumerate(tool_call_response.get('tool_calls', []), 1):
                    tool_name = tool_call.get('function', {}).get('name', 'Unknown')
                    self.log.debug("  " + str(i) + ". " + tool_name)
            elif tool_call_response.get('content'):
                self.log.debug("AI response without tool calls: " + str(tool_call_response.get('content', ''))[:100] + "...")
            self.log.debug("Total tokens used: " + str(total_tokens))
        else:
            self.log.info("Tool call completed with " + str(total_tokens) + " tokens")
        
        # Simplify response format - remove role and content, return only tool_calls
        if tool_call_response.get('tool_calls'):
            if allow_multiple:
                # Return all tool_calls as array for multiple tools
                simplified_response = tool_call_response.get('tool_calls')
            else:
                # Return only first tool_call (not in array) for single tool
                simplified_response = tool_call_response.get('tool_calls')[0]
        else:
            # For non-tool responses, only return content if it's not null or empty
            content = tool_call_response.get('content')
            if content and content.strip():
                simplified_response = {"content": content}
            else:
                simplified_response = None
        
        return simplified_response, total_tokens, details_dict

    def _validate_user_tools(self, user_tools: list) -> list:
        """Validate and format user tools (both UserTool and ExpertTool)"""
        validated_tools = []
        
        for tool in user_tools:
            if isinstance(tool, dict):
                # Ensure all required fields are present for UserTool
                required_fields = ['name', 'description', 'parameters_schema', 'usage_guide', 'when_to_use']
                if all(field in tool for field in required_fields):
                    validated_tools.append(tool)
                else:
                    missing = [field for field in required_fields if field not in tool]
                    self.log.warning("Tool missing required fields: " + str(missing))
                    if self.verbose:
                        self.log.debug("⚠ Tool missing fields: " + str(missing))
            elif hasattr(tool, 'model_dump'):
                # Pydantic model (UserTool or ExpertTool)
                validated_tools.append(tool.model_dump())
            else:
                self.log.warning("Invalid tool format: " + str(type(tool)))
                if self.verbose:
                    self.log.debug("⚠ Invalid tool format: " + str(type(tool)))
        
        return validated_tools