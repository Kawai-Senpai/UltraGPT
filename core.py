from openai import OpenAI 
import tiktoken
from .prompts import (
generate_steps_prompt, 
each_step_prompt, generate_reasoning_prompt, 
generate_conclusion_prompt, combine_all_pipeline_prompts,
generate_single_tool_call_prompt, generate_multiple_tool_call_prompt
)
from .schemas import Steps, Reasoning
from concurrent.futures import ThreadPoolExecutor
from ultraprint.logging import logger
from .providers import ProviderManager, OpenAIProvider, ClaudeProvider
from .tools_manager import ToolManager
from .simple_rag import SimpleRAG
from typing import Optional, List, Dict, Union
from . import config
from .tools.web_search.core import google_search, scrape_url


class UltraGPT:
    def __init__(
        self, 
        api_key: str = None,
        openai_api_key: str = None,
        claude_api_key: str = None,
        google_api_key: str = None,
        search_engine_id: str = None,
        max_tokens: Optional[int] = None,
        input_truncation: Union[str, int] = None,
        verbose: bool = False,
        logger_name: str = 'ultragpt',
        logger_filename: str = 'debug/ultragpt.log',
        log_extra_info: bool = False,
        log_to_file: bool = False,
        log_to_console: bool = False,
        log_level: str = 'DEBUG',
        rag_system: Optional[SimpleRAG] = None,
    ):
        """
        Initialize the UltraGPT class with multi-provider support.
        Args:
            api_key (str, optional): The API key for accessing the OpenAI service.
            claude_api_key (str, optional): The API key for accessing Claude/Anthropic service.
            google_api_key (str, optional): Google Custom Search API key for web search tool.
            search_engine_id (str, optional): Google Custom Search Engine ID for web search tool.
            max_tokens (int, optional): Maximum number of tokens to generate. Set to None to use provider defaults. Defaults to 4096.
            input_truncation (Union[str, int], optional): Input truncation mode. Can be "AUTO" (use model input limits), "OFF" (no truncation), or an integer token count. Defaults to "AUTO".
            verbose (bool, optional): Whether to enable verbose logging. Defaults to False.
            logger_name (str, optional): The name of the logger. Defaults to 'ultragpt'.
            logger_filename (str, optional): The filename for the logger. Defaults to 'debug/ultragpt.log'.
            log_extra_info (bool, optional): Whether to include extra info in logs. Defaults to False.
            log_to_file (bool, optional): Whether to log to a file. Defaults to False.
            log_to_console (bool, optional): Whether to log to console. Defaults to True.
            log_level (str, optional): The logging level. Defaults to 'DEBUG'.
            rag_system (SimpleRAG, optional): Pre-initialized SimpleRAG object for document retrieval. If not provided, RAG functionality will be disabled.
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
        
        # Store max_tokens setting - None means use provider-specific model limits
        self.max_tokens = max_tokens
        
        # Store input_truncation setting - use config default if not provided
        self.input_truncation = input_truncation if input_truncation is not None else config.DEFAULT_INPUT_TRUNCATION
        
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
        
        # Initialize ToolManager
        self.tool_manager = ToolManager(self)
        
        # Initialize RAG system only if provided
        self.rag = None
        if rag_system is not None:
            self.rag = rag_system
            if self.verbose:
                self.log.debug(f"Using provided RAG system with {len(self.rag.documents)} documents")
        else:
            # No RAG system provided - RAG functionality disabled
            if self.verbose:
                self.log.debug("No RAG system provided - RAG functionality disabled")

    def chat_with_ai_sync(
        self,
        messages: list,
        model: str,
        temperature: float,
        tools: list,
        tools_config: dict,
        max_tokens: Optional[int] = None,
        input_truncation: Optional[Union[str, int]] = None,
        deepthink: Optional[bool] = None
    ):
        """
        Sends a synchronous chat request to the specified AI provider and processes the response.
        Args:
            messages (list): A list of message dictionaries to be sent to the AI provider.
            model (str): The model to use (format: "provider:model" or just "model" for OpenAI).
            temperature (float): The temperature for the model's output.
            tools (list): The list of tools to enable.
            tools_config (dict): The configuration for the tools.
            max_tokens (Optional[int]): Maximum tokens to generate. Overrides instance default if provided.
            input_truncation (Optional[Union[str, int]]): Override input truncation setting. Can be "AUTO", "OFF", or a token count.
            deepthink (Optional[bool]): Enable deep thinking mode for supported models.
        Returns:
            tuple: A tuple containing the response content, total tokens, and details dict.
        Raises:
            Exception: If the request to the AI provider fails.
        Logs:
            Debug: Logs the number of messages sent, the number of tokens in the response, and any errors encountered.
            Verbose: Optionally logs detailed steps of the request and response process.
        """
        try:
            # Apply input truncation before processing
            messages = self.apply_input_truncation(messages, model, input_truncation)
            
            self.log.debug("Sending request to AI provider (msgs: " + str(len(messages)) + ")")
            if self.verbose:
                provider_name, model_name = self.provider_manager.parse_model_string(model)
                self.log.debug(f"AI Request → Provider: {provider_name}, Model: {model_name}, Messages: " + str(len(messages)))
                self.log.debug("Checking for tool needs...")
            
            tool_response, tool_usage_details = self.execute_tools(history=messages, tools=tools, tools_config=tools_config)
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
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                deepthink=deepthink
            )
            
            details_dict = {
                "tools_used": tool_usage_details
            }
            
            self.log.debug("Response received (tokens: " + str(tokens) + ")")
            if self.verbose:
                self.log.debug("✓ Response received (" + str(tokens) + " tokens)")
            return content, tokens, details_dict
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
        max_tokens: Optional[int] = None,
        input_truncation: Optional[Union[str, int]] = None,
        deepthink: Optional[bool] = None
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
            max_tokens (Optional[int]): Maximum tokens to generate. Overrides instance default if provided.
            input_truncation (Optional[Union[str, int]]): Override input truncation setting. Can be "AUTO", "OFF", or a token count.
            deepthink (Optional[bool]): Enable deep thinking mode for supported models.
        Returns:
            tuple: A tuple containing the parsed content, total tokens, and details dict.
        Raises:
            Exception: If the parse request fails.
        """
        # Use config defaults if not provided
        model = model or config.DEFAULT_PARSE_MODEL
        temperature = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
        try:
            # Apply input truncation before processing
            messages = self.apply_input_truncation(messages, model, input_truncation)
            
            self.log.debug("Sending parse request with schema: %s", schema)
            
            tool_response, tool_usage_details = self.execute_tools(history=messages, tools=tools, tools_config=tools_config)
            if tool_response:
                tool_response = "Tool Responses:\n" + tool_response
            messages = self.append_message_to_system(messages, tool_response)

            content, tokens = self.provider_manager.chat_completion_with_schema(
                model=model,
                messages=messages,
                schema=schema,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                deepthink=deepthink
            )
            
            details_dict = {
                "tools_used": tool_usage_details
            }
            
            self.log.debug("Parse response received (tokens: " + str(tokens) + ")")
            return content, tokens, details_dict
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
        max_tokens: Optional[int] = None,
        input_truncation: Optional[Union[str, int]] = None,
        parallel_tool_calls: Optional[bool] = None,
        deepthink: Optional[bool] = None
    ):
        """
        Sends a chat message to the model with native tool calling support.
        AI will always be required to choose at least one tool from the provided tools.
        
        Args:
            messages (list): A list of message dictionaries to be sent to the model.
            user_tools (list): List of user-defined tools with schemas and prompts.
            model (str): The model to use (format: "provider:model" or just "model" for OpenAI).
            temperature (float): The temperature for the model's output.
            tools (list): The list of internal tools to enable.
            tools_config (dict): The configuration for internal tools.
            max_tokens (Optional[int]): Maximum tokens to generate. Overrides instance default if provided.
            input_truncation (Optional[Union[str, int]]): Override input truncation setting. Can be "AUTO", "OFF", or a token count.
            parallel_tool_calls (Optional[bool]): Whether to allow parallel tool calls.
            deepthink (Optional[bool]): Enable deep thinking mode for supported models.
        """
        try:
            # Apply input truncation before processing
            messages = self.apply_input_truncation(messages, model, input_truncation)
            
            self.log.debug("Sending native tool calling request")
            
            # Execute UltraGPT tools if any
            tool_response, tool_usage_details = self.execute_tools(
                history=messages, 
                tools=tools, 
                tools_config=tools_config
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
                parallel_tool_calls=parallel_tool_calls,
                deepthink=deepthink
            )
            
            details_dict = {
                "tools_used": tool_usage_details
            }
            
            self.log.debug("Native tool calling response received (tokens: " + str(tokens) + ")")
            return response_message, tokens, details_dict
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
                "description": "Detailed reasoning for why this tool was executed and how it will help solve the user's request. Write this in a user friendly way and organic, as if you are explaining to the user directly why you chose it and so on as if you are humanly reasoning through the problem. This will be shown to the user. Use this as the message tool for the user attached to each tool"
            }
            
            # Add stop_after_tool_call parameter  
            parameters_schema["properties"]["stop_after_tool_call"] = {
                "type": "boolean",
                "description": "Whether to stop execution after this tool call (true if task is complete or user input needed, false to continue with more tools). It is suggested to not set this true immidiately after a important tool call, but rather after reviewing the result and then deciding if the tool worked properly or not."
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

    def append_message_to_system(self, messages: list, new_message: str):
        # add message after system message
        processed = []
        has_system_message = False
        
        for message in messages:
            if message["role"] == "system" or message["role"] == "developer":
                processed.append({
                    "role": message["role"],
                    "content": f"{message['content']}\n{new_message}"
                })
                has_system_message = True
            else:
                processed.append(message)
        
        # If no system message exists, add one at the beginning
        if not has_system_message:
            processed.insert(0, {
                "role": "system",
                "content": new_message
            })
        
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
        steps_model: str = None,
        max_tokens: Optional[int] = None,
        input_truncation: Optional[Union[str, int]] = None,
        deepthink: Optional[bool] = None
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
        all_tools_used = []

        messages = self.turnoff_system_message(messages)
        steps_generator_message = messages + [{"role": "system", "content": generate_steps_prompt()}]

        steps_json, tokens, steps_details = self.chat_with_model_parse(steps_generator_message, schema=Steps, model=active_model, temperature=temperature, tools=tools, tools_config=tools_config, max_tokens=max_tokens, input_truncation=input_truncation, deepthink=deepthink)
        total_tokens += tokens
        all_tools_used.extend(steps_details.get("tools_used", []))
        
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
            step_response, tokens, step_details = self.chat_with_ai_sync(step_message, model=active_model, temperature=temperature, tools=tools, tools_config=tools_config, max_tokens=max_tokens, input_truncation=input_truncation, deepthink=deepthink)
            self.log.debug("Step " + str(idx) + " response: " + step_response[:100] + "...")
            total_tokens += tokens
            all_tools_used.extend(step_details.get("tools_used", []))
            memory.append(
                {
                    "step": step,
                    "answer": step_response
                }
            )

        # Generate final conclusion
        conclusion_prompt = generate_conclusion_prompt(memory)
        conclusion_message = messages + [{"role": "system", "content": conclusion_prompt}]
        conclusion, tokens, conclusion_details = self.chat_with_ai_sync(conclusion_message, model=active_model, temperature=temperature, tools=tools, tools_config=tools_config, max_tokens=max_tokens, input_truncation=input_truncation, deepthink=deepthink)
        total_tokens += tokens
        all_tools_used.extend(conclusion_details.get("tools_used", []))

        if self.verbose:
            self.log.debug("✓ Steps pipeline completed")
        
        return {
            "steps": memory,
            "conclusion": conclusion
        }, total_tokens, {"tools_used": all_tools_used}

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
        deepthink: Optional[bool] = None
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
        all_tools_used = []
        messages = self.turnoff_system_message(messages)

        for iteration in range(reasoning_iterations):
            if self.verbose:
                self.log.debug("Iteration " + str(iteration + 1) + "/" + str(reasoning_iterations))
            self.log.debug("Iteration " + str(iteration + 1) + "/" + str(reasoning_iterations))
            # Generate new thoughts based on all previous thoughts
            reasoning_message = messages + [
                {"role": "system", "content": generate_reasoning_prompt(all_thoughts)}
            ]
            
            reasoning_json, tokens, iteration_details = self.chat_with_model_parse(
                reasoning_message, 
                schema=Reasoning,
                model=active_model,
                temperature=temperature,
                tools=tools,
                tools_config=tools_config,
                max_tokens=max_tokens,
                input_truncation=input_truncation,
                deepthink=deepthink
            )
            total_tokens += tokens
            all_tools_used.extend(iteration_details.get("tools_used", []))
            
            new_thoughts = reasoning_json.get("thoughts", [])
            all_thoughts.extend(new_thoughts)
            
            if self.verbose:
                self.log.debug("Generated " + str(len(new_thoughts)) + " thoughts:")
                for idx, thought in enumerate(new_thoughts, 1):
                    self.log.debug("  " + str(idx) + ". " + thought)
            else:
                self.log.debug("Generated " + str(len(new_thoughts)) + " new thoughts")

        return all_thoughts, total_tokens, {"tools_used": all_tools_used}
    
    #! Main Chat Function ---------------------------------------------------
    def chat(
        self,
        messages: list,
        schema=None,
        model: str = None,
        temperature: float = None,
        max_tokens: Optional[int] = None,  # Override instance default if provided
        input_truncation: Optional[Union[str, int]] = None,  # Override instance input_truncation setting
        reasoning_iterations: int = None,
        steps_pipeline: bool = False,
        reasoning_pipeline: bool = False,
        steps_model: str = None,
        reasoning_model: str = None,
        tools: list = None,
        tools_config: dict = None,
        rag_enabled: bool = True,
        rag_labels: Optional[List[str]] = None,
        rag_top_k: int = 3
    ):
        """
        Initiates a chat session with the given messages and optional schema.
        Args:
            messages (list): A list of message dictionaries to be processed.
            schema (optional): A schema to parse the final output, defaults to None.
            model (str, optional): The model to use. Format: "provider:model" (e.g., "claude:claude-3-sonnet-20240229") or just "model" (defaults to OpenAI). Defaults to "gpt-4o".
            temperature (float, optional): The temperature for the model's output. Defaults to 0.7.
            max_tokens (int, optional): Maximum tokens to generate. Overrides instance default if provided. None uses provider defaults.
            input_truncation (Union[str, int], optional): Override input truncation setting. Can be "AUTO" (use model limits), "OFF" (no truncation), or an integer token count. Defaults to instance setting.
            reasoning_iterations (int, optional): The number of reasoning iterations. Defaults to 3.
            steps_pipeline (bool, optional): Whether to use steps pipeline. Defaults to True.
            reasoning_pipeline (bool, optional): Whether to use reasoning pipeline. Defaults to True.
            steps_model (str, optional): Specific model for steps pipeline. Format: "provider:model" or just "model". Uses main model if None.
            reasoning_model (str, optional): Specific model for reasoning pipeline. Format: "provider:model" or just "model". Uses main model if None.
            tools (list, optional): The list of tools to enable. Defaults to ["web-search", "calculator", "math-operations"].
            tools_config (dict, optional): The configuration for the tools. Each tool's "model" field supports "provider:model" format. Defaults to predefined configurations.
            rag_enabled (bool, optional): Whether to use RAG for context retrieval. Defaults to True.
            rag_labels (Optional[List[str]], optional): Filter RAG search by specific labels. Defaults to None (search all).
            rag_top_k (int, optional): Number of top RAG results to include. Defaults to 3.
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
        reasoning_tools_used = []
        steps_output = {"steps": [], "conclusion": ""}
        steps_tokens = 0
        steps_tools_used = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            
            if reasoning_pipeline:
                futures.append({
                    "type": "reasoning",
                    # Deepthink MUST be disabled inside pipelines to avoid loops
                    "future": executor.submit(self.run_reasoning_pipeline, messages, model, temperature, reasoning_iterations, tools, tools_config, reasoning_model, max_tokens, input_truncation, False)
                })
            
            if steps_pipeline:
                futures.append({
                    "type": "steps",
                    # Deepthink MUST be disabled inside pipelines to avoid loops
                    "future": executor.submit(self.run_steps_pipeline, messages, model, temperature, tools, tools_config, steps_model, max_tokens, input_truncation, False)
                })

            for future in futures:
                if future["type"] == "reasoning":
                    reasoning_output, reasoning_tokens, reasoning_details = future["future"].result()
                    reasoning_tools_used = reasoning_details.get("tools_used", [])
                elif future["type"] == "steps":
                    steps_output, steps_tokens, steps_details = future["future"].result()
                    steps_tools_used = steps_details.get("tools_used", [])

        conclusion = steps_output.get("conclusion", "")
        steps = steps_output.get("steps", [])

        if reasoning_pipeline or steps_pipeline:
            prompt = combine_all_pipeline_prompts(reasoning_output, conclusion)
            messages = self.add_message_before_system(messages, {"role": "user", "content": prompt})

        # Add RAG context if enabled and RAG system exists with documents
        rag_context = ""
        if rag_enabled and self.rag is not None and self.rag.documents:
            # Extract user query for RAG search
            user_query = ""
            system_content = ""
            
            # Get the latest user message and any system message
            for msg in reversed(messages):
                if msg.get("role") == "user" and not user_query:
                    user_query = msg.get("content", "")
                elif msg.get("role") in ["system", "developer"] and not system_content:
                    system_content = msg.get("content", "")
            
            if user_query:
                rag_context = self.rag.get_relevant_context(
                    query=user_query,
                    system_message=system_content,
                    top_k=rag_top_k,
                    labels=rag_labels
                )
                
                if rag_context:
                    messages = self.append_message_to_system(messages, rag_context)
                    if self.verbose:
                        self.log.debug(f"Added RAG context ({rag_top_k} chunks)")
                        self.log.debug(f"RAG context content preview: {rag_context[:200]}...")
                        # Log the system message after RAG injection
                        for msg in messages:
                            if msg.get("role") in ["system", "developer"]:
                                self.log.debug(f"Final system message preview: {msg['content'][:300]}...")

        # Use reasoning_pipeline flag to control deepthink for the final AI call
        final_deepthink = bool(reasoning_pipeline)
        if schema:
            final_output, tokens, final_details = self.chat_with_model_parse(
                messages, schema=schema, model=model, temperature=temperature,
                tools=tools, tools_config=tools_config, max_tokens=max_tokens,
                input_truncation=input_truncation, deepthink=final_deepthink
            )
        else:
            final_output, tokens, final_details = self.chat_with_ai_sync(
                messages, model=model, temperature=temperature,
                tools=tools, tools_config=tools_config, max_tokens=max_tokens,
                input_truncation=input_truncation, deepthink=final_deepthink
            )

        if steps:
            steps.append(conclusion)
        
        # Combine all tools used from all stages
        all_tools_used = reasoning_tools_used + steps_tools_used + final_details.get("tools_used", [])
            
        details_dict = {
            "reasoning": reasoning_output,
            "steps": steps,
            "reasoning_tokens": reasoning_tokens,
            "steps_tokens": steps_tokens,
            "final_tokens": tokens,
            "tools_used": all_tools_used
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

    def execute_tools(
        self,
        history: list,
        tools: list,
        tools_config: dict
    ) -> tuple:
        """Execute tools using native AI tool calling - delegates to ToolManager, returns (result, tool_usage_details)"""
        return self.tool_manager.execute_tools(history, tools, tools_config)

    #! Tool Call Functionality --------------------------------------------
    def tool_call(
        self,
        messages: list,
        user_tools: list,
        allow_multiple: bool = True,
        model: str = None,  # Format: "provider:model" or just "model" (defaults to OpenAI)
        temperature: float = None,
        input_truncation: Optional[Union[str, int]] = None,  # Override instance input_truncation setting
        reasoning_iterations: int = None,
        steps_pipeline: bool = False,
        reasoning_pipeline: bool = False,
        steps_model: str = None,  # Format: "provider:model" or just "model" (defaults to OpenAI)  
        reasoning_model: str = None,  # Format: "provider:model" or just "model" (defaults to OpenAI)
        tools: list = None,
        tools_config: dict = None,
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
            input_truncation (Union[str, int], optional): Override input truncation setting. Can be "AUTO", "OFF", or a token count. Defaults to instance setting.
            reasoning_iterations (int, optional): The number of reasoning iterations. Defaults to 3.
            steps_pipeline (bool, optional): Whether to use steps pipeline. Defaults to True.
            reasoning_pipeline (bool, optional): Whether to use reasoning pipeline. Defaults to True.
            steps_model (str, optional): Specific model for steps pipeline. Format: "provider:model" or just "model". Uses main model if None.
            reasoning_model (str, optional): Specific model for reasoning pipeline. Format: "provider:model" or just "model". Uses main model if None.
            tools (list, optional): The list of internal tools to enable. Defaults to ["web-search", "calculator", "math-operations"].
            tools_config (dict, optional): The configuration for internal tools.
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
        reasoning_tools_used = []
        steps_output = {"steps": [], "conclusion": ""}
        steps_tokens = 0
        steps_tools_used = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            
            if reasoning_pipeline:
                future = executor.submit(
                    self.run_reasoning_pipeline,
                    tool_call_messages, model, temperature, reasoning_iterations,
                    tools, tools_config, reasoning_model, max_tokens, input_truncation, False
                )
                futures.append(("reasoning", future))
            
            if steps_pipeline:
                future = executor.submit(
                    self.run_steps_pipeline,
                    tool_call_messages, model, temperature,
                    tools, tools_config, steps_model, max_tokens, input_truncation, False
                )
                futures.append(("steps", future))
            
            for name, future in futures:
                try:
                    result, tokens, details = future.result()
                    if name == "reasoning":
                        reasoning_output = result
                        reasoning_tokens = tokens
                        reasoning_tools_used = details.get("tools_used", [])
                    elif name == "steps":
                        steps_output = result
                        steps_tokens = tokens
                        steps_tools_used = details.get("tools_used", [])
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
        
        # Use reasoning_pipeline to control deepthink for the final native tool call
        final_deepthink = bool(reasoning_pipeline)
        tool_call_response, tokens, final_details = self.chat_with_model_tools(
            enhanced_messages, 
            user_tools=validated_tools,
            model=model, 
            temperature=temperature,
            tools=tools,
            tools_config=tools_config,
            max_tokens=max_tokens,
            input_truncation=input_truncation,
            parallel_tool_calls=parallel_calls,
            deepthink=final_deepthink
        )

        total_tokens = reasoning_tokens + steps_tokens + tokens
        
        # Combine all tools used from all stages
        all_tools_used = reasoning_tools_used + steps_tools_used + final_details.get("tools_used", [])
        
        # Create details_dict similar to chat method for consistency
        details_dict = {
            "reasoning": reasoning_output,
            "steps": steps_output.get("steps", []),
            "conclusion": steps_output.get("conclusion", ""),
            "reasoning_tokens": reasoning_tokens,
            "steps_tokens": steps_tokens,
            "final_tokens": tokens,
            "tools_used": all_tools_used
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

    #! Web Search Functionality -----------------------------------------------
    def web_search(
        self,
        query: Optional[str] = None,
        url: Optional[str] = None,
        num_results: int = 5,
        enable_scraping: bool = True,
        max_scrape_length: int = 5000,
        scrape_timeout: int = 15,
        return_debug_info: bool = False
    ) -> Union[List[Dict], Dict]:
        """
        Perform web search using Google Custom Search API and/or scrape specific URLs.
        This is a standalone web search functionality that doesn't require AI.
        
        Args:
            query (str, optional): Search query string for Google Custom Search
            url (str, optional): Specific URL to scrape content from
            num_results (int, optional): Number of search results to return (max 10). Defaults to 5.
            enable_scraping (bool, optional): Whether to scrape content from search results. Defaults to True.
            max_scrape_length (int, optional): Maximum length of scraped content per page. Defaults to 5000.
            scrape_timeout (int, optional): Timeout for scraping requests in seconds. Defaults to 15.
            return_debug_info (bool, optional): Whether to include debug information in response. Defaults to False.
            
        Returns:
            Union[List[Dict], Dict]: Search results or scraped content with metadata
            
        Examples:
            # Web search only
            results = ultra.web_search(query="Python tutorials", num_results=3)
            
            # URL scraping only
            content = ultra.web_search(url="https://example.com")
            
            # Web search with scraping disabled
            results = ultra.web_search(query="AI news", enable_scraping=False)
            
        Raises:
            ValueError: If neither query nor url is provided, or if Google API credentials are missing for search
        """
        
        if not query and not url:
            raise ValueError("Either 'query' for web search or 'url' for scraping must be provided")
        
        if self.verbose:
            self.log.debug("=" * 50)
            self.log.debug("Starting web search operation")
            if query:
                self.log.debug(f"Search query: {query}")
            if url:
                self.log.debug(f"Scraping URL: {url}")
        
        # URL scraping mode
        if url:
            if self.verbose:
                self.log.debug(f"Scraping content from: {url}")
            
            try:
                content = scrape_url(url, timeout=scrape_timeout, max_length=max_scrape_length)
                
                result = {
                    "type": "url_scraping",
                    "url": url,
                    "success": content is not None,
                    "content": content or "Unable to scrape content (blocked by robots.txt or error)",
                    "content_length": len(content) if content else 0,
                    "scraped_at": __import__('datetime').datetime.now().isoformat()
                }
                
                if self.verbose:
                    status = "✓ Success" if content else "✗ Failed"
                    length = len(content) if content else 0
                    self.log.debug(f"{status} - Content length: {length} characters")
                
                return result
                
            except Exception as e:
                error_msg = f"Error scraping URL {url}: {str(e)}"
                if self.verbose:
                    self.log.error(error_msg)
                
                return {
                    "type": "url_scraping",
                    "url": url,
                    "success": False,
                    "content": "",
                    "error": error_msg,
                    "scraped_at": __import__('datetime').datetime.now().isoformat()
                }
        
        # Web search mode
        if query:
            # Check for Google API credentials
            api_key = self.google_api_key or __import__('os').getenv('GOOGLE_API_KEY')
            search_engine_id = self.search_engine_id or __import__('os').getenv('GOOGLE_SEARCH_ENGINE_ID')
            
            if not api_key or not search_engine_id:
                error_msg = "Google API credentials not configured. Please provide google_api_key and search_engine_id to UltraGPT constructor or set environment variables GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID."
                if self.verbose:
                    self.log.error(error_msg)
                raise ValueError(error_msg)
            
            if self.verbose:
                self.log.debug(f"Performing Google search with {num_results} results")
            
            try:
                # Perform Google search
                search_results, debug_info = google_search(query, api_key, search_engine_id, num_results)
                
                if self.verbose:
                    self.log.debug(f"Google API returned {len(search_results)} results")
                
                if not search_results:
                    result = {
                        "type": "web_search",
                        "query": query,
                        "results": [],
                        "total_results": 0,
                        "searched_at": __import__('datetime').datetime.now().isoformat(),
                        "message": "No search results found"
                    }
                    
                    if return_debug_info:
                        result["debug_info"] = debug_info
                    
                    if self.verbose:
                        self.log.debug("✗ No search results found")
                    
                    return result
                
                # Process search results
                processed_results = []
                
                # Initialize basic result structure first
                for i, result in enumerate(search_results, 1):
                    title = result.get("title", "")
                    link = result.get("link", "")
                    snippet = result.get("snippet", "")
                    
                    processed_result = {
                        "rank": i,
                        "title": title,
                        "url": link,
                        "snippet": snippet,
                        "scraped_content": None,
                        "scraping_success": False
                    }
                    processed_results.append(processed_result)
                
                # Scrape all URLs in parallel if scraping is enabled
                if enable_scraping:
                    if self.verbose:
                        self.log.debug("Starting parallel scraping of search results...")
                    
                    def scrape_single_result(result_data):
                        """Helper function to scrape a single result"""
                        i, result = result_data
                        link = result["url"]
                        if not link:
                            return i, None, False, "No URL provided"
                        
                        try:
                            scraped_content = scrape_url(link, timeout=scrape_timeout, max_length=max_scrape_length)
                            if scraped_content:
                                if self.verbose:
                                    self.log.debug(f"  ✓ Result {i}: Scraped {len(scraped_content)} characters from {result['title'][:50]}...")
                                return i, scraped_content, True, None
                            else:
                                if self.verbose:
                                    self.log.debug(f"  ✗ Result {i}: Scraping failed (blocked or error) for {result['title'][:50]}...")
                                return i, None, False, "Scraping failed (blocked or error)"
                        except Exception as e:
                            if self.verbose:
                                self.log.debug(f"  ✗ Result {i}: Scraping error for {result['title'][:50]}...: {str(e)}")
                            return i, None, False, str(e)
                    
                    # Use ThreadPoolExecutor for parallel scraping
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    import threading
                    
                    # Create list of (index, result) tuples for parallel processing
                    scraping_tasks = [(result["rank"], result) for result in processed_results]
                    
                    # Use max 5 threads to avoid overwhelming servers
                    max_workers = min(5, len(scraping_tasks))
                    
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all scraping tasks
                        future_to_rank = {executor.submit(scrape_single_result, task): task[0] for task in scraping_tasks}
                        
                        # Collect results as they complete
                        for future in as_completed(future_to_rank):
                            rank = future_to_rank[future]
                            try:
                                result_rank, scraped_content, success, error = future.result()
                                # Update the corresponding result in processed_results
                                for result in processed_results:
                                    if result["rank"] == result_rank:
                                        result["scraped_content"] = scraped_content
                                        result["scraping_success"] = success
                                        if error and self.verbose:
                                            self.log.debug(f"  Error for rank {result_rank}: {error}")
                                        break
                            except Exception as e:
                                if self.verbose:
                                    self.log.debug(f"  Exception in parallel scraping for rank {rank}: {str(e)}")
                    
                    if self.verbose:
                        scraped_count = sum(1 for r in processed_results if r["scraping_success"])
                        self.log.debug(f"Parallel scraping completed: {scraped_count}/{len(processed_results)} URLs scraped successfully")
                
                final_result = {
                    "type": "web_search",
                    "query": query,
                    "results": processed_results,
                    "total_results": len(processed_results),
                    "scraping_enabled": enable_scraping,
                    "searched_at": __import__('datetime').datetime.now().isoformat()
                }
                
                if return_debug_info:
                    final_result["debug_info"] = debug_info
                
                return final_result
                
            except Exception as e:
                error_msg = f"Error performing web search for '{query}': {str(e)}"
                if self.verbose:
                    self.log.error(error_msg)
                
                result = {
                    "type": "web_search",
                    "query": query,
                    "results": [],
                    "total_results": 0,
                    "error": error_msg,
                    "searched_at": __import__('datetime').datetime.now().isoformat()
                }
                
                if return_debug_info:
                    result["debug_info"] = [f"ERROR: {error_msg}"]
                
                return result

    # Token Management Functions
    def count_tokens(self, content: Union[str, List[Dict]], model: str = "gpt-4") -> int:
        """
        Count tokens in a string or list of messages using tiktoken.
        
        Args:
            content (Union[str, List[Dict]]): String content or list of OpenAI message format
            model (str): Model name for tokenizer (default: gpt-4)
            
        Returns:
            int: Number of tokens
        """
        try:
            # Get the encoding for the model
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                # If model not found, use cl100k_base (GPT-4 encoding)
                encoding = tiktoken.get_encoding("cl100k_base")
            
            if isinstance(content, str):
                # Simple string token count
                return len(encoding.encode(content))
            
            elif isinstance(content, list):
                # OpenAI messages format
                total_tokens = 0
                
                for message in content:
                    if isinstance(message, dict):
                        # Count tokens in role
                        if 'role' in message:
                            total_tokens += len(encoding.encode(message['role']))
                        
                        # Count tokens in content
                        if 'content' in message:
                            if isinstance(message['content'], str):
                                total_tokens += len(encoding.encode(message['content']))
                            elif isinstance(message['content'], list):
                                # Handle multi-modal content
                                for item in message['content']:
                                    if isinstance(item, dict) and 'text' in item:
                                        total_tokens += len(encoding.encode(item['text']))
                        
                        # Count tokens in name if present
                        if 'name' in message:
                            total_tokens += len(encoding.encode(message['name']))
                        
                        # Count tokens in function_call if present
                        if 'function_call' in message:
                            if isinstance(message['function_call'], dict):
                                if 'name' in message['function_call']:
                                    total_tokens += len(encoding.encode(message['function_call']['name']))
                                if 'arguments' in message['function_call']:
                                    total_tokens += len(encoding.encode(message['function_call']['arguments']))
                        
                        # Count tokens in tool_calls if present
                        if 'tool_calls' in message and isinstance(message['tool_calls'], list):
                            for tool_call in message['tool_calls']:
                                if isinstance(tool_call, dict):
                                    if 'function' in tool_call and isinstance(tool_call['function'], dict):
                                        if 'name' in tool_call['function']:
                                            total_tokens += len(encoding.encode(tool_call['function']['name']))
                                        if 'arguments' in tool_call['function']:
                                            total_tokens += len(encoding.encode(tool_call['function']['arguments']))
                
                # Add overhead tokens for message formatting (approximate)
                total_tokens += len(content) * 3  # Overhead per message
                total_tokens += 3  # Overhead for the conversation
                
                return total_tokens
            
            else:
                raise ValueError("Content must be a string or list of message dictionaries")
                
        except Exception as e:
            self.log.error(f"Error counting tokens: {str(e)}")
            # Fallback: rough estimation (4 chars per token)
            if isinstance(content, str):
                return len(content) // 4
            elif isinstance(content, list):
                total_chars = sum(len(str(msg)) for msg in content)
                return total_chars // 4
            return 0

    def limit_tokens(
        self, 
        messages: List[Dict], 
        max_tokens: int, 
        model: str = "gpt-4",
        keep_newest: bool = True,
        preserve_system: bool = True
    ) -> List[Dict]:
        """
        Limit messages to fit within a token count by filtering older or newer messages.
        Preserves original message order and handles all OpenAI message formats including
        tool calls, tool results, function calls, and multi-modal content.
        
        Args:
            messages (List[Dict]): List of OpenAI message format dictionaries
            max_tokens (int): Maximum token count to maintain
            model (str): Model name for tokenizer (default: gpt-4)
            keep_newest (bool): If True, keep newest messages. If False, keep oldest (default: True)
            preserve_system (bool): If True, always keep system/developer messages (default: True)
            
        Returns:
            List[Dict]: Filtered list of messages within token limit, preserving original order
        """
        try:
            if not messages:
                return []
            
            if preserve_system:
                # Mark which messages are system/developer messages and their original positions
                system_indices = set()
                system_messages = []
                non_system_messages = []
                
                for i, msg in enumerate(messages):
                    if isinstance(msg, dict) and msg.get('role') in ['system', 'developer']:
                        system_indices.add(i)
                        system_messages.append((i, msg))
                    else:
                        non_system_messages.append((i, msg))
                
                # Calculate tokens for system messages
                system_tokens = self.count_tokens([msg for _, msg in system_messages], model) if system_messages else 0
                
                # Remaining tokens for non-system messages
                remaining_tokens = max_tokens - system_tokens
                
                if remaining_tokens <= 0:
                    # If system messages exceed limit, return only system messages in original order
                    return [msg for _, msg in system_messages]
                
                # Filter non-system messages based on keep_newest preference
                if keep_newest:
                    # Start from the newest non-system messages and work backwards
                    selected_non_system = []
                    current_tokens = 0
                    
                    for i, msg in reversed(non_system_messages):
                        message_tokens = self.count_tokens([msg], model)
                        if current_tokens + message_tokens <= remaining_tokens:
                            selected_non_system.append((i, msg))
                            current_tokens += message_tokens
                        else:
                            break
                    
                    # Reverse to restore chronological order within selected messages
                    selected_non_system.reverse()
                else:
                    # Start from the oldest non-system messages and work forwards
                    selected_non_system = []
                    current_tokens = 0
                    
                    for i, msg in non_system_messages:
                        message_tokens = self.count_tokens([msg], model)
                        if current_tokens + message_tokens <= remaining_tokens:
                            selected_non_system.append((i, msg))
                            current_tokens += message_tokens
                        else:
                            break
                
                # Combine system and selected non-system messages, preserving original order
                all_selected = system_messages + selected_non_system
                all_selected.sort(key=lambda x: x[0])  # Sort by original index
                
                return [msg for _, msg in all_selected]
            
            else:
                # No system message preservation - filter all messages equally
                if keep_newest:
                    # Start from the newest messages and work backwards
                    filtered_messages = []
                    current_tokens = 0
                    
                    for message in reversed(messages):
                        message_tokens = self.count_tokens([message], model)
                        if current_tokens + message_tokens <= max_tokens:
                            filtered_messages.insert(0, message)
                            current_tokens += message_tokens
                        else:
                            break
                    
                    return filtered_messages
                else:
                    # Start from the oldest messages and work forwards
                    filtered_messages = []
                    current_tokens = 0
                    
                    for message in messages:
                        message_tokens = self.count_tokens([message], model)
                        if current_tokens + message_tokens <= max_tokens:
                            filtered_messages.append(message)
                            current_tokens += message_tokens
                        else:
                            break
                    
                    return filtered_messages
                
        except Exception as e:
            self.log.error(f"Error limiting tokens: {str(e)}")
            # Fallback: return original messages (might exceed limit)
            return messages

    def apply_input_truncation(
        self,
        messages: List[Dict],
        model: str,
        input_truncation_override: Optional[Union[str, int]] = None,
        keep_newest: bool = True
    ) -> List[Dict]:
        """
        Apply input truncation to messages based on input_truncation setting.
        
        Args:
            messages (List[Dict]): List of OpenAI message format dictionaries
            model (str): Model name to determine input limits (format: "provider:model" or just "model")
            input_truncation_override (Union[str, int], optional): Override the instance input_truncation setting
            keep_newest (bool): If True, keep newest messages when truncating. If False, keep oldest (default: True)
            
        Returns:
            List[Dict]: Messages truncated according to input limits, preserving original order
        """
        try:
            # Use override if provided, otherwise use instance setting
            truncation_setting = input_truncation_override if input_truncation_override is not None else self.input_truncation
            
            # If truncation is OFF, return messages as-is
            if truncation_setting == "OFF":
                return messages
            
            # Parse model string to get provider and model name
            provider_name, model_name = self.provider_manager.parse_model_string(model)
            
            # Get model input limits
            max_input_tokens = None
            if truncation_setting == "AUTO":
                # Use provider's optimized method to get input limits
                provider = self.provider_manager.get_provider(provider_name)
                max_input_tokens = provider.get_model_input_tokens(model_name)
                
                # Fallback to reasonable default if no limits found
                if max_input_tokens is None:
                    max_input_tokens = 128000  # Common default for modern models
                    
            elif isinstance(truncation_setting, int):
                # Use provided token count
                max_input_tokens = truncation_setting
            else:
                # Invalid setting, return messages as-is
                if self.verbose:
                    self.log.warning(f"Invalid input_truncation setting: {truncation_setting}. Using messages as-is.")
                return messages
            
            # Apply token limiting
            if max_input_tokens:
                # Reserve some tokens for the response and overhead (roughly 20% of input limit)
                effective_limit = int(max_input_tokens * 0.8)
                truncated_messages = self.limit_tokens(
                    messages=messages,
                    max_tokens=effective_limit,
                    model=model_name,
                    keep_newest=keep_newest,
                    preserve_system=True
                )
                
                if self.verbose and len(truncated_messages) < len(messages):
                    self.log.debug(f"Input truncation: {len(messages)} -> {len(truncated_messages)} messages (limit: {effective_limit} tokens)")
                
                return truncated_messages
            
            return messages
            
        except Exception as e:
            self.log.error(f"Error applying input truncation: {str(e)}")
            # Fallback: return original messages
            return messages