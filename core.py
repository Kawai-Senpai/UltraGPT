from openai import OpenAI 
from .prompts import (
generate_steps_prompt, 
each_step_prompt, generate_reasoning_prompt, 
generate_conclusion_prompt, combine_all_pipeline_prompts,
make_tool_analysis_prompt, generate_tool_call_prompt,
generate_single_tool_call_prompt, generate_multiple_tool_call_prompt
)
from pydantic import BaseModel
from .schemas import Steps, Reasoning, ToolAnalysisSchema, ToolCallResponse, SingleToolCallResponse, UserTool
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultraprint.logging import logger

from .tools.web_search.main import _execute as web_search
from .tools.calculator.main import _execute as calculator
from .tools.math_operations.main import _execute as math_operations

from itertools import islice

class UltraGPT:
    def __init__(
        self, 
        api_key: str, 
        google_api_key: str = None,
        search_engine_id: str = None,
        verbose: bool = False,
        logger_name: str = 'ultragpt',
        logger_filename: str = 'debug/ultragpt.log',
        log_extra_info: bool = False,
        log_to_file: bool = False,
        log_to_console: bool = False,
        log_level: str = 'DEBUG',
    ):
        """
        Initialize the UltraGPT class.
        Args:
            api_key (str): The API key for accessing the OpenAI service.
            google_api_key (str, optional): Google Custom Search API key for web search tool.
            search_engine_id (str, optional): Google Custom Search Engine ID for web search tool.
            verbose (bool, optional): Whether to enable verbose logging. Defaults to False.
            logger_name (str, optional): The name of the logger. Defaults to 'ultragpt'.
            logger_filename (str, optional): The filename for the logger. Defaults to 'debug/ultragpt.log'.
            log_extra_info (bool, optional): Whether to include extra info in logs. Defaults to False.
            log_to_file (bool, optional): Whether to log to a file. Defaults to False.
            log_to_console (bool, optional): Whether to log to console. Defaults to True.
            log_level (str, optional): The logging level. Defaults to 'DEBUG'.
        Raises:
            ValueError: If an invalid tool is provided.
        """

        # Create the OpenAI client using the provided API key
        self.openai_client = OpenAI(api_key=api_key)
        
        # Store Google Search credentials
        self.google_api_key = google_api_key
        self.search_engine_id = search_engine_id
        
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

    def chat_with_openai_sync(
        self,
        messages: list,
        model: str,
        temperature: float,
        tools: list,
        tools_config: dict,
        tool_batch_size: int,
        tool_max_workers: int
    ):
        """
        Sends a synchronous chat request to OpenAI and processes the response.
        Args:
            messages (list): A list of message dictionaries to be sent to OpenAI.
            model (str): The model to use.
            temperature (float): The temperature for the model's output.
            tools (list): The list of tools to enable.
            tools_config (dict): The configuration for the tools.
            tool_batch_size (int): The batch size for tool processing.
            tool_max_workers (int): The maximum number of workers for tool processing.
        Returns:
            tuple: A tuple containing the response content (str) and the total number of tokens used (int).
        Raises:
            Exception: If the request to OpenAI fails.
        Logs:
            Debug: Logs the number of messages sent, the number of tokens in the response, and any errors encountered.
            Verbose: Optionally logs detailed steps of the request and response process.
        """
        try:
            self.log.debug("Sending request to OpenAI (msgs: " + str(len(messages)) + ")")
            if self.verbose:
                self.log.debug("OpenAI Request → Messages: " + str(len(messages)))
                self.log.debug("Checking for tool needs...")
            
            tool_response = self.execute_tools(message=messages[-1]["content"], history=messages, tools=tools, tools_config=tools_config, tool_batch_size=tool_batch_size, tool_max_workers=tool_max_workers)
            if tool_response:
                if self.verbose:
                    self.log.debug("Appending tool responses to message")
                tool_response = "Tool Responses:\n" + tool_response
                messages = self.append_message_to_system(messages, tool_response)
            elif self.verbose:
                self.log.debug("No tool responses needed")
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                temperature=temperature
            )
            content = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens
            self.log.debug("Response received (tokens: " + str(tokens) + ")")
            if self.verbose:
                self.log.debug("✓ Response received (" + str(tokens) + " tokens)")
            return content, tokens
        except Exception as e:
            self.log.error("OpenAI sync request failed: " + str(e))
            if self.verbose:
                self.log.debug("✗ OpenAI request failed: " + str(e))
            raise e

    def chat_with_model_parse(
        self,
        messages: list,
        schema=None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        tools: list = [],
        tools_config: dict = {},
        tool_batch_size: int = 3,
        tool_max_workers: int = 10
    ):
        """
        Sends a chat message to the model for parsing and returns the parsed response.
        Args:
            messages (list): A list of message dictionaries to be sent to the model.
            schema (optional): The schema to be used for parsing the response. Defaults to None.
            model (str): The model to use.
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
        try:
            self.log.debug("Sending parse request with schema: %s", schema)
            
            tool_response = self.execute_tools(message=messages[-1]["content"], history=messages, tools=tools, tools_config=tools_config, tool_batch_size=tool_batch_size, tool_max_workers=tool_max_workers)
            if tool_response:
                tool_response = "Tool Responses:\n" + tool_response
            messages = self.append_message_to_system(messages, tool_response)

            response = self.openai_client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=schema,
                temperature=temperature
            )
            content = response.choices[0].message.parsed
            if isinstance(content, BaseModel):
                content = content.model_dump(by_alias=True)
            tokens = response.usage.total_tokens
            
            self.log.debug("Parse response received (tokens: " + str(tokens) + ")")
            return content, tokens
        except Exception as e:
            self.log.error("Parse request failed: " + str(e))
            raise e

    def analyze_tool_need(self, message: str, available_tools: list) -> dict:
        """Analyze if a tool is needed for the message"""
        prompt = make_tool_analysis_prompt(message, available_tools)
        response = self.chat_with_model_parse([{"role": "system", "content": prompt}], schema=ToolAnalysisSchema)
        if not response:
            return {"tools": []}
        return response

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
        steps_model: str = None
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

        steps_json, tokens = self.chat_with_model_parse(steps_generator_message, schema=Steps, model=active_model, temperature=temperature, tools=tools, tools_config=tools_config, tool_batch_size=tool_batch_size, tool_max_workers=tool_max_workers)
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
            step_response, tokens = self.chat_with_openai_sync(step_message, model=active_model, temperature=temperature, tools=tools, tools_config=tools_config, tool_batch_size=tool_batch_size, tool_max_workers=tool_max_workers)
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
        conclusion, tokens = self.chat_with_openai_sync(conclusion_message, model=active_model, temperature=temperature, tools=tools, tools_config=tools_config, tool_batch_size=tool_batch_size, tool_max_workers=tool_max_workers)
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
        reasoning_model: str = None
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
                tool_max_workers=tool_max_workers
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
        model: str = "gpt-4o",
        temperature: float = 0.7,
        reasoning_iterations: int = 3,
        steps_pipeline: bool = True,
        reasoning_pipeline: bool = True,
        steps_model: str = None,
        reasoning_model: str = None,
        tools: list = ["web-search", "calculator", "math-operations"],
        tools_config: dict = {
            "web-search": {
                "max_results": 5, 
                "model": "gpt-4o",
                "enable_scraping": True,  # Enable web scraping of search results
                "max_scrape_length": 5000,  # Max characters per scraped page
                "scrape_timeout": 15,  # Timeout for scraping requests
                "scrape_pause": 1,  # Pause between scraping requests
                "max_history_items": 5  # Max conversation history items to include
            },
            "calculator": {
                "model": "gpt-4o",
                "max_history_items": 5  # Max conversation history items to include
            },
            "math-operations": {
                "model": "gpt-4o",
                "max_history_items": 5  # Max conversation history items to include
            }
        },
        tool_batch_size: int = 3,
        tool_max_workers: int = 10,
    ):
        """
        Initiates a chat session with the given messages and optional schema.
        Args:
            messages (list): A list of message dictionaries to be processed.
            schema (optional): A schema to parse the final output, defaults to None.
            model (str, optional): The model to use. Defaults to "gpt-4o".
            temperature (float, optional): The temperature for the model's output. Defaults to 0.7.
            reasoning_iterations (int, optional): The number of reasoning iterations. Defaults to 3.
            steps_pipeline (bool, optional): Whether to use steps pipeline. Defaults to True.
            reasoning_pipeline (bool, optional): Whether to use reasoning pipeline. Defaults to True.
            steps_model (str, optional): Specific model for steps pipeline. Uses main model if None.
            reasoning_model (str, optional): Specific model for reasoning pipeline. Uses main model if None.
            tools (list, optional): The list of tools to enable. Defaults to ["web-search", "calculator", "math-operations"].
            tools_config (dict, optional): The configuration for the tools. Defaults to predefined configurations.
            tool_batch_size (int, optional): The batch size for tool processing. Defaults to 3.
            tool_max_workers (int, optional): The maximum number of workers for tool processing. Defaults to 10.
        Returns:
            tuple: A tuple containing the final output, total tokens used, and a details dictionary.
                - final_output: The final response from the chat model.
                - total_tokens (int): The total number of tokens used during the session.
                - details_dict (dict): A dictionary with detailed information about the session.
        """
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
                    "future": executor.submit(self.run_reasoning_pipeline, messages, model, temperature, reasoning_iterations, tools, tools_config, tool_batch_size, tool_max_workers, reasoning_model)
                })
            
            if steps_pipeline:
                futures.append({
                    "type": "steps",
                    "future": executor.submit(self.run_steps_pipeline, messages, model, temperature, tools, tools_config, tool_batch_size, tool_max_workers, steps_model)
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
            final_output, tokens = self.chat_with_model_parse(messages, schema=schema, model=model, temperature=temperature, tools=tools, tools_config=tools_config, tool_batch_size=tool_batch_size, tool_max_workers=tool_max_workers)
        else:
            final_output, tokens = self.chat_with_openai_sync(messages, model=model, temperature=temperature, tools=tools, tools_config=tools_config, tool_batch_size=tool_batch_size, tool_max_workers=tool_max_workers)

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
    def execute_tool(self, tool: str, message: str, history: list, tools_config: dict) -> dict:
        """Execute a single tool and return its response"""
        try:
            self.log.debug("Executing tool: " + tool)
            if self.verbose:
                self.log.debug("Executing tool: " + tool)
                
            if tool == "web-search":
                # Merge Google credentials with config, allowing override
                web_search_config = tools_config.get("web-search", {}).copy()
                web_search_config.update({
                    "google_api_key": self.google_api_key,
                    "search_engine_id": web_search_config.get("search_engine_id", self.search_engine_id)
                })
                
                response = web_search(
                    message, 
                    history, 
                    self.openai_client, 
                    web_search_config
                )
                
                # Log specific web search issues for debugging
                if not response:
                    if not self.google_api_key or not web_search_config.get("search_engine_id"):
                        self.log.warning("Web search skipped: Missing Google API credentials")
                        if self.verbose:
                            self.log.debug("⚠ Web search skipped: Missing Google API credentials")
                    else:
                        self.log.warning("Web search returned no results (quota/API error)")
                        if self.verbose:
                            self.log.debug("⚠ Web search returned no results (quota/API error)")
                
                self.log.debug("Tool " + tool + " completed successfully")
                if self.verbose:
                    self.log.debug("✓ " + tool + " returned response:")
                    self.log.debug("-" * 40)
                    self.log.debug(response if response else "(empty - no results)")
                    self.log.debug("-" * 40)
                return {
                    "tool": tool,
                    "response": response
                }
            elif tool == "calculator":
                response = calculator(
                    message, 
                    history, 
                    self.openai_client, 
                    tools_config.get("calculator", {})
                )
                self.log.debug("Tool " + tool + " completed successfully")
                if self.verbose:
                    self.log.debug("✓ " + tool + " returned response:")
                    self.log.debug("-" * 40)
                    self.log.debug(response)
                    self.log.debug("-" * 40)
                return {
                    "tool": tool,
                    "response": response
                }
            elif tool == "math-operations":
                response = math_operations(
                    message, 
                    history, 
                    self.openai_client, 
                    tools_config.get("math-operations", {})
                )
                self.log.debug("Tool " + tool + " completed successfully")
                if self.verbose:
                    self.log.debug("✓ " + tool + " returned response:")
                    self.log.debug("-" * 40)
                    self.log.debug(response)
                    self.log.debug("-" * 40)
                return {
                    "tool": tool,
                    "response": response
                }
            # Add other tool conditions here
            if self.verbose:
                self.log.debug("! Tool " + tool + " not implemented")
            return {
                "tool": tool,
                "response": ""
            }
        except Exception as e:
            self.log.error("Tool " + tool + " failed: " + str(e))
            if self.verbose:
                self.log.debug("✗ Tool " + tool + " failed: " + str(e))
            return {
                "tool": tool,
                "response": f"Tool {tool} failed: {str(e)}"
            }

    def batch_tools(self, tools: list, batch_size: int):
        """Helper function to create batches of tools"""
        iterator = iter(tools)
        while batch := list(islice(iterator, batch_size)):
            yield batch

    def execute_tools(
        self,
        message: str,
        history: list,
        tools: list,
        tools_config: dict,
        tool_batch_size: int,
        tool_max_workers: int
    ) -> str:
        """Execute tools and return formatted responses"""
        if not tools:
            return ""
        
        try:
            total_tools = len(tools)
            self.log.info("Executing " + str(total_tools) + " tools in batches of " + str(tool_batch_size))
            if self.verbose:
                self.log.debug("➤ Executing " + str(total_tools) + " tools in batches of " + str(tool_batch_size))
                self.log.debug("Query: " + (message[:100] + "..." if len(message) > 100 else message))
                self.log.debug("-" * 40)
            
            all_responses = []
            success_count = 0
            
            for batch_idx, tool_batch in enumerate(self.batch_tools(tools, tool_batch_size), 1):
                if self.verbose:
                    self.log.debug("Processing batch " + str(batch_idx) + "...")
                batch_responses = []
                
                with ThreadPoolExecutor(max_workers=min(len(tool_batch), tool_max_workers)) as executor:
                    future_to_tool = {
                        executor.submit(self.execute_tool, tool, message, history, tools_config): tool 
                        for tool in tool_batch
                    }
                    
                    for future in as_completed(future_to_tool):
                        tool = future_to_tool[future]
                        try:
                            result = future.result()
                            if result:
                                batch_responses.append(result)
                                success_count += 1
                        except Exception as e:
                            if self.verbose:
                                self.log.debug("✗ Tool " + tool + " failed: " + str(e))
                            else:
                                self.log.error("Tool " + tool + " failed: " + str(e))
                
                all_responses.extend(batch_responses)
                if self.verbose:
                    self.log.debug("✓ Batch " + str(batch_idx) + ": " + str(len(batch_responses)) + "/" + str(len(tool_batch)) + " tools completed")

            self.log.info("Tools execution completed (" + str(success_count) + "/" + str(total_tools) + " successful)")
            if self.verbose:
                self.log.debug("✓ Tools execution completed (" + str(success_count) + "/" + str(total_tools) + " successful)")
                
            if not all_responses:
                if self.verbose:
                    self.log.debug("No tool responses generated")
                return ""
                
            if self.verbose:
                self.log.debug("Formatted Tool Responses:")
                self.log.debug("=" * 40)
                
            formatted_responses = []
            for r in all_responses:
                tool_name = r['tool'].upper()
                response = r['response'].strip()
                formatted = "[" + tool_name + "]\n" + response
                formatted_responses.append(formatted)
                if self.verbose:
                    self.log.debug(formatted)
                    self.log.debug("-" * 40)
                
            if self.verbose:
                self.log.debug("=" * 40)
                
            return "\n\n".join(formatted_responses)
        except Exception as e:
            self.log.error("Tool execution failed: " + str(e))
            if self.verbose:
                self.log.debug("✗ Tool execution failed: " + str(e))
            return ""

    #! Tool Call Functionality --------------------------------------------
    def tool_call(
        self,
        messages: list,
        user_tools: list,
        allow_multiple: bool = True,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        reasoning_iterations: int = 3,
        steps_pipeline: bool = True,
        reasoning_pipeline: bool = True,
        steps_model: str = None,
        reasoning_model: str = None,
        tools: list = ["web-search", "calculator", "math-operations"],
        tools_config: dict = {
            "web-search": {
                "max_results": 5, 
                "model": "gpt-4o",
                "enable_scraping": True,
                "max_scrape_length": 5000,
                "scrape_timeout": 15,
                "scrape_pause": 1,
                "max_history_items": 5
            },
            "calculator": {
                "model": "gpt-4o",
                "max_history_items": 5
            },
            "math-operations": {
                "model": "gpt-4o",
                "max_history_items": 5
            }
        },
        tool_batch_size: int = 3,
        tool_max_workers: int = 10,
    ):
        """
        Tool call functionality that uses UltraGPT's execution layer to determine 
        which user-defined tools to call and with what parameters.
        
        Args:
            messages (list): A list of message dictionaries to be processed.
            user_tools (list): List of user-defined tools with schemas and prompts.
            allow_multiple (bool, optional): Whether to allow multiple tool calls. Defaults to True.
            model (str, optional): The model to use. Defaults to "gpt-4o".
            temperature (float, optional): The temperature for the model's output. Defaults to 0.7.
            reasoning_iterations (int, optional): The number of reasoning iterations. Defaults to 3.
            steps_pipeline (bool, optional): Whether to use steps pipeline. Defaults to True.
            reasoning_pipeline (bool, optional): Whether to use reasoning pipeline. Defaults to True.
            steps_model (str, optional): Specific model for steps pipeline. Uses main model if None.
            reasoning_model (str, optional): Specific model for reasoning pipeline. Uses main model if None.
            tools (list, optional): The list of internal tools to enable. Defaults to ["web-search", "calculator", "math-operations"].
            tools_config (dict, optional): The configuration for internal tools.
            tool_batch_size (int, optional): The batch size for tool processing. Defaults to 3.
            tool_max_workers (int, optional): The maximum number of workers for tool processing. Defaults to 10.
        
        Returns:
            tuple: A tuple containing the tool call response and total tokens used.
                - tool_call_response: The tool calls with parameters and reasoning.
                - total_tokens (int): The total number of tokens used during the session.
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
        
        # Validate user tools
        validated_tools = self._validate_user_tools(user_tools)
        
        # Create dynamic schemas based on user tools
        dynamic_schema = self._create_dynamic_tool_schema(validated_tools, allow_multiple)
        
        # Create tool call prompt
        if allow_multiple:
            tool_prompt = generate_multiple_tool_call_prompt(validated_tools)
        else:
            tool_prompt = generate_single_tool_call_prompt(validated_tools)
        
        # Add tool call prompt to messages
        tool_call_messages = messages + [{"role": "system", "content": tool_prompt}]
        
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
                    tools, tools_config, tool_batch_size, tool_max_workers, reasoning_model
                )
                futures.append(("reasoning", future))
            
            if steps_pipeline:
                future = executor.submit(
                    self.run_steps_pipeline,
                    tool_call_messages, model, temperature,
                    tools, tools_config, tool_batch_size, tool_max_workers, steps_model
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
        steps = steps_output.get("steps", [])

        if reasoning_pipeline or steps_pipeline:
            combined_prompt = combine_all_pipeline_prompts(reasoning_output, conclusion)
            enhanced_messages = self.append_message_to_system(tool_call_messages, combined_prompt)
        else:
            enhanced_messages = tool_call_messages

        # Generate tool call response using structured output
        tool_call_response, tokens = self.chat_with_model_parse(
            enhanced_messages, 
            schema=dynamic_schema, 
            model=model, 
            temperature=temperature,
            tools=tools,
            tools_config=tools_config,
            tool_batch_size=tool_batch_size,
            tool_max_workers=tool_max_workers
        )

        total_tokens = reasoning_tokens + steps_tokens + tokens
        
        if self.verbose:
            self.log.debug("✓ Tool call analysis completed")
            if allow_multiple:
                self.log.debug("Generated " + str(len(tool_call_response.get('tool_calls', []))) + " tool calls")
                for i, tool_call in enumerate(tool_call_response.get('tool_calls', []), 1):
                    self.log.debug("  " + str(i) + ". " + tool_call.get('tool_name') + " - " + tool_call.get('reasoning'))
            else:
                tool_call = tool_call_response.get('tool_call', {})
                self.log.debug("Selected tool: " + tool_call.get('tool_name') + " - " + tool_call.get('reasoning'))
            self.log.debug("Total tokens used: " + str(total_tokens))
        else:
            self.log.info("Tool call completed with " + str(total_tokens) + " tokens")
        
        return tool_call_response, total_tokens

    def _validate_user_tools(self, user_tools: list) -> list:
        """Validate and format user tools"""
        validated_tools = []
        
        for tool in user_tools:
            if isinstance(tool, dict):
                # Ensure all required fields are present
                required_fields = ['name', 'description', 'parameters_schema', 'usage_guide', 'when_to_use']
                if all(field in tool for field in required_fields):
                    validated_tools.append(tool)
                else:
                    missing = [field for field in required_fields if field not in tool]
                    self.log.warning("Tool missing required fields: " + str(missing))
                    if self.verbose:
                        self.log.debug("⚠ Tool missing fields: " + str(missing))
            elif hasattr(tool, 'model_dump'):
                # Pydantic model
                validated_tools.append(tool.model_dump())
            else:
                self.log.warning("Invalid tool format: " + str(type(tool)))
                if self.verbose:
                    self.log.debug("⚠ Invalid tool format: " + str(type(tool)))
        
        return validated_tools
    
    def _create_dynamic_tool_schema(self, validated_tools: list, allow_multiple: bool):
        """Create dynamic Pydantic schema based on user tools"""
        from pydantic import create_model
        from typing import Any, Dict
        
        # Create individual tool call models for each tool
        tool_models = {}
        
        for tool in validated_tools:
            tool_name = tool['name']
            tool_schema = tool['parameters_schema']
            
            # Create dynamic model for this specific tool's parameters
            param_fields = {}
            for prop_name, prop_def in tool_schema.get('properties', {}).items():
                python_type = self._convert_json_schema_to_python_type(prop_def)
                is_required = prop_name in tool_schema.get('required', [])
                
                if is_required:
                    param_fields[prop_name] = (python_type, ...)
                else:
                    default_value = prop_def.get('default', None)
                    param_fields[prop_name] = (python_type, default_value)
            
            # Create the parameter model
            param_model_name = f"{tool_name.title().replace('_', '')}Params"
            param_model = create_model(param_model_name, **param_fields)
            
            # Create the tool call model
            from pydantic import Field
            from typing import Literal
            
            tool_call_fields = {
                'tool_name': (Literal[tool_name], Field(default=tool_name, description=f"Must be exactly '{tool_name}'")),
                'parameters': (param_model, Field(..., description=f"Parameters for {tool_name} tool")),
                'reasoning': (str, Field(..., description="Reasoning for choosing this tool"))
            }
            
            tool_call_model_name = f"{tool_name.title().replace('_', '')}ToolCall"
            tool_models[tool_name] = create_model(tool_call_model_name, **tool_call_fields)
        
        if allow_multiple:
            # For multiple tools, create a response with a list of any tool call
            if len(tool_models) == 1:
                tool_type = list(tool_models.values())[0]
            else:
                from typing import Union
                tool_type = Union[tuple(tool_models.values())]
            
            response_fields = {
                'tool_calls': (list[tool_type], ...)
            }
            return create_model('DynamicMultipleToolCallResponse', **response_fields)
        else:
            # For single tool, create a response with one tool call
            if len(tool_models) == 1:
                tool_type = list(tool_models.values())[0]
            else:
                from typing import Union
                tool_type = Union[tuple(tool_models.values())]
            
            response_fields = {
                'tool_call': (tool_type, ...)
            }
            return create_model('DynamicSingleToolCallResponse', **response_fields)
    
    def _convert_json_schema_to_python_type(self, schema_def):
        """Convert JSON schema type to Python type"""
        schema_type = schema_def.get('type', 'string')
        
        if schema_type == 'string':
            return str
        elif schema_type == 'integer':
            return int
        elif schema_type == 'number':
            return float
        elif schema_type == 'boolean':
            return bool
        elif schema_type == 'array':
            items_type = self._convert_json_schema_to_python_type(schema_def.get('items', {'type': 'string'}))
            return list[items_type]
        elif schema_type == 'object':
            return dict
        else:
            return str  # Default to string