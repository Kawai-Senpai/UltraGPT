from openai import OpenAI 
import ultraprint.common as p
from .prompts import (
generate_steps_prompt, 
each_step_prompt, generate_reasoning_prompt, 
generate_conclusion_prompt, combine_all_pipeline_prompts,
make_tool_analysis_prompt
)
from pydantic import BaseModel
from .schemas import Steps, Reasoning
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultraprint.logging import logger
from .schemas import ToolAnalysisSchema

from .tools.web_search.main import _execute as web_search
from .tools.calculator.main import _execute as calculator
from .tools.math_operations.main import _execute as math_operations

from itertools import islice

class UltraGPT:
    def __init__(
        self, 
        api_key: str, 
        verbose: bool = False,
        logger_name: str = 'ultragpt',
        logger_filename: str = 'debug/ultragpt.log',
        log_extra_info: bool = False,
        log_to_file: bool = False,
        log_level: str = 'DEBUG',
    ):
        """
        Initialize the UltraGPT class.
        Args:
            api_key (str): The API key for accessing the OpenAI service.
            verbose (bool, optional): Whether to enable verbose logging. Defaults to False.
            logger_name (str, optional): The name of the logger. Defaults to 'ultragpt'.
            logger_filename (str, optional): The filename for the logger. Defaults to 'debug/ultragpt.log'.
            log_extra_info (bool, optional): Whether to include extra info in logs. Defaults to False.
            log_to_file (bool, optional): Whether to log to a file. Defaults to False.
            log_level (str, optional): The logging level. Defaults to 'DEBUG'.
        Raises:
            ValueError: If an invalid tool is provided.
        """

        # Create the OpenAI client using the provided API key
        self.openai_client = OpenAI(api_key=api_key)
        self.verbose = verbose
        self.log = logger(
            name=logger_name,
            filename=logger_filename,
            include_extra_info=log_extra_info,
            write_to_file=log_to_file,
            log_level=log_level,
            log_to_console=False  # Always disable console logging
        )
        
        self.log.info("Initializing UltraGPT")
        if self.verbose:
            p.blue("="*50)
            p.blue("Initializing UltraGPT")
            p.blue("="*50)

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
            self.log.debug("Sending request to OpenAI (msgs: %d)", len(messages))
            if self.verbose:
                p.cyan(f"\nOpenAI Request → Messages: {len(messages)}")
                p.yellow("Checking for tool needs...")
            
            tool_response = self.execute_tools(message=messages[-1]["content"], history=messages, tools=tools, tools_config=tools_config, tool_batch_size=tool_batch_size, tool_max_workers=tool_max_workers)
            if tool_response:
                if self.verbose:
                    p.cyan("\nAppending tool responses to message")
                tool_response = "Tool Responses:\n" + tool_response
                messages = self.append_message_to_system(messages, tool_response)
            elif self.verbose:
                p.dgray("\nNo tool responses needed")
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                temperature=temperature
            )
            content = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens
            self.log.debug("Response received (tokens: %d)", tokens)
            if self.verbose:
                p.green(f"✓ Response received ({tokens} tokens)")
            return content, tokens
        except Exception as e:
            self.log.error("OpenAI sync request failed: %s", str(e))
            if self.verbose:
                p.red(f"✗ OpenAI request failed: {str(e)}")
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
            
            self.log.debug("Parse response received (tokens: %d)", tokens)
            return content, tokens
        except Exception as e:
            self.log.error("Parse request failed: %s", str(e))
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
            p.purple("➤ Starting Steps Pipeline")
            if steps_model:
                p.cyan(f"Using steps model: {steps_model}")
        else:
            self.log.info("Starting steps pipeline")
        total_tokens = 0

        messages = self.turnoff_system_message(messages)
        steps_generator_message = messages + [{"role": "system", "content": generate_steps_prompt()}]

        steps_json, tokens = self.chat_with_model_parse(steps_generator_message, schema=Steps, model=active_model, temperature=temperature, tools=tools, tools_config=tools_config, tool_batch_size=tool_batch_size, tool_max_workers=tool_max_workers)
        total_tokens += tokens
        steps = steps_json.get("steps", [])
        if self.verbose:
            p.yellow(f"Generated {len(steps)} steps:")
            for idx, step in enumerate(steps, 1):
                p.lgray(f"  {idx}. {step}")
        else:
            self.log.debug("Generated %d steps", len(steps))

        memory = []

        for idx, step in enumerate(steps, 1):
            if self.verbose:
                p.cyan(f"Processing step {idx}/{len(steps)}")
            self.log.debug("Processing step %d/%d", idx, len(steps))
            step_prompt = each_step_prompt(memory, step)
            step_message = messages + [{"role": "system", "content": step_prompt}]
            step_response, tokens = self.chat_with_openai_sync(step_message, model=active_model, temperature=temperature, tools=tools, tools_config=tools_config, tool_batch_size=tool_batch_size, tool_max_workers=tool_max_workers)
            self.log.debug("Step %d response: %s...", idx, step_response[:100])
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
            p.green("✓ Steps pipeline completed")
        
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
            p.purple(f"➤ Starting Reasoning Pipeline ({reasoning_iterations} iterations)")
            if reasoning_model:
                p.cyan(f"Using reasoning model: {reasoning_model}")
        else:
            self.log.info("Starting reasoning pipeline (%d iterations)", reasoning_iterations)
        total_tokens = 0
        all_thoughts = []
        messages = self.turnoff_system_message(messages)

        for iteration in range(reasoning_iterations):
            if self.verbose:
                p.yellow(f"Iteration {iteration + 1}/{reasoning_iterations}")
            self.log.debug("Iteration %d/%d", iteration + 1, reasoning_iterations)
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
                p.cyan(f"Generated {len(new_thoughts)} thoughts:")
                for idx, thought in enumerate(new_thoughts, 1):
                    p.lgray(f"  {idx}. {thought}")
            else:
                self.log.debug("Generated %d new thoughts", len(new_thoughts))

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
            "web-search": {"max_results": 1, "model": "gpt-4o"},
            "calculator": {"model": "gpt-4o"},
            "math-operations": {"model": "gpt-4o"}
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
            p.blue("="*50)
            p.blue("Starting Chat Session")
            p.cyan(f"Messages: {len(messages)}")
            p.cyan(f"Schema: {schema}")
            p.cyan(f"Model: {model}")
            p.cyan(f"Tools: {', '.join(tools) if tools else 'None'}")
            p.blue("="*50)
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
            p.blue("="*50)
            p.green("✓ Chat Session Completed")
            p.yellow("Tokens Used:")
            p.lgray(f"  - Reasoning: {reasoning_tokens}")
            p.lgray(f"  - Steps: {steps_tokens}")
            p.lgray(f"  - Final: {tokens}")
            p.lgray(f"  - Total: {total_tokens}")
            p.blue("="*50)
        else:
            self.log.info("Chat completed (total tokens: %d)", total_tokens)
        return final_output, total_tokens, details_dict

    #! Tools ----------------------------------------------------------------
    def execute_tool(self, tool: str, message: str, history: list, tools_config: dict) -> dict:
        """Execute a single tool and return its response"""
        try:
            self.log.debug("Executing tool: %s", tool)
            if self.verbose:
                p.cyan(f"\nExecuting tool: {tool}")
                
            if tool == "web-search":
                response = web_search(
                    message, 
                    history, 
                    self.openai_client, 
                    tools_config.get("web-search", {})
                )
                self.log.debug("Tool %s completed successfully", tool)
                if self.verbose:
                    p.green(f"✓ {tool} returned response:")
                    p.lgray("-" * 40)
                    p.lgray(response)
                    p.lgray("-" * 40)
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
                self.log.debug("Tool %s completed successfully", tool)
                if self.verbose:
                    p.green(f"✓ {tool} returned response:")
                    p.lgray("-" * 40)
                    p.lgray(response)
                    p.lgray("-" * 40)
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
                self.log.debug("Tool %s completed successfully", tool)
                if self.verbose:
                    p.green(f"✓ {tool} returned response:")
                    p.lgray("-" * 40)
                    p.lgray(response)
                    p.lgray("-" * 40)
                return {
                    "tool": tool,
                    "response": response
                }
            # Add other tool conditions here
            if self.verbose:
                p.yellow(f"! Tool {tool} not implemented")
            return {
                "tool": tool,
                "response": ""
            }
        except Exception as e:
            self.log.error("Tool %s failed: %s", tool, str(e))
            if self.verbose:
                p.red(f"\n✗ Tool {tool} failed: {str(e)}")
            raise {
                "tool": tool,
                "response": ""
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
            self.log.info("Executing %d tools in batches of %d", total_tools, tool_batch_size)
            if self.verbose:
                p.purple(f"\n➤ Executing {total_tools} tools in batches of {tool_batch_size}")
                p.yellow("Query: " + message[:100] + "..." if len(message) > 100 else message)
                p.lgray("-" * 40)
            
            all_responses = []
            success_count = 0
            
            for batch_idx, tool_batch in enumerate(self.batch_tools(tools, tool_batch_size), 1):
                if self.verbose:
                    p.yellow(f"\nProcessing batch {batch_idx}...")
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
                                p.red(f"✗ Tool {tool} failed: {str(e)}")
                            else:
                                self.log.error("Tool %s failed: %s", tool, str(e))
                
                all_responses.extend(batch_responses)
                if self.verbose:
                    p.green(f"✓ Batch {batch_idx}: {len(batch_responses)}/{len(tool_batch)} tools completed\n")

            self.log.info("Tools execution completed (%d/%d successful)", success_count, total_tools)
            if self.verbose:
                p.green(f"\n✓ Tools execution completed ({success_count}/{total_tools} successful)")
                
            if not all_responses:
                if self.verbose:
                    p.yellow("\nNo tool responses generated")
                return ""
                
            if self.verbose:
                p.cyan("\nFormatted Tool Responses:")
                p.lgray("=" * 40)
                
            formatted_responses = []
            for r in all_responses:
                tool_name = r['tool'].upper()
                response = r['response'].strip()
                formatted = f"[{tool_name}]\n{response}"
                formatted_responses.append(formatted)
                if self.verbose:
                    p.lgray(formatted)
                    p.lgray("-" * 40)
                
            if self.verbose:
                p.lgray("=" * 40 + "\n")
                
            return "\n\n".join(formatted_responses)
        except Exception as e:
            self.log.error("Tool execution failed: %s", str(e))
            if self.verbose:
                p.red(f"✗ Tool execution failed: {str(e)}")
            return ""