from openai import OpenAI 
import ultraprint.common as p
from prompts import (
generate_steps_prompt, 
each_step_prompt, generate_reasoning_prompt, 
generate_conclusion_prompt, combine_all_pipeline_prompts,
make_tool_analysis_prompt
)
from pydantic import BaseModel
from schemas import Steps, Reasoning
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultraprint.logging import logger
from schemas import ToolAnalysisSchema

from tools.web_search.main import _execute as web_search
from tools.calculator.main import _execute as calculator

from itertools import islice

class UltraGPT:
    
    def __init__(
        self, 
        api_key: str, 
        model: str = None, 
        temperature: float = 0.7, 
        reasoning_iterations: int = 3,
        steps_pipeline: bool = True,
        reasoning_pipeline: bool = True,
        verbose: bool = False,
        logger_name: str = 'ultragpt',
        logger_filename: str = 'debug/ultragpt.log',
        log_extra_info: bool = False,
        log_to_file: bool = False,
        log_level: str = 'DEBUG',
        tools: list = ["web-search", "calculator"],
        tools_config: dict = {
            "web-search": {
                "max_results": 1,
                "model": "gpt-4o"
            },
            "calculator": {
                "model": "gpt-4o"
            }
        },
        tool_batch_size: int = 3,  # New parameter for controlling batch size
        tool_max_workers: int = 10, # New parameter for controlling max workers
    ):
        # Create the OpenAI client using the provided API key
        self.openai_client = OpenAI(api_key=api_key)
        self.model = model or "gpt-4o"
        self.temperature = temperature
        self.reasoning_iterations = reasoning_iterations
        self.steps_pipeline = steps_pipeline
        self.reasoning_pipeline = reasoning_pipeline
        self.tools = tools
        self.tools_config = tools_config
        self.tool_batch_size = tool_batch_size
        self.tool_max_workers = tool_max_workers
        
        supported_tools = ["web-search", "calculator"]
        for tool in tools:
            if tool not in supported_tools:
                raise ValueError(f"Invalid tool: {tools}. Supported tools: {', '.join(supported_tools)}")

        self.verbose = verbose
        self.log = logger(
            name=logger_name,
            filename=logger_filename,
            include_extra_info=log_extra_info,
            write_to_file=log_to_file,
            log_level=log_level,
            log_to_console=False  # Always disable console logging
        )
        
        self.log.info("Initializing UltraGPT with model: %s", self.model)
        if self.verbose:
            p.blue("="*50)
            p.blue("Initializing UltraGPT")
            p.cyan(f"Model: {self.model}")
            if tools:
                p.cyan(f"Tools enabled: {', '.join(tools)}")
            p.blue("="*50)

    def chat_with_openai_sync(self, messages: list):
        try:
            self.log.debug("Sending request to OpenAI (msgs: %d)", len(messages))
            if self.verbose:
                p.cyan(f"\nOpenAI Request → Messages: {len(messages)}")
                p.yellow("Checking for tool needs...")
            
            tool_response = self.execute_tools(messages[-1]["content"], messages)
            if tool_response:
                if self.verbose:
                    p.cyan("\nAppending tool responses to message")
                tool_response = "Tool Responses:\n" + tool_response
                messages = self.append_message_to_system(messages, tool_response)
            elif self.verbose:
                p.dgray("\nNo tool responses needed")
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                temperature=self.temperature
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

    def chat_with_model_parse(self, messages: list, schema=None):
        try:
            self.log.debug("Sending parse request with schema: %s", schema)
            
            tool_response = self.execute_tools(messages[-1]["content"], messages)
            if tool_response:
                tool_response = "Tool Responses:\n" + tool_response
            messages = self.append_message_to_system(messages, tool_response)

            response = self.openai_client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=schema,
                temperature=self.temperature
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
            if message["role"] == "system":
                processed.append(new_message)
            processed.append(message)
        return processed

    def append_message_to_system(self, messages: list, new_message: dict):
        # add message after system message
        processed = []
        for message in messages:
            if message["role"] == "system":
                processed.append({
                    "role": "system",
                    "content": f"{message['content']}\n{new_message}"
                })
            else:
                processed.append(message)
        return processed
    
    #! Pipelines -----------------------------------------------------------
    def run_steps_pipeline(self, messages: list):
        if self.verbose:
            p.purple("➤ Starting Steps Pipeline")
        else:
            self.log.info("Starting steps pipeline")
        total_tokens = 0

        messages = self.turnoff_system_message(messages)
        steps_generator_message = messages + [{"role": "system", "content": generate_steps_prompt()}]

        steps_json, tokens = self.chat_with_model_parse(steps_generator_message, schema=Steps)
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
            step_response, tokens = self.chat_with_openai_sync(step_message)
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
        conclusion, tokens = self.chat_with_openai_sync(conclusion_message)
        total_tokens += tokens

        if self.verbose:
            p.green("✓ Steps pipeline completed")
        
        return {
            "steps": memory,
            "conclusion": conclusion
        }, total_tokens

    def run_reasoning_pipeline(self, messages: list):
        if self.verbose:
            p.purple(f"➤ Starting Reasoning Pipeline ({self.reasoning_iterations} iterations)")
        else:
            self.log.info("Starting reasoning pipeline (%d iterations)", self.reasoning_iterations)
        total_tokens = 0
        all_thoughts = []
        messages = self.turnoff_system_message(messages)

        for iteration in range(self.reasoning_iterations):
            if self.verbose:
                p.yellow(f"Iteration {iteration + 1}/{self.reasoning_iterations}")
            self.log.debug("Iteration %d/%d", iteration + 1, self.reasoning_iterations)
            # Generate new thoughts based on all previous thoughts
            reasoning_message = messages + [
                {"role": "system", "content": generate_reasoning_prompt(all_thoughts)}
            ]
            
            reasoning_json, tokens = self.chat_with_model_parse(
                reasoning_message, 
                schema=Reasoning
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
    def chat(self, messages: list, schema=None):
        if self.verbose:
            p.blue("="*50)
            p.blue("Starting Chat Session")
            p.cyan(f"Messages: {len(messages)}")
            p.cyan(f"Schema: {schema}")
            p.blue("="*50)
        else:
            self.log.info("Starting chat session")
        reasoning_output = []
        reasoning_tokens = 0
        steps_output = {"steps": [], "conclusion": ""}
        steps_tokens = 0

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            
            if self.reasoning_pipeline:
                futures.append({
                    "type": "reasoning",
                    "future": executor.submit(self.run_reasoning_pipeline, messages)
                })
            
            if self.steps_pipeline:
                futures.append({
                    "type": "steps",
                    "future": executor.submit(self.run_steps_pipeline, messages)
                })

            for future in futures:
                if future["type"] == "reasoning":
                    reasoning_output, reasoning_tokens = future["future"].result()
                elif future["type"] == "steps":
                    steps_output, steps_tokens = future["future"].result()

        conclusion = steps_output.get("conclusion", "")
        steps = steps_output.get("steps", [])

        if self.reasoning_pipeline or self.steps_pipeline:
            prompt = combine_all_pipeline_prompts(reasoning_output, conclusion)
            messages = self.add_message_before_system(messages, {"role": "user", "content": prompt})

        if schema:
            final_output, tokens = self.chat_with_model_parse(messages, schema=schema)
        else:
            final_output, tokens = self.chat_with_openai_sync(messages)

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
    def execute_tool(self, tool: str, message: str, history: list) -> dict:
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
                    self.tools_config.get("web-search", {})
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
                    self.tools_config.get("calculator", {})
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
            return None
        except Exception as e:
            self.log.error("Tool %s failed: %s", tool, str(e))
            if self.verbose:
                p.red(f"\n✗ Tool {tool} failed: {str(e)}")
            raise e

    def batch_tools(self, tools: list, batch_size: int):
        """Helper function to create batches of tools"""
        iterator = iter(tools)
        while batch := list(islice(iterator, batch_size)):
            yield batch

    def execute_tools(self, message: str, history: list) -> str:
        """Execute tools and return formatted responses"""
        if not self.tools:
            return ""
            
        total_tools = len(self.tools)
        self.log.info("Executing %d tools in batches of %d", total_tools, self.tool_batch_size)
        if self.verbose:
            p.purple(f"\n➤ Executing {total_tools} tools in batches of {self.tool_batch_size}")
            p.yellow("Query: " + message[:100] + "..." if len(message) > 100 else message)
            p.lgray("-" * 40)
        
        all_responses = []
        success_count = 0
        
        for batch_idx, tool_batch in enumerate(self.batch_tools(self.tools, self.tool_batch_size), 1):
            if self.verbose:
                p.yellow(f"\nProcessing batch {batch_idx}...")
            batch_responses = []
            
            with ThreadPoolExecutor(max_workers=min(len(tool_batch), self.tool_max_workers)) as executor:
                future_to_tool = {
                    executor.submit(self.execute_tool, tool, message, history): tool 
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