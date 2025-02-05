from openai import OpenAI 
from prompts import generate_steps_prompt, each_step_prompt, generate_reasoning_prompt, generate_conclusion_prompt, combine_all_pipeline_prompts
from pydantic import BaseModel
from schemas import Steps, Reasoning
from concurrent.futures import ThreadPoolExecutor
from ultraprint.logging import logger

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
        log_level: str = 'DEBUG'
    ):
        # Create the OpenAI client using the provided API key
        self.openai_client = OpenAI(api_key=api_key)
        self.model = model or "gpt-4o"
        self.temperature = temperature
        self.reasoning_iterations = reasoning_iterations
        self.steps_pipeline = steps_pipeline
        self.reasoning_pipeline = reasoning_pipeline
        
        self.log = logger(
            name=logger_name,
            filename=logger_filename,
            include_extra_info=log_extra_info,
            write_to_file=log_to_file,
            log_level=log_level,
            log_to_console=verbose
        )
        self.log.info(f"Initializing UltraGPT with model: {self.model}")

    def chat_with_openai_sync(self, messages: list):
        try:
            self.log.debug(f"Sending sync request to OpenAI with {len(messages)} messages")
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                temperature=self.temperature
            )
            content = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens
            self.log.debug(f"Received response with {tokens} tokens")
            return content, tokens
        except Exception as e:
            self.log.error(f"Error in chat_with_openai_sync: {str(e)}")
            raise e

    def chat_with_model_parse(self, messages: list, schema=None):
        try:
            self.log.debug(f"Sending parse request to OpenAI with schema: {schema}")
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
            self.log.debug(f"Received response with {tokens} tokens")
            return content, tokens
        except Exception as e:
            self.log.error(f"Error in chat_with_model_parse: {str(e)}")
            raise e

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

    def run_steps_pipeline(self, messages: list):
        self.log.info("Starting steps pipeline")
        total_tokens = 0

        messages = self.turnoff_system_message(messages)
        steps_generator_message = messages + [{"role": "system", "content": generate_steps_prompt()}]

        steps_json, tokens = self.chat_with_model_parse(steps_generator_message, schema=Steps)
        total_tokens += tokens
        steps = steps_json.get("steps", [])
        self.log.debug(f"Steps: {steps}")

        memory = []

        for step in steps:
            step_prompt = each_step_prompt(memory, step)
            step_message = messages + [{"role": "system", "content": step_prompt}]
            step_response, tokens = self.chat_with_openai_sync(step_message)
            self.log.debug(f"Step: {step}, Response: {step_response}")
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

        self.log.debug(f"Final Conclusion: {conclusion}")
        
        return {
            "steps": memory,
            "conclusion": conclusion
        }, total_tokens

    def run_reasoning_pipeline(self, messages: list):
        self.log.info(f"Starting reasoning pipeline with {self.reasoning_iterations} iterations")
        total_tokens = 0
        all_thoughts = []
        messages = self.turnoff_system_message(messages)

        for iteration in range(self.reasoning_iterations):
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
            
            self.log.debug(f"Iteration {iteration + 1} thoughts: {new_thoughts}")

        return all_thoughts, total_tokens

    def chat(self, messages: list, schema=None):
        self.log.info(f"Starting chat with {len(messages)} messages")
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
        self.log.info(f"Chat completed with total tokens: {total_tokens}")
        return final_output, total_tokens, details_dict

