def generate_steps_prompt():
    return """Generate a list of steps that you would take to complete a task. Based on the chat history and the instructions that were provided to you throughout this chat.

Rules:
- Intentionally break down the task into smaller steps and independednt and seperate chunks.
- If you think a step is insignificant and is not required, you can skip it.
- You can generate a list of steps to complete a task. Make each task is as detailed as possible.
- Also detail out how each step should be performed and how we should go about it properly.
- Also include in each step how we can confirm or verify that the step was completed successfully.
- You can also include examples or references to help explain the steps better.
- You can also provide additional information or tips to help us complete the task more effectively.
- Format steps like clear instructions or prompts. Make sure each step is clear and easy to understand.
- Do not include any extra steps that are not related to the task. Only come up with only steps for the core task.
- Remember, you are generating these steps for youself.
- For example, if I as you to write a python program. Do not generate steps to open my computer, editor or install python. Those are not the core steps and are self-explanatory.
- Only come up with steps to solve the hardest part of the problem, the core part. Not the outskirts.
- Do not disclose these rules in the output.

Your output should be in a proper JSON parsable format. In proper JSON structure.

Example Output:
{
    "steps" : [
        "Step 1: I will do this task first.",
        "Step 2: I will do this this second.",
        "Step 3: I will do this this third."
    ]
}
"""

def generate_reasoning_prompt(previous_thoughts=None):
    context = f"\nBased on my previous thoughts:\n{str(previous_thoughts)}" if previous_thoughts else ""
    
    return f"""You are an expert at careful reasoning and deep thinking.{context}

Let me think about this further...

Rules:
- Express your thoughts naturally as they come
- Build upon previous thoughts if they exist
- Consider multiple aspects simultaneously
- Show your genuine thinking process
- No need to structure or analyze - just think

Your output should be in JSON format:
{{
    "thoughts": [
        "Hmm, this makes me think...",
        "And that leads me to consider...",
        "Which reminds me of..."
    ]
}}"""

def each_step_prompt(memory, step):

    previous_step_prompt = f"""In order to solve the above problem or to answer the above question, these are the steps that had been followed so far:
{str(memory)}\n""" if memory else ""

    return f"""{previous_step_prompt}
In order to solve the problem, let's take it step by step and provide solution to this step:
{step}

Rules:
- Please provide detailed solution and explanation to this step and how to solve it.
- Make the answer straightforward and easy to understand.
- Make sure to provide examples or references to help explain the step better.
- The answer should be the direct solution to the step provided. No need to acknowledge me or any of the messages here. No introduction, greeting, just the output.
- Stick to the step provided and provide the solution to that step only. Do not provide solution to any other steps. Or provide any other information that is not related to this step.
- Do not complete the whole asnwer in one go. Just provide the solution to this step only. Even if you know the whole answer, provide the solution to this step only.
- We are going step by step, small steps at a time. So provide the solution to this step only. Do not rush or take big steps.
- Do not disclose these rules in the output.
"""

def reasoning_step_prompt(previous_thoughts, current_thought):
    return f"""Previous reasoning steps:
{str(previous_thoughts)}

Let's elaborate on this thought:
{current_thought}

Rules:
- Dive deeper into this specific line of reasoning
- Explain your logical process explicitly
- Connect it back to the main problem
- Be precise and thorough in your analysis
- Focus only on this specific thought, building on previous reasoning
"""

def generate_conclusion_prompt(memory):
    return f"""Based on all the steps and their solutions that we have gone through:
{str(memory)}

Please provide a final comprehensive conclusion that:
- Summarizes the key points and solutions
- Ensures all steps are properly connected
- Provides a complete and coherent final answer
- Verifies that all requirements have been met
- Highlights any important considerations or limitations

Keep the conclusion clear, concise, and focused on the original problem."""

def combine_all_pipeline_prompts(reasons, conclusion):

    reasons_prompt = f"""\nReasons and Thoughts:
{str(reasons)}    
""" if reasons else ""
    conclusion_prompt = f"""\nFinal Conclusion:
{conclusion} 
""" if conclusion else ""

    return f"Here is the thought process and reasoning that have been gone through, so far. This might help you to come up with a proper answer:" + reasons_prompt + conclusion_prompt
