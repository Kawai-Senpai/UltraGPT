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

Why this is important:
- This will help you to break down the task into smaller steps and make it easier to complete the task. So, think about the task and come up with the steps that you would take to complete it. So, only output task that you are capable of doing.
- Break down complex tasks into smaller, manageable steps to ensure clarity and focus. But actively move towards the solution and make progress.
- Assume that when you actually perform the task, the result from the previous step will be used as the input for the next step. So, make sure to provide the output of each step in a way that it can be used as input for the next step. So, we can build upon the previous steps and make progress towards the final solution.

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

    return f"""You are an expert problem solver using a private scratchpad to reason step-by-step.{context}

Private Scratchpad Rules:
- These notes are never exposed to the user, so do NOT mention secrecy, refusal, policies, or any limitation about sharing thoughts.
- Think honestly and concretely about the task, performing intermediate calculations, assumptions, and checks.
- Expand on prior thoughts when present so each new thought advances the solution.
- Explore alternative approaches, edge cases, and potential pitfalls, then converge on the most promising plan.
- Avoid filler, apologies, or meta-commentary; focus purely on useful reasoning content.
- Do not write phrases like "Final answer", "I can't share", or anything that sounds like a public response; this is internal analysis only.

Focus Areas:
- Brainstorm multiple solution paths or strategies and compare their trade-offs.
- Identify requirements, constraints, risks, and the data you still need.
- When a path seems viable, sketch the concrete steps or sub-calculations required to execute it.
- Note validations or tests you would run to confirm the result.
- Reference available tools or resources if they unlock better answers.

Formatting:
- Produce 3 to 5 concise thought entries.
- Each entry must start with "Thought X:" (where X is 1, 2, 3, ...).
- Each entry should capture a meaningful piece of reasoning such as a sub-problem, calculation, assumption, or validation.
- Keep the focus on exploration and intermediate logic, not on re-stating the final answer.
- Vary the content across entries (e.g., consider alternative approaches, sanity checks, tool usage) instead of repeating the same sentence.
- If you arrive at a conclusion, rewrite it as reasoning steps ("Thought X: Multiply..."), not as a final answer statement.
- If you detect that two thoughts are identical, revise them so each adds new reasoning signal. (Do not repeat the same thought.)

Your output must be valid JSON. Example:
{{
    "thoughts": [
        "Thought 1: Convert 20% to decimal (0.2) so multiplication is straightforward.",
        "Thought 2: Multiply 0.2 by 150 to produce 30 and note units remain consistent.",
        "Thought 3: Sanity-check by seeing that 10% would be 15, so doubling confirms 30."
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

def generate_tool_call_prompt(user_tools: list, allow_multiple: bool = True) -> str:
    """Generate prompt for tool calling functionality"""
    tools_info = ""
    for tool in user_tools:
        # Build tool info with support for both UserTool and ExpertTool
        # Note: Description is already sent via native tool format, no need to repeat here
        tools_info += f"""
Tool: {tool['name']}
When to use: {tool['when_to_use']}
Usage Guide: {tool['usage_guide']}"""
        
        # Add ExpertTool specific fields if present
        if 'expert_category' in tool:
            tools_info += f"\nExpert Category: {tool['expert_category']}"
        if 'prerequisites' in tool and tool['prerequisites']:
            tools_info += f"\nPrerequisites (Tools): {', '.join(tool['prerequisites'])}"
    
        # Add newline for separation between tools
        tools_info += "\n"

    multiple_instruction = "You can call multiple tools if needed." if allow_multiple else "You can only call ONE tool at a time."
    
    return f"""Available Tools:
{tools_info}

Tool Instructions:
- {multiple_instruction}
- Use 'stop_after_tool_call' param only when no more tool calls are needed.
- It ends the session immediately and you will not see the tool’s result, so only use it after you have seen the results and are sure."""

def generate_single_tool_call_prompt(user_tools: list) -> str:
    """Generate prompt for single tool calling"""
    return generate_tool_call_prompt(user_tools, allow_multiple=False)

def generate_multiple_tool_call_prompt(user_tools: list) -> str:
    """Generate prompt for multiple tool calling"""
    return generate_tool_call_prompt(user_tools, allow_multiple=True)
