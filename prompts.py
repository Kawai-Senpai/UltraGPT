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
    
    return f"""You are an expert at careful reasoning and deep thinking.{context}

Let me think about this further...

Rules:
- Express your thoughts naturally as they come
- Build upon previous thoughts if they exist
- Consider multiple aspects simultaneously
- Show your genuine thinking process
- No need to structure or analyze - just think

What your thoughts must follow / focus on:
- Your thoughts must move towards solving the problem or answering the question at hand. You need to brainstorm and think out loud.
- You must bran storm through multiple possible solutions and approaches to the problem. You can think about multiple poossible solutions and approaches to the problem. And out put them.
- You can also think about the possible limitations or challenges that you might face while solving the problem. And how to overcome them.
- You can reiterate rules, directions and instructions that you have been given so far. But only how it effects your thoughts and your ability to solve the problem. And what you should do.
- If your previous thoughts were of, let's say 'thinking of doing X', you can think of 'how to do X' or straight up 'do X' and output the result in your thoughts.
- You can also use tools if provided to you.
- The goal of this is to expose as much possible reasons, thoughts, ideas, approaches, solutions, limitations, challenges and anything else that you can think of that might help you to solve the problem or answer the question at hand. This may also contain multiple solution itself!

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

def make_tool_analysis_prompt(message: str, available_tools: list) -> str:
    """Format prompt for tool analysis"""
    tools_str = str(available_tools)
    example = """Your output should look like this (example):
{
    "tools": ["web-search", "tool_name2"]
}"""
    return f"""This is a user message. Analyze if this user message requires any tools. Available tools: {tools_str}

Message: "{message}"

{example}

Rules:
- Only include the tool names under "tools" that are needed to respond to the message.
- If no tools are needed, return empty array. You can use multiple tools if needed.
- Only use tools that are available to you. Do not use any other tools.
- It is not necessary to use a tool for every message. Only use a tool if it is truly needed.
- Your output should be in parsable proper JSON format like the given example.
"""

def format_tool_response(tool_response: str) -> str:
    """Format tool response for inclusion in context"""
    return f"\n\nTool response: {tool_response}" if tool_response else ""

def generate_tool_call_prompt(user_tools: list, allow_multiple: bool = True) -> str:
    """Generate prompt for tool calling functionality"""
    tools_info = ""
    for tool in user_tools:
        tools_info += f"""
Tool Name: {tool['name']}
Description: {tool['description']}
When to use: {tool['when_to_use']}
Usage Guide: {tool['usage_guide']}
Parameters Schema: {tool['parameters_schema']}
---
"""
    
    multiple_instruction = "You can call multiple tools if needed." if allow_multiple else "You can only call ONE tool at a time."
    
    return f"""You are an expert AI assistant with access to the following tools. Your task is to analyze the user's request and determine which tool(s) to call and with what parameters.

Available Tools:
{tools_info}

Instructions:
- Carefully analyze the user's request to understand what they need
- Use your reasoning capabilities to determine which tool(s) would be most appropriate
- Generate the exact parameters needed for each tool call based on the user's request
- {multiple_instruction}
- If no further tools are needed after this, use the 'stop_after_tool_call' tool to indicate no action is required after calling a tool.

For each tool call, you must:
1. Identify the most appropriate tool for the task
2. Extract or generate the exact parameters needed from the user's request
3. Provide clear, user-friendly reasoning that explains how you'll help them accomplish their goal

IMPORTANT: Your reasoning should be written as if you're a helpful assistant talking directly to the user. 
- Don't mention technical tool names or parameter details
- Focus on explaining how you'll help them achieve their goal
- Sound natural and conversational
- Example: "I'll help you send that email to John with your project update" instead of "The 'send_email' tool is appropriate with parameters recipient, subject, body"

Your response must be in the specified JSON format with tool calls and reasoning."""

def generate_single_tool_call_prompt(user_tools: list) -> str:
    """Generate prompt for single tool calling"""
    return generate_tool_call_prompt(user_tools, allow_multiple=False)

def generate_multiple_tool_call_prompt(user_tools: list) -> str:
    """Generate prompt for multiple tool calling"""
    return generate_tool_call_prompt(user_tools, allow_multiple=True)
