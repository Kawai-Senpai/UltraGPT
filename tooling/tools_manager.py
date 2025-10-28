"""
UltraGPT Tools Module

This module contains all tool-related functionality including:
- Internal tool loading and execution
- User tool validation and conversion
- Native tool format conversions
- Tool execution orchestration

All these functions were moved from core.py to keep the main class organized.
"""

import os
import json
import inspect
import importlib
from typing import Any, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from .. import config
from ..schemas import (
    normalize_pydantic_optional_fields,
    prepare_schema_for_openai,
)

class ToolManager:
    """
    Manages all tool-related operations for UltraGPT.
    This class handles both internal tools and user-defined tools.
    """
    
    def __init__(self, ultragpt_instance):
        """
        Initialize the ToolManager with a reference to the UltraGPT instance.
        
        Args:
            ultragpt_instance: The UltraGPT instance that owns this tool manager
        """
        self.ultragpt = ultragpt_instance
        self.log = ultragpt_instance.log
        self.verbose = ultragpt_instance.verbose
        self.provider_manager = ultragpt_instance.provider_manager
        self.max_tokens = ultragpt_instance.max_tokens
        self.google_api_key = ultragpt_instance.google_api_key
        self.search_engine_id = ultragpt_instance.search_engine_id



    def load_internal_tools(self, tools: list) -> dict:
        """Dynamically load internal tool configurations from tool directories"""
        package_root = os.path.dirname(os.path.dirname(__file__))
        tools_base_path = os.path.join(package_root, "tools")
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
                
                # Normalize Optional fields for OpenAI compatibility
                normalize_pydantic_optional_fields(schema_dict)
                
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

    def convert_internal_tools_to_native_format(self, loaded_tools: dict) -> list:
        """Convert loaded internal tools to native AI provider tool format"""
        native_tools = []
        
        for tool_name, tool_config in loaded_tools.items():
            # Use the native_schema that was already created in load_internal_tools
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

    def execute_internal_tool_with_params(
        self, 
        tool_config: dict, 
        parameters: dict, 
        tools_config: dict
    ) -> str:
        """Execute an internal tool directly with AI-provided parameters"""
        # but kept for future extensibility
        try:
            # Get tool-specific config
            tool_name_normalized = tool_config['name']  # Use normalized name
            config_data = tools_config.get(tool_name_normalized, {})
            
            # Import and execute the tool's execute_tool function
            try:
                core_module = importlib.import_module(f'.tools.{tool_name_normalized}.core', package='ultragpt')
                execute_function = getattr(core_module, 'execute_tool', None)
                
                if execute_function:
                    # Set credentials for web search tool in thread-local context
                    if tool_name_normalized == "web_search":
                        try:
                            context_module = importlib.import_module(f'.tools.{tool_name_normalized}.context', package='ultragpt')
                            context_module.set_credentials(self.google_api_key, self.search_engine_id)
                        except ImportError:
                            pass  # Context module not available
                    
                    result = execute_function(parameters)
                    
                    # Clear credentials after execution
                    if tool_name_normalized == "web_search":
                        try:
                            context_module.clear_credentials()
                        except:
                            pass
                    
                    return result if result else ""
                else:
                    return f"No execute_tool function found for {tool_name_normalized}"
                    
            except Exception as e:
                return f"Error executing tool {tool_name_normalized}: {str(e)}"
                
        except Exception as e:
            return f"Tool execution error: {str(e)}"

    def execute_tools(
        self,
        history: List[BaseMessage],
        tools: list,
        tools_config: dict
    ) -> tuple:
        """Execute tools using native AI tool calling - returns (result_string, tool_usage_details)"""
        tool_usage_details = []

        if not tools or not history:
            return "", tool_usage_details

        message = ""
        for msg in reversed(history):
            if isinstance(msg, HumanMessage) and msg.content:
                message = str(msg.content)
                break
        
        if not message:
            return "", tool_usage_details
        
        try:
            self.log.info(f"Loading and executing {len(tools)} tools using native AI tool calling")
            if self.verbose:
                self.log.debug(f"➤ Loading {len(tools)} tools for native AI tool calling")
                self.log.debug(f"Query: {message[:100] + '...' if len(message) > 100 else message}")
                self.log.debug("-" * 40)
            
            # Load internal tools dynamically
            loaded_tools = self.load_internal_tools(tools)
            if not loaded_tools:
                self.log.warning("No tools could be loaded")
                return "", tool_usage_details
            
            # Convert to native tool format
            native_tools = self.convert_internal_tools_to_native_format(loaded_tools)
            
            # Create tool selection prompt
            tool_descriptions = []
            for tool_name, tool_config in loaded_tools.items():
                tool_descriptions.append(f"- {tool_config['display_name']}: {tool_config['description']}")
            
            tool_selection_prompt = (
                "Available internal tools:\n"
                f"{chr(10).join(tool_descriptions)}\n\n"
                "Analyze the user's message and select the appropriate tools with their parameters. Each tool should be called with the specific parameters needed to help answer the user's question.\n\n"
                f"User message: \"{message}\"\n\n"
                "IMPORTANT:\n"
                "- Only call tools that meaningfully contribute to solving the user's request\n"
                "- Follow each tool's schema when providing parameters\n"
                "- Skip tool calls when they are unnecessary\n"
                "- Multiple tools may be called sequentially when required\n"
            )

            tool_messages: List[BaseMessage] = [SystemMessage(content=tool_selection_prompt)]

            context_window = history[-config.MAX_CONTEXT_MESSAGES : ] if config.MAX_CONTEXT_MESSAGES else history
            trimmed_context: List[BaseMessage] = []
            for context_msg in context_window:
                if isinstance(context_msg, (HumanMessage, AIMessage)) and context_msg.content:
                    trimmed_context.append(context_msg)

            if trimmed_context and isinstance(trimmed_context[-1], HumanMessage) and str(trimmed_context[-1].content) == message:
                trimmed_context = trimmed_context[:-1]

            tool_messages.extend(trimmed_context)
            tool_messages.append(HumanMessage(content=message))
            
            # Get model from tools_config or use default
            model = config.DEFAULT_TOOLS_MODEL
            parallel_tool_calls = True  # Default to enabling parallel tool calls
            for tool_config in tools_config.values():
                if 'model' in tool_config:
                    model = tool_config['model']
                if 'parallel_tool_calls' in tool_config:
                    parallel_tool_calls = tool_config['parallel_tool_calls']
                if 'model' in tool_config or 'parallel_tool_calls' in tool_config:
                    break
            
            # Make native tool call
            try:
                response_message, tokens = self.provider_manager.chat_completion_with_tools(
                    model=model,
                    messages=tool_messages,
                    tools=native_tools,
                    temperature=config.TOOL_SELECTION_TEMPERATURE,  # Low temperature for tool selection
                    max_tokens=self.max_tokens,
                    parallel_tool_calls=parallel_tool_calls  # Enable parallel tool calls
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
                return "", tool_usage_details
            
            # Execute the selected tools
            tool_results = []
            for tool_call in response_message.get('tool_calls', []):
                function_name = tool_call.get('function', {}).get('name')
                function_args = tool_call.get('function', {}).get('arguments', {})
                
                # Parse arguments if they're in string format
                if isinstance(function_args, str):
                    try:
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
                        tool_result = self.execute_internal_tool_with_params(
                            tool_config, function_args, tools_config
                        )
                        
                        tool_results.append({
                            "tool": tool_config['display_name'],
                            "response": tool_result
                        })
                        
                        # Track tool usage
                        tool_usage_details.append({
                            "tool_name": function_name,
                            "display_name": tool_config['display_name'],
                            "parameters": function_args,
                            "result": tool_result,
                            "success": True,
                            "error": None
                        })
                        
                        if self.verbose:
                            self.log.debug(f"✓ {function_name} completed")
                            self.log.debug("-" * 40)
                            self.log.debug(tool_result if tool_result else "(empty result)")
                            self.log.debug("-" * 40)
                            
                    except Exception as e:
                        error_msg = f"Tool execution failed: {str(e)}"
                        self.log.error(f"Tool {function_name} execution failed: {str(e)}")
                        tool_results.append({
                            "tool": tool_config['display_name'],
                            "response": error_msg
                        })
                        
                        # Track failed tool usage
                        tool_usage_details.append({
                            "tool_name": function_name,
                            "display_name": tool_config['display_name'],
                            "parameters": function_args,
                            "result": None,
                            "success": False,
                            "error": str(e)
                        })
                else:
                    self.log.warning(f"Unknown tool called: {function_name}")
            
            # Format results
            if not tool_results:
                return "", tool_usage_details
                
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
                
            return "\n\n".join(formatted_responses), tool_usage_details
                
        except Exception as e:
            self.log.error(f"Tool execution failed: {str(e)}")
            if self.verbose:
                self.log.debug(f"✗ Tool execution failed: {str(e)}")
            return "", tool_usage_details

    def convert_user_tools_to_native_format(self, user_tools: list) -> list:
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
            
            # Normalize Pydantic v2 Optional field schemas to OpenAI-compatible format
            normalize_pydantic_optional_fields(parameters_schema)
            
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
            
            # Finalize schema for strict mode: inline refs, flatten allOf, rebuild required, disallow extras
            parameters_schema = prepare_schema_for_openai(parameters_schema)
            
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

    def validate_user_tools(self, user_tools: list) -> list:
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
