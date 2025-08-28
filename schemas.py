from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Optional, Union, Type

class Steps(BaseModel):
    steps: List[str]

class Reasoning(BaseModel):
    thoughts: List[str]

class ToolAnalysisSchema(BaseModel):
    tools: List[str]

class UserTool(BaseModel):
    name: str
    description: str
    parameters_schema: Union[Type[BaseModel], Dict[str, Any]]
    usage_guide: str
    when_to_use: str
    
    @field_validator('parameters_schema')
    @classmethod
    def validate_parameters_schema(cls, v):
        """Convert Pydantic class to JSON schema if needed"""
        if isinstance(v, type) and issubclass(v, BaseModel):
            return v.model_json_schema()
        elif isinstance(v, dict):
            return v
        else:
            raise ValueError("parameters_schema must be either a Pydantic BaseModel class or a dict")

class ExpertTool(UserTool):
    """Extended UserTool for expert systems with additional metadata"""
    expert_category: str
    prerequisites: Optional[List[str]] = None
    
    # ExpertTool inherits all validation from UserTool including parameters_schema

# Schemas for native tool calling responses
class FunctionCall(BaseModel):
    """Function call within a tool call"""
    name: str
    arguments: str  # JSON string

class NativeToolCall(BaseModel):
    """Native tool call from AI providers"""
    id: str
    type: str  # "function"
    function: FunctionCall

class NativeToolCallMessage(BaseModel):
    """Message containing native tool calls"""
    role: str  # "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[NativeToolCall]] = None

class ToolResult(BaseModel):
    """Result from executing a tool"""
    tool_call_id: str
    content: str
    error: Optional[str] = None