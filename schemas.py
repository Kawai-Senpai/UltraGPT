from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Optional, Union, Type

class Steps(BaseModel):
    steps: List[str]

class Reasoning(BaseModel):
    thoughts: List[str]

class ToolAnalysisSchema(BaseModel):
    tools: List[str]

class ToolCall(BaseModel):
    """
    Base ToolCall schema. In actual use, dynamic schemas are created with:
    - tool_name: str
    - parameters: <DynamicParameterModel> (based on user's parameter schema)
    - reasoning: str
    """
    tool_name: str
    reasoning: str
    # Note: 'parameters' field is added dynamically based on user tool schemas

class ToolCallResponse(BaseModel):
    tool_calls: List[ToolCall]

class SingleToolCallResponse(BaseModel):
    tool_call: ToolCall

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