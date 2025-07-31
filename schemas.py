from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Steps(BaseModel):
    steps: List[str]

class Reasoning(BaseModel):
    thoughts: List[str]

class ToolAnalysisSchema(BaseModel):
    tools: List[str]

class ToolCall(BaseModel):
    tool_name: str
    reasoning: str

class ToolCallResponse(BaseModel):
    tool_calls: List[ToolCall]

class SingleToolCallResponse(BaseModel):
    tool_call: ToolCall

class UserTool(BaseModel):
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    usage_guide: str
    when_to_use: str