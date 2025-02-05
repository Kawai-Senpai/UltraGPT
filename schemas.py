from pydantic import BaseModel
from typing import List

class Steps(BaseModel):
    steps: List[str]

class Reasoning(BaseModel):
    thoughts: List[str]

class ToolAnalysisSchema(BaseModel):
    tools: List[str]