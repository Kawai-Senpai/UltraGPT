from pydantic import BaseModel

class Steps(BaseModel):
    steps: list[str]

class Reasoning(BaseModel):
    thoughts: list[str]