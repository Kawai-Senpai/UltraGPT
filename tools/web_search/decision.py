from .prompts import make_query
from .schemas import ToolQuery
from pydantic import BaseModel

#! Initialize ---------------------------------------------------------------

def query_finder(message, client, config):
    prompt = make_query(message)
    response = client.beta.chat.completions.parse(
        model=config.get("model", "gpt-4o"),
        messages=[{"role": "system", "content": prompt}],
        response_format=ToolQuery
    )
    content = response.choices[0].message.parsed
    if not content:
        return {"query": []}
    if isinstance(content, BaseModel):
        content = content.model_dump(by_alias=True)
    return content