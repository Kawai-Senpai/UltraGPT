from .prompts import make_query
from .schemas import CalculatorQuery
from pydantic import BaseModel

def query_finder(message, client, config):
    prompt = make_query(message)
    response = client.beta.chat.completions.parse(
        model=config.get("model", "gpt-4o"),
        messages=[{"role": "system", "content": prompt}],
        response_format=CalculatorQuery
    )
    content = response.choices[0].message.parsed
    if not content:
        return {
            "add": [],
            "sub": [],
            "mul": [],
            "div": []
        }
    if isinstance(content, BaseModel):
        content = content.model_dump(by_alias=True)
    return content

