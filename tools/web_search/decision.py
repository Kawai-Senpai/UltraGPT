from .prompts import make_query
from .schemas import ToolQuery
from pydantic import BaseModel

#! Initialize ---------------------------------------------------------------

def query_finder(message, client, config, history=None):
    prompt = make_query(message)
    
    # Build messages array with history as user messages, followed by system message
    messages = []
    if history:
        # Get max history items from config (default: 5)
        max_history_items = config.get("max_history_items", 5)
        
        # Take only the last N history items to stay within limit
        recent_history = history[-max_history_items:] if len(history) > max_history_items else history
        
        for hist_msg in recent_history:
            if isinstance(hist_msg, dict) and hist_msg.get("content"):
                messages.append({"role": "user", "content": hist_msg["content"]})
    
    # Add the system message at the end
    messages.append({"role": "system", "content": prompt})
    
    response = client.beta.chat.completions.parse(
        model=config.get("model", "gpt-4o"),
        messages=messages,
        response_format=ToolQuery
    )
    content = response.choices[0].message.parsed
    if not content:
        return {"query": []}
    if isinstance(content, BaseModel):
        content = content.model_dump(by_alias=True)
    return content