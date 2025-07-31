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
    
    # Get model from config - can now include provider specification
    model = config.get("model", "gpt-4o")
    
    # Use the provider manager if available (new multi-provider approach)
    if hasattr(client, 'provider_manager'):
        response_content, tokens = client.provider_manager.chat_completion_with_schema(
            model=model,
            messages=messages,
            schema=ToolQuery
        )
        content = response_content
    else:
        # Fallback to legacy OpenAI approach
        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=ToolQuery
        )
        content = response.choices[0].message.parsed
        if isinstance(content, BaseModel):
            content = content.model_dump(by_alias=True)
    
    if not content:
        return {"query": []}
    
    return content