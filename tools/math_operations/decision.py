from .prompts import make_math_operations_query
from .schemas import MathOperationsQuery
from pydantic import BaseModel

def query_finder(message, client, config, history=None):
    """Parse the message to find what mathematical operations are needed"""
    try:
        prompt = make_math_operations_query(message)
        
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
            response_format=MathOperationsQuery
        )
        content = response.choices[0].message.parsed
        
        # Default structure
        default_operations = {
            "range_checks": [],
            "proximity_checks": [],
            "statistical_analyses": [],
            "prime_checks": [],
            "factor_analyses": [],
            "sequence_analyses": [],
            "percentage_operations": [],
            "outlier_detections": []
        }
        
        if not content:
            return default_operations
            
        if isinstance(content, BaseModel):
            content = content.model_dump(by_alias=True)
        
        # Ensure all fields exist and are lists, not None
        for key in default_operations:
            if key not in content or content[key] is None:
                content[key] = []
                
        return content
        
    except Exception as e:
        # Return default structure on any error
        return {
            "range_checks": [],
            "proximity_checks": [],
            "statistical_analyses": [],
            "prime_checks": [],
            "factor_analyses": [],
            "sequence_analyses": [],
            "percentage_operations": [],
            "outlier_detections": []
        }
