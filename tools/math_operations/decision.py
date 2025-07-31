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
        
        # Get model from config - can now include provider specification
        model = config.get("model", "gpt-4o")
        
        # Use the provider manager if available (new multi-provider approach)
        if hasattr(client, 'provider_manager'):
            response_content, tokens = client.provider_manager.chat_completion_with_schema(
                model=model,
                messages=messages,
                schema=MathOperationsQuery
            )
            content = response_content
        else:
            # Fallback to legacy OpenAI approach
            response = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=MathOperationsQuery
            )
            content = response.choices[0].message.parsed
            if isinstance(content, BaseModel):
                content = content.model_dump(by_alias=True)
        
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
