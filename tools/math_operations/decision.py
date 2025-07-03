from .prompts import make_math_operations_query
from .schemas import MathOperationsQuery
from pydantic import BaseModel

def query_finder(message, client, config):
    """Parse the message to find what mathematical operations are needed"""
    try:
        prompt = make_math_operations_query(message)
        response = client.beta.chat.completions.parse(
            model=config.get("model", "gpt-4o"),
            messages=[{"role": "system", "content": prompt}],
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
