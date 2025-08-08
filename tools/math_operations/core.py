#* Math operations ---------------------------------------------------------------
def execute_tool(parameters):
    """Standard entry point for math operations tool - takes AI-provided parameters directly"""
    try:
        expression = parameters.get("expression")
        variables = parameters.get("variables", {})
        
        if not expression:
            return "Please provide a mathematical expression to evaluate."
        
        try:
            # Import math functions for evaluation
            import math
            
            # Create a safe evaluation environment
            safe_dict = {
                '__builtins__': {},
                'math': math,
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'log': math.log, 'log10': math.log10, 'sqrt': math.sqrt,
                'abs': abs, 'pow': pow,
                'pi': math.pi, 'e': math.e,
                'factorial': math.factorial,
                'exp': math.exp,
                'floor': math.floor, 'ceil': math.ceil,
                'max': max, 'min': min,
                'sum': sum
            }
            
            # Add any provided variables
            safe_dict.update(variables)
            
            # Evaluate the expression
            result = eval(expression, safe_dict)
            return f"Result: {result}"
                
        except Exception as e:
            return f"Error evaluating expression '{expression}': {str(e)}"
    except Exception as e:
        return f"Math operations tool error: {str(e)}"

def math_operations(message, client, config, history=None):
    """Legacy function - now serves as fallback for direct calls"""
    return "Math operations tool is now using native AI tool calling. Please use the UltraGPT chat interface to access math functions."

def perform_math_operations(parameters):
    """Legacy function - redirects to standard entry point"""
    return execute_tool(parameters)
