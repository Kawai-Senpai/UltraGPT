#* Calculator operations ---------------------------------------------------------------
def execute_tool(parameters):
    """Standard entry point for calculator tool - takes AI-provided parameters directly"""
    try:
        addition = parameters.get("add", [])
        subtract = parameters.get("sub", [])
        multiply = parameters.get("mul", [])
        divide = parameters.get("div", [])

        formatted_results = []
        if addition:
            add = 0
            for a in addition:
                add += a
            formatted_results.append(f"Addition of {addition} = {add}")
        if subtract:
            sub = subtract[0]
            for s in subtract[1:]:
                sub -= s
            formatted_results.append(f"Subtraction of {subtract} = {sub}")
        if multiply:
            mul = 1
            for m in multiply:
                mul *= m
            formatted_results.append(f"Multiplication of {multiply} = {mul}")
        if divide:
            div = divide[0]
            for d in divide[1:]:
                if d == 0:
                    div = "Infinity"
                    break
                div /= d
            formatted_results.append(f"Division of {divide} = {div}")

        return "\n".join(formatted_results) if formatted_results else ""
    except Exception as e:
        return f"Calculator error: {str(e)}"

def calculate(message, client, config, history=None):
    """Legacy function - now serves as fallback for direct calls"""
    return "Calculator tool is now using native AI tool calling. Please use the UltraGPT chat interface to access calculator functions."

def perform_calculations(parameters):
    """Legacy function - redirects to standard entry point"""
    return execute_tool(parameters)