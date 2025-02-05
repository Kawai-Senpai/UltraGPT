from .core import calculate

#? Required ------------------------------------------------------------------
_info = "This allows you to perform mathematic calculations."

def extract_system_message(history):
    for message in history:
        if message["role"] == "system":
            return message["content"]

def _execute(message, history, client, config):
    """Main function to execute the web search tool"""
    system_message = extract_system_message(history)
    return calculate(message + "\n" + system_message, client, config)