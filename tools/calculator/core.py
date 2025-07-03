from .decision import query_finder

#* Web search ---------------------------------------------------------------
def calculate(message, client, config, history=None):
    """Perform web search using DuckDuckGo"""
    try:
        querys = query_finder(message, client, config, history)
        addition = querys.get("add", [])
        subtract = querys.get("sub", [])
        multiply = querys.get("mul", [])
        divide = querys.get("div", [])

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
        return ""