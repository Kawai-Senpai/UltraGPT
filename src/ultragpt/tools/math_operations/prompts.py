#* Math Operations Prompts ---------------------------------------------------------------
_info = "This allows you to perform advanced mathematical calculations and operations"

_description = """This is a mathematical operations tool that allows you to:
1. Perform complex mathematical calculations
2. Solve equations and mathematical expressions
3. Handle advanced mathematical operations beyond basic arithmetic

Parameters:
- expression: The mathematical expression to evaluate (required)
- variables: Dictionary of variable values if the expression contains variables (optional)

Examples:
- "Calculate 2^3 + sqrt(16)" → expression="2**3 + sqrt(16)"
- "Solve x^2 + 5x + 6 = 0" → expression="x**2 + 5*x + 6", variables={"x": [values]}
- "Evaluate sin(pi/2)" → expression="sin(pi/2)"

Supported functions: sin, cos, tan, log, ln, sqrt, abs, pow, factorial, and more."""
