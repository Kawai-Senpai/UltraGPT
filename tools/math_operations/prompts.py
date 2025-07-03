def make_math_operations_query(message):
    return f"""You are a mathematical operations parser. Analyze the user's request and history and perform all the needed and necessary mathematical operations, which will initially help the user in their request.

Available operations:
1. "range_checks" - Check if numbers lie between ranges (a, b)
2. "proximity_checks" - Check if numbers are close to target values within tolerance
3. "statistical_analyses" - Get mean, median, mode, standard deviation of numbers
4. "prime_checks" - Check if numbers are prime
5. "factor_analyses" - Get factors and prime factorization
6. "sequence_analyses" - Check if numbers form arithmetic or geometric sequences
7. "percentage_operations" - Calculate percentages using ONLY these operation types:
   - "percentage_of_total" (calculate what percentage each number is of the total)
   - "percentage_change" (calculate percentage change between consecutive numbers)
8. "outlier_detections" - Find numbers that are outside normal range using ONLY these methods:
   - "iqr" (Interquartile Range method)
   - "zscore" (Z-score method)

IMPORTANT: For percentage_operations, ONLY use "percentage_of_total" or "percentage_change" as operation_type.
IMPORTANT: For outlier_detections, ONLY use "iqr" or "zscore" as method.
DO NOT use "subtract", "division", "add", "multiply" or any other operation types.

User request: "{message}"

Parse the request and extract all mathematical operations needed. You can perform multiple operations of the same type or different types in one request.

JSON Schema Examples:

Example 1 - Multiple range checks:
{{
    "range_checks": [
        {{"numbers": [1, 5, 8], "range_min": 0, "range_max": 10}},
        {{"numbers": [15, 20, 25], "range_min": 10, "range_max": 30}}
    ],
    "proximity_checks": [],
    "statistical_analyses": [],
    "prime_checks": [],
    "factor_analyses": [],
    "sequence_analyses": [],
    "percentage_operations": [],
    "outlier_detections": []
}}

Example 2 - Mixed operations:
{{
    "range_checks": [],
    "proximity_checks": [],
    "statistical_analyses": [
        {{"numbers": [1, 2, 3, 4, 5]}}
    ],
    "prime_checks": [
        {{"numbers": [17, 23, 29]}}
    ],
    "factor_analyses": [],
    "sequence_analyses": [],
    "percentage_operations": [],
    "outlier_detections": []
}}

Example 3 - Outlier detection and sequence analysis:
{{
    "range_checks": [],
    "proximity_checks": [],
    "statistical_analyses": [],
    "prime_checks": [],
    "factor_analyses": [],
    "sequence_analyses": [
        {{"numbers": [2, 4, 6, 8]}}
    ],
    "percentage_operations": [],
    "outlier_detections": [
        {{"numbers": [1, 2, 3, 100], "method": "iqr"}}
    ]
}}

Example 4 - Proximity check:
{{
    "range_checks": [],
    "proximity_checks": [
        {{"numbers": [9.8, 10.2, 9.9], "target": 10, "tolerance": 0.5}}
    ],
    "statistical_analyses": [],
    "prime_checks": [],
    "factor_analyses": [],
    "sequence_analyses": [],
    "percentage_operations": [],
    "outlier_detections": []
}}

Example 5 - Percentage operations:
{{
    "range_checks": [],
    "proximity_checks": [],
    "statistical_analyses": [],
    "prime_checks": [],
    "factor_analyses": [],
    "sequence_analyses": [],
    "percentage_operations": [
        {{"numbers": [10, 20, 30, 40], "operation_type": "percentage_of_total"}},
        {{"numbers": [100, 110, 121], "operation_type": "percentage_change"}}
    ],
    "outlier_detections": []
}}

Your response must match exactly the MathOperationsQuery schema format shown above."""

def execute_math_operation_prompt(operation_request):
    return f"""Execute the mathematical operation based on this request:
{operation_request}

Perform the calculation and provide:
1. The result of the operation
2. A clear explanation of what was calculated
3. Additional details if relevant

Be precise and show your work where applicable.
Your output should be in JSON format matching the MathOperationResult schema.
"""
