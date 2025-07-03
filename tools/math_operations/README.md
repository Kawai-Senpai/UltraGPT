# Math Operations Tool

This tool provides advanced mathematical operations for UltraGPT, extending beyond basic calculations. **Now supports multiple operations in a single request!**

## Key Features

- **Multiple Operations**: Perform several different mathematical operations in one request
- **Structured Parsing**: Uses OpenAI's structured output for reliable operation detection
- **Comprehensive Results**: Detailed explanations and results for each operation

## Available Operations

### 1. Range Checking (`range_checks`)
Check if numbers lie between specified bounds.

**Example**: "Check if [1, 5, 8] lie between 0 and 10, and [15, 20, 25] lie between 10 and 30"

### 2. Outlier Detection (`outlier_detections`)
Find numbers that are statistical outliers using IQR or z-score methods.

**Example**: "Find outliers in [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]"

### 3. Proximity Checking (`proximity_checks`)
Check if numbers are close to a target value within a tolerance.

**Example**: "Are the numbers [9.8, 10.2, 9.9] close to 10 within 0.5?"

### 4. Statistical Analysis (`statistical_analyses`)
Get comprehensive statistics including mean, median, mode, standard deviation, etc.

**Example**: "Get statistical summary of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"

### 5. Prime Number Checking (`prime_checks`)
Check which numbers in a list are prime.

**Example**: "Are 17, 23, 29, 30 prime numbers?"

### 6. Factor Analysis (`factor_analyses`)
Get factors and prime factorization of numbers.

**Example**: "Find factors and prime factorization of 24, 36, 48"

### 7. Sequence Analysis (`sequence_analyses`)
Check if numbers form arithmetic or geometric sequences.

**Example**: "Do [2, 4, 6, 8, 10] form an arithmetic sequence?"

### 8. Percentage Operations (`percentage_operations`)
Calculate percentages, ratios, and percentage changes.

**Example**: "Calculate percentage of total for [10, 20, 30, 40]"

## Multiple Operations Examples

You can now perform multiple operations in a single request:

```python
# Multiple different operations
response = ultragpt.chat([{
    "role": "user", 
    "content": """
    Please do these calculations:
    1. Check if [1, 5, 8] lie between 0 and 10
    2. Are 17, 23, 29 prime numbers?
    3. Get statistical summary of [1, 2, 3, 4, 5]
    4. Find outliers in [1, 2, 3, 100]
    """
}], tools=["math-operations"])

# Multiple operations of the same type
response = ultragpt.chat([{
    "role": "user", 
    "content": """
    Check these ranges:
    - Do [1, 5, 8] lie between 0 and 10?
    - Do [15, 20, 25] lie between 10 and 30?
    - Do [35, 40, 45] lie between 30 and 50?
    """
}], tools=["math-operations"])
```

## Usage

The tool automatically parses natural language requests and determines the appropriate mathematical operations to perform.

```python
from ultragpt import UltraGPT

ultragpt = UltraGPT(api_key="your-key")

# Single operation
response, tokens, details = ultragpt.chat(
    messages=[{"role": "user", "content": "Check if [1, 5, 8, 12] lie between 0 and 10"}],
    tools=["math-operations"]
)

# Multiple operations
response, tokens, details = ultragpt.chat(
    messages=[{"role": "user", "content": "Check if [1, 5, 8] are between 0-10 and are 17, 23, 29 prime?"}],
    tools=["math-operations"]
)
```

## Configuration

You can configure the math operations tool in the `tools_config`:

```python
tools_config = {
    "math-operations": {
        "model": "gpt-4o"  # Model to use for parsing operations
    }
}
```

## Technical Implementation

The tool uses:
- **Structured Output**: OpenAI's `beta.chat.completions.parse()` for reliable parsing
- **Pydantic Schemas**: Type-safe operation definitions
- **Batch Processing**: Efficient handling of multiple operations
- **Detailed Results**: Comprehensive formatting of results
