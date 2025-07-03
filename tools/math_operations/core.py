import math
import statistics
from typing import List, Union, Tuple
from .schemas import MathOperationResult

def check_range(numbers: List[Union[int, float]], range_min: Union[int, float], range_max: Union[int, float]) -> MathOperationResult:
    """Check if numbers lie between a range"""
    results = [range_min <= num <= range_max for num in numbers]
    in_range = [num for num, is_in in zip(numbers, results) if is_in]
    out_of_range = [num for num, is_in in zip(numbers, results) if not is_in]
    
    explanation = f"Checked if numbers {numbers} lie between {range_min} and {range_max}"
    details = f"In range: {in_range}, Out of range: {out_of_range}"
    
    return MathOperationResult(
        operation="check_range",
        result=results,
        explanation=explanation,
        details=details
    )

def find_outliers(numbers: List[Union[int, float]], method: str = "iqr") -> MathOperationResult:
    """Find outliers in a dataset"""
    if len(numbers) < 4:
        return MathOperationResult(
            operation="find_outliers",
            result=[],
            explanation="Need at least 4 numbers to detect outliers reliably",
            details="Insufficient data"
        )
    
    if method == "iqr":
        sorted_nums = sorted(numbers)
        q1 = statistics.quantiles(sorted_nums, n=4)[0]
        q3 = statistics.quantiles(sorted_nums, n=4)[2]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [num for num in numbers if num < lower_bound or num > upper_bound]
    else:  # z-score method
        mean = statistics.mean(numbers)
        std_dev = statistics.stdev(numbers) if len(numbers) > 1 else 0
        if std_dev == 0:
            outliers = []
        else:
            outliers = [num for num in numbers if abs((num - mean) / std_dev) > 2]
    
    return MathOperationResult(
        operation="find_outliers",
        result=outliers,
        explanation=f"Found outliers using {method} method",
        details=f"Outliers: {outliers}, Method: {method}"
    )

def check_proximity(numbers: List[Union[int, float]], target: Union[int, float], tolerance: Union[int, float]) -> MathOperationResult:
    """Check if numbers are close to target within tolerance"""
    results = [abs(num - target) <= tolerance for num in numbers]
    close_numbers = [num for num, is_close in zip(numbers, results) if is_close]
    far_numbers = [num for num, is_close in zip(numbers, results) if not is_close]
    
    explanation = f"Checked if numbers {numbers} are within {tolerance} of target {target}"
    details = f"Close to target: {close_numbers}, Far from target: {far_numbers}"
    
    return MathOperationResult(
        operation="check_proximity",
        result=results,
        explanation=explanation,
        details=details
    )

def statistical_summary(numbers: List[Union[int, float]]) -> MathOperationResult:
    """Get statistical summary of numbers"""
    if not numbers:
        return MathOperationResult(
            operation="statistical_summary",
            result="No numbers provided",
            explanation="Cannot compute statistics for empty list"
        )
    
    mean_val = statistics.mean(numbers)
    median_val = statistics.median(numbers)
    
    try:
        mode_val = statistics.mode(numbers)
    except statistics.StatisticsError:
        mode_val = "No unique mode"
    
    if len(numbers) > 1:
        std_dev = statistics.stdev(numbers)
        variance = statistics.variance(numbers)
    else:
        std_dev = 0
        variance = 0
    
    min_val = min(numbers)
    max_val = max(numbers)
    range_val = max_val - min_val
    
    summary = {
        "mean": mean_val,
        "median": median_val,
        "mode": mode_val,
        "std_dev": std_dev,
        "variance": variance,
        "min": min_val,
        "max": max_val,
        "range": range_val,
        "count": len(numbers)
    }
    
    explanation = f"Statistical summary of {len(numbers)} numbers"
    details = f"Mean: {mean_val:.2f}, Median: {median_val}, Std Dev: {std_dev:.2f}"
    
    return MathOperationResult(
        operation="statistical_summary",
        result=summary,
        explanation=explanation,
        details=details
    )

def is_prime(n: int) -> bool:
    """Check if a number is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def prime_check(numbers: List[Union[int, float]]) -> MathOperationResult:
    """Check which numbers are prime"""
    integer_numbers = [int(num) for num in numbers if num == int(num) and num > 0]
    results = [is_prime(num) for num in integer_numbers]
    prime_numbers = [num for num, is_prime_val in zip(integer_numbers, results) if is_prime_val]
    composite_numbers = [num for num, is_prime_val in zip(integer_numbers, results) if not is_prime_val and num > 1]
    
    explanation = f"Checked primality of numbers {integer_numbers}"
    details = f"Prime: {prime_numbers}, Composite: {composite_numbers}"
    
    # Convert integer keys to string keys for Pydantic validation
    result_dict = {str(num): is_prime_val for num, is_prime_val in zip(integer_numbers, results)}
    
    return MathOperationResult(
        operation="prime_check",
        result=result_dict,
        explanation=explanation,
        details=details
    )

def get_factors(n: int) -> List[int]:
    """Get all factors of a number"""
    factors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
    return sorted(factors)

def prime_factorization(n: int) -> List[int]:
    """Get prime factorization of a number"""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def factor_analysis(numbers: List[Union[int, float]]) -> MathOperationResult:
    """Analyze factors and prime factorization"""
    integer_numbers = [int(num) for num in numbers if num == int(num) and num > 0]
    analysis = {}
    
    for num in integer_numbers:
        factors = get_factors(num)
        prime_factors = prime_factorization(num)
        # Use string keys for Pydantic validation
        analysis[str(num)] = {
            "factors": factors,
            "prime_factorization": prime_factors,
            "factor_count": len(factors)
        }
    
    explanation = f"Factor analysis of numbers {integer_numbers}"
    
    return MathOperationResult(
        operation="factor_analysis",
        result=analysis,
        explanation=explanation
    )

def check_sequence(numbers: List[Union[int, float]]) -> MathOperationResult:
    """Check if numbers form arithmetic or geometric sequence"""
    if len(numbers) < 3:
        return MathOperationResult(
            operation="check_sequence",
            result="Need at least 3 numbers to check sequence",
            explanation="Insufficient numbers for sequence analysis"
        )
    
    # Check arithmetic sequence
    is_arithmetic = True
    arithmetic_diff = numbers[1] - numbers[0]
    for i in range(2, len(numbers)):
        if numbers[i] - numbers[i-1] != arithmetic_diff:
            is_arithmetic = False
            break
    
    # Check geometric sequence
    is_geometric = True
    geometric_ratio = None
    if numbers[0] != 0 and numbers[1] != 0:
        geometric_ratio = numbers[1] / numbers[0]
        for i in range(2, len(numbers)):
            if numbers[i-1] == 0 or numbers[i] / numbers[i-1] != geometric_ratio:
                is_geometric = False
                break
    else:
        is_geometric = False
    
    result = {
        "is_arithmetic": is_arithmetic,
        "is_geometric": is_geometric,
        "arithmetic_difference": arithmetic_diff if is_arithmetic else None,
        "geometric_ratio": geometric_ratio if is_geometric else None
    }
    
    sequence_type = []
    if is_arithmetic:
        sequence_type.append(f"arithmetic (diff: {arithmetic_diff})")
    if is_geometric and geometric_ratio is not None:
        sequence_type.append(f"geometric (ratio: {geometric_ratio})")
    
    explanation = f"Sequence analysis: {', '.join(sequence_type) if sequence_type else 'No pattern detected'}"
    
    return MathOperationResult(
        operation="check_sequence",
        result=result,
        explanation=explanation
    )

def percentage_operations(numbers: List[Union[int, float]], operation_type: str = "percentage_of_total") -> MathOperationResult:
    """Perform percentage calculations"""
    if not numbers:
        return MathOperationResult(
            operation="percentage_operations",
            result="No numbers provided",
            explanation="Cannot perform percentage operations on empty list",
            details="Please provide a list of numbers"
        )
    
    # Validate operation type
    allowed_operations = ["percentage_of_total", "percentage_change"]
    if operation_type not in allowed_operations:
        return MathOperationResult(
            operation="percentage_operations",
            result=f"Unsupported operation type: {operation_type}",
            explanation=f"Invalid percentage operation type: {operation_type}",
            details=f"Supported types: {', '.join(allowed_operations)}"
        )
    
    result = None
    explanation = ""
    details = None
    
    if operation_type == "percentage_of_total":
        total = sum(numbers)
        if total == 0:
            percentages = [0] * len(numbers)
        else:
            percentages = [(num / total) * 100 for num in numbers]
        
        # Use string keys for Pydantic validation
        result = {str(num): percentage for num, percentage in zip(numbers, percentages)}
        explanation = f"Calculated percentage of total for each number"
        details = f"Total: {total}, Percentages: {[f'{p:.2f}%' for p in percentages]}"
        
    elif operation_type == "percentage_change":
        if len(numbers) < 2:
            return MathOperationResult(
                operation="percentage_operations",
                result="Need at least 2 numbers for percentage change",
                explanation="Insufficient numbers for percentage change calculation",
                details="Please provide at least 2 numbers"
            )
        
        changes = []
        for i in range(1, len(numbers)):
            if numbers[i-1] == 0:
                changes.append(float('inf') if numbers[i] > 0 else float('-inf') if numbers[i] < 0 else 0)
            else:
                change = ((numbers[i] - numbers[i-1]) / numbers[i-1]) * 100
                changes.append(change)
        
        result = changes
        explanation = f"Calculated percentage changes between consecutive numbers"
        details = f"Changes: {[f'{c:.2f}%' if abs(c) != float('inf') else 'infinite' for c in changes]}"
    
    return MathOperationResult(
        operation="percentage_operations",
        result=result,
        explanation=explanation,
        details=details
    )
