#* Math operations ---------------------------------------------------------------
def execute_tool(parameters):
    """Standard entry point for math operations tool - takes AI-provided parameters directly"""
    try:
        results = []
        
        # Handle prime checks
        if 'prime_checks' in parameters and parameters['prime_checks']:
            for prime_check in parameters['prime_checks']:
                numbers = prime_check.get('numbers', [])
                for num in numbers:
                    is_prime = check_prime(num)
                    results.append(f"{num} is {'prime' if is_prime else 'not prime'}")
        
        # Handle factor analyses  
        if 'factor_analyses' in parameters and parameters['factor_analyses']:
            for factor_analysis in parameters['factor_analyses']:
                numbers = factor_analysis.get('numbers', [])
                for num in numbers:
                    factors = get_factors(int(num))
                    results.append(f"Factors of {num}: {factors}")
        
        # Handle statistical analyses
        if 'statistical_analyses' in parameters and parameters['statistical_analyses']:
            for stat_analysis in parameters['statistical_analyses']:
                numbers = stat_analysis.get('numbers', [])
                if numbers:
                    import statistics
                    mean_val = sum(numbers) / len(numbers)
                    median_val = statistics.median(numbers)
                    results.append(f"Statistics for {numbers}: mean={mean_val:.2f}, median={median_val}")
        
        # Handle range checks
        if 'range_checks' in parameters and parameters['range_checks']:
            for range_check in parameters['range_checks']:
                numbers = range_check.get('numbers', [])
                range_min = range_check.get('range_min')
                range_max = range_check.get('range_max')
                for num in numbers:
                    in_range = range_min <= num <= range_max
                    results.append(f"{num} is {'within' if in_range else 'outside'} range [{range_min}, {range_max}]")
        
        # Handle proximity checks
        if 'proximity_checks' in parameters and parameters['proximity_checks']:
            for prox_check in parameters['proximity_checks']:
                numbers = prox_check.get('numbers', [])
                target = prox_check.get('target')
                tolerance = prox_check.get('tolerance', 1.0)
                for num in numbers:
                    is_close = abs(num - target) <= tolerance
                    results.append(f"{num} is {'close to' if is_close else 'far from'} {target} (tolerance: {tolerance})")
        
        # Handle sequence analyses
        if 'sequence_analyses' in parameters and parameters['sequence_analyses']:
            for seq_analysis in parameters['sequence_analyses']:
                numbers = seq_analysis.get('numbers', [])
                if len(numbers) >= 3:
                    is_arithmetic = is_arithmetic_sequence(numbers)
                    is_geometric = is_geometric_sequence(numbers)
                    results.append(f"Sequence {numbers}: arithmetic={is_arithmetic}, geometric={is_geometric}")
        
        # Handle percentage operations
        if 'percentage_operations' in parameters and parameters['percentage_operations']:
            for percent_op in parameters['percentage_operations']:
                numbers = percent_op.get('numbers', [])
                op_type = percent_op.get('operation_type', 'percentage_of_total')
                if op_type == 'percentage_of_total' and numbers:
                    total = sum(numbers)
                    percentages = [(num/total)*100 for num in numbers]
                    results.append(f"Percentages of total for {numbers}: {[f'{p:.1f}%' for p in percentages]}")
        
        # Handle outlier detection
        if 'outlier_detections' in parameters and parameters['outlier_detections']:
            for outlier_det in parameters['outlier_detections']:
                numbers = outlier_det.get('numbers', [])
                method = outlier_det.get('method', 'iqr')
                outliers = detect_outliers(numbers, method)
                results.append(f"Outliers in {numbers} (method: {method}): {outliers}")
        
        if results:
            return "\n".join(results)
        else:
            return "No valid mathematical operations found in the parameters."
            
    except Exception as e:
        return f"Math operations tool error: {str(e)}"

def check_prime(n):
    """Check if a number is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def get_factors(n):
    """Get all factors of a number"""
    factors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
    return sorted(factors)

def is_arithmetic_sequence(numbers):
    """Check if numbers form an arithmetic sequence"""
    if len(numbers) < 3:
        return False
    diff = numbers[1] - numbers[0]
    for i in range(2, len(numbers)):
        if numbers[i] - numbers[i-1] != diff:
            return False
    return True

def is_geometric_sequence(numbers):
    """Check if numbers form a geometric sequence"""
    if len(numbers) < 3 or 0 in numbers:
        return False
    ratio = numbers[1] / numbers[0]
    for i in range(2, len(numbers)):
        if abs(numbers[i] / numbers[i-1] - ratio) > 1e-10:
            return False
    return True

def detect_outliers(numbers, method='iqr'):
    """Detect outliers using IQR or Z-score method"""
    if len(numbers) < 4:
        return []
    
    if method == 'iqr':
        sorted_nums = sorted(numbers)
        n = len(sorted_nums)
        q1 = sorted_nums[n//4]
        q3 = sorted_nums[3*n//4]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return [x for x in numbers if x < lower_bound or x > upper_bound]
    elif method == 'zscore':
        import statistics
        mean = statistics.mean(numbers)
        stdev = statistics.stdev(numbers)
        return [x for x in numbers if abs((x - mean) / stdev) > 2]
    else:
        return []

def math_operations(message, client, config, history=None):
    """Legacy function - now serves as fallback for direct calls"""
    return "Math operations tool is now using native AI tool calling. Please use the UltraGPT chat interface to access math functions."

def perform_math_operations(parameters):
    """Legacy function - redirects to standard entry point"""
    return execute_tool(parameters)
