from .prompts import make_math_operations_query
from .schemas import MathOperationsQuery, MathOperationResult
from .decision import query_finder
from .core import (
    check_range, find_outliers, check_proximity, statistical_summary,
    prime_check, factor_analysis, check_sequence, percentage_operations
)

#? Required ------------------------------------------------------------------
_info = "This allows you to perform advanced mathematical operations like range checking, statistical analysis, prime checking, sequence analysis, and more."

def extract_system_message(history):
    for message in history:
        if message["role"] == "system" or message["role"] == "developer":
            return message["content"]
    return ""

def perform_math_operations(message, client, config):
    """Perform mathematical operations using structured parsing"""
    try:
        # Get all operations from the message
        operations = query_finder(message, client, config)
        
        if not operations or not isinstance(operations, dict):
            return "No mathematical operations detected in the request."
        
        formatted_results = []
        
        # Process range checks
        range_checks = operations.get("range_checks") or []
        for i, range_check in enumerate(range_checks):
            if not isinstance(range_check, dict):
                continue
            numbers = range_check.get("numbers", [])
            range_min = range_check.get("range_min")
            range_max = range_check.get("range_max")
            if numbers and range_min is not None and range_max is not None:
                result = check_range(numbers, range_min, range_max)
                formatted_results.append(f"Range Check {i+1}: {result.explanation}")
                formatted_results.append(f"  Result: {result.result}")
                if result.details:
                    formatted_results.append(f"  Details: {result.details}")
        
        # Process proximity checks
        proximity_checks = operations.get("proximity_checks") or []
        for i, prox_check in enumerate(proximity_checks):
            if not isinstance(prox_check, dict):
                continue
            numbers = prox_check.get("numbers", [])
            target = prox_check.get("target")
            tolerance = prox_check.get("tolerance", 1.0)
            if numbers and target is not None:
                result = check_proximity(numbers, target, tolerance)
                formatted_results.append(f"Proximity Check {i+1}: {result.explanation}")
                formatted_results.append(f"  Result: {result.result}")
                if result.details:
                    formatted_results.append(f"  Details: {result.details}")
        
        # Process statistical analyses
        statistical_analyses = operations.get("statistical_analyses") or []
        for i, stat_analysis in enumerate(statistical_analyses):
            if not isinstance(stat_analysis, dict):
                continue
            numbers = stat_analysis.get("numbers", [])
            if numbers:
                result = statistical_summary(numbers)
                formatted_results.append(f"Statistical Analysis {i+1}: {result.explanation}")
                if isinstance(result.result, dict):
                    for key, value in result.result.items():
                        formatted_results.append(f"  {key}: {value}")
                else:
                    formatted_results.append(f"  Result: {result.result}")
        
        # Process prime checks
        prime_checks = operations.get("prime_checks") or []
        for i, prime_check_req in enumerate(prime_checks):
            if not isinstance(prime_check_req, dict):
                continue
            numbers = prime_check_req.get("numbers", [])
            if numbers:
                result = prime_check(numbers)
                formatted_results.append(f"Prime Check {i+1}: {result.explanation}")
                if isinstance(result.result, dict):
                    for num, is_prime in result.result.items():
                        formatted_results.append(f"  {num}: {'Prime' if is_prime else 'Not Prime'}")
                else:
                    formatted_results.append(f"  Result: {result.result}")
                if result.details:
                    formatted_results.append(f"  Details: {result.details}")
        
        # Process factor analyses
        factor_analyses = operations.get("factor_analyses") or []
        for i, factor_analysis_req in enumerate(factor_analyses):
            if not isinstance(factor_analysis_req, dict):
                continue
            numbers = factor_analysis_req.get("numbers", [])
            if numbers:
                result = factor_analysis(numbers)
                formatted_results.append(f"Factor Analysis {i+1}: {result.explanation}")
                if isinstance(result.result, dict):
                    for num, analysis in result.result.items():
                        formatted_results.append(f"  {num}:")
                        formatted_results.append(f"    Factors: {analysis.get('factors', [])}")
                        formatted_results.append(f"    Prime Factorization: {analysis.get('prime_factorization', [])}")
                else:
                    formatted_results.append(f"  Result: {result.result}")
        
        # Process sequence analyses
        sequence_analyses = operations.get("sequence_analyses") or []
        for i, seq_analysis in enumerate(sequence_analyses):
            if not isinstance(seq_analysis, dict):
                continue
            numbers = seq_analysis.get("numbers", [])
            if numbers:
                result = check_sequence(numbers)
                formatted_results.append(f"Sequence Analysis {i+1}: {result.explanation}")
                if isinstance(result.result, dict):
                    formatted_results.append(f"  Arithmetic: {result.result.get('is_arithmetic', False)}")
                    formatted_results.append(f"  Geometric: {result.result.get('is_geometric', False)}")
                    if result.result.get('arithmetic_difference'):
                        formatted_results.append(f"  Arithmetic Difference: {result.result['arithmetic_difference']}")
                    if result.result.get('geometric_ratio'):
                        formatted_results.append(f"  Geometric Ratio: {result.result['geometric_ratio']}")
                else:
                    formatted_results.append(f"  Result: {result.result}")
        
        # Process percentage operations
        percentage_operations_list = operations.get("percentage_operations") or []
        for i, perc_op in enumerate(percentage_operations_list):
            if not isinstance(perc_op, dict):
                continue
            numbers = perc_op.get("numbers", [])
            operation_type = perc_op.get("operation_type", "percentage_of_total")
            if numbers:
                result = percentage_operations(numbers, operation_type)
                formatted_results.append(f"Percentage Operation {i+1}: {result.explanation}")
                if isinstance(result.result, dict):
                    for num, percentage in result.result.items():
                        formatted_results.append(f"  {num}: {percentage:.2f}%")
                elif isinstance(result.result, list):
                    for j, perc in enumerate(result.result):
                        formatted_results.append(f"  Change {j+1}: {perc:.2f}%")
                else:
                    formatted_results.append(f"  Result: {result.result}")
                if result.details:
                    formatted_results.append(f"  Details: {result.details}")
        
        # Process outlier detections
        outlier_detections = operations.get("outlier_detections") or []
        for i, outlier_detection in enumerate(outlier_detections):
            if not isinstance(outlier_detection, dict):
                continue
            numbers = outlier_detection.get("numbers", [])
            method = outlier_detection.get("method", "iqr")
            if numbers:
                result = find_outliers(numbers, method)
                formatted_results.append(f"Outlier Detection {i+1}: {result.explanation}")
                formatted_results.append(f"  Outliers Found: {result.result}")
                if result.details:
                    formatted_results.append(f"  Details: {result.details}")
        
        return "\n".join(formatted_results) if formatted_results else "No mathematical operations detected in the request."
        
    except Exception as e:
        return f"Error performing mathematical operations: {str(e)}"
        proximity_checks = operations.get("proximity_checks", [])
        for i, prox_check in enumerate(proximity_checks):
            numbers = prox_check.get("numbers", [])
            target = prox_check.get("target")
            tolerance = prox_check.get("tolerance", 1.0)
            if numbers and target is not None:
                result = check_proximity(numbers, target, tolerance)
                formatted_results.append(f"Proximity Check {i+1}: {result.explanation}")
                formatted_results.append(f"  Result: {result.result}")
                if result.details:
                    formatted_results.append(f"  Details: {result.details}")
        
        # Process statistical analyses
        statistical_analyses = operations.get("statistical_analyses", [])
        for i, stat_analysis in enumerate(statistical_analyses):
            numbers = stat_analysis.get("numbers", [])
            if numbers:
                result = statistical_summary(numbers)
                formatted_results.append(f"Statistical Analysis {i+1}: {result.explanation}")
                if isinstance(result.result, dict):
                    for key, value in result.result.items():
                        formatted_results.append(f"  {key}: {value}")
                else:
                    formatted_results.append(f"  Result: {result.result}")
        
        # Process prime checks
        prime_checks = operations.get("prime_checks", [])
        for i, prime_check_req in enumerate(prime_checks):
            numbers = prime_check_req.get("numbers", [])
            if numbers:
                result = prime_check(numbers)
                formatted_results.append(f"Prime Check {i+1}: {result.explanation}")
                if isinstance(result.result, dict):
                    for num, is_prime in result.result.items():
                        formatted_results.append(f"  {num}: {'Prime' if is_prime else 'Not Prime'}")
                else:
                    formatted_results.append(f"  Result: {result.result}")
                if result.details:
                    formatted_results.append(f"  Details: {result.details}")
        
        # Process factor analyses
        factor_analyses = operations.get("factor_analyses", [])
        for i, factor_analysis_req in enumerate(factor_analyses):
            numbers = factor_analysis_req.get("numbers", [])
            if numbers:
                result = factor_analysis(numbers)
                formatted_results.append(f"Factor Analysis {i+1}: {result.explanation}")
                if isinstance(result.result, dict):
                    for num, analysis in result.result.items():
                        formatted_results.append(f"  {num}:")
                        formatted_results.append(f"    Factors: {analysis.get('factors', [])}")
                        formatted_results.append(f"    Prime Factorization: {analysis.get('prime_factorization', [])}")
                else:
                    formatted_results.append(f"  Result: {result.result}")
        
        # Process sequence analyses
        sequence_analyses = operations.get("sequence_analyses", [])
        for i, seq_analysis in enumerate(sequence_analyses):
            numbers = seq_analysis.get("numbers", [])
            if numbers:
                result = check_sequence(numbers)
                formatted_results.append(f"Sequence Analysis {i+1}: {result.explanation}")
                if isinstance(result.result, dict):
                    formatted_results.append(f"  Arithmetic: {result.result.get('is_arithmetic', False)}")
                    formatted_results.append(f"  Geometric: {result.result.get('is_geometric', False)}")
                    if result.result.get('arithmetic_difference'):
                        formatted_results.append(f"  Arithmetic Difference: {result.result['arithmetic_difference']}")
                    if result.result.get('geometric_ratio'):
                        formatted_results.append(f"  Geometric Ratio: {result.result['geometric_ratio']}")
                else:
                    formatted_results.append(f"  Result: {result.result}")
        
        # Process percentage operations
        percentage_operations_list = operations.get("percentage_operations", [])
        for i, perc_op in enumerate(percentage_operations_list):
            numbers = perc_op.get("numbers", [])
            operation_type = perc_op.get("operation_type", "percentage_of_total")
            if numbers:
                result = percentage_operations(numbers, operation_type)
                formatted_results.append(f"Percentage Operation {i+1}: {result.explanation}")
                if isinstance(result.result, dict):
                    for num, percentage in result.result.items():
                        formatted_results.append(f"  {num}: {percentage:.2f}%")
                elif isinstance(result.result, list):
                    for j, perc in enumerate(result.result):
                        formatted_results.append(f"  Change {j+1}: {perc:.2f}%")
                else:
                    formatted_results.append(f"  Result: {result.result}")
                if result.details:
                    formatted_results.append(f"  Details: {result.details}")
        
        # Process outlier detections
        outlier_detections = operations.get("outlier_detections", [])
        for i, outlier_detection in enumerate(outlier_detections):
            numbers = outlier_detection.get("numbers", [])
            method = outlier_detection.get("method", "iqr")
            if numbers:
                result = find_outliers(numbers, method)
                formatted_results.append(f"Outlier Detection {i+1}: {result.explanation}")
                formatted_results.append(f"  Outliers Found: {result.result}")
                if result.details:
                    formatted_results.append(f"  Details: {result.details}")
        
        return "\n".join(formatted_results) if formatted_results else "No mathematical operations detected in the request."
        
    except Exception as e:
        return f"Error performing mathematical operations: {str(e)}"

def _execute(message, history, client, config):
    """Main function to execute the math operations tool"""
    system_message = extract_system_message(history)
    if system_message == message:
        return perform_math_operations(message, client, config)
    else:
        full_message = message + "\n" + system_message
        return perform_math_operations(full_message, client, config)
