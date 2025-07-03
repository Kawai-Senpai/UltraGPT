from pydantic import BaseModel, Field
from typing import List, Union, Optional, Dict, Any

class RangeCheck(BaseModel):
    numbers: List[Union[int, float]]
    range_min: Union[int, float]
    range_max: Union[int, float]

class ProximityCheck(BaseModel):
    numbers: List[Union[int, float]]
    target: Union[int, float]
    tolerance: Optional[Union[int, float]] = 1.0

class StatisticalAnalysis(BaseModel):
    numbers: List[Union[int, float]]

class PrimeCheck(BaseModel):
    numbers: List[Union[int, float]]

class FactorAnalysis(BaseModel):
    numbers: List[Union[int, float]]

class SequenceAnalysis(BaseModel):
    numbers: List[Union[int, float]]

class PercentageOperation(BaseModel):
    numbers: List[Union[int, float]]
    operation_type: str = Field(default="percentage_of_total", description="Type of percentage operation: 'percentage_of_total' or 'percentage_change'")

class OutlierDetection(BaseModel):
    numbers: List[Union[int, float]]
    method: str = Field(default="iqr", description="Method for outlier detection: 'iqr' or 'zscore'")

class MathOperationsQuery(BaseModel):
    range_checks: List[RangeCheck] = Field(default_factory=list, description="Check if numbers lie between ranges")
    proximity_checks: List[ProximityCheck] = Field(default_factory=list, description="Check if numbers are close to target values within tolerance")
    statistical_analyses: List[StatisticalAnalysis] = Field(default_factory=list, description="Get mean, median, mode, standard deviation of numbers")
    prime_checks: List[PrimeCheck] = Field(default_factory=list, description="Check if numbers are prime")
    factor_analyses: List[FactorAnalysis] = Field(default_factory=list, description="Get factors and prime factorization")
    sequence_analyses: List[SequenceAnalysis] = Field(default_factory=list, description="Check if numbers form arithmetic or geometric sequences")
    percentage_operations: List[PercentageOperation] = Field(default_factory=list, description="Calculate percentages, ratios, proportions")
    outlier_detections: List[OutlierDetection] = Field(default_factory=list, description="Find numbers that are outside normal range")

class MathOperationResult(BaseModel):
    operation: str
    result: Union[bool, List[bool], float, List[float], str, Dict[str, Any]]
    explanation: str
    details: Optional[str] = None
