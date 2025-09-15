"""
Shared metrics utilities for model evaluation
"""

import numpy as np
from typing import Dict, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from .constants import CPU_USAGE_MIN, CPU_USAGE_MAX


def calculate_metrics(y_true: Union[np.array, list], y_pred: Union[np.array, list]) -> Dict[str, float]:
    """
    Calculate standard evaluation metrics for forecasting models

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary containing MAE, RMSE, and MAPE metrics
    """
    # Convert to numpy arrays if needed
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true({len(y_true)}) != y_pred({len(y_pred)})")

    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics for empty arrays")

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape)
    }


def clip_cpu_forecasts(forecasts: Union[np.array, list],
                      min_val: float = CPU_USAGE_MIN,
                      max_val: float = CPU_USAGE_MAX) -> np.ndarray:
    """
    Clip forecasts to reasonable bounds for CPU usage

    Args:
        forecasts: Forecast values to clip
        min_val: Minimum allowed value (default: 0.0)
        max_val: Maximum allowed value (default: 100.0)

    Returns:
        Clipped forecast values as numpy array
    """
    if not isinstance(forecasts, np.ndarray):
        forecasts = np.array(forecasts)

    return np.clip(forecasts, min_val, max_val)


# Backward compatibility aliases
def calculate_forecast_metrics(y_true: Union[np.array, list],
                              y_pred: Union[np.array, list]) -> Dict[str, float]:
    """Alias for calculate_metrics for backward compatibility"""
    return calculate_metrics(y_true, y_pred)