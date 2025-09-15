"""
Helper utilities for forecast operations and common patterns
"""

from typing import List, Dict, Any, Union
import pandas as pd
from datetime import datetime


def create_error_forecast_response(metric: str, forecast_horizon: int, error_message: str,
                                 model_name: str = "Error") -> 'MetricForecast':
    """
    Create a standardized MetricForecast object for error cases.

    This eliminates the repeated pattern of creating error forecasts
    throughout the application.

    Args:
        metric: Metric name
        forecast_horizon: Forecast horizon value
        error_message: Error description
        model_name: Model name for the error (defaults to "Error")

    Returns:
        MetricForecast with error information
    """
    # Import here to avoid circular imports
    from schemas import MetricForecast, ModelPerformance

    return MetricForecast(
        metric=metric,
        predictions=[],
        model_name=model_name,
        model_performance=ModelPerformance(mae=0.0, rmse=0.0, mape=0.0),
        has_confidence_intervals=False,
        total_models_compared=0,
        forecast_horizon=forecast_horizon,
        error=error_message
    )


def create_successful_forecast_response(metric: str, predictions_result: Dict[str, Any],
                                      forecast_horizon: int) -> 'MetricForecast':
    """
    Create a standardized MetricForecast object for successful predictions.

    This eliminates code duplication in forecast response creation.

    Args:
        metric: Metric name
        predictions_result: Forecasting results dictionary
        forecast_horizon: Forecast horizon value

    Returns:
        MetricForecast with successful predictions
    """
    # Import here to avoid circular imports
    from schemas import MetricForecast, ModelPerformance, PredictionPoint

    metadata = predictions_result.get('metadata', {})
    predictions = []

    for pred in predictions_result['predictions']:
        prediction_point = PredictionPoint(
            date=pred.get('date', ''),
            value=pred.get('value', 0.0),
            lower_ci=pred.get('lower_bound'),
            upper_ci=pred.get('upper_bound')
        )
        predictions.append(prediction_point)

    model_perf = metadata.get('model_performance', {})

    return MetricForecast(
        metric=metric,
        predictions=predictions,
        model_name=metadata.get('model_name', 'Unknown'),
        model_performance=ModelPerformance(
            mae=model_perf.get('mae', 0.0),
            rmse=model_perf.get('rmse', 0.0),
            mape=model_perf.get('mape', 0.0)
        ),
        has_confidence_intervals=metadata.get('has_confidence_intervals', False),
        total_models_compared=metadata.get('total_models_compared', 0),
        forecast_horizon=forecast_horizon
    )


def validate_metrics_list(metrics: Union[str, List[str]],
                         valid_metrics: List[str] = None) -> List[str]:
    """
    Validate and normalize metrics to a list format.

    Common pattern used across multiple endpoints.

    Args:
        metrics: Single metric string or list of metrics
        valid_metrics: List of allowed metrics (defaults to standard set)

    Returns:
        List of validated metrics

    Raises:
        ValueError: If metrics are invalid
    """
    if valid_metrics is None:
        valid_metrics = ["cpu", "memory", "io"]

    # Convert single metric to list for consistent processing
    if isinstance(metrics, str):
        metric_list = [metrics]
    else:
        metric_list = metrics

    if len(metric_list) == 0:
        raise ValueError("At least one metric must be specified")

    # Validate all metrics
    for metric in metric_list:
        if metric not in valid_metrics:
            raise ValueError(f"Metric '{metric}' must be one of: {', '.join(valid_metrics)}")

    return metric_list


def safe_execute_with_fallback(operation_name: str, primary_func, fallback_func=None,
                              *args, **kwargs) -> Any:
    """
    Safely execute an operation with optional fallback.

    Common pattern for operations that might fail but have graceful degradation.

    Args:
        operation_name: Name of the operation for logging
        primary_func: Primary function to execute
        fallback_func: Optional fallback function if primary fails
        *args, **kwargs: Arguments to pass to functions

    Returns:
        Result from primary function or fallback

    Raises:
        Exception: If both primary and fallback fail, or if no fallback provided
    """
    from utils.logging_config import get_logger

    logger = get_logger(__name__)

    try:
        logger.debug(f"Executing {operation_name}")
        return primary_func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"{operation_name} failed: {e}")

        if fallback_func is not None:
            try:
                logger.info(f"Attempting fallback for {operation_name}")
                return fallback_func(*args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback for {operation_name} also failed: {fallback_error}")
                raise fallback_error
        else:
            raise e


def extract_request_config(request: 'ForecastRequest', settings: Any) -> Dict[str, Any]:
    """
    Extract and prepare configuration from request and settings.

    Common pattern used in multiple forecast endpoints.

    Args:
        request: Forecast request object
        settings: Application settings

    Returns:
        Configuration dictionary
    """
    # Prepare headers for authentication if available
    headers = {}
    if hasattr(settings, 'AUTH_TOKEN') and settings.AUTH_TOKEN:
        headers['Authorization'] = f'Bearer {settings.AUTH_TOKEN}'

    # Use provided values or defaults from config
    return {
        'headers': headers if headers else None,
        'timeout': getattr(request, 'timeout', None) or getattr(settings, 'REQUEST_TIMEOUT', 30),
        'forecast_horizon': getattr(request, 'forecast_horizon', None) or getattr(settings, 'DEFAULT_FORECAST_HORIZON', 30),
        'use_enhanced_ttm': getattr(request, 'use_enhanced_ttm', None) if getattr(request, 'use_enhanced_ttm', None) is not None else getattr(settings, 'DEFAULT_USE_ENHANCED_TTM', False)
    }


class ForecastErrorHandler:
    """
    Centralized error handling for forecast operations.

    Provides consistent error handling patterns across the application.
    """

    @staticmethod
    def handle_data_loading_error(metric: str, data_url: str, error: Exception,
                                 forecast_horizon: int) -> 'MetricForecast':
        """Handle errors during data loading"""
        error_message = f"Failed to load data from {data_url}: {str(error)}"
        return create_error_forecast_response(metric, forecast_horizon, error_message)

    @staticmethod
    def handle_forecasting_error(metric: str, error: Exception,
                                forecast_horizon: int) -> 'MetricForecast':
        """Handle errors during forecasting process"""
        error_message = f"Forecasting failed: {str(error)}"
        return create_error_forecast_response(metric, forecast_horizon, error_message)

    @staticmethod
    def handle_internal_error(metric: str, error: Exception,
                             forecast_horizon: int) -> 'MetricForecast':
        """Handle internal/unexpected errors"""
        error_message = f"Internal error: {str(error)}"
        return create_error_forecast_response(metric, forecast_horizon, error_message)

    @staticmethod
    def handle_prediction_error(metric: str, prediction_result: Dict[str, Any],
                               forecast_horizon: int) -> 'MetricForecast':
        """Handle errors in prediction results"""
        error_message = f"Forecasting failed: {prediction_result.get('error', 'Unknown error')}"
        return create_error_forecast_response(metric, forecast_horizon, error_message)


def log_operation_metrics(operation_name: str, start_time: float,
                         success: bool, **metadata) -> None:
    """
    Log standardized operation metrics.

    Common pattern for tracking operation performance and success rates.

    Args:
        operation_name: Name of the operation
        start_time: Operation start time (from time.time())
        success: Whether operation succeeded
        **metadata: Additional metadata to log
    """
    import time
    from utils.logging_config import get_logger

    logger = get_logger(__name__)

    duration = time.time() - start_time
    status = "SUCCESS" if success else "FAILURE"

    log_data = {
        'operation': operation_name,
        'status': status,
        'duration_seconds': round(duration, 3),
        **metadata
    }

    if success:
        logger.info(f"Operation completed", extra=log_data)
    else:
        logger.warning(f"Operation failed", extra=log_data)