"""
FastAPI application for TTM Forecasting System
"""

from typing import List, Union, Dict, Any
from fastapi import FastAPI, HTTPException
import pandas as pd
from datetime import datetime

from forecaster import ExponentialSmoothingForecaster
from utils.data_loader import DataLoader
from utils.error_handlers import get_error_handlers
from utils.forecast_helpers import (
    create_successful_forecast_response,
    validate_metrics_list, extract_request_config, ForecastErrorHandler
)
from schemas import (
    ForecastRequest, ForecastResponse,
    HealthResponse, AvailableModelsResponse, MetricsResponse, MetricForecast
)
import psutil
import time
from settings import settings

# Track application start time for uptime calculation
app_start_time = time.time()


# FastAPI app
app = FastAPI(
    title="Optimized Exponential Smoothing Forecasting API",
    description="High-performance Time Series Forecasting API using Enhanced Exponential Smoothing with parallel processing, smart caching, and batch optimization",
    version="2.0.0"
)

# Register error handlers
error_handlers = get_error_handlers()
for exception_class, handler in error_handlers.items():
    app.add_exception_handler(exception_class, handler)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint with system metrics"""
    try:
        # Calculate uptime
        uptime = time.time() - app_start_time

        # Get memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024

        # Get available models (focused on exponential smoothing)
        available_models = ["ExponentialSmoothing"]

        # Check if other models are available
        try:
            from models.exponential_smoothing_model import EXPONENTIAL_SMOOTHING_AVAILABLE
            if not EXPONENTIAL_SMOOTHING_AVAILABLE:
                available_models = ["Fallback"]
        except:
            pass

        return HealthResponse(
            status="healthy",
            version="1.0.0",
            timestamp=datetime.now().isoformat(),
            uptime_seconds=uptime,
            memory_usage_mb=memory_usage_mb,
            available_models=available_models
        )

    except Exception:
        # If health check itself fails, return minimal response
        return HealthResponse(
            status="degraded",
            version="1.0.0",
            timestamp=datetime.now().isoformat()
        )


@app.get("/models", response_model=AvailableModelsResponse)
async def get_available_models():
    """Get list of available forecasting models"""
    return AvailableModelsResponse(
        models=["ExponentialSmoothing"],
        enhanced_models=["ExponentialSmoothing_Enhanced", "ExponentialSmoothing_Parallel", "ExponentialSmoothing_Cached"]
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_system_metrics():
    """Get detailed system metrics for monitoring"""
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)

    # Memory usage
    memory = psutil.virtual_memory()
    memory_used_mb = (memory.total - memory.available) / 1024 / 1024
    memory_available_mb = memory.available / 1024 / 1024

    # Disk usage
    disk = psutil.disk_usage('/')
    disk_usage_percent = (disk.used / disk.total) * 100

    # Uptime
    uptime = time.time() - app_start_time

    return MetricsResponse(
        cpu_percent=cpu_percent,
        memory_percent=memory.percent,
        memory_used_mb=memory_used_mb,
        memory_available_mb=memory_available_mb,
        disk_usage_percent=disk_usage_percent,
        uptime_seconds=uptime
    )


def _validate_and_normalize_metrics(metrics: Union[str, List[str]]) -> List[str]:
    """Validate and normalize metrics - wrapper for HTTPException handling"""
    try:
        return validate_metrics_list(metrics)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _load_data_for_metric(instance_name: str, metric: str, config: Dict[str, Any]) -> pd.Series:
    """
    Load data for a specific metric.

    Args:
        instance_name: Instance identifier
        metric: Metric name
        config: Configuration dictionary

    Returns:
        Time series data

    Raises:
        Exception: If data loading fails
    """
    data_url = settings.get_data_url(instance_name, metric)
    return DataLoader.load_data(
        data_url,
        timeout=config['timeout'],
        headers=config['headers']
    )


def _run_forecasting_pipeline(series: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the optimized exponential smoothing forecasting pipeline for a time series.

    Args:
        series: Time series data
        config: Configuration dictionary

    Returns:
        Forecasting results dictionary
    """
    # Calculate test size from the series length (30% default)
    test_size = max(14, int(len(series) * 0.3))  # Minimum 14 days test

    # Initialize exponential smoothing forecaster with optimizations
    forecaster = ExponentialSmoothingForecaster(
        series,
        forecast_horizon=config['forecast_horizon'],
        test_size=test_size
    )

    # Configure plotting for API usage
    forecaster.configure_plotting(save_plots=settings.SAVE_PLOTS, show_plots=settings.SHOW_PLOTS)

    # Run optimized forecasting
    forecaster.run_forecast()

    # Generate plots if enabled
    if settings.SAVE_PLOTS or settings.SHOW_PLOTS:
        forecaster.plot_results()
        forecaster.plot_model_comparison()
        forecaster.create_interactive_plot()

    # Get model predictions
    return forecaster.get_model_predictions()




def _process_single_metric(instance_name: str, metric: str, config: Dict[str, Any]) -> MetricForecast:
    """
    Process forecasting for a single metric using centralized error handling.

    Args:
        instance_name: Instance identifier
        metric: Metric name
        config: Configuration dictionary

    Returns:
        MetricForecast for the processed metric
    """
    forecast_horizon = config['forecast_horizon']

    try:
        # Load data from URL
        try:
            series = _load_data_for_metric(instance_name, metric, config)
        except Exception as e:
            data_url = settings.get_data_url(instance_name, metric)
            return ForecastErrorHandler.handle_data_loading_error(
                metric, data_url, e, forecast_horizon
            )

        # Run forecasting pipeline
        predictions_result = _run_forecasting_pipeline(series, config)

        if 'error' in predictions_result:
            return ForecastErrorHandler.handle_prediction_error(
                metric, predictions_result, forecast_horizon
            )

        # Create successful forecast result
        return create_successful_forecast_response(metric, predictions_result, forecast_horizon)

    except Exception as e:
        return ForecastErrorHandler.handle_internal_error(metric, e, forecast_horizon)


@app.post("/forecast", response_model=ForecastResponse)
async def forecast_metric(request: ForecastRequest):
    """
    Main forecasting endpoint - now clean and focused.

    Fetches data for the specified instance and metric(s), then generates forecasts
    """
    try:
        # Validate and normalize metrics
        metrics = _validate_and_normalize_metrics(request.metric)

        # Prepare configuration
        config = extract_request_config(request, settings)

        # Process each metric independently
        forecasts = []
        for metric in metrics:
            forecast_result = _process_single_metric(request.instance_name, metric, config)
            forecasts.append(forecast_result)

        return ForecastResponse(
            instance_name=request.instance_name,
            forecasts=forecasts,
            total_forecasts=len(forecasts),
            generated_at=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/forecast/batch")
async def forecast_batch_metrics(requests: List[ForecastRequest]):
    """
    Batch forecasting endpoint for multiple instances/metrics
    """
    results = []

    for req in requests:
        try:
            result = await forecast_metric(req)
            results.append({"success": True, "data": result})
        except HTTPException as e:
            results.append({
                "success": False,
                "error": e.detail,
                "instance_name": req.instance_name,
                "metric": req.metric
            })

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    if settings.API_DEBUG:
        # Use import string for reload functionality
        uvicorn.run("app:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
    else:
        # Use app object for production (no reload)
        uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT, reload=False)
