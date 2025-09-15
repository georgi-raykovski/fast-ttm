"""
FastAPI application for TTM Forecasting System
"""

import os
from typing import List, Union, Optional
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import pandas as pd
from datetime import datetime
from decouple import config as env_config

from forecaster import DailyCPUForecaster
from utils.data_loader import DataLoader
import psutil
import time

# Track application start time for uptime calculation
app_start_time = time.time()


# Configuration
class Config:
    """Configuration for the API"""

    def __init__(self):
        # Configurable base URL for data fetching
        self.data_base_url = env_config('DATA_BASE_URL', default='http://localhost:8080')
        self.default_timeout = env_config('REQUEST_TIMEOUT', default=30, cast=int)
        self.auth_token = env_config('AUTH_TOKEN', default=None)

        # API Server configuration
        self.api_host = env_config('API_HOST', default='0.0.0.0')
        self.api_port = env_config('API_PORT', default=8000, cast=int)
        self.api_debug = env_config('API_DEBUG', default=False, cast=bool)

        # Forecasting defaults
        self.default_forecast_horizon = env_config('DEFAULT_FORECAST_HORIZON', default=30, cast=int)
        self.default_use_enhanced_ttm = env_config('DEFAULT_USE_ENHANCED_TTM', default=False, cast=bool)

        # Plotting configuration
        self.save_plots = env_config('SAVE_PLOTS', default=False, cast=bool)
        self.show_plots = env_config('SHOW_PLOTS', default=False, cast=bool)

    def get_data_url(self, instance_name: str, metric: str) -> str:
        """Build the data URL based on instance and metric"""
        return f"{self.data_base_url}/api/metrics?instance={instance_name}&metric={metric}"


config = Config()


# Pydantic models for request/response
class ForecastRequest(BaseModel):
    """Request model for forecasting"""
    instance_name: str = Field(..., description="Name of the instance to forecast")
    metric: Union[str, List[str]] = Field(..., description="Metric(s) to forecast: cpu, memory, or io")
    forecast_horizon: Optional[int] = Field(default=None, ge=1, le=365, description="Number of days to forecast")
    use_enhanced_ttm: Optional[bool] = Field(default=None, description="Whether to use enhanced TTM models")
    timeout: Optional[int] = Field(default=None, ge=1, le=300, description="Request timeout in seconds")


class TestForecastRequest(BaseModel):
    """Request model for test forecasting with local data"""
    data_file: str = Field(default="data.json", description="Local data file to use (data.json or data-long.json)")
    forecast_horizon: Optional[int] = Field(default=None, ge=1, le=365, description="Number of days to forecast")
    use_enhanced_ttm: Optional[bool] = Field(default=None, description="Whether to use enhanced TTM models")


class ModelPerformance(BaseModel):
    """Model performance metrics"""
    mae: float
    rmse: float
    mape: float


class PredictionPoint(BaseModel):
    """Single prediction point"""
    date: str
    value: float
    lower_ci: Optional[float] = None
    upper_ci: Optional[float] = None


class MetricForecast(BaseModel):
    """Individual metric forecast result"""
    metric: str
    predictions: List[PredictionPoint]
    model_name: str
    model_performance: ModelPerformance
    has_confidence_intervals: bool
    total_models_compared: int
    forecast_horizon: int
    error: Optional[str] = None


class ForecastResponse(BaseModel):
    """Response model for forecasting - consistent array structure"""
    instance_name: str
    forecasts: List[MetricForecast]
    total_forecasts: int
    generated_at: str


class TestForecastResponse(BaseModel):
    """Response model for test forecasting - consistent array structure"""
    data_file: str
    data_points: int
    forecasts: List[MetricForecast]
    total_forecasts: int
    generated_at: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    uptime_seconds: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    available_models: Optional[List[str]] = None


class AvailableModelsResponse(BaseModel):
    """Available models response"""
    models: List[str]
    enhanced_models: List[str]


class MetricsResponse(BaseModel):
    """System metrics response"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    uptime_seconds: float
    request_count: Optional[int] = None


# FastAPI app
app = FastAPI(
    title="TTM Forecasting API",
    description="Time Series Forecasting API for CPU, Memory, and IO metrics",
    version="1.0.0"
)


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

        # Get available models (simplified check)
        available_models = ["SeasonalNaive", "NaiveBayes"]

        # Check if TTM is available
        try:
            from models.ttm_model import TTM_AVAILABLE
            if TTM_AVAILABLE:
                available_models.append("TTM")
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

    except Exception as e:
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
        models=["SeasonalNaive", "NaiveBayes", "TTM"],
        enhanced_models=["TTMFineTuned", "TTMAugmented", "TTMEnsemble"]
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


@app.post("/forecast", response_model=ForecastResponse)
async def forecast_metric(request: ForecastRequest):
    """
    Main forecasting endpoint

    Fetches data for the specified instance and metric(s), then generates forecasts
    """
    try:
        # Convert single metric to list for consistent processing
        if isinstance(request.metric, str):
            metrics = [request.metric]
        else:
            metrics = request.metric

        if len(metrics) == 0:
            raise HTTPException(status_code=400, detail="At least one metric must be specified")

        # Validate all metrics
        valid_metrics = ["cpu", "memory", "io"]
        for metric in metrics:
            if metric not in valid_metrics:
                raise HTTPException(status_code=400, detail=f"Metric '{metric}' must be one of: {', '.join(valid_metrics)}")

        # Prepare headers for authentication if available
        headers = {}
        if config.auth_token:
            headers['Authorization'] = f'Bearer {config.auth_token}'

        # Use provided values or defaults from config
        timeout = request.timeout or config.default_timeout
        forecast_horizon = request.forecast_horizon or config.default_forecast_horizon
        use_enhanced_ttm = request.use_enhanced_ttm if request.use_enhanced_ttm is not None else config.default_use_enhanced_ttm

        forecasts = []

        # Process each metric independently
        for metric in metrics:
            try:
                # Build data URL for this metric
                data_url = config.get_data_url(request.instance_name, metric)

                # Load data from URL
                try:
                    series = DataLoader.load_data(
                        data_url,
                        timeout=timeout,
                        headers=headers if headers else None
                    )
                except Exception as e:
                    # Add error to forecasts and continue with other metrics
                    forecasts.append(MetricForecast(
                        metric=metric,
                        predictions=[],
                        model_name="Error",
                        model_performance=ModelPerformance(mae=0.0, rmse=0.0, mape=0.0),
                        has_confidence_intervals=False,
                        total_models_compared=0,
                        forecast_horizon=forecast_horizon,
                        error=f"Failed to load data from {data_url}: {str(e)}"
                    ))
                    continue

                # Initialize forecaster
                forecaster = DailyCPUForecaster(
                    series,
                    forecast_horizon=forecast_horizon,
                    use_enhanced_ttm=use_enhanced_ttm
                )

                # Configure plotting for API usage
                forecaster.configure_plotting(save_plots=config.save_plots, show_plots=config.show_plots)

                # Run forecasting
                forecaster.run_all_models()

                # Get best model predictions
                predictions_result = forecaster.get_best_model_predictions()

                if 'error' in predictions_result:
                    # Add error to forecasts and continue
                    forecasts.append(MetricForecast(
                        metric=metric,
                        predictions=[],
                        model_name="Error",
                        model_performance=ModelPerformance(mae=0.0, rmse=0.0, mape=0.0),
                        has_confidence_intervals=False,
                        total_models_compared=0,
                        forecast_horizon=forecast_horizon,
                        error=f"Forecasting failed: {predictions_result['error']}"
                    ))
                    continue

                # Format predictions
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

                # Add successful forecast
                forecasts.append(MetricForecast(
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
                ))

            except Exception as e:
                # Add error to forecasts and continue with other metrics
                forecasts.append(MetricForecast(
                    metric=metric,
                    predictions=[],
                    model_name="Error",
                    model_performance=ModelPerformance(mae=0.0, rmse=0.0, mape=0.0),
                    has_confidence_intervals=False,
                    total_models_compared=0,
                    forecast_horizon=forecast_horizon,
                    error=f"Internal error: {str(e)}"
                ))

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


@app.post("/forecast/test", response_model=TestForecastResponse)
async def test_forecast_with_local_data(request: TestForecastRequest):
    """
    Test forecasting endpoint using local data files

    Uses existing data.json or data-long.json files for testing without external dependencies
    """
    try:
        # Use provided values or defaults from config
        forecast_horizon = request.forecast_horizon or config.default_forecast_horizon
        use_enhanced_ttm = request.use_enhanced_ttm if request.use_enhanced_ttm is not None else config.default_use_enhanced_ttm

        # Validate data file choice
        if request.data_file not in ["data.json", "data-long.json"]:
            raise HTTPException(status_code=400, detail="data_file must be 'data.json' or 'data-long.json'")

        # Load data from local file
        try:
            series = DataLoader.load_data(f"./{request.data_file}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load local data from {request.data_file}: {str(e)}")

        # Initialize forecaster
        forecaster = DailyCPUForecaster(
            series,
            forecast_horizon=forecast_horizon,
            use_enhanced_ttm=use_enhanced_ttm
        )

        # Configure plotting for API usage
        forecaster.configure_plotting(save_plots=config.save_plots, show_plots=config.show_plots)

        # Run forecasting
        forecaster.run_all_models()

        # Get best model predictions
        predictions_result = forecaster.get_best_model_predictions()

        if 'error' in predictions_result:
            raise HTTPException(status_code=500, detail=f"Forecasting failed: {predictions_result['error']}")

        # Format response
        metadata = predictions_result.get('metadata', {})
        predictions = []

        for pred in predictions_result['predictions']:
            # Parse prediction format - assuming it's a string like "2024-01-01: 45.2 [40.1, 50.3]"
            # You may need to adjust this based on actual format
            prediction_point = PredictionPoint(
                date=pred.get('date', ''),
                value=pred.get('value', 0.0),
                lower_ci=pred.get('lower_bound'),
                upper_ci=pred.get('upper_bound')
            )
            predictions.append(prediction_point)

        model_perf = metadata.get('model_performance', {})

        # Create test metric forecast (using 'test' as metric name)
        metric_forecast = MetricForecast(
            metric="test",
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

        return TestForecastResponse(
            data_file=request.data_file,
            data_points=len(series),
            forecasts=[metric_forecast],
            total_forecasts=1,
            generated_at=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    if config.api_debug:
        # Use import string for reload functionality
        uvicorn.run("app:app", host=config.api_host, port=config.api_port, reload=True)
    else:
        # Use app object for production (no reload)
        uvicorn.run(app, host=config.api_host, port=config.api_port, reload=False)
