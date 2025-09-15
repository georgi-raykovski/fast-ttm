"""
Pydantic schemas for API request/response models.

This module contains all the data validation and serialization models
used by the FastAPI application, separated from the main app logic.
"""

from typing import List, Union, Optional
from pydantic import BaseModel, Field


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
    description: str = "List of available forecasting models"
    total_models: int = Field(..., description="Total number of available models")

    def __init__(self, **data):
        if 'total_models' not in data and 'models' in data:
            data['total_models'] = len(data['models'])
        super().__init__(**data)


class MetricsResponse(BaseModel):
    """System metrics response"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    uptime_seconds: float
    request_count: Optional[int] = None