"""
Base model class for forecasting models
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


class BaseModel(ABC):
    """Base class for all forecasting models"""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, data: pd.Series) -> None:
        """Fit the model to training data"""
        pass

    @abstractmethod
    def forecast(self, horizon: int) -> np.ndarray:
        """Generate forecasts for the specified horizon"""
        pass

    def calculate_metrics(self, y_true: np.array, y_pred: np.array) -> Dict:
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    def fit_and_forecast(self, train_data: pd.Series, test_data: pd.Series,
                        forecast_horizon: int) -> Dict:
        """Fit model and generate forecasts with metrics"""
        # Fit the model
        self.fit(train_data)

        # Generate forecasts
        test_forecast = self.forecast(len(test_data))
        future_forecast = self.forecast(forecast_horizon)

        # Calculate metrics
        metrics = self.calculate_metrics(test_data, test_forecast)

        # Clip forecasts to reasonable bounds for CPU usage
        test_forecast = np.clip(test_forecast, 0, 100)
        future_forecast = np.clip(future_forecast, 0, 100)

        return {
            'metrics': metrics,
            'test_forecast': test_forecast,
            'future_forecast': future_forecast
        }