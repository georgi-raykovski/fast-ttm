"""
Seasonal Naive Forecasting Model
"""

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import mean_absolute_error
from .base_model import BaseModel
from utils.logging_config import get_logger

logger = get_logger(__name__)


class SeasonalNaiveModel(BaseModel):
    """Enhanced Seasonal Naive model with automatic pattern selection and multiple seasonalities"""

    def __init__(self):
        super().__init__("SeasonalNaive")
        self.best_season_length = 7
        self.seasonal_pattern = None
        self.multiple_patterns = {}
        self.pattern_weights = {}

    def fit(self, data: pd.Series) -> None:
        """Fit the model by finding the best seasonal pattern(s)"""
        # Determine seasonal lengths based on data size
        seasonal_lengths = self._get_seasonal_lengths(len(data))

        # Test different seasonal patterns to find the best one
        best_mae = float('inf')
        best_season = 7
        pattern_performance = {}

        logger.info(f"Testing seasonal patterns: {seasonal_lengths} on {len(data)} data points")

        # Reserve last 20% for validation (minimum 3 points)
        val_size = max(3, len(data) // 5)
        train_data = data[:-val_size]
        val_data = data[-val_size:]

        for s_len in seasonal_lengths:
            if len(train_data) >= s_len:
                # Create validation forecast
                last_season = train_data.iloc[-s_len:]
                repeats = len(val_data) // s_len + 1
                val_forecast = np.tile(last_season.values, repeats)[:len(val_data)]

                mae = mean_absolute_error(val_data, val_forecast)
                if mae < best_mae:
                    best_mae = mae
                    best_season = s_len

        self.best_season_length = best_season
        self.seasonal_pattern = data.iloc[-best_season:].values
        self.is_fitted = True

        logger.info(f"Best seasonal pattern: {self.best_season_length} days")

    def _get_seasonal_lengths(self, data_length: int) -> list:
        """Determine appropriate seasonal lengths based on data size"""
        if data_length >= 365:
            # With yearly data, test all major patterns
            return [7, 14, 30, 91, 182, 365]
        elif data_length >= 182:
            # With 6+ months, test up to quarterly
            return [7, 14, 30, 91]
        elif data_length >= 90:
            # With 3+ months, test up to monthly
            return [7, 14, 30]
        elif data_length >= 30:
            # With 1+ month, test weekly and bi-weekly
            return [7, 14]
        else:
            # Limited data, just weekly
            return [7]

    def forecast(self, horizon: int) -> np.ndarray:
        """Generate forecasts using the best seasonal pattern"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        # Generate forecast by repeating the seasonal pattern
        repeats = horizon // self.best_season_length + 1
        forecast = np.tile(self.seasonal_pattern, repeats)[:horizon]

        return forecast