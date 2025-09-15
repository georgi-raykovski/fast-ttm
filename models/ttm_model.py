"""
TTM (Tiny Time Mixer) Zero-Shot Forecasting Model
"""

import numpy as np
import pandas as pd
import torch
import warnings
from typing import Dict
from .base_model import BaseModel
from utils.logging_config import get_logger
from utils.constants import TTM_DEFAULT_CONTEXT_LENGTH
from utils.exceptions import TTMLibraryError, ForecastingError

logger = get_logger(__name__)

try:
    from tsfm_public import TinyTimeMixerForPrediction
    TTM_AVAILABLE = True
except ImportError:
    TTM_AVAILABLE = False


class TTMModel(BaseModel):
    """TTM zero-shot forecasting model"""

    def __init__(self):
        super().__init__("TTM_ZeroShot")
        self.model = None
        self.context_length = TTM_DEFAULT_CONTEXT_LENGTH
        self.train_data = None

        if not TTM_AVAILABLE:
            raise TTMLibraryError()

    def fit(self, data: pd.Series) -> None:
        """Fit the model by loading pre-trained TTM and preparing data"""
        try:
            # Load pre-trained TTM model
            model_name = "ibm-granite/granite-timeseries-ttm-r1"
            self.model = TinyTimeMixerForPrediction.from_pretrained(model_name)

            # Store training data for forecasting
            self.train_data = data.values
            self.is_fitted = True

            logger.info(f"TTM model loaded: {model_name}")

        except Exception as e:
            logger.error(f"Failed to load TTM model: {e}")
            raise

    def _prepare_input_data(self) -> np.ndarray:
        """Prepare input data with proper context length"""
        if len(self.train_data) < self.context_length:
            # Simple repetition to reach required length
            repeats_needed = self.context_length // len(self.train_data) + 1
            padded_data = np.tile(self.train_data, repeats_needed)[-self.context_length:]
        else:
            padded_data = self.train_data[-self.context_length:]

        # Reshape for TTM (batch_size, context_length, num_features)
        return padded_data.reshape(1, self.context_length, 1)

    def forecast(self, horizon: int) -> np.ndarray:
        """Generate forecasts using TTM model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        try:
            # Prepare input data
            input_data = self._prepare_input_data()

            # Generate forecast
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                input_tensor = torch.tensor(input_data, dtype=torch.float32)
                with torch.no_grad():
                    forecast_output = self.model(input_tensor)

            # Extract predictions (TTM outputs 96 predictions)
            predictions = forecast_output.prediction_outputs.squeeze().numpy()

            # Handle cases where we need more or fewer predictions
            if len(predictions) >= horizon:
                forecast = predictions[:horizon]
            else:
                # If TTM predicts fewer points, repeat last prediction
                forecast = np.concatenate([
                    predictions,
                    np.full(horizon - len(predictions), predictions[-1])
                ])

            logger.info(f"TTM used context length: {self.context_length}, predicted {len(predictions)} points")
            return forecast

        except Exception as e:
            logger.error(f"TTM forecasting failed: {e}")
            raise ForecastingError("TTM_ZeroShot", str(e))
            # Fallback to naive forecast
            return np.full(horizon, self.train_data[-1])