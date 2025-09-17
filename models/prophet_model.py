"""
Prophet Model for Time Series Forecasting

Facebook's Prophet is a robust forecasting tool that automatically handles
seasonality, holidays, and trend changes. Excellent for daily data with
strong seasonal patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .base_model import BaseModel
from utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet model requires 'prophet'. Install with: pip install prophet")


class ProphetModel(BaseModel):
    """
    Facebook Prophet model for time series forecasting.

    Features:
    - Automatic seasonality detection (daily, weekly, yearly)
    - Holiday effects handling
    - Robust to missing data and outliers
    - Confidence intervals included
    - Trend changepoint detection
    """

    def __init__(self, yearly_seasonality: Optional[bool] = 'auto',
                 weekly_seasonality: Optional[bool] = 'auto',
                 daily_seasonality: Optional[bool] = False,
                 seasonality_mode: str = 'additive',
                 changepoint_prior_scale: float = 0.05):
        super().__init__("Prophet")
        self.model = None
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model_params = {}

    def fit(self, data: pd.Series) -> None:
        """Fit Prophet model"""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet model requires prophet. Install with: pip install prophet")

        try:
            logger.info(f"Fitting Prophet model on {len(data)} data points")

            # Prepare data in Prophet format
            df = pd.DataFrame({
                'ds': data.index,
                'y': data.values
            })

            # Configure Prophet model
            self.model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                seasonality_mode=self.seasonality_mode,
                changepoint_prior_scale=self.changepoint_prior_scale,
                interval_width=0.95,  # 95% confidence intervals
                mcmc_samples=0,  # Use MAP estimation (faster)
                uncertainty_samples=1000
            )

            # Add custom seasonalities if we have enough data
            if len(data) >= 365:
                # Add monthly seasonality
                self.model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

            if len(data) >= 90:
                # Add quarterly seasonality
                self.model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)

            # Suppress Prophet's verbose output
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
            logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

            # Fit the model
            self.model.fit(df)

            self.is_fitted = True

            # Store model information
            self.model_params = {
                'yearly_seasonality': self.model.yearly_seasonality,
                'weekly_seasonality': self.model.weekly_seasonality,
                'daily_seasonality': self.model.daily_seasonality,
                'seasonality_mode': self.seasonality_mode,
                'changepoint_prior_scale': self.changepoint_prior_scale,
                'n_changepoints': len(self.model.changepoints),
                'data_length': len(data)
            }

            logger.info(f"Prophet fitted successfully: {self.model_params}")

        except Exception as e:
            logger.error(f"Prophet model fitting failed: {e}")
            raise

    def forecast(self, horizon: int) -> np.ndarray:
        """Generate Prophet forecasts"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        try:
            # Create future dates dataframe
            future = self.model.make_future_dataframe(periods=horizon)

            # Generate forecast
            forecast = self.model.predict(future)

            # Extract the future predictions (last `horizon` points)
            future_forecast = forecast['yhat'].iloc[-horizon:].values

            # Store confidence intervals for later use
            self._last_confidence_intervals = {
                'lower': forecast['yhat_lower'].iloc[-horizon:].values,
                'upper': forecast['yhat_upper'].iloc[-horizon:].values
            }

            logger.info(f"Generated Prophet forecast for {horizon} periods")

            return future_forecast

        except Exception as e:
            logger.error(f"Prophet forecasting failed: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        info = super().get_model_info()

        if self.is_fitted:
            info.update({
                'model_type': 'Facebook Prophet',
                'parameters': self.model_params,
                'has_confidence_intervals': hasattr(self, '_last_confidence_intervals'),
                'seasonalities': self._get_seasonality_info()
            })

        return info

    def _get_seasonality_info(self) -> Dict[str, Any]:
        """Get information about detected seasonalities"""
        if not self.is_fitted:
            return {}

        seasonalities = {}
        for name, seasonality in self.model.seasonalities.items():
            seasonalities[name] = {
                'period': seasonality['period'],
                'fourier_order': seasonality['fourier_order'],
                'mode': seasonality['mode']
            }

        return seasonalities

    def get_confidence_intervals(self) -> Optional[Dict[str, np.ndarray]]:
        """Get the last generated confidence intervals"""
        return getattr(self, '_last_confidence_intervals', None)

    def get_components(self, data: pd.Series) -> Optional[pd.DataFrame]:
        """Get Prophet's component decomposition"""
        if not self.is_fitted:
            return None

        try:
            # Prepare data
            df = pd.DataFrame({
                'ds': data.index,
                'y': data.values
            })

            # Generate forecast to get components
            forecast = self.model.predict(df)

            # Select relevant components
            components = ['trend']
            if 'yearly' in forecast.columns:
                components.append('yearly')
            if 'weekly' in forecast.columns:
                components.append('weekly')
            if 'monthly' in forecast.columns:
                components.append('monthly')
            if 'quarterly' in forecast.columns:
                components.append('quarterly')

            return forecast[components]

        except Exception as e:
            logger.warning(f"Could not extract Prophet components: {e}")
            return None

    def get_changepoints(self) -> Optional[pd.DataFrame]:
        """Get detected trend changepoints"""
        if not self.is_fitted:
            return None

        try:
            changepoints = pd.DataFrame({
                'ds': self.model.changepoints,
                'trend_change': self.model.params['delta'][0, :]
            })

            # Only return significant changepoints
            significant_changes = np.abs(changepoints['trend_change']) > 0.01
            return changepoints[significant_changes]

        except Exception as e:
            logger.warning(f"Could not extract changepoints: {e}")
            return None

    def plot_components(self, data: pd.Series, save_path: Optional[str] = None):
        """Plot Prophet's component decomposition"""
        if not self.is_fitted:
            logger.warning("Model must be fitted before plotting components")
            return

        try:
            # Prepare data
            df = pd.DataFrame({
                'ds': data.index,
                'y': data.values
            })

            # Generate forecast
            forecast = self.model.predict(df)

            # Create components plot
            fig = self.model.plot_components(forecast)

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Prophet components plot saved to {save_path}")

            return fig

        except Exception as e:
            logger.warning(f"Could not create components plot: {e}")
            return None