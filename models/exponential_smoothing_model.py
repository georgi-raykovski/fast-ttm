"""
Exponential Smoothing (Holt-Winters) Model for Time Series Forecasting

Implements Triple Exponential Smoothing with automatic parameter optimization
for trend and seasonal patterns. Excellent for data with clear seasonality.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from .base_model import BaseModel
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Optional import - will gracefully fail if not available
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    EXPONENTIAL_SMOOTHING_AVAILABLE = True
except ImportError:
    EXPONENTIAL_SMOOTHING_AVAILABLE = False
    logger.warning("Exponential Smoothing model requires 'statsmodels'. Install with: pip install statsmodels")


class ExponentialSmoothingModel(BaseModel):
    """
    Holt-Winters Exponential Smoothing model with automatic configuration.

    Features:
    - Automatic trend detection (additive/multiplicative/none)
    - Automatic seasonality detection (additive/multiplicative/none)
    - Multiple seasonal period testing (7, 30, 91, 365 days)
    - Robust parameter optimization
    - Confidence intervals for forecasts
    """

    def __init__(self, seasonal_periods: Optional[List[int]] = None):
        super().__init__("ExponentialSmoothing")
        self.model = None
        self.seasonal_periods = seasonal_periods or [7, 30, 91]  # Weekly, monthly, quarterly
        self.best_seasonal_period = 7
        self.best_config = {}
        self.model_info = {}

    def fit(self, data: pd.Series) -> None:
        """Fit Exponential Smoothing model with automatic configuration"""
        if not EXPONENTIAL_SMOOTHING_AVAILABLE:
            raise ImportError("Exponential Smoothing model requires statsmodels. Install with: pip install statsmodels")

        try:
            logger.info(f"Fitting Exponential Smoothing on {len(data)} data points")

            # Determine appropriate seasonal periods
            seasonal_periods = self._get_seasonal_periods(len(data))

            # Test different configurations
            best_aic = float('inf')
            best_model = None
            best_config = {}

            configs = self._generate_configurations(seasonal_periods)

            for config in configs:
                try:
                    model = self._fit_single_config(data, config)
                    if model is not None and hasattr(model, 'aic'):
                        aic = model.aic

                        if aic < best_aic:
                            best_aic = aic
                            best_model = model
                            best_config = config

                            logger.info(f"New best ES config: period={config['seasonal_periods']}, "
                                      f"trend={config['trend']}, seasonal={config['seasonal']}, AIC={aic:.2f}")

                except Exception as e:
                    logger.warning(f"ES config {config} failed: {e}")
                    continue

            if best_model is None:
                raise ValueError("Could not fit Exponential Smoothing with any configuration")

            self.model = best_model
            self.best_config = best_config
            self.best_seasonal_period = best_config.get('seasonal_periods', 7)
            self.is_fitted = True

            # Store model information
            self.model_info = {
                'aic': best_aic,
                'bic': getattr(best_model, 'bic', None),
                'seasonal_periods': self.best_seasonal_period,
                'trend': best_config.get('trend', 'add'),
                'seasonal': best_config.get('seasonal', 'add'),
                'damped_trend': best_config.get('damped_trend', False)
            }

            logger.info(f"Exponential Smoothing fitted successfully: {self.model_info}")

        except Exception as e:
            logger.error(f"Exponential Smoothing fitting failed: {e}")
            raise

    def _get_seasonal_periods(self, data_length: int) -> List[int]:
        """Determine appropriate seasonal periods based on data length"""
        if data_length >= 365:
            # With yearly data, test multiple seasonalities
            return [7, 30, 91, 365]
        elif data_length >= 182:
            # With 6+ months, test up to quarterly
            return [7, 30, 91]
        elif data_length >= 90:
            # With 3+ months, test up to monthly
            return [7, 30]
        else:
            # Limited data, weekly only
            return [7]

    def _generate_configurations(self, seasonal_periods: List[int]) -> List[Dict[str, Any]]:
        """Generate different model configurations to test"""
        configs = []

        # Trend options
        trend_options = ['add', 'mul', None]
        seasonal_options = ['add', 'mul', None]
        damped_options = [True, False]

        for period in seasonal_periods:
            for trend in trend_options:
                for seasonal in seasonal_options:
                    # Skip invalid combinations
                    if seasonal is None and period > 7:
                        continue

                    for damped in damped_options:
                        # Damped trend only makes sense with trend
                        if damped and trend is None:
                            continue

                        configs.append({
                            'trend': trend,
                            'seasonal': seasonal,
                            'seasonal_periods': period,
                            'damped_trend': damped
                        })

        # Sort by complexity (simpler models first)
        return sorted(configs, key=lambda x: self._config_complexity(x))

    def _config_complexity(self, config: Dict[str, Any]) -> int:
        """Calculate configuration complexity for sorting"""
        complexity = 0
        if config['trend'] is not None:
            complexity += 1
        if config['seasonal'] is not None:
            complexity += 1
        if config['damped_trend']:
            complexity += 1
        if config['seasonal'] == 'mul':
            complexity += 1
        if config['trend'] == 'mul':
            complexity += 1
        return complexity

    def _fit_single_config(self, data: pd.Series, config: Dict[str, Any]) -> Optional[Any]:
        """Fit a single Exponential Smoothing configuration"""
        try:
            # Handle edge cases
            if len(data) < 2 * config['seasonal_periods']:
                return None

            model = ExponentialSmoothing(
                data,
                trend=config['trend'],
                seasonal=config['seasonal'],
                seasonal_periods=config['seasonal_periods'],
                damped_trend=config['damped_trend']
            )

            fitted_model = model.fit(
                optimized=True,
                remove_bias=False,
                use_brute=False
            )

            return fitted_model

        except Exception as e:
            logger.debug(f"Config {config} failed: {e}")
            return None

    def forecast(self, horizon: int) -> np.ndarray:
        """Generate Exponential Smoothing forecasts"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        try:
            # Generate forecast
            forecast = self.model.forecast(steps=horizon)

            # Generate prediction intervals if possible
            try:
                prediction_intervals = self.model.get_prediction(
                    start=len(self.model.fittedvalues),
                    end=len(self.model.fittedvalues) + horizon - 1
                )

                # Store confidence intervals
                conf_int = prediction_intervals.conf_int()
                self._last_confidence_intervals = {
                    'lower': conf_int.iloc[:, 0].values,
                    'upper': conf_int.iloc[:, 1].values
                }

            except Exception as e:
                logger.warning(f"Could not generate confidence intervals: {e}")
                self._last_confidence_intervals = None

            logger.info(f"Generated Exponential Smoothing forecast for {horizon} periods")

            return np.array(forecast)

        except Exception as e:
            logger.error(f"Exponential Smoothing forecasting failed: {e}")
            # Fallback: use last fitted value
            if hasattr(self.model, 'fittedvalues') and len(self.model.fittedvalues) > 0:
                last_value = self.model.fittedvalues.iloc[-1]
                return np.full(horizon, last_value)
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        info = super().get_model_info()

        if self.is_fitted:
            info.update({
                'model_type': 'Exponential Smoothing (Holt-Winters)',
                'parameters': self.model_info,
                'best_config': self.best_config,
                'has_confidence_intervals': hasattr(self, '_last_confidence_intervals') and self._last_confidence_intervals is not None
            })

        return info

    def get_confidence_intervals(self) -> Optional[Dict[str, np.ndarray]]:
        """Get the last generated confidence intervals"""
        return getattr(self, '_last_confidence_intervals', None)

    def decompose_series(self, data: pd.Series) -> Optional[Dict[str, pd.Series]]:
        """Perform seasonal decomposition of the series"""
        if not EXPONENTIAL_SMOOTHING_AVAILABLE:
            return None

        try:
            # Use the best seasonal period for decomposition
            period = self.best_seasonal_period if self.is_fitted else 7

            if len(data) < 2 * period:
                return None

            decomposition = seasonal_decompose(
                data,
                model='additive',  # Could be made configurable
                period=period
            )

            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'observed': decomposition.observed
            }

        except Exception as e:
            logger.warning(f"Series decomposition failed: {e}")
            return None

    def get_fitted_values(self) -> Optional[pd.Series]:
        """Get fitted values from the model"""
        if self.is_fitted and hasattr(self.model, 'fittedvalues'):
            return self.model.fittedvalues
        return None

    def get_residuals(self) -> Optional[pd.Series]:
        """Get model residuals"""
        if self.is_fitted and hasattr(self.model, 'resid'):
            return self.model.resid
        return None