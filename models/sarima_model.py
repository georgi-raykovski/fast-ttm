"""
SARIMA (Seasonal AutoRegressive Integrated Moving Average) Model

This model is excellent for time series with strong seasonal patterns and trends.
It automatically selects the best parameters and seasonal periods for yearly data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from .base_model import BaseModel
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Optional import - will gracefully fail if not available
try:
    from pmdarima import auto_arima
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.stats.diagnostic import acorr_ljungbox
    SARIMA_AVAILABLE = True
except ImportError:
    SARIMA_AVAILABLE = False
    logger.warning("SARIMA model requires 'pmdarima' and 'statsmodels'. Install with: pip install pmdarima statsmodels")


class SARIMAModel(BaseModel):
    """
    SARIMA model with automatic parameter selection for yearly time series data.

    Features:
    - Auto parameter selection (p,d,q) and seasonal (P,D,Q,s)
    - Multiple seasonal period detection (7, 30, 365 days)
    - Handles trend and seasonality automatically
    - Robust error handling and fallback
    """

    def __init__(self, seasonal_periods: Optional[List[int]] = None):
        super().__init__("SARIMA")
        self.model = None
        self.seasonal_periods = seasonal_periods or [7, 30, 365]  # Weekly, monthly, yearly
        self.best_seasonal_period = 7
        self.model_params = {}
        self.residuals_info = {}

    def fit(self, data: pd.Series) -> None:
        """Fit SARIMA model with automatic parameter selection"""
        if not SARIMA_AVAILABLE:
            raise ImportError("SARIMA model requires pmdarima. Install with: pip install pmdarima")

        try:
            logger.info(f"Fitting SARIMA model on {len(data)} data points")

            # Determine appropriate seasonal periods based on data length
            seasonal_periods = self._get_seasonal_periods(len(data))

            best_aic = float('inf')
            best_model = None
            best_period = 7

            for period in seasonal_periods:
                try:
                    logger.info(f"Testing SARIMA with seasonal period {period}")

                    # Auto-ARIMA with seasonal parameters
                    temp_model = auto_arima(
                        data,
                        seasonal=True,
                        m=period,  # Seasonal period
                        max_p=3, max_q=3,  # Non-seasonal AR and MA
                        max_P=2, max_Q=2,  # Seasonal AR and MA
                        max_d=2, max_D=1,  # Differencing
                        stepwise=True,
                        suppress_warnings=True,
                        error_action='ignore',
                        information_criterion='aic',
                        n_fits=50
                    )

                    if temp_model.aic() < best_aic:
                        best_aic = temp_model.aic()
                        best_model = temp_model
                        best_period = period

                        logger.info(f"New best SARIMA: period={period}, AIC={best_aic:.2f}")

                except Exception as e:
                    logger.warning(f"SARIMA failed for period {period}: {e}")
                    continue

            if best_model is None:
                raise ValueError("Could not fit SARIMA model with any seasonal period")

            self.model = best_model
            self.best_seasonal_period = best_period
            self.is_fitted = True

            # Store model information
            self.model_params = {
                'order': self.model.order,
                'seasonal_order': self.model.seasonal_order,
                'aic': self.model.aic(),
                'bic': self.model.bic(),
                'seasonal_period': best_period
            }

            # Analyze residuals
            self._analyze_residuals(data)

            logger.info(f"SARIMA fitted successfully: {self.model_params}")

        except Exception as e:
            logger.error(f"SARIMA model fitting failed: {e}")
            raise

    def _get_seasonal_periods(self, data_length: int) -> List[int]:
        """Determine appropriate seasonal periods based on data length"""
        if data_length >= 365:
            # With yearly data, test multiple seasonalities
            return [7, 30, 365]
        elif data_length >= 90:
            # With 3+ months, test weekly and monthly
            return [7, 30]
        elif data_length >= 30:
            # With 1+ month, test weekly
            return [7]
        else:
            # Limited data, use simple weekly
            return [7]

    def _analyze_residuals(self, original_data: pd.Series) -> None:
        """Analyze model residuals for quality assessment"""
        try:
            residuals = self.model.resid()

            # Ljung-Box test for residual autocorrelation
            ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)

            self.residuals_info = {
                'residuals_mean': float(np.mean(residuals)),
                'residuals_std': float(np.std(residuals)),
                'ljung_box_pvalue': float(ljung_box['lb_pvalue'].iloc[-1]) if len(ljung_box) > 0 else None
            }

            logger.info(f"Residuals analysis: mean={self.residuals_info['residuals_mean']:.4f}, "
                       f"std={self.residuals_info['residuals_std']:.4f}")

        except Exception as e:
            logger.warning(f"Residuals analysis failed: {e}")
            self.residuals_info = {}

    def forecast(self, horizon: int) -> np.ndarray:
        """Generate SARIMA forecasts with confidence intervals"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        try:
            # Generate forecasts
            forecast_result = self.model.predict(n_periods=horizon, return_conf_int=True)

            if isinstance(forecast_result, tuple):
                forecasts, conf_intervals = forecast_result

                # Store confidence intervals for later use
                self._last_confidence_intervals = {
                    'lower': conf_intervals[:, 0],
                    'upper': conf_intervals[:, 1]
                }
            else:
                forecasts = forecast_result
                self._last_confidence_intervals = None

            logger.info(f"Generated SARIMA forecast for {horizon} periods")

            return np.array(forecasts)

        except Exception as e:
            logger.error(f"SARIMA forecasting failed: {e}")
            # Fallback to seasonal naive if SARIMA fails
            if hasattr(self, '_fallback_pattern'):
                repeats = horizon // len(self._fallback_pattern) + 1
                return np.tile(self._fallback_pattern, repeats)[:horizon]
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        info = super().get_model_info()

        if self.is_fitted:
            info.update({
                'model_type': 'SARIMA',
                'parameters': self.model_params,
                'residuals_analysis': self.residuals_info,
                'best_seasonal_period': self.best_seasonal_period,
                'has_confidence_intervals': hasattr(self, '_last_confidence_intervals') and self._last_confidence_intervals is not None
            })

        return info

    def get_confidence_intervals(self) -> Optional[Dict[str, np.ndarray]]:
        """Get the last generated confidence intervals"""
        return getattr(self, '_last_confidence_intervals', None)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get model parameter significance (approximate)"""
        if not self.is_fitted:
            return {}

        try:
            # Get parameter statistics from fitted model
            summary = self.model.summary()

            # Extract coefficient p-values as importance measure
            importance = {}
            if hasattr(self.model, 'pvalues'):
                for i, pval in enumerate(self.model.pvalues):
                    param_name = f"param_{i}"
                    importance[param_name] = 1.0 - pval  # Higher significance = higher importance

            return importance

        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {}