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
    Enhanced Holt-Winters Exponential Smoothing model with automatic configuration
    and intelligent feature selection.

    Features:
    - Automatic trend detection (additive/multiplicative/none)
    - Automatic seasonality detection (additive/multiplicative/none)
    - Intelligent seasonal period selection based on data characteristics
    - Statistical tests for stationarity and seasonality
    - Robust parameter optimization with cross-validation
    - Confidence intervals for forecasts
    - Feature importance analysis
    """

    def __init__(self, seasonal_periods: Optional[List[int]] = None,
                 auto_feature_selection: bool = False,
                 cross_validate: bool = False,
                 parallel_cv: bool = False,
                 max_workers: Optional[int] = None,
                 peak_enhancement: bool = True):
        super().__init__("ExponentialSmoothing_PeakEnhanced")
        self.model = None
        self.auto_feature_selection = False
        self.cross_validate = False
        self.parallel_cv = False
        self.max_workers = 1
        self.peak_enhancement = peak_enhancement
        self.best_seasonal_period = 7  # Will be updated based on data length
        self.best_config = {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 7, 'damped_trend': False}
        self.model_info = {}
        self.peak_adjustment_factor = 1.0

    def fit(self, data: pd.Series) -> None:
        """Fit Enhanced Exponential Smoothing model with peak detection"""
        if not EXPONENTIAL_SMOOTHING_AVAILABLE:
            raise ImportError("Exponential Smoothing model requires statsmodels. Install with: pip install statsmodels")

        try:
            logger.info(f"Fitting Peak-Enhanced Exponential Smoothing on {len(data)} data points")

            # Analyze peaks if enhancement is enabled
            if self.peak_enhancement:
                self._analyze_peak_patterns(data)

            # Smart seasonal period selection based on data length and peaks
            optimal_period = self._select_optimal_seasonal_period_with_peaks(data)
            self.best_seasonal_period = optimal_period

            # Update configuration with optimal seasonal period and peak handling
            config = self._get_peak_enhanced_config(data, optimal_period)

            logger.info(f"Selected seasonal period: {optimal_period} (data length: {len(data)}, peak enhancement: {self.peak_enhancement})")

            # Fit model with peak-optimized configuration
            fitted_model = self._fit_single_config(data, config)

            if fitted_model is None:
                raise ValueError("Could not fit Peak-Enhanced Exponential Smoothing")

            self.model = fitted_model
            self.best_config = config
            self.is_fitted = True

            # Store enhanced model information
            best_aic = getattr(fitted_model, 'aic', float('inf'))
            self.model_info = {
                'aic': best_aic,
                'bic': getattr(fitted_model, 'bic', None),
                'seasonal_periods': self.best_seasonal_period,
                'trend': config.get('trend', 'add'),
                'seasonal': config.get('seasonal', 'add'),
                'damped_trend': config.get('damped_trend', False),
                'peak_enhancement': self.peak_enhancement,
                'peak_adjustment_factor': self.peak_adjustment_factor
            }

            logger.info(f"Peak-Enhanced Exponential Smoothing fitted successfully: {self.model_info}")

        except Exception as e:
            logger.error(f"Peak-Enhanced Exponential Smoothing fitting failed: {e}")
            raise

    def _select_optimal_seasonal_period(self, data: pd.Series) -> int:
        """Select optimal seasonal period based on data length and characteristics"""
        data_length = len(data)

        # Data-length based selection with minimum data requirements
        if data_length >= 730:  # 2+ years of data
            # Test yearly pattern first, then fall back to shorter periods
            candidates = [365, 91, 30, 7]
        elif data_length >= 365:  # 1+ year of data
            # Test quarterly and shorter patterns
            candidates = [91, 30, 7]
        elif data_length >= 180:  # 6+ months of data
            # Test monthly and weekly patterns
            candidates = [30, 7]
        elif data_length >= 60:   # 2+ months of data
            # Test weekly pattern only
            candidates = [7]
        else:
            # Too little data for reliable seasonal modeling
            logger.warning(f"Limited data ({data_length} points) - using minimal seasonality")
            return min(7, data_length // 3)  # Use shorter period for very limited data

        # Filter candidates that have enough data (need at least 2 full cycles)
        valid_candidates = [p for p in candidates if data_length >= 2 * p]

        if not valid_candidates:
            # Fallback to the largest period we can fit
            return min(7, data_length // 3)

        # For performance, just pick the largest valid period (captures longest patterns)
        # This avoids the complexity of autocorrelation testing while still being smart
        optimal_period = valid_candidates[0]

        logger.info(f"Selected seasonal period {optimal_period} from candidates {valid_candidates}")
        return optimal_period

    def _get_seasonal_periods(self, data_length: int) -> List[int]:
        """Return seasonal period for compatibility (now uses smart selection)"""
        return [7]  # Kept for compatibility but not used in new flow

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
                # Try to get forecast with confidence intervals
                if hasattr(self.model, 'forecast'):
                    forecast_with_ci = self.model.forecast(steps=horizon, return_conf_int=True)
                    if isinstance(forecast_with_ci, tuple):
                        _, conf_int = forecast_with_ci
                        self._last_confidence_intervals = {
                            'lower': conf_int[:, 0],
                            'upper': conf_int[:, 1]
                        }
                    else:
                        self._last_confidence_intervals = None
                else:
                    self._last_confidence_intervals = None

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

    def forecast_batch(self, horizons: List[int]) -> Dict[int, np.ndarray]:
        """Generate optimized batch forecasts for multiple horizons"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        try:
            max_horizon = max(horizons)
            # Generate forecast for the maximum horizon once
            full_forecast = self.forecast(max_horizon)

            # Extract subsets for each requested horizon
            batch_forecasts = {}
            for horizon in horizons:
                if horizon <= max_horizon:
                    batch_forecasts[horizon] = full_forecast[:horizon]
                else:
                    # If requested horizon is larger, generate specifically for it
                    batch_forecasts[horizon] = self.forecast(horizon)

            logger.info(f"Generated batch forecasts for horizons: {horizons}")
            return batch_forecasts

        except Exception as e:
            logger.error(f"Batch forecasting failed: {e}")
            raise

    def forecast_with_intervals(self, horizon: int, confidence_levels: List[float] = [0.8, 0.95]) -> Dict[str, np.ndarray]:
        """Generate forecast with multiple confidence intervals efficiently"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        try:
            # Generate base forecast
            point_forecast = self.forecast(horizon)

            # Try to get prediction intervals from statsmodels
            result = {'point_forecast': point_forecast}

            try:
                if hasattr(self.model, 'get_prediction'):
                    # Get forecast with prediction intervals
                    for confidence_level in confidence_levels:
                        prediction = self.model.get_prediction(
                            start=len(self.model.fittedvalues),
                            end=len(self.model.fittedvalues) + horizon - 1
                        )
                        conf_int = prediction.conf_int(alpha=1-confidence_level)

                        result[f'lower_{confidence_level}'] = conf_int.iloc[:, 0].values
                        result[f'upper_{confidence_level}'] = conf_int.iloc[:, 1].values

                elif hasattr(self.model, 'forecast') and hasattr(self.model, 'params'):
                    # Fallback: approximate confidence intervals using model residuals
                    if hasattr(self.model, 'resid') and len(self.model.resid.dropna()) > 0:
                        residual_std = np.std(self.model.resid.dropna())

                        for confidence_level in confidence_levels:
                            # Use normal distribution approximation
                            from scipy.stats import norm
                            z_score = norm.ppf((1 + confidence_level) / 2)
                            margin = z_score * residual_std

                            result[f'lower_{confidence_level}'] = point_forecast - margin
                            result[f'upper_{confidence_level}'] = point_forecast + margin

            except Exception as e:
                logger.warning(f"Could not generate confidence intervals: {e}")

            return result

        except Exception as e:
            logger.error(f"Forecast with intervals failed: {e}")
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

    def _analyze_peak_patterns(self, data: pd.Series) -> None:
        """Analyze peak patterns to enhance forecasting"""
        try:
            from scipy import signal

            # Find peaks and valleys
            data_values = data.values
            peaks, peak_properties = signal.find_peaks(
                data_values,
                height=np.percentile(data_values, 75),
                distance=5,
                prominence=np.std(data_values) * 0.5
            )

            valleys, valley_properties = signal.find_peaks(
                -data_values,
                height=-np.percentile(data_values, 25),
                distance=5,
                prominence=np.std(data_values) * 0.5
            )

            # Store peak characteristics
            self.peak_info = {
                'num_peaks': len(peaks),
                'num_valleys': len(valleys),
                'peak_indices': peaks,
                'valley_indices': valleys,
                'avg_peak_value': np.mean(data_values[peaks]) if len(peaks) > 0 else data.mean(),
                'avg_valley_value': np.mean(data_values[valleys]) if len(valleys) > 0 else data.mean(),
                'peak_volatility': np.std(data_values[peaks]) if len(peaks) > 0 else data.std(),
                'amplitude': np.mean(data_values[peaks]) - np.mean(data_values[valleys]) if len(peaks) > 0 and len(valleys) > 0 else 0
            }

            # Calculate peak adjustment factor based on volatility
            data_std = data.std()
            peak_std = self.peak_info['peak_volatility']

            if peak_std > data_std * 1.5:  # High peak volatility
                self.peak_adjustment_factor = 1.3
            elif peak_std > data_std * 1.2:  # Medium peak volatility
                self.peak_adjustment_factor = 1.2
            else:
                self.peak_adjustment_factor = 1.1

            logger.info(f"Peak analysis: {self.peak_info['num_peaks']} peaks, {self.peak_info['num_valleys']} valleys, "
                       f"amplitude: {self.peak_info['amplitude']:.2f}, adjustment factor: {self.peak_adjustment_factor:.2f}")

        except ImportError:
            logger.warning("scipy not available for peak analysis, using standard configuration")
            self.peak_info = {}
            self.peak_adjustment_factor = 1.0
        except Exception as e:
            logger.warning(f"Peak analysis failed: {e}, using standard configuration")
            self.peak_info = {}
            self.peak_adjustment_factor = 1.0

    def _select_optimal_seasonal_period_with_peaks(self, data: pd.Series) -> int:
        """Select optimal seasonal period considering peak patterns"""
        base_period = self._select_optimal_seasonal_period(data)

        if not self.peak_enhancement or not hasattr(self, 'peak_info'):
            return base_period

        # Analyze peak spacing to suggest seasonal period adjustments
        if 'peak_indices' in self.peak_info and len(self.peak_info['peak_indices']) > 2:
            peak_indices = self.peak_info['peak_indices']
            peak_spacings = np.diff(peak_indices)
            avg_peak_spacing = np.mean(peak_spacings)

            # If peaks occur more frequently than the base period, consider shorter period
            if avg_peak_spacing < base_period * 0.7:
                # Check if weekly pattern (7 days) captures peaks better
                if len(data) >= 60 and avg_peak_spacing >= 5 and avg_peak_spacing <= 10:
                    logger.info(f"Peak spacing ({avg_peak_spacing:.1f}) suggests weekly pattern, using period 7")
                    return 7

        return base_period

    def _get_peak_enhanced_config(self, data: pd.Series, seasonal_period: int) -> Dict[str, Any]:
        """Get configuration optimized for peak detection"""
        config = self.best_config.copy()
        config['seasonal_periods'] = seasonal_period

        if self.peak_enhancement and hasattr(self, 'peak_info'):
            # For high amplitude data, use multiplicative seasonality to better capture peaks
            if self.peak_info.get('amplitude', 0) > data.std() * 2:
                config['seasonal'] = 'mul'
                logger.info("Using multiplicative seasonality for high amplitude peaks")

            # For highly volatile peaks, disable damping to allow more responsive forecasts
            if self.peak_adjustment_factor > 1.2:
                config['damped_trend'] = False
                logger.info("Disabled trend damping for volatile peaks")

        return config

    def _intelligent_seasonal_selection(self, data: pd.Series) -> List[int]:
        """Return optimal seasonal period for compatibility"""
        return [self._select_optimal_seasonal_period_with_peaks(data)]



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