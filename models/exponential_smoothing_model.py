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
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller, kpss
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
                 auto_feature_selection: bool = True,
                 cross_validate: bool = True):
        super().__init__("ExponentialSmoothing_Enhanced")
        self.model = None
        self.seasonal_periods = seasonal_periods
        self.auto_feature_selection = auto_feature_selection
        self.cross_validate = cross_validate
        self.best_seasonal_period = 7
        self.best_config = {}
        self.model_info = {}
        self.data_characteristics = {}
        self.feature_analysis = {}

    def fit(self, data: pd.Series) -> None:
        """Fit Enhanced Exponential Smoothing model with intelligent feature selection"""
        if not EXPONENTIAL_SMOOTHING_AVAILABLE:
            raise ImportError("Exponential Smoothing model requires statsmodels. Install with: pip install statsmodels")

        try:
            logger.info(f"Fitting Enhanced Exponential Smoothing on {len(data)} data points")

            # Analyze data characteristics for feature selection
            if self.auto_feature_selection:
                self._analyze_data_characteristics(data)

            # Intelligent seasonal period selection
            seasonal_periods = self._intelligent_seasonal_selection(data)

            # Test different configurations with cross-validation if enabled
            if self.cross_validate and len(data) > 50:
                best_aic, best_model, best_config = self._cross_validate_configurations(data, seasonal_periods)
            else:
                best_aic, best_model, best_config = self._evaluate_configurations(data, seasonal_periods)

            if best_model is None:
                raise ValueError("Could not fit Enhanced Exponential Smoothing with any configuration")

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

            logger.info(f"Enhanced Exponential Smoothing fitted successfully: {self.model_info}")

        except Exception as e:
            logger.error(f"Enhanced Exponential Smoothing fitting failed: {e}")
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

    def _analyze_data_characteristics(self, data: pd.Series) -> None:
        """Analyze data characteristics for intelligent feature selection"""
        try:
            logger.info("Analyzing data characteristics for feature selection")

            # Basic statistics
            self.data_characteristics = {
                'length': len(data),
                'mean': float(data.mean()),
                'std': float(data.std()),
                'cv': float(data.std() / data.mean()) if data.mean() != 0 else float('inf'),
                'skewness': float(data.skew()),
                'kurtosis': float(data.kurtosis())
            }

            # Stationarity tests
            try:
                adf_result = adfuller(data.dropna())
                self.data_characteristics['adf_statistic'] = adf_result[0]
                self.data_characteristics['adf_pvalue'] = adf_result[1]
                self.data_characteristics['is_stationary_adf'] = adf_result[1] < 0.05

                kpss_result = kpss(data.dropna())
                self.data_characteristics['kpss_statistic'] = kpss_result[0]
                self.data_characteristics['kpss_pvalue'] = kpss_result[1]
                self.data_characteristics['is_stationary_kpss'] = kpss_result[1] > 0.05
            except Exception as e:
                logger.warning(f"Stationarity tests failed: {e}")
                self.data_characteristics['is_stationary_adf'] = False
                self.data_characteristics['is_stationary_kpss'] = False

            # Seasonality detection
            self._detect_seasonality_patterns(data)

            # Trend analysis
            self._analyze_trend_characteristics(data)

        except Exception as e:
            logger.warning(f"Data characteristics analysis failed: {e}")
            self.data_characteristics = {}

    def _detect_seasonality_patterns(self, data: pd.Series) -> None:
        """Detect seasonal patterns in the data"""
        try:
            # Auto-correlation analysis for different lags
            seasonal_strength = {}

            # Test common seasonal periods
            test_periods = [7, 14, 30, 91, 182, 365]

            for period in test_periods:
                if len(data) >= 3 * period:
                    try:
                        # Calculate auto-correlation at this lag
                        autocorr = data.autocorr(lag=period)
                        seasonal_strength[period] = abs(autocorr) if not pd.isna(autocorr) else 0
                    except:
                        seasonal_strength[period] = 0

            self.data_characteristics['seasonal_strength'] = seasonal_strength

            # Identify strongest seasonal period
            if seasonal_strength:
                best_period = max(seasonal_strength.keys(), key=lambda k: seasonal_strength[k])
                self.data_characteristics['strongest_seasonal_period'] = best_period
                self.data_characteristics['max_seasonal_strength'] = seasonal_strength[best_period]

        except Exception as e:
            logger.warning(f"Seasonality detection failed: {e}")

    def _analyze_trend_characteristics(self, data: pd.Series) -> None:
        """Analyze trend characteristics"""
        try:
            # Linear trend analysis
            x = np.arange(len(data))
            trend_coef = np.polyfit(x, data.values, 1)[0]

            self.data_characteristics['linear_trend_slope'] = float(trend_coef)
            self.data_characteristics['has_strong_trend'] = abs(trend_coef) > data.std() / len(data)

            # Moving average trend
            if len(data) >= 30:
                ma_30 = data.rolling(30).mean()
                trend_strength = (ma_30.iloc[-1] - ma_30.iloc[29]) / ma_30.std()
                self.data_characteristics['ma_trend_strength'] = float(trend_strength)

        except Exception as e:
            logger.warning(f"Trend analysis failed: {e}")

    def _intelligent_seasonal_selection(self, data: pd.Series) -> List[int]:
        """Intelligently select seasonal periods based on data analysis"""
        if self.seasonal_periods is not None:
            return self.seasonal_periods

        # Start with data-length-based selection
        base_periods = self._get_seasonal_periods(len(data))

        # If we have seasonality analysis, refine the selection
        if 'seasonal_strength' in self.data_characteristics:
            seasonal_strength = self.data_characteristics['seasonal_strength']

            # Filter periods with significant seasonal strength (> 0.1)
            significant_periods = [p for p, s in seasonal_strength.items() if s > 0.1]

            if significant_periods:
                # Combine base periods with significant periods, prioritize significant ones
                all_periods = list(set(base_periods + significant_periods))
                # Sort by seasonal strength (descending) then by period (ascending)
                all_periods.sort(key=lambda p: (-seasonal_strength.get(p, 0), p))
                return all_periods[:5]  # Limit to top 5 periods

        return base_periods

    def _cross_validate_configurations(self, data: pd.Series, seasonal_periods: List[int]) -> tuple:
        """Cross-validate different model configurations"""
        try:
            logger.info("Cross-validating model configurations")

            configs = self._generate_configurations(seasonal_periods)
            best_score = float('inf')
            best_model = None
            best_config = {}

            # Use time series cross-validation
            cv_window = min(len(data) // 4, 30)  # Use 1/4 of data or 30 points as validation
            train_size = len(data) - cv_window

            for config in configs:
                try:
                    cv_scores = []

                    # Perform 3-fold time series CV
                    for fold in range(3):
                        fold_train_end = train_size - fold * cv_window // 3
                        fold_train_start = max(0, fold_train_end - train_size)
                        fold_val_end = fold_train_end + cv_window // 3

                        if fold_train_end - fold_train_start < 2 * config.get('seasonal_periods', 7):
                            continue

                        train_fold = data.iloc[fold_train_start:fold_train_end]
                        val_fold = data.iloc[fold_train_end:fold_val_end]

                        if len(train_fold) < 2 * config.get('seasonal_periods', 7) or len(val_fold) == 0:
                            continue

                        fold_model = self._fit_single_config(train_fold, config)
                        if fold_model is not None:
                            try:
                                forecast = fold_model.forecast(len(val_fold))
                                mse = np.mean((val_fold.values - forecast) ** 2)
                                cv_scores.append(mse)
                            except:
                                continue

                    if cv_scores:
                        avg_score = np.mean(cv_scores)
                        if avg_score < best_score:
                            best_score = avg_score
                            # Refit on full training data
                            best_model = self._fit_single_config(data, config)
                            best_config = config

                            logger.info(f"New best CV config: period={config['seasonal_periods']}, "
                                      f"trend={config['trend']}, seasonal={config['seasonal']}, CV_MSE={avg_score:.2f}")

                except Exception as e:
                    logger.debug(f"CV failed for config {config}: {e}")
                    continue

            # If CV failed, fall back to AIC selection
            if best_model is None:
                logger.warning("Cross-validation failed, falling back to AIC selection")
                return self._evaluate_configurations(data, seasonal_periods)

            # Get AIC for the best model
            best_aic = getattr(best_model, 'aic', float('inf'))
            return best_aic, best_model, best_config

        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}, falling back to AIC selection")
            return self._evaluate_configurations(data, seasonal_periods)

    def _evaluate_configurations(self, data: pd.Series, seasonal_periods: List[int]) -> tuple:
        """Evaluate configurations using AIC (fallback method)"""
        configs = self._generate_configurations(seasonal_periods)
        best_aic = float('inf')
        best_model = None
        best_config = {}

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

        return best_aic, best_model, best_config

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on model characteristics"""
        if not self.is_fitted:
            return {}

        importance = {}

        # Seasonal component importance
        if self.best_config.get('seasonal') is not None:
            seasonal_strength = self.data_characteristics.get('max_seasonal_strength', 0.5)
            importance['seasonality'] = float(seasonal_strength)

        # Trend component importance
        if self.best_config.get('trend') is not None:
            trend_strength = abs(self.data_characteristics.get('linear_trend_slope', 0))
            trend_strength = min(trend_strength / self.data_characteristics.get('std', 1), 1.0)
            importance['trend'] = float(trend_strength)

        # Damping importance
        if self.best_config.get('damped_trend', False):
            importance['damping'] = 0.3

        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance

    def get_data_characteristics(self) -> Dict[str, Any]:
        """Get analyzed data characteristics"""
        return self.data_characteristics.copy()

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