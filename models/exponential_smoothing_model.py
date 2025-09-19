"""
Exponential Smoothing (Holt-Winters) Model for Time Series Forecasting

Implements Triple Exponential Smoothing with automatic parameter optimization
for trend and seasonal patterns. Excellent for data with clear seasonality.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import hashlib
import json
from .base_model import BaseModel
from utils.logging_config import get_logger
from utils.cache import get_cache

logger = get_logger(__name__)

# Optional import - will gracefully fail if not available
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
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
                 cross_validate: bool = True,
                 parallel_cv: bool = True,
                 max_workers: Optional[int] = None):
        super().__init__("ExponentialSmoothing_Enhanced")
        self.model = None
        self.seasonal_periods = seasonal_periods
        self.auto_feature_selection = auto_feature_selection
        self.cross_validate = cross_validate
        self.parallel_cv = parallel_cv
        self.max_workers = max_workers or min(mp.cpu_count(), 4)  # Limit to avoid overwhelming
        self.best_seasonal_period = 7
        self.best_config = {}
        self.model_info = {}
        self.data_characteristics = {}
        self.feature_analysis = {}
        self._config_cache = get_cache()  # Smart caching for configurations

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
                if self.parallel_cv and len(seasonal_periods) > 2:
                    best_aic, best_model, best_config = self._parallel_cross_validate_configurations(data, seasonal_periods)
                else:
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
        """Detect seasonal patterns in the data using optimized vectorized operations"""
        try:
            # Auto-correlation analysis for different lags - vectorized approach
            seasonal_strength = {}

            # Test common seasonal periods
            test_periods = [7, 14, 30, 91, 182, 365]

            # Convert to numpy for faster operations
            data_values = data.values
            data_length = len(data_values)

            for period in test_periods:
                if data_length >= 3 * period:
                    try:
                        # Vectorized auto-correlation calculation
                        if period < data_length:
                            x = data_values[:-period]
                            y = data_values[period:]
                            # Use numpy's corrcoef for faster computation
                            correlation_matrix = np.corrcoef(x, y)
                            autocorr = correlation_matrix[0, 1] if correlation_matrix.shape == (2, 2) else 0
                            seasonal_strength[period] = abs(autocorr) if not np.isnan(autocorr) else 0
                        else:
                            seasonal_strength[period] = 0
                    except:
                        seasonal_strength[period] = 0

            self.data_characteristics['seasonal_strength'] = seasonal_strength

            # Identify strongest seasonal period using numpy argmax for speed
            if seasonal_strength:
                strengths = np.array(list(seasonal_strength.values()))
                periods = np.array(list(seasonal_strength.keys()))
                best_idx = np.argmax(strengths)
                best_period = periods[best_idx]
                self.data_characteristics['strongest_seasonal_period'] = int(best_period)
                self.data_characteristics['max_seasonal_strength'] = float(strengths[best_idx])

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

    def _get_data_fingerprint(self, data: pd.Series) -> str:
        """Generate a unique fingerprint for the data for caching purposes"""
        try:
            # Create hash based on data characteristics
            data_stats = {
                'length': len(data),
                'mean': float(data.mean()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'first_values': data.head(10).tolist(),
                'last_values': data.tail(10).tolist()
            }

            # Convert to JSON string and hash
            data_string = json.dumps(data_stats, sort_keys=True)
            return hashlib.md5(data_string.encode()).hexdigest()[:16]
        except:
            # Fallback to simple hash
            return hashlib.md5(str(data.values).encode()).hexdigest()[:16]

    def _cache_key_for_config(self, data_fingerprint: str, config: Dict[str, Any]) -> str:
        """Generate cache key for a specific configuration"""
        config_str = json.dumps(config, sort_keys=True)
        return f"es_config_{data_fingerprint}_{hashlib.md5(config_str.encode()).hexdigest()[:8]}"

    def _get_cached_config_result(self, data: pd.Series, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached result for a configuration if available"""
        try:
            data_fingerprint = self._get_data_fingerprint(data)
            cache_key = self._cache_key_for_config(data_fingerprint, config)

            cached_result = self._config_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for config: {config}")
                return cached_result
            return None
        except Exception as e:
            logger.debug(f"Cache lookup failed: {e}")
            return None

    def _cache_config_result(self, data: pd.Series, config: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Cache the result for a configuration"""
        try:
            data_fingerprint = self._get_data_fingerprint(data)
            cache_key = self._cache_key_for_config(data_fingerprint, config)

            # Cache for 1 hour (3600 seconds)
            self._config_cache.set(cache_key, result, ttl=3600)
            logger.debug(f"Cached result for config: {config}")
        except Exception as e:
            logger.debug(f"Cache store failed: {e}")

    def _parallel_cross_validate_configurations(self, data: pd.Series, seasonal_periods: List[int]) -> tuple:
        """Parallel cross-validate different model configurations using multiprocessing"""
        try:
            logger.info(f"Parallel cross-validating model configurations using {self.max_workers} workers")

            configs = self._generate_configurations(seasonal_periods)
            cv_window = min(len(data) // 4, 30)
            train_size = len(data) - cv_window

            def evaluate_config_cv(config_data_tuple):
                """Function to evaluate a single configuration in parallel"""
                config, data_values, train_size, cv_window = config_data_tuple

                try:
                    cv_scores = []

                    # Convert back to Series with proper index
                    data_series = pd.Series(data_values, index=range(len(data_values)))

                    # Perform 3-fold time series CV
                    for fold in range(3):
                        fold_train_end = train_size - fold * cv_window // 3
                        fold_train_start = max(0, fold_train_end - train_size)
                        fold_val_end = fold_train_end + cv_window // 3

                        if fold_train_end - fold_train_start < 2 * config.get('seasonal_periods', 7):
                            continue

                        train_fold = data_series.iloc[fold_train_start:fold_train_end]
                        val_fold = data_series.iloc[fold_train_end:fold_val_end]

                        if len(train_fold) < 2 * config.get('seasonal_periods', 7) or len(val_fold) == 0:
                            continue

                        # Create temporary model instance for this worker
                        temp_model = ExponentialSmoothingModel(parallel_cv=False)
                        fold_model = temp_model._fit_single_config(train_fold, config)

                        if fold_model is not None:
                            try:
                                forecast = fold_model.forecast(len(val_fold))
                                mse = np.mean((val_fold.values - forecast) ** 2)
                                cv_scores.append(mse)
                            except:
                                continue

                    if cv_scores:
                        avg_score = np.mean(cv_scores)
                        return (config, avg_score)
                    else:
                        return (config, float('inf'))

                except Exception as e:
                    logger.debug(f"Parallel CV failed for config {config}: {e}")
                    return (config, float('inf'))

            # Prepare data for parallel processing
            config_data_tuples = [
                (config, data.values, train_size, cv_window)
                for config in configs
            ]

            best_score = float('inf')
            best_config = {}

            # Use ProcessPoolExecutor for parallel evaluation
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_config = {
                    executor.submit(evaluate_config_cv, config_tuple): config_tuple[0]
                    for config_tuple in config_data_tuples
                }

                # Collect results as they complete
                for future in as_completed(future_to_config):
                    try:
                        config, score = future.result()
                        if score < best_score:
                            best_score = score
                            best_config = config
                            logger.info(f"New best parallel CV config: period={config['seasonal_periods']}, "
                                      f"trend={config['trend']}, seasonal={config['seasonal']}, CV_MSE={score:.2f}")
                    except Exception as e:
                        logger.debug(f"Parallel CV task failed: {e}")

            # Refit best model on full data
            if best_config:
                best_model = self._fit_single_config(data, best_config)
                best_aic = getattr(best_model, 'aic', float('inf')) if best_model else float('inf')
                return best_aic, best_model, best_config
            else:
                logger.warning("Parallel cross-validation failed, falling back to sequential")
                return self._cross_validate_configurations(data, seasonal_periods)

        except Exception as e:
            logger.warning(f"Parallel cross-validation failed: {e}, falling back to sequential")
            return self._cross_validate_configurations(data, seasonal_periods)

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
        """Evaluate configurations using AIC with smart caching"""
        configs = self._generate_configurations(seasonal_periods)
        best_aic = float('inf')
        best_model = None
        best_config = {}

        for config in configs:
            try:
                # Check cache first
                cached_result = self._get_cached_config_result(data, config)
                if cached_result is not None:
                    aic = cached_result.get('aic', float('inf'))
                    if aic < best_aic:
                        best_aic = aic
                        best_config = config
                        # Need to refit the model since we can't cache the actual model object
                        best_model = self._fit_single_config(data, config)
                        logger.info(f"New best ES config (cached): period={config['seasonal_periods']}, "
                                  f"trend={config['trend']}, seasonal={config['seasonal']}, AIC={aic:.2f}")
                    continue

                # Not in cache, evaluate
                model = self._fit_single_config(data, config)
                if model is not None and hasattr(model, 'aic'):
                    aic = model.aic

                    # Cache the result
                    self._cache_config_result(data, config, {'aic': aic})

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