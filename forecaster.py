"""
Main forecaster class that orchestrates all models
"""

import pandas as pd
from datetime import timedelta
from typing import Dict, Any, Tuple, Optional
import asyncio
import concurrent.futures
from models import (SeasonalNaiveModel, FeatureRichNaiveBayesModel,
                     ExponentialSmoothingModel, PROPHET_AVAILABLE, LIGHTGBM_AVAILABLE)

# Import optional models only if available
if PROPHET_AVAILABLE:
    from models import ProphetModel
if LIGHTGBM_AVAILABLE:
    from models import LightGBMQuantileModel
from models.ensemble import EnsembleMethods
from visualization import ForecastVisualizer
from utils.logging_config import get_logger
from utils.cache import get_cache, hash_dataframe, hash_dataframe_fast
from utils.exceptions import TTMLibraryError, ModelNotAvailableError, InsufficientDataError, DataValidationError
from utils.constants import MIN_DATA_POINTS_FOR_RELIABLE_FORECAST, FORECAST_CACHE_MAX_AGE

logger = get_logger(__name__)


class DailyCPUForecaster:
    """Main forecaster class that coordinates all models and ensemble methods"""

    def __init__(self, data: pd.Series, forecast_horizon: int = 30, test_size: int = 14,
                 use_enhanced_ttm: bool = False):
        # Validation
        if not isinstance(data, pd.Series):
            raise DataValidationError("Data must be a pandas Series")

        min_required = test_size + MIN_DATA_POINTS_FOR_RELIABLE_FORECAST
        if len(data) < min_required:
            raise InsufficientDataError(len(data), min_required)

        if forecast_horizon <= 0:
            raise ValueError("Forecast horizon must be positive")
        if test_size <= 0:
            raise ValueError("Test size must be positive")
        if test_size >= len(data):
            raise DataValidationError("Test size cannot be larger than data length")

        self.data = data
        self.forecast_horizon = forecast_horizon
        self.test_size = test_size

        # Split data
        self.train = data[:-test_size]
        self.test = data[-test_size:]

        # Initialize components
        self.results = {}
        self.ensemble_methods = EnsembleMethods()
        self.visualizer = ForecastVisualizer(save_plots=True, show_plots=False, metric_name='cpu')

        # Initialize core models
        self.models = {
            'SeasonalNaive': SeasonalNaiveModel(),
            'FeatureRichNaiveBayes': FeatureRichNaiveBayesModel(),
            'ExponentialSmoothing': ExponentialSmoothingModel()
        }

        # Add enhanced models with graceful fallback
        self._add_enhanced_models()

        logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")

    def _add_enhanced_models(self) -> None:
        """Add enhanced models with graceful fallback"""
        # Try to add Prophet model
        if PROPHET_AVAILABLE:
            try:
                self.models['Prophet'] = ProphetModel()
                logger.info("Added Prophet model")
            except Exception as e:
                logger.warning(f"Prophet model failed to initialize: {e}")
        else:
            logger.warning("Prophet model unavailable: library not installed")

        # Try to add LightGBM Quantile model
        if LIGHTGBM_AVAILABLE:
            try:
                self.models['LightGBM_Quantile'] = LightGBMQuantileModel()
                logger.info("Added LightGBM Quantile model")
            except Exception as e:
                logger.warning(f"LightGBM Quantile model failed to initialize: {e}")
        else:
            logger.warning("LightGBM Quantile model unavailable: library not installed or missing dependencies")

    def _run_single_model(self, name: str, model: Any) -> Tuple[str, Dict[str, Any]]:
        """Run a single model and return results"""
        try:
            # Check cache first
            cache = get_cache()
            cache.max_age = FORECAST_CACHE_MAX_AGE
            data_hash = hash_dataframe_fast(self.train)
            cache_key = cache.cache_key_for_forecast(
                name, data_hash, self.forecast_horizon, test_size=self.test_size
            )

            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"Using cached results for {name}")
                return name, cached_result

            logger.info(f"Running {name} forecast...")
            result = model.fit_and_forecast(
                self.train, self.test, self.forecast_horizon
            )

            # Cache the result
            cache.set(cache_key, result)
            return name, result

        except Exception as e:
            logger.error(f"{name} failed: {e}")
            return name, None

    def run_all_models(self, use_async: bool = True) -> None:
        """Run all individual models with optional async execution"""
        logger.info("Running all forecasting models...")

        if use_async and len(self.models) > 1:
            # Run models concurrently
            self._run_models_async()
        else:
            # Run models sequentially
            self._run_models_sequential()

        # Create ensemble forecasts with graceful degradation
        successful_results = {k: v for k, v in self.results.items() if v is not None}

        if len(successful_results) >= 2:
            logger.info("Creating ensemble forecasts...")
            try:
                ensemble_results = self.ensemble_methods.create_ensemble_forecasts(
                    successful_results, self.test
                )
                self.results.update(ensemble_results)

                logger.info("Adding confidence intervals...")
                self.ensemble_methods.add_confidence_intervals(self.results, ensemble_results)
            except Exception as e:
                logger.error(f"Ensemble creation failed: {e}. Continuing with individual models only.")
        elif len(successful_results) == 1:
            logger.warning("Only one model succeeded. Ensemble forecasts unavailable.")
        else:
            logger.error("No models succeeded. Unable to generate forecasts.")

        # Log final status
        total_models = len(self.models)
        successful_models = len(successful_results)
        logger.info(f"Forecasting completed: {successful_models}/{total_models} models succeeded")

    def _run_models_sequential(self) -> None:
        """Run models one by one"""
        for name, model in self.models.items():
            model_name, result = self._run_single_model(name, model)
            if result is not None:
                self.results[model_name] = result

    def _run_models_async(self) -> None:
        """Run models concurrently using thread pool"""
        logger.info("Running models concurrently for better performance...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(self.models))) as executor:
            # Submit all model runs
            future_to_name = {
                executor.submit(self._run_single_model, name, model): name
                for name, model in self.models.items()
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_name):
                model_name, result = future.result()
                if result is not None:
                    self.results[model_name] = result

    def get_summary(self) -> pd.DataFrame:
        """Get summary table of model performance"""
        return self.visualizer.create_summary_table(self.results)

    def plot_results(self):
        """Plot individual model forecasts (separate plots)"""
        self.visualizer.plot_results(
            self.data, self.results, self.forecast_horizon, self.test_size
        )


    def plot_model_comparison(self):
        """Plot model performance comparison bar charts"""
        self.visualizer.plot_model_comparison(self.results)

    def create_interactive_plot(self):
        """Create interactive HTML plot with zoom and pan"""
        self.visualizer.create_interactive_plot(
            self.data, self.results, self.forecast_horizon, self.test_size
        )

    def get_best_model(self) -> str:
        """Get the name of the best performing model (including ensemble models)"""
        available_results = {k: v for k, v in self.results.items() if v is not None}

        if not available_results:
            return None

        best_mae = float('inf')
        best_model = None
        for name, res in available_results.items():
            if res and 'metrics' in res and res['metrics']['MAE'] < best_mae:
                best_mae = res['metrics']['MAE']
                best_model = name
        return best_model

    def get_best_model_predictions(self, include_confidence_intervals: bool = True,
                                   include_metadata: bool = True) -> dict:
        """Get predictions from the best performing model with optional confidence intervals and metadata"""
        best_model_name = self.get_best_model()
        if not best_model_name or best_model_name not in self.results:
            return {"error": "No models available or results empty"}

        best_result = self.results[best_model_name]

        # Create future dates
        future_dates = pd.date_range(
            self.data.index[-1] + timedelta(days=1),
            periods=self.forecast_horizon
        )

        # Format predictions as list of dictionaries
        predictions = []
        for i, (date, value) in enumerate(zip(future_dates, best_result['future_forecast'])):
            prediction = {
                'date': date.strftime('%Y-%m-%d'),
                'value': float(value)
            }

            # Add confidence intervals if available and requested
            if include_confidence_intervals and 'future_upper' in best_result and 'future_lower' in best_result:
                prediction['upper_bound'] = float(best_result['future_upper'][i])
                prediction['lower_bound'] = float(best_result['future_lower'][i])

            predictions.append(prediction)

        result = {'predictions': predictions}

        # Add metadata if requested
        if include_metadata:
            result['metadata'] = {
                'model_name': best_model_name,
                'model_performance': {
                    'mae': float(best_result['metrics']['MAE']),
                    'rmse': float(best_result['metrics']['RMSE']),
                    'mape': float(best_result['metrics']['MAPE'])
                },
                'forecast_horizon_days': self.forecast_horizon,
                'total_models_compared': len(self.results),
                'has_confidence_intervals': 'future_upper' in best_result and 'future_lower' in best_result
            }

        return result

    def configure_plotting(self, save_plots: bool = True, show_plots: bool = False,
                          output_dir: str = './plots', metric_name: str = 'cpu'):
        """Configure plotting behavior"""
        self.visualizer = ForecastVisualizer(
            save_plots=save_plots,
            show_plots=show_plots,
            output_dir=output_dir,
            metric_name=metric_name
        )