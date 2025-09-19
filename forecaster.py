"""
Exponential Smoothing forecaster class
"""

import pandas as pd
from datetime import timedelta
from typing import Dict, Any, Tuple
from models import ExponentialSmoothingModel
from visualization import ForecastVisualizer
from utils.logging_config import get_logger
from utils.cache import get_cache, hash_dataframe_fast
from utils.exceptions import InsufficientDataError, DataValidationError
from utils.constants import MIN_DATA_POINTS_FOR_RELIABLE_FORECAST, FORECAST_CACHE_MAX_AGE

logger = get_logger(__name__)


class ExponentialSmoothingForecaster:
    """Exponential Smoothing focused forecaster class"""

    def __init__(self, data: pd.Series, forecast_horizon: int = 30, test_size: int = 14):
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
        self.visualizer = ForecastVisualizer(save_plots=True, show_plots=False, metric_name='cpu')

        # Initialize the Exponential Smoothing model
        self.model = ExponentialSmoothingModel()

        logger.info("Loaded Exponential Smoothing model")

    def _run_model(self) -> Tuple[str, Dict[str, Any]]:
        """Run the Exponential Smoothing model and return results"""
        try:
            # Check cache first
            cache = get_cache()
            cache.max_age = FORECAST_CACHE_MAX_AGE
            data_hash = hash_dataframe_fast(self.train)
            cache_key = cache.cache_key_for_forecast(
                "ExponentialSmoothing", data_hash, self.forecast_horizon, test_size=self.test_size
            )

            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.info("Using cached results for Exponential Smoothing")
                return "ExponentialSmoothing", cached_result

            logger.info("Running Exponential Smoothing forecast...")
            result = self.model.fit_and_forecast(
                self.train, self.test, self.forecast_horizon
            )

            # Cache the result
            cache.set(cache_key, result)
            return "ExponentialSmoothing", result

        except Exception as e:
            logger.error(f"Exponential Smoothing failed: {e}")
            return "ExponentialSmoothing", None

    def run_forecast(self) -> None:
        """Run the Exponential Smoothing forecast"""
        logger.info("Running Exponential Smoothing forecast...")

        # Run the model
        model_name, result = self._run_model()

        if result is not None:
            self.results[model_name] = result
            logger.info("Exponential Smoothing forecast completed successfully")
        else:
            logger.error("Exponential Smoothing forecast failed")
            self.results = {}


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

    def get_model_name(self) -> str:
        """Get the model name"""
        if self.results and 'ExponentialSmoothing' in self.results:
            return 'ExponentialSmoothing'
        return None

    def get_model_predictions(self, include_confidence_intervals: bool = True,
                             include_metadata: bool = True) -> dict:
        """Get predictions from the Exponential Smoothing model with optional confidence intervals and metadata"""
        model_name = self.get_model_name()
        if not model_name or model_name not in self.results:
            return {"error": "Model not available or results empty"}

        result = self.results[model_name]

        # Create future dates
        future_dates = pd.date_range(
            self.data.index[-1] + timedelta(days=1),
            periods=self.forecast_horizon
        )

        # Format predictions as list of dictionaries
        predictions = []
        for i, (date, value) in enumerate(zip(future_dates, result['future_forecast'])):
            prediction = {
                'date': date.strftime('%Y-%m-%d'),
                'value': float(value)
            }

            # Add confidence intervals if available and requested
            if include_confidence_intervals and 'future_upper' in result and 'future_lower' in result:
                prediction['upper_bound'] = float(result['future_upper'][i])
                prediction['lower_bound'] = float(result['future_lower'][i])

            predictions.append(prediction)

        output = {'predictions': predictions}

        # Add metadata if requested
        if include_metadata:
            output['metadata'] = {
                'model_name': model_name,
                'model_performance': {
                    'mae': float(result['metrics']['MAE']),
                    'rmse': float(result['metrics']['RMSE']),
                    'mape': float(result['metrics']['MAPE'])
                },
                'forecast_horizon_days': self.forecast_horizon,
                'has_confidence_intervals': 'future_upper' in result and 'future_lower' in result
            }

        return output

    def configure_plotting(self, save_plots: bool = True, show_plots: bool = False,
                          output_dir: str = './plots', metric_name: str = 'cpu'):
        """Configure plotting behavior"""
        self.visualizer = ForecastVisualizer(
            save_plots=save_plots,
            show_plots=show_plots,
            output_dir=output_dir,
            metric_name=metric_name
        )