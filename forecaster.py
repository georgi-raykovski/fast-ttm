"""
Main forecaster class that orchestrates all models
"""

import pandas as pd
from datetime import timedelta
from typing import Dict
from models import (SeasonalNaiveModel, TTMModel, NaiveBayesModel,
                     TTMFineTunedModel, TTMAugmentedModel, TTMEnsembleModel)
from models.ensemble import EnsembleMethods
from visualization import ForecastVisualizer
from utils.logging_config import get_logger
from utils.exceptions import TTMLibraryError, ModelNotAvailableError, InsufficientDataError, DataValidationError
from utils.constants import MIN_DATA_POINTS_FOR_RELIABLE_FORECAST

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
        self.visualizer = ForecastVisualizer(save_plots=True, show_plots=False)

        # Initialize models with TTM error handling
        self.models = {
            'SeasonalNaive': SeasonalNaiveModel(),
            'NaiveBayes': NaiveBayesModel()
        }

        # Try to add TTM models with graceful fallback
        try:
            if use_enhanced_ttm:
                logger.info("Attempting to load enhanced TTM models...")

                # Try basic TTM first
                try:
                    self.models['TTM_ZeroShot'] = TTMModel()
                except (ImportError, TTMLibraryError, ModelNotAvailableError) as e:
                    logger.warning(f"Basic TTM model unavailable: {e}")

                # Try enhanced TTM models
                try:
                    self.models['TTM_Ensemble'] = TTMEnsembleModel()
                except (ImportError, TTMLibraryError, ModelNotAvailableError) as e:
                    logger.warning(f"TTM Ensemble model unavailable: {e}")

                try:
                    self.models['TTM_Augmented'] = TTMAugmentedModel()
                except (ImportError, TTMLibraryError, ModelNotAvailableError) as e:
                    logger.warning(f"TTM Augmented model unavailable: {e}")

                # Add fine-tuning model if we have sufficient data
                if len(self.train) >= 30:
                    try:
                        self.models['TTM_FineTuned'] = TTMFineTunedModel()
                    except (ImportError, TTMLibraryError, ModelNotAvailableError) as e:
                        logger.warning(f"TTM Fine-tuned model unavailable: {e}")

            else:
                # Try to add basic TTM model
                try:
                    self.models['TTM_ZeroShot'] = TTMModel()
                except (ImportError, TTMLibraryError, ModelNotAvailableError) as e:
                    logger.warning(f"TTM library not available: {e}")
                    logger.info("Continuing with non-TTM models only")

        except Exception as e:
            logger.error(f"Error loading TTM models: {e}")
            logger.info("Continuing with SeasonalNaive and NaiveBayes only")

        logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")

    def run_all_models(self):
        """Run all individual models"""
        logger.info("Running all forecasting models...")

        for name, model in self.models.items():
            try:
                logger.info(f"Running {name} forecast...")
                result = model.fit_and_forecast(
                    self.train, self.test, self.forecast_horizon
                )
                self.results[name] = result

            except Exception as e:
                logger.error(f"{name} failed: {e}")
                # Continue with other models

        # Create ensemble forecasts
        if len(self.results) >= 2:
            logger.info("Creating ensemble forecasts...")
            ensemble_results = self.ensemble_methods.create_ensemble_forecasts(
                self.results, self.test
            )
            self.results.update(ensemble_results)

            logger.info("Adding confidence intervals...")
            self.ensemble_methods.add_confidence_intervals(self.results, ensemble_results)

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
        if not self.results:
            return None

        best_mae = float('inf')
        best_model = None
        for name, res in self.results.items():
            if res['metrics']['MAE'] < best_mae:
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
                          output_dir: str = './plots'):
        """Configure plotting behavior"""
        self.visualizer = ForecastVisualizer(
            save_plots=save_plots,
            show_plots=show_plots,
            output_dir=output_dir
        )