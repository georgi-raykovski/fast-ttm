"""
Main forecaster class that orchestrates all models
"""

import pandas as pd
from typing import Dict
from models import (SeasonalNaiveModel, TTMModel, NaiveBayesModel,
                     TTMFineTunedModel, TTMAugmentedModel, TTMEnsembleModel)
from ensemble import EnsembleMethods
from visualization import ForecastVisualizer


class DailyCPUForecaster:
    """Main forecaster class that coordinates all models and ensemble methods"""

    def __init__(self, data: pd.Series, forecast_horizon: int = 30, test_size: int = 14,
                 use_enhanced_ttm: bool = False):
        self.data = data
        self.forecast_horizon = forecast_horizon
        self.test_size = test_size

        # Split data
        self.train = data[:-test_size]
        self.test = data[-test_size:]

        # Initialize components
        self.results = {}
        self.ensemble_methods = EnsembleMethods()
        self.visualizer = ForecastVisualizer()

        # Initialize models with TTM error handling
        self.models = {
            'SeasonalNaive': SeasonalNaiveModel(),
            'NaiveBayes': NaiveBayesModel()
        }

        # Try to add TTM models with graceful fallback
        try:
            if use_enhanced_ttm:
                print("Attempting to load enhanced TTM models...")

                # Try basic TTM first
                try:
                    self.models['TTM_ZeroShot'] = TTMModel()
                except ImportError as e:
                    print(f"Basic TTM model unavailable: {e}")

                # Try enhanced TTM models
                try:
                    self.models['TTM_Ensemble'] = TTMEnsembleModel()
                except ImportError as e:
                    print(f"TTM Ensemble model unavailable: {e}")

                try:
                    self.models['TTM_Augmented'] = TTMAugmentedModel()
                except ImportError as e:
                    print(f"TTM Augmented model unavailable: {e}")

                # Add fine-tuning model if we have sufficient data
                if len(self.train) >= 30:
                    try:
                        self.models['TTM_FineTuned'] = TTMFineTunedModel()
                    except ImportError as e:
                        print(f"TTM Fine-tuned model unavailable: {e}")

            else:
                # Try to add basic TTM model
                try:
                    self.models['TTM_ZeroShot'] = TTMModel()
                except ImportError as e:
                    print(f"TTM library not available: {e}")
                    print("Continuing with non-TTM models only")

        except Exception as e:
            print(f"Error loading TTM models: {e}")
            print("Continuing with SeasonalNaive and NaiveBayes only")

        print(f"Loaded {len(self.models)} models: {list(self.models.keys())}")

    def run_all_models(self):
        """Run all individual models"""
        print("\nRunning all forecasting models...")

        for name, model in self.models.items():
            try:
                print(f"\nRunning {name} Forecast...")
                result = model.fit_and_forecast(
                    self.train, self.test, self.forecast_horizon
                )
                self.results[name] = result

            except Exception as e:
                print(f"{name} failed: {e}")
                # Continue with other models

        # Create ensemble forecasts
        if len(self.results) >= 2:
            print("\nCreating Ensemble Forecasts...")
            ensemble_results = self.ensemble_methods.create_ensemble_forecasts(
                self.results, self.test
            )
            self.results.update(ensemble_results)

            print("\nAdding Confidence Intervals...")
            self.ensemble_methods.add_confidence_intervals(self.results, ensemble_results)

    def get_summary(self) -> pd.DataFrame:
        """Get summary table of model performance"""
        return self.visualizer.create_summary_table(self.results)

    def plot_results(self):
        """Plot individual model forecasts (separate plots)"""
        self.visualizer.plot_results(
            self.data, self.results, self.forecast_horizon, self.test_size
        )

    def plot_overview(self):
        """Plot overview with all models for comparison"""
        self.visualizer.plot_overview(
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
        """Get the name of the best performing model"""
        return self.visualizer.get_best_model(self.results)

    def get_best_model_predictions(self) -> list:
        """Get predictions from the best performing model in {date, value} format"""
        best_model_name = self.get_best_model()
        if not best_model_name or best_model_name not in self.results:
            return []

        best_result = self.results[best_model_name]

        # Create future dates
        import pandas as pd
        from datetime import timedelta

        future_dates = pd.date_range(
            self.data.index[-1] + timedelta(days=1),
            periods=self.forecast_horizon
        )

        # Format as list of dictionaries
        predictions = []
        for date, value in zip(future_dates, best_result['future_forecast']):
            predictions.append({
                'date': date.strftime('%Y-%m-%d'),
                'value': float(value)
            })

        return predictions