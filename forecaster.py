"""
Main forecaster class that orchestrates all models
"""

import pandas as pd
from typing import Dict
from models import SeasonalNaiveModel, TTMModel, NaiveBayesModel
from ensemble import EnsembleMethods
from visualization import ForecastVisualizer


class DailyCPUForecaster:
    """Main forecaster class that coordinates all models and ensemble methods"""

    def __init__(self, data: pd.Series, forecast_horizon: int = 30, test_size: int = 14):
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

        # Initialize models
        self.models = {
            'SeasonalNaive': SeasonalNaiveModel(),
            'TTM_ZeroShot': TTMModel(),
            'NaiveBayes': NaiveBayesModel()
        }

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
        """Plot forecasting results"""
        self.visualizer.plot_results(
            self.data, self.results, self.forecast_horizon, self.test_size
        )

    def plot_model_comparison(self):
        """Plot model performance comparison"""
        self.visualizer.plot_model_comparison(self.results)

    def get_best_model(self) -> str:
        """Get the name of the best performing model"""
        return self.visualizer.get_best_model(self.results)