"""
Ensemble methods for combining multiple forecasting models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


class EnsembleMethods:
    """Methods for combining forecasts from multiple models"""

    @staticmethod
    def calculate_metrics(y_true: np.array, y_pred: np.array) -> Dict:
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    @staticmethod
    def simple_average(forecasts: List[np.ndarray]) -> np.ndarray:
        """Simple average of all forecasts"""
        return np.mean(forecasts, axis=0)

    @staticmethod
    def weighted_average(forecasts: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """Weighted average based on model performance"""
        weights = np.array(weights) / np.sum(weights)  # Normalize
        return np.average(forecasts, axis=0, weights=weights)

    @staticmethod
    def median_ensemble(forecasts: List[np.ndarray]) -> np.ndarray:
        """Median ensemble (robust to outliers)"""
        return np.median(forecasts, axis=0)

    def create_ensemble_forecasts(self, results: Dict, test_data: pd.Series) -> Dict:
        """Create ensemble forecasts from individual model results"""
        ensemble_results = {}

        # Extract individual forecasts
        test_forecasts = []
        future_forecasts = []
        model_names = []
        model_maes = []

        for name, result in results.items():
            if 'Ensemble' not in name:  # Only use individual models
                test_forecasts.append(result['test_forecast'])
                future_forecasts.append(result['future_forecast'])
                model_names.append(name)
                model_maes.append(result['metrics']['MAE'])

        if len(test_forecasts) < 2:
            return ensemble_results

        test_forecasts = np.array(test_forecasts)
        future_forecasts = np.array(future_forecasts)

        # 1. Simple Average
        avg_test = self.simple_average(test_forecasts)
        avg_future = self.simple_average(future_forecasts)

        ensemble_results['Ensemble_Average'] = {
            'metrics': self.calculate_metrics(test_data, avg_test),
            'test_forecast': avg_test,
            'future_forecast': avg_future
        }

        # 2. Weighted Average (inverse MAE weighting)
        weights = [1 / (mae + 1e-8) for mae in model_maes]  # Small epsilon to avoid division by zero
        weighted_test = self.weighted_average(test_forecasts, weights)
        weighted_future = self.weighted_average(future_forecasts, weights)

        ensemble_results['Ensemble_Weighted'] = {
            'metrics': self.calculate_metrics(test_data, weighted_test),
            'test_forecast': weighted_test,
            'future_forecast': weighted_future
        }

        # 3. Median (robust to outliers)
        median_test = self.median_ensemble(test_forecasts)
        median_future = self.median_ensemble(future_forecasts)

        ensemble_results['Ensemble_Median'] = {
            'metrics': self.calculate_metrics(test_data, median_test),
            'test_forecast': median_test,
            'future_forecast': median_future
        }

        return ensemble_results

    def add_confidence_intervals(self, results: Dict, ensemble_results: Dict) -> None:
        """Add confidence intervals to ensemble forecasts"""
        # Get individual model predictions for variance calculation
        test_forecasts = []
        future_forecasts = []

        for name, result in results.items():
            if 'Ensemble' not in name:  # Only individual models
                test_forecasts.append(result['test_forecast'])
                future_forecasts.append(result['future_forecast'])

        if len(test_forecasts) > 1:
            # Calculate standard deviation across models
            test_std = np.std(test_forecasts, axis=0)
            future_std = np.std(future_forecasts, axis=0)

            # Add 95% confidence intervals to ensemble results
            for name, result in ensemble_results.items():
                # 95% confidence intervals (±1.96 std)
                result['test_lower'] = np.clip(result['test_forecast'] - 1.96 * test_std, 0, 100)
                result['test_upper'] = np.clip(result['test_forecast'] + 1.96 * test_std, 0, 100)
                result['future_lower'] = np.clip(result['future_forecast'] - 1.96 * future_std, 0, 100)
                result['future_upper'] = np.clip(result['future_forecast'] + 1.96 * future_std, 0, 100)
