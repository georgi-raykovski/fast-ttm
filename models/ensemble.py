"""
Enhanced Ensemble methods for combining multiple forecasting models with improved boundaries
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from utils.metrics import calculate_metrics, clip_cpu_forecasts
from utils.constants import CONFIDENCE_Z_SCORE


class EnsembleMethods:
    """Enhanced methods for combining forecasts from multiple models with intelligent boundaries"""


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

    @staticmethod
    def percentile_ensemble(forecasts: List[np.ndarray], percentile: float = 50.0) -> np.ndarray:
        """Percentile-based ensemble for robust forecasting"""
        return np.percentile(forecasts, percentile, axis=0)

    @staticmethod
    def trimmed_mean_ensemble(forecasts: List[np.ndarray], trim_fraction: float = 0.2) -> np.ndarray:
        """Trimmed mean ensemble (removes extreme values)"""
        from scipy.stats import trim_mean
        return np.array([trim_mean(forecasts[:, i], trim_fraction) for i in range(forecasts.shape[1])])

    @staticmethod
    def performance_weighted_ensemble(forecasts: List[np.ndarray], performance_scores: List[float]) -> np.ndarray:
        """Ensemble with weights based on inverse performance scores (lower is better)"""
        # Convert performance scores to weights (inverse relationship)
        weights = [1.0 / (score + 1e-8) for score in performance_scores]
        weights = np.array(weights) / np.sum(weights)  # Normalize
        return np.average(forecasts, axis=0, weights=weights)

    def create_ensemble_forecasts(self, results: Dict, test_data: pd.Series) -> Dict:
        """Create enhanced ensemble forecasts with intelligent boundaries"""
        ensemble_results = {}

        # Extract individual forecasts
        test_forecasts = []
        future_forecasts = []
        model_names = []
        model_maes = []
        model_rmses = []
        model_mapes = []

        for name, result in results.items():
            if 'Ensemble' not in name:  # Only use individual models
                test_forecasts.append(result['test_forecast'])
                future_forecasts.append(result['future_forecast'])
                model_names.append(name)
                model_maes.append(result['metrics']['MAE'])
                model_rmses.append(result['metrics']['RMSE'])
                model_mapes.append(result['metrics']['MAPE'])

        if len(test_forecasts) < 2:
            return ensemble_results

        test_forecasts = np.array(test_forecasts)
        future_forecasts = np.array(future_forecasts)

        # 1. Simple Average
        avg_test = self.simple_average(test_forecasts)
        avg_future = self.simple_average(future_forecasts)

        ensemble_results['Ensemble_Average'] = {
            'metrics': calculate_metrics(test_data, avg_test),
            'test_forecast': avg_test,
            'future_forecast': avg_future
        }

        # 2. Performance-Weighted Average (using combined performance score)
        combined_scores = [mae + rmse + mape for mae, rmse, mape in zip(model_maes, model_rmses, model_mapes)]
        perf_weighted_test = self.performance_weighted_ensemble(test_forecasts, combined_scores)
        perf_weighted_future = self.performance_weighted_ensemble(future_forecasts, combined_scores)

        ensemble_results['Ensemble_PerformanceWeighted'] = {
            'metrics': calculate_metrics(test_data, perf_weighted_test),
            'test_forecast': perf_weighted_test,
            'future_forecast': perf_weighted_future
        }

        # 3. Robust Median (robust to outliers)
        median_test = self.median_ensemble(test_forecasts)
        median_future = self.median_ensemble(future_forecasts)

        ensemble_results['Ensemble_Robust'] = {
            'metrics': calculate_metrics(test_data, median_test),
            'test_forecast': median_test,
            'future_forecast': median_future
        }

        # 4. Trimmed Mean (removes extreme predictions)
        try:
            trimmed_test = self.trimmed_mean_ensemble(test_forecasts, trim_fraction=0.2)
            trimmed_future = self.trimmed_mean_ensemble(future_forecasts, trim_fraction=0.2)

            ensemble_results['Ensemble_TrimmedMean'] = {
                'metrics': calculate_metrics(test_data, trimmed_test),
                'test_forecast': trimmed_test,
                'future_forecast': trimmed_future
            }
        except Exception as e:
            print(f"Trimmed mean ensemble failed: {e}")

        # 5. Best Model Selection (dynamic ensemble)
        best_model_idx = np.argmin(model_maes)
        best_test = test_forecasts[best_model_idx]
        best_future = future_forecasts[best_model_idx]

        ensemble_results['Ensemble_BestModel'] = {
            'metrics': calculate_metrics(test_data, best_test),
            'test_forecast': best_test,
            'future_forecast': best_future,
            'best_model': model_names[best_model_idx]
        }

        return ensemble_results

    def add_confidence_intervals(self, results: Dict, ensemble_results: Dict) -> None:
        """Add enhanced confidence intervals with adaptive boundaries"""
        # Get individual model predictions for variance calculation
        test_forecasts = []
        future_forecasts = []
        model_performance = []

        for name, result in results.items():
            if 'Ensemble' not in name:  # Only individual models
                test_forecasts.append(result['test_forecast'])
                future_forecasts.append(result['future_forecast'])
                model_performance.append(result['metrics']['MAE'])

        if len(test_forecasts) > 1:
            test_forecasts = np.array(test_forecasts)
            future_forecasts = np.array(future_forecasts)

            # Calculate multiple types of uncertainty measures
            # 1. Standard deviation across models
            test_std = np.std(test_forecasts, axis=0)
            future_std = np.std(future_forecasts, axis=0)

            # 2. Percentile-based intervals (more robust)
            test_q25 = np.percentile(test_forecasts, 25, axis=0)
            test_q75 = np.percentile(test_forecasts, 75, axis=0)
            future_q25 = np.percentile(future_forecasts, 25, axis=0)
            future_q75 = np.percentile(future_forecasts, 75, axis=0)

            # 3. Performance-weighted uncertainty
            performance_weights = np.array([1.0 / (mae + 1e-8) for mae in model_performance])
            performance_weights = performance_weights / np.sum(performance_weights)

            # Weighted variance calculation
            weighted_mean_test = np.average(test_forecasts, axis=0, weights=performance_weights)
            weighted_mean_future = np.average(future_forecasts, axis=0, weights=performance_weights)

            weighted_var_test = np.average((test_forecasts - weighted_mean_test)**2, axis=0, weights=performance_weights)
            weighted_var_future = np.average((future_forecasts - weighted_mean_future)**2, axis=0, weights=performance_weights)

            weighted_std_test = np.sqrt(weighted_var_test)
            weighted_std_future = np.sqrt(weighted_var_future)

            # Add enhanced confidence intervals to ensemble results
            for name, result in ensemble_results.items():
                # Choose interval type based on ensemble method
                if 'Robust' in name or 'Median' in name:
                    # Use percentile-based intervals for robust methods
                    test_width = (test_q75 - test_q25) / 1.35  # IQR to std approximation
                    future_width = (future_q75 - future_q25) / 1.35
                elif 'PerformanceWeighted' in name:
                    # Use weighted uncertainty for performance-weighted methods
                    test_width = weighted_std_test
                    future_width = weighted_std_future
                else:
                    # Use standard deviation for other methods
                    test_width = test_std
                    future_width = future_std

                # Apply adaptive confidence multiplier
                confidence_multiplier = self._adaptive_confidence_multiplier(len(test_forecasts), model_performance)

                # Calculate boundaries with enhanced clipping
                test_lower = self._enhanced_clipping(result['test_forecast'] - confidence_multiplier * test_width)
                test_upper = self._enhanced_clipping(result['test_forecast'] + confidence_multiplier * test_width)
                future_lower = self._enhanced_clipping(result['future_forecast'] - confidence_multiplier * future_width)
                future_upper = self._enhanced_clipping(result['future_forecast'] + confidence_multiplier * future_width)

                result['test_lower'] = test_lower
                result['test_upper'] = test_upper
                result['future_lower'] = future_lower
                result['future_upper'] = future_upper

                # Add uncertainty metrics
                result['uncertainty_metrics'] = {
                    'avg_test_width': float(np.mean(test_upper - test_lower)),
                    'avg_future_width': float(np.mean(future_upper - future_lower)),
                    'max_test_width': float(np.max(test_upper - test_lower)),
                    'max_future_width': float(np.max(future_upper - future_lower)),
                    'confidence_level': 0.95,
                    'method': 'enhanced_adaptive'
                }

    def _adaptive_confidence_multiplier(self, n_models: int, performance_scores: List[float]) -> float:
        """Calculate adaptive confidence multiplier based on model count and performance"""
        # Base multiplier for 95% confidence
        base_multiplier = CONFIDENCE_Z_SCORE

        # Adjust based on number of models (more models = more confidence)
        model_adjustment = 1.0 - 0.1 * min(n_models - 2, 3) / 3  # Max 10% reduction

        # Adjust based on performance variance (high variance = less confidence)
        if len(performance_scores) > 1:
            perf_cv = np.std(performance_scores) / np.mean(performance_scores)
            variance_adjustment = 1.0 + 0.2 * min(perf_cv, 1.0)  # Max 20% increase
        else:
            variance_adjustment = 1.0

        return base_multiplier * model_adjustment * variance_adjustment

    def _enhanced_clipping(self, values: np.ndarray) -> np.ndarray:
        """Enhanced clipping with intelligent boundaries"""
        # Apply standard CPU forecasting bounds
        clipped = clip_cpu_forecasts(values)

        # Additional intelligent bounds based on typical CPU usage patterns
        # Ensure lower bound is never negative
        clipped = np.maximum(clipped, 0.0)

        # Ensure upper bound respects typical maximum CPU usage
        clipped = np.minimum(clipped, 100.0)

        # Apply smoothing to prevent unrealistic jumps
        if len(clipped) > 1:
            # Limit rate of change between consecutive points
            max_change_rate = 20.0  # Max 20% change between consecutive points
            for i in range(1, len(clipped)):
                max_change = max_change_rate
                if clipped[i] > clipped[i-1] + max_change:
                    clipped[i] = clipped[i-1] + max_change
                elif clipped[i] < clipped[i-1] - max_change:
                    clipped[i] = clipped[i-1] - max_change

        return clipped
