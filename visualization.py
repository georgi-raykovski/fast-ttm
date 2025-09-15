"""
Visualization utilities for forecasting results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from typing import Dict


class ForecastVisualizer:
    """Visualization utilities for forecasting results"""

    def __init__(self):
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def get_best_model(self, results: Dict) -> str:
        """Return the name of the best performing individual model"""
        best_mae = float('inf')
        best_model = None
        for name, res in results.items():
            if 'Ensemble' not in name:  # Only individual models
                if res['metrics']['MAE'] < best_mae:
                    best_mae = res['metrics']['MAE']
                    best_model = name
        return best_model

    def plot_results(self, data: pd.Series, results: Dict, forecast_horizon: int,
                    test_size: int) -> None:
        """Plot historical data and forecasting results with confidence intervals"""

        plt.figure(figsize=(16, 8))

        # Plot historical data
        plt.plot(data.index, data.values, label='Historical CPU', color='blue', linewidth=2)

        # Create future dates
        future_dates = pd.date_range(
            data.index[-1] + timedelta(days=1),
            periods=forecast_horizon
        )

        # Create test dates
        test_dates = data.index[-test_size:]

        # Get best individual model name
        best_model = self.get_best_model(results)

        for name, res in results.items():
            # Only plot ensemble models and best individual model to avoid clutter
            if 'Ensemble' in name or name == best_model:
                # Plot test forecasts
                plt.plot(test_dates, res['test_forecast'],
                        label=f'{name} Test', linestyle='--', linewidth=1.5)

                # Plot future forecasts
                plt.plot(future_dates, res['future_forecast'],
                        label=f'{name} Future', linestyle=':', linewidth=1.5)

                # Add confidence intervals for ensemble models
                if 'test_upper' in res:
                    plt.fill_between(test_dates, res['test_lower'], res['test_upper'],
                                   alpha=0.2, label=f'{name} 95% CI')
                    plt.fill_between(future_dates, res['future_lower'], res['future_upper'],
                                   alpha=0.2)

        plt.title("Daily CPU Forecasts with Confidence Intervals", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("CPU Usage (%)", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_model_comparison(self, results: Dict) -> None:
        """Create a bar chart comparing model performance"""

        models = []
        maes = []
        rmses = []
        mapes = []

        for name, res in results.items():
            models.append(name)
            maes.append(res['metrics']['MAE'])
            rmses.append(res['metrics']['RMSE'])
            mapes.append(res['metrics']['MAPE'])

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # MAE
        ax1.bar(range(len(models)), maes, color='skyblue')
        ax1.set_title('Mean Absolute Error (MAE)')
        ax1.set_ylabel('MAE')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')

        # RMSE
        ax2.bar(range(len(models)), rmses, color='lightcoral')
        ax2.set_title('Root Mean Square Error (RMSE)')
        ax2.set_ylabel('RMSE')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')

        # MAPE
        ax3.bar(range(len(models)), mapes, color='lightgreen')
        ax3.set_title('Mean Absolute Percentage Error (MAPE)')
        ax3.set_ylabel('MAPE (%)')
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels(models, rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

    def create_summary_table(self, results: Dict) -> pd.DataFrame:
        """Create a summary DataFrame of model performance"""
        summary = []
        for name, res in results.items():
            summary.append({
                'Model': name,
                'MAE': round(res['metrics']['MAE'], 3),
                'RMSE': round(res['metrics']['RMSE'], 3),
                'MAPE': round(res['metrics']['MAPE'], 2)
            })

        df = pd.DataFrame(summary).sort_values('MAE')
        return df