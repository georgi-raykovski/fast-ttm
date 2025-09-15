"""
Visualization utilities for forecasting results
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from typing import Dict
import os


class ForecastVisualizer:
    """Visualization utilities for forecasting results"""

    def __init__(self, save_plots: bool = True, show_plots: bool = False, output_dir: str = './plots'):
        # Set matplotlib backend to prevent GUI windows when not showing plots
        if not show_plots:
            current_backend = matplotlib.get_backend()
            if current_backend != 'Agg':
                try:
                    matplotlib.use('Agg', force=True)  # Non-interactive backend
                except:
                    # If backend switch fails, we'll still work but might show windows
                    pass

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        # Configure matplotlib for interactivity
        plt.rcParams['figure.figsize'] = [16, 8]
        plt.rcParams['font.size'] = 10

        # Setup output directory
        self.save_plots = save_plots
        self.show_plots = show_plots
        self.output_dir = output_dir
        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)


    def plot_results(self, data: pd.Series, results: Dict, forecast_horizon: int,
                    test_size: int) -> None:
        """Plot historical data and forecasting results with confidence intervals"""

        # Create future dates and test dates
        future_dates = pd.date_range(
            data.index[-1] + timedelta(days=1),
            periods=forecast_horizon
        )
        test_dates = data.index[-test_size:]

        # Separate individual models and ensemble models
        individual_models = {k: v for k, v in results.items() if 'Ensemble' not in k}
        ensemble_models = {k: v for k, v in results.items() if 'Ensemble' in k}

        # Plot 1: Individual Models (separate subplots)
        if individual_models:
            n_models = len(individual_models)
            fig, axes = plt.subplots(n_models, 1, figsize=(16, 4 * n_models))
            if n_models == 1:
                axes = [axes]

            for i, (name, res) in enumerate(individual_models.items()):
                ax = axes[i]

                # Plot historical data
                ax.plot(data.index, data.values, label='Historical CPU',
                       color='blue', linewidth=2)

                # Plot test forecasts
                ax.plot(test_dates, res['test_forecast'],
                       label='Test Forecast', color='red', linestyle='--', linewidth=2)

                # Plot future forecasts
                ax.plot(future_dates, res['future_forecast'],
                       label='Future Forecast', color='green', linestyle=':', linewidth=2)

                ax.set_title(f"{name} Model Forecast", fontsize=14)
                ax.set_xlabel("Date", fontsize=12)
                ax.set_ylabel("CPU Usage (%)", fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.suptitle("Individual Model Forecasts", fontsize=16)
            plt.tight_layout()

            # Enable interactive navigation (only for interactive backends)
            if self.show_plots and matplotlib.get_backend() != 'Agg':
                try:
                    fig_manager = plt.get_current_fig_manager()
                    if fig_manager and hasattr(fig_manager, 'toolbar') and fig_manager.toolbar:
                        fig_manager.toolbar.pan()
                except:
                    # Skip interactive features if they fail
                    pass

            # Save plot
            if self.save_plots:
                plt.savefig(f'{self.output_dir}/individual_models.png', dpi=300, bbox_inches='tight')
                print(f"Individual models plot saved to {self.output_dir}/individual_models.png")

            if self.show_plots:
                plt.show()
            else:
                plt.close(fig)

        # Plot 2: Ensemble Models with Confidence Intervals
        if ensemble_models:
            fig, axes = plt.subplots(len(ensemble_models), 1, figsize=(16, 5 * len(ensemble_models)))
            if len(ensemble_models) == 1:
                axes = [axes]

            for i, (name, res) in enumerate(ensemble_models.items()):
                ax = axes[i]

                # Plot historical data
                ax.plot(data.index, data.values, label='Historical CPU',
                       color='blue', linewidth=2)

                # Plot test forecasts
                ax.plot(test_dates, res['test_forecast'],
                       label='Test Forecast', color='red', linestyle='--', linewidth=2)

                # Plot future forecasts
                ax.plot(future_dates, res['future_forecast'],
                       label='Future Forecast', color='green', linestyle=':', linewidth=2)

                # Add confidence intervals if available
                if 'test_upper' in res:
                    ax.fill_between(test_dates, res['test_lower'], res['test_upper'],
                                   alpha=0.3, color='red', label='Test 95% CI')
                    ax.fill_between(future_dates, res['future_lower'], res['future_upper'],
                                   alpha=0.3, color='green', label='Future 95% CI')

                ax.set_title(f"{name} Forecast with Confidence Intervals", fontsize=14)
                ax.set_xlabel("Date", fontsize=12)
                ax.set_ylabel("CPU Usage (%)", fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.suptitle("Ensemble Model Forecasts", fontsize=16)
            plt.tight_layout()

            # Enable interactive navigation (only for interactive backends)
            if self.show_plots and matplotlib.get_backend() != 'Agg':
                try:
                    fig_manager = plt.get_current_fig_manager()
                    if fig_manager and hasattr(fig_manager, 'toolbar') and fig_manager.toolbar:
                        fig_manager.toolbar.pan()
                except:
                    # Skip interactive features if they fail
                    pass

            # Save plot
            if self.save_plots:
                plt.savefig(f'{self.output_dir}/ensemble_models.png', dpi=300, bbox_inches='tight')
                print(f"Ensemble models plot saved to {self.output_dir}/ensemble_models.png")

            if self.show_plots:
                plt.show()
            else:
                plt.close(fig)


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

        # Save plot
        if self.save_plots:
            plt.savefig(f'{self.output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {self.output_dir}/model_comparison.png")

        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

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

    def create_interactive_plot(self, data: pd.Series, results: Dict, forecast_horizon: int,
                              test_size: int) -> None:
        """Create an interactive HTML plot using plotly (if available)"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.offline as pyo

            # Create subplots
            fig = make_subplots(
                rows=len(results), cols=1,
                subplot_titles=list(results.keys()),
                vertical_spacing=0.05
            )

            # Create dates
            future_dates = pd.date_range(
                data.index[-1] + timedelta(days=1),
                periods=forecast_horizon
            )
            test_dates = data.index[-test_size:]

            for i, (name, res) in enumerate(results.items(), 1):
                # Historical data
                fig.add_trace(
                    go.Scatter(x=data.index, y=data.values, name=f'{name} - Historical',
                              line=dict(color='blue', width=2), showlegend=(i == 1)),
                    row=i, col=1
                )

                # Test forecast
                fig.add_trace(
                    go.Scatter(x=test_dates, y=res['test_forecast'],
                              name=f'{name} - Test', line=dict(color='red', dash='dash'),
                              showlegend=(i == 1)),
                    row=i, col=1
                )

                # Future forecast
                fig.add_trace(
                    go.Scatter(x=future_dates, y=res['future_forecast'],
                              name=f'{name} - Future', line=dict(color='green', dash='dot'),
                              showlegend=(i == 1)),
                    row=i, col=1
                )

                # Confidence intervals if available
                if 'test_upper' in res:
                    fig.add_trace(
                        go.Scatter(x=test_dates, y=res['test_upper'],
                                  mode='lines', line=dict(width=0), showlegend=False),
                        row=i, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=test_dates, y=res['test_lower'],
                                  fill='tonexty', fillcolor='rgba(255,0,0,0.2)',
                                  mode='lines', line=dict(width=0),
                                  name=f'{name} - 95% CI' if i == 1 else '',
                                  showlegend=(i == 1)),
                        row=i, col=1
                    )

                    fig.add_trace(
                        go.Scatter(x=future_dates, y=res['future_upper'],
                                  mode='lines', line=dict(width=0), showlegend=False),
                        row=i, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=future_dates, y=res['future_lower'],
                                  fill='tonexty', fillcolor='rgba(0,255,0,0.2)',
                                  mode='lines', line=dict(width=0), showlegend=False),
                        row=i, col=1
                    )

            # Update layout
            fig.update_layout(
                title="Interactive CPU Forecasting Results",
                height=400 * len(results),
                showlegend=True
            )

            for i in range(len(results)):
                fig.update_yaxes(title_text="CPU Usage (%)", row=i+1, col=1)

            fig.update_xaxes(title_text="Date", row=len(results), col=1)

            # Save as HTML
            if self.save_plots:
                output_file = f'{self.output_dir}/interactive_forecast.html'
                pyo.plot(fig, filename=output_file, auto_open=False)
                print(f"Interactive plot saved to {output_file}")
            elif self.show_plots:
                fig.show()

        except ImportError:
            print("Plotly not installed. Install with: pip install plotly")
            print("Falling back to matplotlib plots.")