#!/usr/bin/env python3
"""
Zero-shot Time Series Forecasting with Classical Methods

This script implements robust forecasting methods that work well with limited data:
1. Simple Moving Average - Very robust with limited data
2. Exponential Smoothing (ETS) - Works well with 2-3 months of data
3. Naive/Seasonal Naive - Uses recent values as forecasts
4. Simple Linear Regression - For clear trend patterns

These methods don't require training and work immediately with any time series data.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


class ZeroShotTimeSeriesForecaster:
    """
    Zero-shot forecasting using classical methods that don't require training
    """

    def __init__(self, data_path='data.json'):
        """Initialize forecaster and load data"""
        self.data_path = data_path
        self.df = self.load_data()
        self.results = {}

    def load_data(self):
        """Load time series data from JSON"""
        with open(self.data_path, 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        print(f"Loaded data: {len(df)} points from {df['date'].min()} to {df['date'].max()}")
        return df

    def simple_moving_average(self, window=7, forecast_horizon=7):
        """
        Simple Moving Average forecast
        Very robust with limited data, uses last N values to predict
        """
        values = self.df['value'].values

        if len(values) < window:
            window = max(1, len(values) // 2)

        # Use last 'window' values to compute average
        last_values = values[-window:]
        forecast_value = float(np.mean(last_values))

        # Generate forecast dates
        last_date = self.df['date'].iloc[-1]
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_horizon)]

        # Simple MA assumes constant forecast
        forecast_values = [forecast_value] * forecast_horizon

        return {
            'method': 'Simple Moving Average',
            'window': window,
            'dates': forecast_dates,
            'values': forecast_values,
            'parameters': f'window={window}'
        }

    def exponential_smoothing(self, alpha=0.3, beta=0.1, forecast_horizon=7):
        """
        Simple Exponential Smoothing with trend (Holt's method)
        Works well with 2-3 months of data
        """
        values = self.df['value'].values
        n = len(values)

        if n < 2:
            # Fallback to naive method
            return self.naive_forecast(forecast_horizon)

        # Initialize level and trend
        level = values[0]
        trend = values[1] - values[0] if n > 1 else 0

        # Apply exponential smoothing
        for i in range(1, n):
            prev_level = level
            level = alpha * values[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend

        # Generate forecasts
        last_date = self.df['date'].iloc[-1]
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_horizon)]
        forecast_values = [float(level + (i+1) * trend) for i in range(forecast_horizon)]

        return {
            'method': 'Exponential Smoothing',
            'dates': forecast_dates,
            'values': forecast_values,
            'parameters': f'alpha={alpha}, beta={beta}'
        }

    def naive_forecast(self, forecast_horizon=7, seasonal=False, season_length=7):
        """
        Enhanced Naive forecasting methods for capturing highs and lows
        - Simple: Use last value
        - Seasonal: Use historical seasonal patterns with volatility preservation
        """
        values = self.df['value'].values

        if not seasonal or len(values) < season_length:
            # Simple naive: repeat last value
            forecast_value = float(values[-1])
            forecast_values = [forecast_value] * forecast_horizon
        else:
            # Non-repetitive seasonal approach using full dataset knowledge
            forecast_values = []
            n = len(values)

            # Learn patterns from entire dataset, not just cycles
            # 1. Overall data characteristics
            global_mean = np.mean(values)
            global_std = np.std(values)
            global_trend = np.polyfit(np.arange(n), values, 1)[0]

            # 2. Volatility evolution over entire dataset
            volatility_windows = []
            window_size = max(7, n // 10)
            for i in range(window_size, n, max(1, window_size//2)):
                vol = np.std(values[i-window_size:i])
                volatility_windows.append(vol)

            current_vol = volatility_windows[-1] if volatility_windows else global_std

            # 3. Analyze high/low frequencies across entire dataset
            # Find when highs and lows typically occur
            high_threshold = np.percentile(values, 75)
            low_threshold = np.percentile(values, 25)

            high_prob_by_position = np.zeros(season_length)
            low_prob_by_position = np.zeros(season_length)

            for i, val in enumerate(values):
                pos = i % season_length
                if val >= high_threshold:
                    high_prob_by_position[pos] += 1
                elif val <= low_threshold:
                    low_prob_by_position[pos] += 1

            # Normalize probabilities
            cycle_count = max(1, n // season_length)
            high_prob_by_position /= cycle_count
            low_prob_by_position /= cycle_count

            print(f"  Using full dataset: {n} points, global trend: {global_trend:.3f}")
            print(f"  High/low probabilities learned from {cycle_count} cycles")

            for i in range(forecast_horizon):
                day_in_season = i % season_length

                # Start with trend projection
                base_value = values[-1] + global_trend * (i + 1)

                # Add mean reversion
                mean_reversion = (global_mean - base_value) * 0.1

                # Probabilistic high/low bias based on learned patterns
                high_bias = high_prob_by_position[day_in_season] * current_vol * 0.8
                low_bias = -low_prob_by_position[day_in_season] * current_vol * 0.8

                # Combine biases (they can partially cancel out)
                position_bias = high_bias + low_bias

                # Add realistic volatility (not repeating exact values)
                volatility_component = np.random.normal(0, current_vol * 0.4)

                # Momentum from recent data
                if i == 0:
                    momentum = (values[-1] - values[-3]) * 0.3 if n >= 3 else 0
                else:
                    momentum = (forecast_values[i-1] - values[-1]) * 0.2

                forecast_value = (base_value + mean_reversion + position_bias +
                                volatility_component + momentum)

                # Ensure reasonable bounds
                data_range = values.max() - values.min()
                forecast_value = np.clip(forecast_value,
                                       values.min() - data_range * 0.2,
                                       values.max() + data_range * 0.2)

                forecast_values.append(float(forecast_value))

        last_date = self.df['date'].iloc[-1]
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_horizon)]

        method_name = f"{'Full Dataset Seasonal ' if seasonal else ''}Naive"
        params = f"full_dataset, global_learning" if seasonal else "last_value"

        return {
            'method': method_name,
            'dates': forecast_dates,
            'values': forecast_values,
            'parameters': params
        }

    def linear_regression_forecast(self, forecast_horizon=7):
        """
        Simple Linear Regression for trend patterns
        Fits a line through the time series data
        """
        values = self.df['value'].values
        n = len(values)

        # Create time features (days since start)
        X = np.arange(n).reshape(-1, 1)
        y = values

        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)

        # Generate forecasts
        future_X = np.arange(n, n + forecast_horizon).reshape(-1, 1)
        forecast_values = [float(x) for x in model.predict(future_X)]

        last_date = self.df['date'].iloc[-1]
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_horizon)]

        return {
            'method': 'Linear Regression',
            'dates': forecast_dates,
            'values': forecast_values,
            'parameters': f'slope={model.coef_[0]:.2f}, intercept={model.intercept_:.2f}'
        }

    def volatility_aware_seasonal_forecast(self, forecast_horizon=7, season_length=7):
        """
        Non-repetitive volatility forecasting that learns from entire dataset
        Uses local patterns, global trends, and stochastic modeling
        """
        values = self.df['value'].values
        n = len(values)

        if n < 20:  # Need enough data
            return self.naive_forecast(forecast_horizon, seasonal=False)

        forecast_values = []

        # Learn from entire dataset, not just recent patterns
        # 1. Global trend analysis
        x = np.arange(n)
        global_trend = np.polyfit(x, values, 1)[0]  # Linear trend over all data

        # 2. Volatility regimes - analyze how volatility changes over time
        window_size = max(7, n // 8)  # Dynamic window size
        volatility_series = []

        for i in range(window_size, n):
            window_vol = np.std(values[i-window_size:i])
            volatility_series.append(window_vol)

        current_volatility = volatility_series[-1] if volatility_series else np.std(values)
        volatility_trend = 0
        if len(volatility_series) > 10:
            vol_x = np.arange(len(volatility_series))
            volatility_trend = np.polyfit(vol_x, volatility_series, 1)[0]

        # 3. Learn multi-scale patterns (not just weekly)
        # Short-term momentum (3-day)
        short_momentum = (values[-3:].mean() - values[-6:-3].mean()) if n >= 6 else 0

        # Medium-term trend (2-week)
        medium_trend = (values[-7:].mean() - values[-14:-7].mean()) / 7 if n >= 14 else global_trend

        # 4. Identify volatility clusters and persistence
        recent_changes = np.abs(np.diff(values[-min(20, n):]))
        volatility_persistence = np.corrcoef(recent_changes[:-1], recent_changes[1:])[0,1] if len(recent_changes) > 1 else 0

        print(f"  Global trend: {global_trend:.3f}/day, Medium trend: {medium_trend:.3f}/day")
        print(f"  Current volatility: {current_volatility:.2f}, Vol trend: {volatility_trend:.4f}")
        print(f"  Volatility persistence: {volatility_persistence:.3f}")

        # 5. Generate non-repetitive forecasts
        base_level = values[-1]  # Start from last observed value

        for i in range(forecast_horizon):
            day_ahead = i + 1

            # Combine multiple trend components
            trend_component = (global_trend * 0.3 + medium_trend * 0.5 + short_momentum * 0.2) * day_ahead

            # Evolving volatility based on learned patterns
            current_vol = current_volatility + volatility_trend * day_ahead
            current_vol = max(current_vol, current_volatility * 0.5)  # Don't let volatility go too low

            # Stochastic component that doesn't repeat
            # Use different noise sources and scales
            base_noise = np.random.normal(0, current_vol * 0.3)

            # Add momentum-based variation (volatility clustering)
            if i > 0:
                momentum_factor = volatility_persistence * 0.3
                prev_change = forecast_values[i-1] - (forecast_values[i-2] if i > 1 else base_level)
                momentum_noise = momentum_factor * prev_change * np.random.uniform(-0.5, 0.5)
            else:
                momentum_noise = 0

            # Occasional "regime switches" - dramatic changes that break patterns
            regime_switch = 0
            if np.random.random() < 0.05:  # 5% chance of regime switch
                regime_direction = 1 if np.random.random() < 0.5 else -1
                regime_switch = regime_direction * current_vol * np.random.uniform(1, 2)

            # Mean reversion force - prevents forecasts from going too extreme
            current_mean = np.mean(values)
            mean_reversion = (current_mean - base_level) * 0.02 * day_ahead

            # Combine all components
            forecast_value = (base_level + trend_component + base_noise +
                            momentum_noise + regime_switch + mean_reversion)

            # Ensure realistic bounds
            data_range = values.max() - values.min()
            forecast_value = np.clip(forecast_value,
                                   values.min() - data_range * 0.3,
                                   values.max() + data_range * 0.3)

            forecast_values.append(float(forecast_value))

        last_date = self.df['date'].iloc[-1]
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_horizon)]

        return {
            'method': 'Stochastic Volatility Forecast',
            'dates': forecast_dates,
            'values': forecast_values,
            'parameters': f'non_repetitive, global_trend={global_trend:.3f}'
        }

    def evaluate_method_on_historical(self, method_func, test_size=7, **kwargs):
        """
        Evaluate a forecasting method on historical data with focus on highs/lows
        """
        if len(self.df) <= test_size:
            return None

        # Split data
        train_df = self.df[:-test_size].copy()
        test_values = self.df['value'].values[-test_size:]

        # Temporarily replace data for method evaluation
        original_df = self.df
        self.df = train_df

        try:
            # Get forecast
            result = method_func(forecast_horizon=test_size, **kwargs)
            predicted = result['values'][:test_size]

            # Standard metrics
            mse = float(mean_squared_error(test_values, predicted))
            mae = float(mean_absolute_error(test_values, predicted))
            rmse = float(np.sqrt(mse))

            # High/Low specific metrics
            test_max = max(test_values)
            test_min = min(test_values)
            pred_max = max(predicted)
            pred_min = min(predicted)

            # How well does it predict the range?
            range_error = abs((test_max - test_min) - (pred_max - pred_min))
            range_accuracy = 1 - (range_error / max(test_max - test_min, 1))

            # How well does it predict extremes?
            max_error = abs(test_max - pred_max)
            min_error = abs(test_min - pred_min)
            extreme_accuracy = 1 - ((max_error + min_error) / (2 * (test_max - test_min + 1)))

            # Volatility preservation
            test_volatility = np.std(test_values)
            pred_volatility = np.std(predicted)
            volatility_ratio = min(pred_volatility, test_volatility) / max(pred_volatility, test_volatility)

            # Restore original data
            self.df = original_df

            return {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'range_accuracy': max(0, range_accuracy),
                'extreme_accuracy': max(0, extreme_accuracy),
                'volatility_preservation': volatility_ratio,
                'test_range': test_max - test_min,
                'pred_range': pred_max - pred_min,
                'method': result['method']
            }
        except Exception as e:
            self.df = original_df
            return None

    def forecast_all_methods(self, forecast_horizon=7):
        """
        Run all forecasting methods and return results
        """
        # Only keep methods that can predict highs and lows
        methods = [
            ('full_dataset_seasonal', self.naive_forecast, {'seasonal': True, 'season_length': 7}),
            ('stochastic_volatility', self.volatility_aware_seasonal_forecast, {'season_length': 7}),
        ]

        results = {}

        print(f"\nGenerating {forecast_horizon}-day forecasts using all methods:")
        print("-" * 60)

        for method_key, method_func, params in methods:
            try:
                # Get forecast
                forecast = method_func(forecast_horizon=forecast_horizon, **params)

                # Evaluate on historical data if possible
                evaluation = self.evaluate_method_on_historical(method_func, **params)

                results[method_key] = {
                    'forecast': forecast,
                    'evaluation': evaluation
                }

                # Print results
                print(f"{forecast['method']:25} | {forecast['parameters']}")
                if evaluation:
                    print(f"{'':25} | RMSE: {evaluation['rmse']:.2f}, MAE: {evaluation['mae']:.2f}")
                    print(f"{'':25} | Range Accuracy: {evaluation['range_accuracy']:.1%}, Extreme Accuracy: {evaluation['extreme_accuracy']:.1%}")
                    print(f"{'':25} | Volatility Preservation: {evaluation['volatility_preservation']:.1%}")
                    print(f"{'':25} | Test Range: {evaluation['test_range']:.1f}, Predicted Range: {evaluation['pred_range']:.1f}")

                forecast_range = max(forecast['values']) - min(forecast['values'])
                print(f"{'':25} | 30-day Forecast Range: {forecast_range:.1f} (Low: {min(forecast['values']):.1f}, High: {max(forecast['values']):.1f})")
                print()

            except Exception as e:
                print(f"{method_key} failed: {e}")
                results[method_key] = None

        return results

    def create_visualization(self, results, forecast_horizon=7):
        """
        Create visualization of historical data and forecasts
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Historical data
        ax1.plot(self.df['date'], self.df['value'], 'k-', linewidth=2, label='Historical Data')
        ax1.set_title('Historical Time Series Data')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Forecasts comparison
        colors = ['red', 'blue', 'green', 'orange', 'brown', 'purple']

        # Plot historical data on forecast chart too
        ax2.plot(self.df['date'], self.df['value'], 'k-', linewidth=2, label='Historical Data', alpha=0.7)

        # Plot each forecast
        for i, (method_key, result) in enumerate(results.items()):
            if result and result['forecast']:
                forecast = result['forecast']
                color = colors[i % len(colors)]

                ax2.plot(forecast['dates'], forecast['values'],
                        color=color, linewidth=2, linestyle='--',
                        marker='o', markersize=4,
                        label=f"{forecast['method']}")

        ax2.set_title(f'{forecast_horizon}-Day Forecasts Comparison')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Value')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('ttm_forecasts.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\nVisualization saved as 'ttm_forecasts.png'")

    def get_best_method(self, results):
        """
        Select the best performing method based on historical evaluation
        """
        valid_methods = [(k, v) for k, v in results.items()
                        if v and v['evaluation'] and v['evaluation']['rmse']]

        if not valid_methods:
            print("No methods could be evaluated on historical data")
            return None

        # Sort by RMSE (lower is better)
        valid_methods.sort(key=lambda x: x[1]['evaluation']['rmse'])
        best_method, best_result = valid_methods[0]

        print(f"\nBest performing method: {best_result['forecast']['method']}")
        print(f"Historical RMSE: {best_result['evaluation']['rmse']:.2f}")
        print(f"Historical MAE: {best_result['evaluation']['mae']:.2f}")

        return best_method, best_result

    def run_forecasting(self, forecast_horizon=7, save_results=True):
        """
        Main method to run all forecasting approaches
        """
        print("="*60)
        print("ZERO-SHOT TIME SERIES FORECASTING")
        print("="*60)
        print(f"Data points: {len(self.df)}")
        print(f"Date range: {self.df['date'].min().date()} to {self.df['date'].max().date()}")
        print(f"Value range: {self.df['value'].min():.1f} to {self.df['value'].max():.1f}")

        # Run all methods
        results = self.forecast_all_methods(forecast_horizon)

        # Find best method
        best = self.get_best_method(results)

        # Create visualization
        self.create_visualization(results, forecast_horizon)

        # Save results
        if save_results:
            output = {
                'timestamp': datetime.now().isoformat(),
                'data_summary': {
                    'points': len(self.df),
                    'date_range': [self.df['date'].min().isoformat(),
                                  self.df['date'].max().isoformat()],
                    'value_range': [float(self.df['value'].min()),
                                   float(self.df['value'].max())]
                },
                'forecast_horizon': forecast_horizon,
                'results': {}
            }

            for method_key, result in results.items():
                if result:
                    # Convert dates to strings for JSON serialization
                    forecast_data = result['forecast'].copy()
                    forecast_data['dates'] = [d.isoformat() for d in forecast_data['dates']]

                    output['results'][method_key] = {
                        'forecast': forecast_data,
                        'evaluation': result['evaluation']
                    }

            with open('ttm_forecast_results.json', 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\nResults saved to 'ttm_forecast_results.json'")

        return results


def main():
    """
    Main function to run zero-shot forecasting
    """
    # Initialize forecaster
    forecaster = ZeroShotTimeSeriesForecaster('data.json')

    # Run forecasting with 30-day horizon to capture highs and lows
    results = forecaster.run_forecasting(forecast_horizon=30)

    print("\n" + "="*60)
    print("FORECASTING COMPLETE!")
    print("="*60)
    print("Check 'ttm_forecasts.png' for visualization")
    print("Check 'ttm_forecast_results.json' for detailed results")


if __name__ == "__main__":
    main()