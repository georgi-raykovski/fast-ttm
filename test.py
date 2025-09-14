"""
Time Series Forecasting Script for Short Data (60 days)
Tests multiple recommended approaches for limited historical data

Installation Instructions:
-------------------------
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels 

Note: This version doesn't require pmdarima to avoid compatibility issues.
      It uses statsmodels ARIMA with automatic parameter selection instead.
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import seaborn as sns
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Statistical models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class TimeSeriesForecaster:
    """Class to handle time series forecasting with limited data"""
    
    def __init__(self, data: pd.Series, forecast_horizon: int = 7, test_size: int = 14):
        """
        Initialize the forecaster
        
        Parameters:
        -----------
        data : pd.Series
            Time series data with DatetimeIndex
        forecast_horizon : int
            Number of periods to forecast
        test_size : int
            Number of periods to hold out for testing
        """
        self.data = data
        self.forecast_horizon = forecast_horizon
        self.test_size = test_size
        
        # Split data
        self.train = data[:-test_size]
        self.test = data[-test_size:]
        
        # Store results
        self.results = {}
        self.forecasts = {}
        
    def calculate_metrics(self, y_true: np.array, y_pred: np.array) -> Dict:
        """Calculate forecast accuracy metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def simple_moving_average(self, window_sizes: List[int] = [3, 5, 7]) -> Dict:
        """Simple Moving Average forecasting"""
        best_window = None
        best_mae = float('inf')
        
        for window in window_sizes:
            if window > len(self.train) // 2:
                continue
                
            # Calculate moving average
            ma = self.train.rolling(window=window).mean()
            
            # Use last MA value as forecast
            forecast = np.full(len(self.test), ma.iloc[-1])
            
            mae = mean_absolute_error(self.test, forecast)
            if mae < best_mae:
                best_mae = mae
                best_window = window
        
        # Generate final forecast with best window
        ma = self.train.rolling(window=best_window).mean()
        test_forecast = np.full(len(self.test), ma.iloc[-1])
        future_forecast = np.full(self.forecast_horizon, ma.iloc[-1])
        
        metrics = self.calculate_metrics(self.test, test_forecast)
        
        self.results['SMA'] = {
            'metrics': metrics,
            'best_window': best_window,
            'test_forecast': test_forecast,
            'future_forecast': future_forecast
        }
        
        return self.results['SMA']
    
    def exponential_smoothing(self) -> Dict:
        """Exponential Smoothing (ETS) forecasting"""
        try:
            # Try different configurations
            configs = [
                {'trend': None, 'seasonal': None},
                {'trend': 'add', 'seasonal': None},
                {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 7},
            ]
            
            best_config = None
            best_mae = float('inf')
            best_model = None
            
            for config in configs:
                try:
                    if config.get('seasonal') and config.get('seasonal_periods', 7) > len(self.train) // 2:
                        continue
                        
                    model = ExponentialSmoothing(
                        self.train,
                        **config
                    )
                    fitted_model = model.fit(optimized=True)
                    test_forecast = fitted_model.forecast(len(self.test))
                    
                    mae = mean_absolute_error(self.test, test_forecast)
                    if mae < best_mae:
                        best_mae = mae
                        best_config = config
                        best_model = fitted_model
                        
                except:
                    continue
            
            if best_model is None:
                raise Exception("No valid ETS model found")
            
            # Generate forecasts with best model
            test_forecast = best_model.forecast(len(self.test))
            future_forecast = best_model.forecast(self.forecast_horizon)
            
            metrics = self.calculate_metrics(self.test, test_forecast)
            
            self.results['ETS'] = {
                'metrics': metrics,
                'config': best_config,
                'test_forecast': test_forecast,
                'future_forecast': future_forecast
            }
            
        except Exception as e:
            print(f"ETS failed: {e}")
            # Fallback to simple exponential smoothing
            alpha = 0.3
            forecast = [self.train.iloc[0]]
            for i in range(1, len(self.train)):
                forecast.append(alpha * self.train.iloc[i-1] + (1-alpha) * forecast[-1])
            
            test_forecast = np.full(len(self.test), forecast[-1])
            future_forecast = np.full(self.forecast_horizon, forecast[-1])
            
            metrics = self.calculate_metrics(self.test, test_forecast)
            
            self.results['ETS'] = {
                'metrics': metrics,
                'config': {'simple': True, 'alpha': alpha},
                'test_forecast': test_forecast,
                'future_forecast': future_forecast
            }
        
        return self.results['ETS']
    
    def naive_forecast(self, season_length=30) -> Dict:
        """Seasonal Naive forecast: repeats last season's values."""
        last_season = self.train.iloc[-season_length:]

        # Repeat for test set
        test_forecast = np.tile(last_season.values, len(self.test) // season_length + 1)[:len(self.test)]

        # Repeat for future
        future_forecast = np.tile(last_season.values, self.forecast_horizon // season_length + 1)[:self.forecast_horizon]

        # Compute full metrics (MAE, RMSE, MAPE)
        metrics = self.calculate_metrics(self.test, test_forecast)

        self.results['Naive'] = {
            'metrics': metrics,
            'type': 'SeasonalNaive',
            'test_forecast': test_forecast,
            'future_forecast': future_forecast
        }

        return self.results['Naive']

    
    def arima_forecast(self) -> Dict:
        """ARIMA with automatic parameter selection (without pmdarima)"""
        try:
            # Grid search for best ARIMA parameters
            best_aic = float('inf')
            best_order = None
            best_model = None
            
            # Try different combinations (keeping it simple for short series)
            p_values = [0, 1, 2]
            d_values = [0, 1]
            q_values = [0, 1, 2]
            
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        # Skip ARIMA(0,0,0)
                        if p == 0 and d == 0 and q == 0:
                            continue
                        
                        try:
                            model = ARIMA(self.train, order=(p, d, q))
                            fitted_model = model.fit(method_kwargs={"warn_convergence": False})
                            
                            # Use AIC for model selection
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_order = (p, d, q)
                                best_model = fitted_model
                        except:
                            continue
            
            if best_model is None:
                # If no model worked, try simplest ARIMA(1,0,0)
                model = ARIMA(self.train, order=(1, 0, 0))
                best_model = model.fit()
                best_order = (1, 0, 0)
            
            # Generate test forecast
            test_forecast = best_model.forecast(len(self.test))
            
            # Refit on all training data for future forecast
            model_full = ARIMA(self.train, order=best_order)
            fitted_full = model_full.fit(method_kwargs={"warn_convergence": False})
            future_forecast = fitted_full.forecast(self.forecast_horizon)
            
            metrics = self.calculate_metrics(self.test, test_forecast)
            
            self.results['ARIMA'] = {
                'metrics': metrics,
                'order': best_order,
                'AIC': best_aic,
                'test_forecast': test_forecast,
                'future_forecast': future_forecast
            }
            
        except Exception as e:
            print(f"ARIMA failed: {e}")
            # Fallback to naive forecast
            return self.naive_forecast()
        
        return self.results['ARIMA']
    
    def linear_trend_forecast(self) -> Dict:
        """Simple linear regression forecast"""
        try:
            # Create time index
            X = np.arange(len(self.train)).reshape(-1, 1)
            y = self.train.values
            
            # Fit linear regression
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict test set
            X_test = np.arange(len(self.train), len(self.train) + len(self.test)).reshape(-1, 1)
            test_forecast = model.predict(X_test)
            
            # Future forecast
            X_future = np.arange(len(self.train) + len(self.test), 
                               len(self.train) + len(self.test) + self.forecast_horizon).reshape(-1, 1)
            future_forecast = model.predict(X_future)
            
            metrics = self.calculate_metrics(self.test, test_forecast)
            
            self.results['LinearTrend'] = {
                'metrics': metrics,
                'test_forecast': test_forecast,
                'future_forecast': future_forecast,
                'slope': model.coef_[0],
                'intercept': model.intercept_
            }
            
        except Exception as e:
            print(f"Linear trend failed: {e}")
            self.results['LinearTrend'] = self.results.get('Naive', self.naive_forecast())
        
        return self.results['LinearTrend']
    
    def ensemble_forecast(self, models=("SMA", "ETS", "ARIMA"), weights=None):
        forecasts = []
        for model in models:
            forecasts.append(self.results[model]['future_forecast'])

        forecasts = np.array(forecasts)  # shape: (n_models, horizon)

        if weights is None:
            weights = np.ones(len(models)) / len(models)
        else:
            weights = np.array(weights) / np.sum(weights)

        # Weighted average per day across models
        future_forecast = np.average(forecasts, axis=0, weights=weights)

        # Same for test set
        test_forecasts = [self.results[m]['test_forecast'] for m in models]
        test_forecasts = np.array(test_forecasts)
        test_forecast = np.average(test_forecasts, axis=0, weights=weights)

        metrics = self.calculate_metrics(self.test, test_forecast)

        self.results['Ensemble'] = {
            'metrics': metrics,
            'type': f"Models: {', '.join(models)}",
            'test_forecast': test_forecast,
            'future_forecast': future_forecast,
            'models_used': list(models),
        }

        return self.results['Ensemble']
    
    def run_all_models(self):
        """Run all forecasting models"""
        print("\n" + "="*60)
        print("Running forecasting models...")
        print("="*60)
        
        print("1. Simple Moving Average...")
        self.simple_moving_average()
        
        print("2. Exponential Smoothing...")
        self.exponential_smoothing()
        
        print("3. Naive Forecast...")
        self.naive_forecast()
        
        print("4. Linear Trend...")
        self.linear_trend_forecast()
        
        print("5. ARIMA (Grid Search)...")
        self.arima_forecast()
        
        
        print("7. Creating Ensemble...")
        self.ensemble_forecast()
    
    def plot_results(self):
        """Plot forecasting results"""
        n_models = len(self.results)
        n_cols = 2
        n_rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (name, result) in enumerate(self.results.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Plot historical data
            ax.plot(self.train.index, self.train.values, 
                   label='Training Data', color='blue', alpha=0.7)
            ax.plot(self.test.index, self.test.values, 
                   label='Actual Test Data', color='green', alpha=0.7)
            
            # Plot test forecast
            ax.plot(self.test.index, result['test_forecast'], 
                   label=f'{name} Forecast', color='red', alpha=0.7)
            
            # Add future forecast
            future_dates = pd.date_range(
                start=self.data.index[-1] + timedelta(days=1),
                periods=self.forecast_horizon,
                freq=pd.infer_freq(self.data.index) or 'D'
            )
            ax.plot(future_dates, result['future_forecast'], 
                   label='Future Forecast', color='orange', linestyle='--')
            
            
            ax.set_title(f'{name} (MAE: {result["metrics"]["MAE"]:.2f})')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            
        # Hide any unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
            
        plt.tight_layout()
        plt.show()
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all model performances"""
        summary_data = []
        
        for name, result in self.results.items():
            row = {
                'Model': name,
                'MAE': result['metrics']['MAE'],
                'RMSE': result['metrics']['RMSE'],
                'MAPE': result['metrics']['MAPE']
            }
            
            # Add model-specific info
            if name == 'SMA':
                row['Details'] = f"Window: {result['best_window']}"
            elif name == 'ARIMA':
                row['Details'] = f"Order: {result.get('order', 'N/A')}"
            elif name == 'ETS':
                row['Details'] = f"Config: {result['config']}"
            elif name == 'Naive':
                row['Details'] = f"Type: {result['type']}"
            elif name == 'LinearTrend':
                row['Details'] = f"Slope: {result.get('slope', 0):.4f}"
            elif name == 'Ensemble':
                row['Details'] = f"Models: {', '.join(result['models_used'])}"
            else:
                row['Details'] = ''
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('MAE')
        summary_df['Rank'] = range(1, len(summary_df) + 1)
        
        return summary_df


def generate_sample_data(n_days: int = 60, seed: int = 42) -> pd.Series:
    """
    Generate sample time series data for testing
    
    Parameters:
    -----------
    n_days : int
        Number of days of data to generate
    seed : int
        Random seed for reproducibility
    """
    np.random.seed(seed)
    
    # Create date range
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='D')
    
    # Generate synthetic time series with trend and seasonality
    trend = np.linspace(100, 120, n_days)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_days) / 7)  # Weekly pattern
    noise = np.random.normal(0, 5, n_days)
    
    values = trend + seasonal + noise
    values = np.maximum(values, 0)  # Ensure non-negative
    
    return pd.Series(values, index=dates, name='value')


def check_stationarity(data: pd.Series) -> Dict:
    """Check if time series is stationary using Augmented Dickey-Fuller test"""
    result = adfuller(data.dropna())
    
    return {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4],
        'is_stationary': result[1] < 0.05
    }


def main():
    """Main execution function"""
    print("=" * 60)
    print("Time Series Forecasting with 60 Days of Data")
    print("=" * 60)
    
    # Generate or load your data
    print("\nGenerating sample data...")
    # data = generate_sample_data(n_days=60)
    with open('./data.json', 'r') as f:
            data = json.load(f)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Create time series with date index
    data = pd.Series(df['value'].values, index=df['date'], name='value')

    print(f"Data shape: {len(data)} days")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"\nData statistics:")
    print(f"  Mean: {data.mean():.2f}")
    print(f"  Std: {data.std():.2f}")
    print(f"  Min: {data.min():.2f}")
    print(f"  Max: {data.max():.2f}")
    
    # Check stationarity
    print("\nChecking stationarity...")
    stationarity = check_stationarity(data)
    print(f"  ADF Statistic: {stationarity['ADF Statistic']:.4f}")
    print(f"  p-value: {stationarity['p-value']:.4f}")
    print(f"  Is stationary: {stationarity['is_stationary']}")
    
    # Initialize forecaster
    print("\nInitializing forecaster...")
    forecaster = TimeSeriesForecaster(
        data=data,
        forecast_horizon=7,  # Forecast 7 days ahead
        test_size=14  # Use last 14 days for testing
    )
    
    # Run all models
    forecaster.run_all_models()
    
    # Get summary
    print("\n" + "=" * 60)
    print("Model Performance Summary")
    print("=" * 60)
    summary = forecaster.get_summary()
    print(summary.to_string(index=False))
    
    # Plot results
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)
    forecaster.plot_results()
    
    # Get best model
    best_model = summary.iloc[0]['Model']
    print(f"\n{'='*60}")
    print(f"Best performing model: {best_model}")
    print(f"{'='*60}")
    print(f"MAE: {summary.iloc[0]['MAE']:.2f}")
    print(f"RMSE: {summary.iloc[0]['RMSE']:.2f}")
    print(f"MAPE: {summary.iloc[0]['MAPE']:.2f}%")
    
    # Print future forecasts from best model
    print("\n" + "=" * 60)
    print(f"7-Day Forecast from {best_model}")
    print("=" * 60)
    
    future_dates = pd.date_range(
        start=data.index[-1] + timedelta(days=1),
        periods=7,
        freq='D'
    )
    
    best_forecast = forecaster.results[best_model]['future_forecast']
    
    for date, value in zip(future_dates, best_forecast):
        print(f"{date.date()}: {value:.2f}")
    
    return forecaster, summary


# Example usage with custom data
def example_with_custom_data():
    with open('./data.json', 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"Loaded data: {len(df)} points from {df['date'].min()} to {df['date'].max()}")

    # ✅ Convert to Series with DatetimeIndex
    series = pd.Series(df['value'].values, index=df['date'], name='value')

    # Run forecasting
    forecaster = TimeSeriesForecaster(series, forecast_horizon=7, test_size=14)
    forecaster.run_all_models()
    
    summary = forecaster.get_summary()
    print(summary)
    
    return forecaster


if __name__ == "__main__":
    # Run main analysis
    forecaster, summary = main()
    
    # Uncomment to run with custom data
    custom_forecaster = example_with_custom_data()
