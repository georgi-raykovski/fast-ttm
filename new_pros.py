"""
Daily CPU Usage Forecasting Script for 60-Day Data
Includes baseline, seasonal naive, Holt-Winters, and peak-aware forecasts

Installation Instructions:
-------------------------
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels 
"""

import numpy as np
import pandas as pd
import warnings
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class DailyCPUForecaster:
    """Forecast daily CPU usage with limited historical data (60 days)"""
    
    def __init__(self, data: pd.Series, forecast_horizon: int = 30, test_size: int = 14):
        self.data = data
        self.forecast_horizon = forecast_horizon
        self.test_size = test_size
        
        # Split train/test
        self.train = data[:-test_size]
        self.test = data[-test_size:]
        
        self.results = {}
        
    def calculate_metrics(self, y_true: np.array, y_pred: np.array) -> Dict:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    def seasonal_naive_forecast(self, season_length: int = 7) -> Dict:
        """Weekly seasonal naive forecast: repeats same weekday"""
        last_season = self.train.iloc[-season_length:]
        
        # Forecast for test set
        repeats = len(self.test) // season_length + 1
        test_forecast = np.tile(last_season.values, repeats)[:len(self.test)]
        
        # Forecast future
        repeats_future = self.forecast_horizon // season_length + 1
        future_forecast = np.tile(last_season.values, repeats_future)[:self.forecast_horizon]
        
        metrics = self.calculate_metrics(self.test, test_forecast)
        
        self.results['SeasonalNaive'] = {
            'metrics': metrics,
            'test_forecast': test_forecast,
            'future_forecast': future_forecast
        }
        return self.results['SeasonalNaive']
    
    def holt_winters_forecast(self, seasonal_periods: int = 7) -> Dict:
        """Holt-Winters weekly seasonality forecast"""
        try:
            model = ExponentialSmoothing(
                self.train,
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods
            )
            fitted = model.fit(optimized=True)
            
            test_forecast = fitted.forecast(len(self.test))
            future_forecast = fitted.forecast(self.forecast_horizon)
            
            metrics = self.calculate_metrics(self.test, test_forecast)
            
            self.results['HoltWinters'] = {
                'metrics': metrics,
                'test_forecast': test_forecast,
                'future_forecast': future_forecast,
                'fitted_model': fitted
            }
            
        except Exception as e:
            print(f"Holt-Winters failed: {e}")
            # Fallback to naive
            return self.seasonal_naive_forecast(season_length=seasonal_periods)
        
        return self.results['HoltWinters']
    
    def quantile_forecast(self, fitted_model, quantile: float = 0.95) -> np.array:
        """Generate upper-bound forecast for peak-aware right-sizing"""
        # Holt-Winters does not directly provide quantiles, so approximate
        residuals = self.train - fitted_model.fittedvalues
        std_dev = np.std(residuals)
        forecast = fitted_model.forecast(self.forecast_horizon)
        upper_bound = forecast + 1.65 * std_dev  # approx 95th percentile
        return upper_bound
    
    def run_all_models(self):
        """Run all forecasting models"""
        print("\nRunning Seasonal Naive Forecast...")
        self.seasonal_naive_forecast()
        
        print("\nRunning Holt-Winters Forecast...")
        hw_result = self.holt_winters_forecast()
        
        print("\nGenerating 95th Percentile Forecast (Peak-Aware)...")
        peak_forecast = self.quantile_forecast(hw_result['fitted_model'], quantile=0.95)
        self.results['HoltWintersPeak95'] = {
            'metrics': self.calculate_metrics(self.test, peak_forecast[:len(self.test)]),
            'test_forecast': peak_forecast[:len(self.test)],
            'future_forecast': peak_forecast
        }
        
    def plot_results(self):
        """Plot all forecasts"""
        plt.figure(figsize=(15,6))
        plt.plot(self.data.index, self.data.values, label='Historical CPU', color='blue')
        
        for name, res in self.results.items():
            # Align test forecast
            plt.plot(self.test.index, res['test_forecast'], label=f'{name} Test Forecast', linestyle='--')
            # Future forecast dates
            future_dates = pd.date_range(self.data.index[-1]+timedelta(days=1),
                                         periods=self.forecast_horizon)
            plt.plot(future_dates, res['future_forecast'], label=f'{name} Future Forecast', linestyle=':')
        
        plt.title("Daily CPU Forecasts")
        plt.xlabel("Date")
        plt.ylabel("CPU Usage")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def get_summary(self) -> pd.DataFrame:
        """Return metrics summary"""
        summary = []
        for name, res in self.results.items():
            summary.append({
                'Model': name,
                'MAE': res['metrics']['MAE'],
                'RMSE': res['metrics']['RMSE'],
                'MAPE': res['metrics']['MAPE']
            })
        return pd.DataFrame(summary).sort_values('MAE')

# Sample usage with JSON data
def load_and_forecast(data_path: str = './data.json', forecast_horizon: int = 30):
    import json
    with open(data_path, 'r') as f:
        raw = json.load(f)
    
    df = pd.DataFrame(raw)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    series = pd.Series(df['value'].values, index=df['date'], name='value')
    
    print(f"Loaded {len(series)} days of data from {series.index.min()} to {series.index.max()}")
    
    forecaster = DailyCPUForecaster(series, forecast_horizon=forecast_horizon)
    forecaster.run_all_models()
    
    print("\nModel Performance Summary:")
    print(forecaster.get_summary().to_string(index=False))
    
    forecaster.plot_results()
    
    return forecaster

# Example run
if __name__ == "__main__":
    forecaster = load_and_forecast()
