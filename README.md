# Daily CPU Usage Forecasting

A modular time series forecasting system for CPU usage prediction using multiple models and ensemble methods.

## Features

- **Multiple Models**: Seasonal Naive, TTM Zero-Shot, and Naive Bayes
- **Ensemble Methods**: Simple average, weighted average, and median ensemble
- **Confidence Intervals**: 95% prediction intervals for ensemble models
- **Modular Architecture**: Clean separation of models, ensemble methods, and visualization
- **Automatic Model Selection**: Finds the best seasonal patterns and parameters

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from main import load_and_forecast

# Load from local file
forecaster = load_and_forecast('./data.json', forecast_horizon=30)

# Load from URL
forecaster = load_and_forecast('https://api.example.com/cpu-data', forecast_horizon=30)

# Load from URL with custom headers/timeout
forecaster = load_and_forecast(
    'https://api.example.com/cpu-data',
    forecast_horizon=30,
    timeout=60,
    headers={'Authorization': 'Bearer your-token'}
)
```

### Advanced Usage

```python
import pandas as pd
from forecaster import DailyCPUForecaster

# Load your data
series = pd.Series(data, index=dates)

# Initialize forecaster
forecaster = DailyCPUForecaster(series, forecast_horizon=30, test_size=14)

# Run all models
forecaster.run_all_models()

# Get results
print(forecaster.get_summary())
forecaster.plot_results()
forecaster.plot_model_comparison()
```

## Architecture

```
├── main.py                 # Entry point
├── forecaster.py          # Main orchestrator class
├── ensemble.py            # Ensemble methods
├── visualization.py       # Plotting utilities
├── models/                # Model classes
│   ├── __init__.py
│   ├── base_model.py      # Base model interface
│   ├── seasonal_naive.py  # Seasonal naive model
│   ├── ttm_model.py       # TTM zero-shot model
│   └── naive_bayes_model.py # Naive Bayes model
└── requirements.txt       # Dependencies
```

## Models

### Seasonal Naive
- Tests multiple seasonal patterns (7, 14, 30 days)
- Automatically selects best performing pattern
- Simple and robust baseline

### TTM Zero-Shot
- Uses pre-trained Tiny Time Mixer model
- Handles context length requirements via data repetition
- State-of-the-art zero-shot forecasting

### Naive Bayes
- Uses lag features (1, 2, 3, 7, 14, 30 days)
- Bins target variable for classification
- Handles missing historical data gracefully

## Ensemble Methods

1. **Simple Average**: Equal weight to all models
2. **Weighted Average**: Weights based on inverse MAE
3. **Median**: Robust to outlier predictions

## Data Format

The system supports multiple JSON formats:

### Format 1: Array of Objects
```json
[
  {"date": "2024-01-01", "value": 45.2},
  {"date": "2024-01-02", "value": 52.1},
  ...
]
```

### Format 2: Object with Data Array
```json
{
  "data": [
    {"date": "2024-01-01", "value": 45.2},
    {"date": "2024-01-02", "value": 52.1},
    ...
  ]
}
```

### Format 3: Separate Arrays
```json
{
  "dates": ["2024-01-01", "2024-01-02", ...],
  "values": [45.2, 52.1, ...]
}
```

### URL Data Sources

You can load data from any URL that returns JSON in one of the above formats:

```python
# Public API
forecaster = load_and_forecast('https://api.example.com/metrics/cpu')

# With authentication
forecaster = load_and_forecast(
    'https://api.private.com/metrics',
    headers={'Authorization': 'Bearer your-token'}
)

# With custom timeout
forecaster = load_and_forecast(
    'https://slow-api.com/data',
    timeout=120
)
```

## Results

The system provides:
- Performance metrics (MAE, RMSE, MAPE) for all models
- Visual plots with confidence intervals
- Model comparison charts
- Interactive HTML plots with zoom/pan capabilities
- Automatic plot saving to `./plots/` directory
- Best model recommendations

## Visualization Features

### Static Plots (Matplotlib)
- **Individual model plots**: Each model gets its own clear subplot
- **Ensemble plots**: With confidence intervals
- **Overview comparison**: All models on one chart
- **Performance bar charts**: MAE, RMSE, MAPE comparison

### Interactive Plots (Plotly)
- **Zoom and pan**: Full interactivity with mouse controls
- **HTML output**: Saved as `./plots/interactive_forecast.html`
- **Auto-opens in browser**: No scrolling limitations
- **Hover tooltips**: Detailed information on data points

### Plot Files
All plots are automatically saved to `./plots/`:
- `individual_models.png`
- `ensemble_models.png`
- `overview_comparison.png`
- `model_comparison.png`
- `interactive_forecast.html` (interactive)