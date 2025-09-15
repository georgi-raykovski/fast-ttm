# Daily CPU Usage Forecasting

A modular time series forecasting system for CPU usage prediction using multiple models and ensemble methods.

## Features

- **Multiple Models**: Seasonal Naive, TTM Zero-Shot, and Naive Bayes
- **Ensemble Methods**: Simple average, weighted average, and median ensemble
- **Confidence Intervals**: 95% prediction intervals for ensemble models
- **Production-Ready API**: FastAPI web service with health checks and monitoring
- **Professional Logging**: Configurable logging system with structured output
- **Error Handling**: Custom exceptions with detailed error context
- **Modular Architecture**: Clean separation of models, utilities, and services
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

# Create interactive plot
forecaster.create_interactive_plot()
```

## Architecture

```
├── main.py                 # Entry point
├── forecaster.py          # Main orchestrator class
├── visualization.py       # Plotting utilities
├── app.py                 # FastAPI web service
├── models/                # Model classes
│   ├── __init__.py
│   ├── base_model.py      # Base model interface
│   ├── ensemble.py        # Ensemble methods
│   ├── seasonal_naive.py  # Seasonal naive model
│   ├── ttm_model.py       # TTM zero-shot model
│   ├── ttm_enhanced.py    # Enhanced TTM models
│   └── naive_bayes_model.py # Naive Bayes model
├── utils/                 # Shared utilities
│   ├── constants.py       # Application constants
│   ├── data_loader.py     # Data loading utilities
│   ├── exceptions.py      # Custom exceptions
│   ├── logging_config.py  # Logging configuration
│   └── metrics.py         # Shared metrics utilities
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
- Structured logging with configurable levels
- Professional error handling with custom exceptions
- Health and performance monitoring endpoints

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

## Production Deployment

The application includes a FastAPI web service ready for production deployment.

### Quick Start with Docker

```bash
# Build and run the container
docker-compose up -d

# Check application health
curl http://localhost:8000/health

# View logs
docker-compose logs -f ttm-api
```

### Production Server Options

#### Option 1: Docker (Recommended)
```bash
# Production deployment
docker-compose up -d

# Scale workers
docker-compose up -d --scale ttm-api=3
```

#### Option 2: Gunicorn (Direct)
```bash
# Install dependencies
pip install -r requirements.txt

# Run production server
gunicorn -c gunicorn.conf.py app:app
```

#### Option 3: Development Server
```bash
# For testing only
python app.py
```

### API Endpoints

- **POST /forecast** - Generate forecasts for CPU/memory/IO metrics
- **POST /forecast/test** - Test with local data files
- **POST /forecast/batch** - Batch forecasting for multiple metrics
- **GET /health** - Health check with system metrics
- **GET /metrics** - Detailed system monitoring
- **GET /models** - Available forecasting models

### Configuration

Copy `.env.example` to `.env` and configure for your environment:

```bash
# Production example
DATA_BASE_URL=https://your-metrics-api.com
API_DEBUG=false
LOG_LEVEL=INFO
WORKERS=4
```

### Monitoring

The application provides built-in monitoring:

- **Health Check**: `GET /health` - Application status and metrics
- **System Metrics**: `GET /metrics` - CPU, memory, disk usage
- **Docker Health Check**: Automatic container health monitoring
