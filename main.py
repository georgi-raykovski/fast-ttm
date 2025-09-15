"""
Main entry point for Daily CPU Usage Forecasting
"""

from forecaster import DailyCPUForecaster
from data_loader import DataLoader


def load_and_forecast(data_source: str = './data.json', forecast_horizon: int = 30,
                     use_enhanced_ttm: bool = False, **kwargs):
    """
    Load data and run all forecasting models

    Args:
        data_source: File path or URL to data
        forecast_horizon: Number of days to forecast
        use_enhanced_ttm: Whether to use enhanced TTM models (fine-tuning, ensemble, augmentation)
        **kwargs: Additional arguments for URL loading (timeout, headers)
    """
    # Load data (auto-detects file vs URL)
    series = DataLoader.load_data(data_source, **kwargs)

    print(f"Loaded {len(series)} days of data from {series.index.min()} to {series.index.max()}")

    # Initialize forecaster and run models
    forecaster = DailyCPUForecaster(series, forecast_horizon=forecast_horizon,
                                   use_enhanced_ttm=use_enhanced_ttm)
    forecaster.run_all_models()

    # Display results
    print("\nModel Performance Summary:")
    print(forecaster.get_summary().to_string(index=False))

    # Plot results (separate plots for each model) - saves but doesn't show
    forecaster.plot_results()

    # Create interactive HTML plot (with zoom and pan) - saves but doesn't show
    print("\nCreating interactive plot...")
    forecaster.create_interactive_plot()

    # Get best model predictions
    best_model_name = forecaster.get_best_model()
    print(f"\nBest performing model: {best_model_name}")

    best_predictions = forecaster.get_best_model_predictions()
    print("\nBest model predictions:")
    for prediction in best_predictions:
        print(prediction)

    return forecaster


def load_and_forecast_from_url(url: str, forecast_horizon: int = 30,
                              timeout: int = 30, headers: dict = None):
    """
    Convenience function to load data from URL and run forecasting

    Args:
        url: URL to fetch data from
        forecast_horizon: Number of days to forecast
        timeout: Request timeout in seconds
        headers: Optional HTTP headers
    """
    return load_and_forecast(url, forecast_horizon, timeout=timeout, headers=headers)


if __name__ == "__main__":
    forecaster = load_and_forecast("./data.json", use_enhanced_ttm=True)
