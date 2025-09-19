"""
Main entry point for Exponential Smoothing Time Series Forecasting
"""

from typing import List
from forecaster import ExponentialSmoothingForecaster
from utils.data_loader import DataLoader
from utils.logging_config import get_logger

logger = get_logger(__name__)


def load_and_forecast(data_source: str = './data.json', forecast_horizon: int = 30,
                     test_split: float = 0.3, generate_plots: bool = True,
                     save_plots: bool = True, show_plots: bool = False,
                     parallel_cv: bool = True, **kwargs):
    """
    Load data and run Exponential Smoothing forecasting

    Args:
        data_source: File path or URL to data
        forecast_horizon: Number of days to forecast
        test_split: Fraction of data to use for testing (default 0.3 = 30%)
        generate_plots: Whether to generate plots at all (for performance)
        save_plots: Whether to save plots to disk
        show_plots: Whether to display plots interactively
        parallel_cv: Whether to use parallel cross-validation for better performance
        **kwargs: Additional arguments for URL loading (timeout, headers)
    """
    # Load data (auto-detects file vs URL)
    series = DataLoader.load_data(data_source, **kwargs)

    logger.info(f"Loaded {len(series)} days of data from {series.index.min()} to {series.index.max()}")

    # Initialize forecaster and run model with configurable test data split and optimizations
    test_size = int(len(series) * test_split)
    forecaster = ExponentialSmoothingForecaster(series, forecast_horizon=forecast_horizon,
                                               test_size=test_size)

    # Configure model optimizations
    forecaster.model.parallel_cv = parallel_cv

    # Configure plotting behavior
    if generate_plots:
        forecaster.configure_plotting(save_plots=save_plots, show_plots=show_plots)

    forecaster.run_forecast()

    # Display results
    logger.info("Exponential Smoothing Performance:")
    summary = forecaster.get_summary().to_string(index=False)
    logger.info(f"\n{summary}")
    print(summary)  # Keep console output for user

    # Generate plots only if requested
    if generate_plots:
        # Plot results (separate plots for each model)
        forecaster.plot_results()

        # Create interactive HTML plot (with zoom and pan)
        logger.info("Creating interactive plot...")
        forecaster.create_interactive_plot()
    else:
        logger.info("Skipping plot generation for better performance.")

    # Get model predictions with metadata
    predictions_result = forecaster.get_model_predictions()

    if 'error' in predictions_result:
        logger.error(f"Error getting predictions: {predictions_result['error']}")
        print(f"\nError getting predictions: {predictions_result['error']}")
    else:
        metadata = predictions_result.get('metadata', {})
        model_name = metadata.get('model_name', 'Unknown')
        logger.info(f"Model: {model_name}")
        print(f"\nModel: {model_name}")

        if 'model_performance' in metadata:
            perf = metadata['model_performance']
            perf_str = f"Model performance - MAE: {perf['mae']:.3f}, RMSE: {perf['rmse']:.3f}, MAPE: {perf['mape']:.2f}%"
            logger.info(perf_str)
            print(perf_str)

        ci_available = metadata.get('has_confidence_intervals', False)
        logger.info(f"Confidence intervals available: {ci_available}")
        print(f"Confidence intervals available: {ci_available}")

        logger.info("Model predictions generated")
        print("\nModel predictions:")
        for prediction in predictions_result['predictions']:
            print(prediction)  # Keep console output for predictions

    return forecaster


def get_predictions_only(data_source: str = './data.json', forecast_horizon: int = 30,
                        test_split: float = 0.3, **kwargs):
    """
    Fast prediction-only function that skips all plotting for maximum performance

    Args:
        data_source: File path or URL to data
        forecast_horizon: Number of days to forecast
        test_split: Fraction of data to use for testing (default 0.3 = 30%)
        **kwargs: Additional arguments for URL loading (timeout, headers)

    Returns:
        Dict with predictions and metadata from Exponential Smoothing model
    """
    forecaster = load_and_forecast(
        data_source=data_source,
        forecast_horizon=forecast_horizon,
        test_split=test_split,
        generate_plots=False,
        parallel_cv=True,  # Enable optimizations for speed
        **kwargs
    )
    return forecaster.get_model_predictions()


def get_batch_predictions(data_source: str = './data.json', horizons: List[int] = [7, 14, 30],
                         test_split: float = 0.3, **kwargs):
    """
    Generate optimized batch predictions for multiple forecast horizons

    Args:
        data_source: File path or URL to data
        horizons: List of forecast horizons in days
        test_split: Fraction of data to use for testing
        **kwargs: Additional arguments for URL loading

    Returns:
        Dict with batch predictions for all horizons
    """
    forecaster = load_and_forecast(
        data_source=data_source,
        forecast_horizon=max(horizons),  # Use max horizon for training
        test_split=test_split,
        generate_plots=False,
        parallel_cv=True,
        **kwargs
    )

    # Get batch forecasts
    batch_results = forecaster.get_batch_forecasts(horizons)

    logger.info(f"Generated batch predictions for horizons: {horizons}")
    return batch_results


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
    forecaster = load_and_forecast("./data.json")
