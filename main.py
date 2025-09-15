"""
Main entry point for Daily CPU Usage Forecasting
"""

from forecaster import DailyCPUForecaster
from data_loader import DataLoader


def load_and_forecast(data_source: str = './data.json', forecast_horizon: int = 30,
                     use_enhanced_ttm: bool = False, generate_plots: bool = True,
                     save_plots: bool = True, show_plots: bool = False, **kwargs):
    """
    Load data and run all forecasting models

    Args:
        data_source: File path or URL to data
        forecast_horizon: Number of days to forecast
        use_enhanced_ttm: Whether to use enhanced TTM models (fine-tuning, ensemble, augmentation)
        generate_plots: Whether to generate plots at all (for performance)
        save_plots: Whether to save plots to disk
        show_plots: Whether to display plots interactively
        **kwargs: Additional arguments for URL loading (timeout, headers)
    """
    # Load data (auto-detects file vs URL)
    series = DataLoader.load_data(data_source, **kwargs)

    print(f"Loaded {len(series)} days of data from {series.index.min()} to {series.index.max()}")

    # Initialize forecaster and run models
    forecaster = DailyCPUForecaster(series, forecast_horizon=forecast_horizon,
                                   use_enhanced_ttm=use_enhanced_ttm)

    # Configure plotting behavior
    if generate_plots:
        forecaster.configure_plotting(save_plots=save_plots, show_plots=show_plots)

    forecaster.run_all_models()

    # Display results
    print("\nModel Performance Summary:")
    print(forecaster.get_summary().to_string(index=False))

    # Generate plots only if requested
    if generate_plots:
        # Plot results (separate plots for each model)
        forecaster.plot_results()

        # Create interactive HTML plot (with zoom and pan)
        print("\nCreating interactive plot...")
        forecaster.create_interactive_plot()
    else:
        print("\nSkipping plot generation for better performance.")

    # Get best model predictions with metadata
    predictions_result = forecaster.get_best_model_predictions()

    if 'error' in predictions_result:
        print(f"\nError getting predictions: {predictions_result['error']}")
    else:
        metadata = predictions_result.get('metadata', {})
        print(f"\nBest performing model: {metadata.get('model_name', 'Unknown')}")

        if 'model_performance' in metadata:
            perf = metadata['model_performance']
            print(f"Model performance - MAE: {perf['mae']:.3f}, RMSE: {perf['rmse']:.3f}, MAPE: {perf['mape']:.2f}%")

        print(f"Confidence intervals available: {metadata.get('has_confidence_intervals', False)}")
        print(f"Total models compared: {metadata.get('total_models_compared', 0)}")

        print("\nBest model predictions:")
        for prediction in predictions_result['predictions']:
            print(prediction)

    return forecaster


def get_predictions_only(data_source: str = './data.json', forecast_horizon: int = 30,
                        use_enhanced_ttm: bool = False, **kwargs):
    """
    Fast prediction-only function that skips all plotting for maximum performance

    Args:
        data_source: File path or URL to data
        forecast_horizon: Number of days to forecast
        use_enhanced_ttm: Whether to use enhanced TTM models
        **kwargs: Additional arguments for URL loading (timeout, headers)

    Returns:
        Dict with predictions and metadata from best performing model
    """
    forecaster = load_and_forecast(
        data_source=data_source,
        forecast_horizon=forecast_horizon,
        use_enhanced_ttm=use_enhanced_ttm,
        generate_plots=False,
        **kwargs
    )
    return forecaster.get_best_model_predictions()


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
