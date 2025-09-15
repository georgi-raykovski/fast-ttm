"""
Custom exceptions for TTM Forecasting System
"""


class TTMForecastingError(Exception):
    """Base exception for TTM forecasting system"""
    pass


class ModelNotAvailableError(TTMForecastingError):
    """Raised when a required model library is not available"""

    def __init__(self, model_name: str, install_command: str = None):
        self.model_name = model_name
        self.install_command = install_command

        message = f"{model_name} library not available"
        if install_command:
            message += f". Install with: {install_command}"

        super().__init__(message)


class TTMLibraryError(ModelNotAvailableError):
    """Raised when TTM library is not available"""

    def __init__(self):
        super().__init__(
            model_name="TTM library (tsfm_public)",
            install_command="pip install git+https://github.com/IBM/tsfm.git"
        )


class DataValidationError(TTMForecastingError):
    """Raised when data validation fails"""
    pass


class InsufficientDataError(DataValidationError):
    """Raised when there's insufficient data for reliable forecasting"""

    def __init__(self, available_points: int, required_points: int):
        self.available_points = available_points
        self.required_points = required_points
        super().__init__(
            f"Insufficient data: {available_points} points available, "
            f"but {required_points} required for reliable forecasting"
        )


class ModelFittingError(TTMForecastingError):
    """Raised when model fitting fails"""

    def __init__(self, model_name: str, reason: str = None):
        self.model_name = model_name
        self.reason = reason

        message = f"Failed to fit {model_name} model"
        if reason:
            message += f": {reason}"

        super().__init__(message)


class ForecastingError(TTMForecastingError):
    """Raised when forecasting fails"""

    def __init__(self, model_name: str, reason: str = None):
        self.model_name = model_name
        self.reason = reason

        message = f"Forecasting failed for {model_name}"
        if reason:
            message += f": {reason}"

        super().__init__(message)


class MetricsCalculationError(TTMForecastingError):
    """Raised when metrics calculation fails"""
    pass