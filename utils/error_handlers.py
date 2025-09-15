"""
Error handling utilities for TTM Forecasting API
"""

import time
import traceback
import uuid
from typing import Dict, Any, Optional
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from utils.logging_config import get_logger
from utils.exceptions import (
    TTMForecastingError, DataValidationError, InsufficientDataError,
    ModelNotAvailableError, TTMLibraryError, ForecastingError
)

logger = get_logger(__name__)


class ErrorDetail:
    """Standard error detail structure"""

    def __init__(self,
                 error_type: str,
                 message: str,
                 details: Optional[Dict[str, Any]] = None,
                 suggestions: Optional[list] = None):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        self.suggestions = suggestions or []

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON response"""
        result = {
            "error_type": self.error_type,
            "message": self.message,
            "timestamp": time.time()
        }

        if self.details:
            result["details"] = self.details

        if self.suggestions:
            result["suggestions"] = self.suggestions

        return result


class APIErrorHandler:
    """Centralized API error handling"""

    @staticmethod
    def handle_validation_error(error: DataValidationError, request: Request = None) -> JSONResponse:
        """Handle data validation errors"""
        error_detail = ErrorDetail(
            error_type="VALIDATION_ERROR",
            message=str(error),
            suggestions=[
                "Check that your data is a valid pandas Series",
                "Ensure data has sufficient points for forecasting",
                "Verify date format is correct"
            ]
        )

        logger.warning(f"Validation error: {error}")
        return JSONResponse(
            status_code=400,
            content=error_detail.to_dict()
        )

    @staticmethod
    def handle_insufficient_data_error(error: InsufficientDataError, request: Request = None) -> JSONResponse:
        """Handle insufficient data errors"""
        error_detail = ErrorDetail(
            error_type="INSUFFICIENT_DATA",
            message=str(error),
            details={
                "available_points": error.available_points,
                "required_points": error.required_points,
                "shortage": error.required_points - error.available_points
            },
            suggestions=[
                f"Provide at least {error.required_points} data points",
                "Consider reducing the test_size parameter",
                "Use a shorter forecast horizon"
            ]
        )

        logger.warning(f"Insufficient data: {error}")
        return JSONResponse(
            status_code=400,
            content=error_detail.to_dict()
        )

    @staticmethod
    def handle_model_unavailable_error(error: ModelNotAvailableError, request: Request = None) -> JSONResponse:
        """Handle model not available errors"""
        error_detail = ErrorDetail(
            error_type="MODEL_UNAVAILABLE",
            message=str(error),
            details={
                "model_name": error.model_name,
                "install_command": error.install_command
            },
            suggestions=[
                f"Install the required library: {error.install_command}" if error.install_command else "Install required dependencies",
                "Use alternative models that are available",
                "Check system requirements"
            ]
        )

        logger.error(f"Model unavailable: {error}")
        return JSONResponse(
            status_code=503,
            content=error_detail.to_dict()
        )

    @staticmethod
    def handle_forecasting_error(error: ForecastingError, request: Request = None) -> JSONResponse:
        """Handle forecasting execution errors"""
        error_detail = ErrorDetail(
            error_type="FORECASTING_ERROR",
            message=str(error),
            details={
                "model_name": error.model_name,
                "reason": error.reason
            },
            suggestions=[
                "Try with different model parameters",
                "Check data quality and format",
                "Use alternative forecasting models",
                "Contact support if error persists"
            ]
        )

        logger.error(f"Forecasting error: {error}")
        return JSONResponse(
            status_code=422,
            content=error_detail.to_dict()
        )

    @staticmethod
    def handle_generic_ttm_error(error: TTMForecastingError, request: Request = None) -> JSONResponse:
        """Handle generic TTM forecasting errors"""
        error_detail = ErrorDetail(
            error_type="FORECASTING_SYSTEM_ERROR",
            message=str(error),
            suggestions=[
                "Check input parameters",
                "Verify system resources",
                "Try again with different settings"
            ]
        )

        logger.error(f"TTM system error: {error}")
        return JSONResponse(
            status_code=500,
            content=error_detail.to_dict()
        )

    @staticmethod
    def handle_unexpected_error(error: Exception, request: Request = None) -> JSONResponse:
        """
        Handle unexpected errors with environment-aware information disclosure.

        In production, sensitive details are hidden to prevent information leakage.
        In development, full details are shown for debugging.
        """
        # Generate unique error ID for tracking
        error_id = str(uuid.uuid4())

        # Always log full details internally for debugging
        logger.error(f"Unexpected error [{error_id}]: {error}")
        logger.error(f"Error class: {error.__class__.__name__}")
        logger.error(f"Full traceback: {traceback.format_exc()}")

        # Check if we're in debug mode (you can also check environment variables)
        try:
            from settings import Settings
            settings = Settings()
            is_debug = settings.API_DEBUG
        except:
            # Fallback: assume production if settings unavailable
            is_debug = False

        # Prepare error details based on environment
        if is_debug:
            # Development: show detailed error information
            error_details = {
                "error_id": error_id,
                "error_class": error.__class__.__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc().split('\n')
            }
            message = f"Development Error: {str(error)}"
            suggestions = [
                "Check the traceback for debugging information",
                "Verify input parameters and data format",
                "Check logs for additional context"
            ]
        else:
            # Production: hide sensitive details
            error_details = {
                "error_id": error_id,
                "timestamp": time.time()
            }
            message = "An unexpected error occurred while processing your request"
            suggestions = [
                "Try your request again",
                f"Contact support with error ID: {error_id}",
                "Ensure all required parameters are provided correctly"
            ]

        error_detail = ErrorDetail(
            error_type="INTERNAL_SERVER_ERROR",
            message=message,
            details=error_details,
            suggestions=suggestions
        )

        return JSONResponse(
            status_code=500,
            content=error_detail.to_dict()
        )

    @staticmethod
    def create_error_handler(error_class, handler_method):
        """Create a custom error handler for FastAPI"""
        async def error_handler(request: Request, exc: error_class):
            return handler_method(exc, request)
        return error_handler


def get_error_handlers():
    """Get all error handlers for FastAPI registration"""
    handler = APIErrorHandler()

    return {
        DataValidationError: handler.create_error_handler(DataValidationError, handler.handle_validation_error),
        InsufficientDataError: handler.create_error_handler(InsufficientDataError, handler.handle_insufficient_data_error),
        ModelNotAvailableError: handler.create_error_handler(ModelNotAvailableError, handler.handle_model_unavailable_error),
        TTMLibraryError: handler.create_error_handler(TTMLibraryError, handler.handle_model_unavailable_error),
        ForecastingError: handler.create_error_handler(ForecastingError, handler.handle_forecasting_error),
        TTMForecastingError: handler.create_error_handler(TTMForecastingError, handler.handle_generic_ttm_error),
        Exception: handler.create_error_handler(Exception, handler.handle_unexpected_error)
    }


# Quick retry decorator for external API calls
def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry function on failure"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")

            # Re-raise the last exception if all retries failed
            raise last_exception
        return wrapper
    return decorator