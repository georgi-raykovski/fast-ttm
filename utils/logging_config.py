"""
Centralized logging configuration for TTM Forecasting System
"""

import logging
import sys
from pathlib import Path
from decouple import config


def setup_logging(
    log_level: str = None,
    log_file: str = None,
    enable_console: bool = True,
    format_string: str = None
) -> logging.Logger:
    """
    Set up centralized logging configuration

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        enable_console: Whether to log to console
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance
    """

    # Get configuration from environment or use defaults
    log_level = log_level or config('LOG_LEVEL', default='INFO')
    log_file = log_file or config('LOG_FILE', default=None)

    # Default format string
    if not format_string:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Create logger
    logger = logging.getLogger('ttm_forecaster')
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance for a specific module

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f'ttm_forecaster.{name}')
    return logging.getLogger('ttm_forecaster')


# Initialize default logger
default_logger = setup_logging()