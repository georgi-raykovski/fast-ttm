"""
Constants and shared messages for TTM Forecasting System
"""

# Model bounds
CPU_USAGE_MIN = 0.0
CPU_USAGE_MAX = 100.0

# Confidence interval settings
CONFIDENCE_LEVEL = 0.95
CONFIDENCE_Z_SCORE = 1.96  # For 95% confidence interval

# TTM Model settings
TTM_DEFAULT_CONTEXT_LENGTH = 512

# Error messages moved to utils/exceptions.py for better exception handling

# Seasonal patterns for testing
DEFAULT_SEASONAL_PATTERNS = [7, 14, 30]

# Default forecast settings
DEFAULT_FORECAST_HORIZON = 30
DEFAULT_TEST_SIZE = 14

# Validation settings
MIN_DATA_POINTS_FOR_RELIABLE_FORECAST = 7

# Caching settings
DEFAULT_CACHE_MAX_AGE = 3600  # 1 hour in seconds
FORECAST_CACHE_MAX_AGE = 1800  # 30 minutes for forecasts
DATA_CACHE_MAX_AGE = 7200  # 2 hours for data
