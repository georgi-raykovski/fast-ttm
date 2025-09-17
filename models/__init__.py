"""
Models package for time series forecasting
"""

from .base_model import BaseModel
from .seasonal_naive import SeasonalNaiveModel
from .feature_rich_naive_bayes import FeatureRichNaiveBayesModel
from .exponential_smoothing_model import ExponentialSmoothingModel

# Optional models - import with graceful error handling
try:
    from .prophet_model import ProphetModel
    PROPHET_AVAILABLE = True
except ImportError:
    ProphetModel = None
    PROPHET_AVAILABLE = False

try:
    from .lightgbm_quantile_model import LightGBMQuantileModel
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError):  # OSError for macOS libomp issue
    LightGBMQuantileModel = None
    LIGHTGBM_AVAILABLE = False

# TTM models - commented out due to insufficient context length requirements
# from .ttm_model import TTMModel
# from .ttm_enhanced import TTMFineTunedModel, TTMAugmentedModel, TTMEnsembleModel

__all__ = [
    'BaseModel', 'SeasonalNaiveModel', 'FeatureRichNaiveBayesModel',
    'ExponentialSmoothingModel'
]

# Add optional models to __all__ if available
if PROPHET_AVAILABLE:
    __all__.append('ProphetModel')
if LIGHTGBM_AVAILABLE:
    __all__.append('LightGBMQuantileModel')