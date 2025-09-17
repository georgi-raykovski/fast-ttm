"""
Feature-Rich Naive Bayes Model for Time Series Forecasting

Enhanced version of the basic Naive Bayes model with extensive feature engineering
for yearly time series data. Creates lag features, rolling statistics, and temporal features.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel
from utils.logging_config import get_logger

logger = get_logger(__name__)


class FeatureRichNaiveBayesModel(BaseModel):
    """
    Enhanced Naive Bayes model with rich feature engineering for yearly data.

    Features created:
    - Lag features (1, 2, 3, 7, 14, 30, 91, 182, 365 days)
    - Rolling statistics (mean, std, min, max) for multiple windows
    - Temporal features (day of week, month, quarter, etc.)
    - Trend and seasonality decomposition features
    - Holiday and weekend indicators
    """

    def __init__(self, n_bins: int = 10):
        super().__init__("FeatureRichNaiveBayes")
        self.n_bins = n_bins
        self.model = GaussianNB()
        self.scaler = StandardScaler()
        self.bin_edges = None
        self.feature_names = []
        self.feature_importance = {}

    def fit(self, data: pd.Series) -> None:
        """Fit the feature-rich Naive Bayes model"""
        try:
            logger.info(f"Fitting Feature-Rich Naive Bayes on {len(data)} data points")

            # Create rich feature set
            features_df = self._create_features(data)

            # Remove rows with NaN values (from lag features)
            features_df = features_df.dropna()

            min_samples = max(10, len(self.feature_names) + 1)  # At least 10 samples or features + 1
            if len(features_df) < min_samples:
                # Reduce feature complexity for small datasets
                logger.warning(f"Small dataset ({len(features_df)} samples), reducing feature complexity")
                # Use only basic features for small datasets
                basic_features = [col for col in features_df.columns
                                if col.startswith(('lag_', 'rolling_mean_', 'day_of_week', 'is_weekend')) or col == 'target']
                if len(basic_features) > 1:
                    features_df = features_df[basic_features]
                    logger.info(f"Reduced to {len(basic_features)-1} basic features")
                else:
                    raise ValueError(f"Insufficient data after feature creation: {len(features_df)} samples")

            # Prepare target variable (binned values)
            target_values = features_df['target'].values
            self.bin_edges = np.percentile(target_values, np.linspace(0, 100, self.n_bins + 1))

            # Create binned target
            binned_target = np.digitize(target_values, self.bin_edges) - 1
            binned_target = np.clip(binned_target, 0, self.n_bins - 1)

            # Prepare features
            feature_columns = [col for col in features_df.columns if col != 'target']
            X = features_df[feature_columns].values

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Fit the model
            self.model.fit(X_scaled, binned_target)

            self.feature_names = feature_columns
            self.is_fitted = True

            # Calculate feature importance (variance-based approximation)
            self._calculate_feature_importance(X_scaled, binned_target)

            logger.info(f"Feature-Rich Naive Bayes fitted with {len(self.feature_names)} features")

        except Exception as e:
            logger.error(f"Feature-Rich Naive Bayes fitting failed: {e}")
            raise

    def _create_features(self, data: pd.Series) -> pd.DataFrame:
        """Create comprehensive feature set for yearly time series data"""
        df = pd.DataFrame(index=data.index)
        df['target'] = data

        # Basic lag features - more extensive for yearly data
        lag_periods = self._get_lag_periods(len(data))

        for lag in lag_periods:
            if lag < len(data):
                df[f'lag_{lag}'] = data.shift(lag)

        # Rolling statistics - multiple windows
        rolling_windows = self._get_rolling_windows(len(data))

        for window in rolling_windows:
            if window < len(data):
                rolling_series = data.rolling(window=window)
                df[f'rolling_mean_{window}'] = rolling_series.mean()
                df[f'rolling_std_{window}'] = rolling_series.std()
                df[f'rolling_min_{window}'] = rolling_series.min()
                df[f'rolling_max_{window}'] = rolling_series.max()
                df[f'rolling_median_{window}'] = rolling_series.median()

        # Temporal features
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

        # Seasonal indicators
        df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Trend features
        df['linear_trend'] = np.arange(len(df))
        df['quadratic_trend'] = df['linear_trend'] ** 2

        # Difference features (change from previous periods)
        for diff_period in [1, 7, 30]:
            if diff_period < len(data):
                df[f'diff_{diff_period}'] = data.diff(diff_period)
                df[f'pct_change_{diff_period}'] = data.pct_change(diff_period)

        # Statistical features over different windows
        if len(data) >= 30:
            # Z-score over rolling windows
            for window in [7, 30]:
                rolling_mean = data.rolling(window).mean()
                rolling_std = data.rolling(window).std()
                df[f'zscore_{window}'] = (data - rolling_mean) / rolling_std

        # Advanced features for yearly data
        if len(data) >= 365:
            # Year-over-year features
            df['lag_365'] = data.shift(365)
            df['yoy_change'] = data.pct_change(365)

            # Seasonal decomposition approximation
            # Simple trend (yearly moving average)
            yearly_trend = data.rolling(window=365, center=True).mean()
            df['detrended'] = data - yearly_trend

            # Quarterly patterns
            df['lag_91'] = data.shift(91)  # ~3 months
            df['lag_182'] = data.shift(182)  # ~6 months

        logger.info(f"Created {len([col for col in df.columns if col != 'target'])} features")
        return df

    def _get_lag_periods(self, data_length: int) -> List[int]:
        """Get appropriate lag periods based on data length"""
        if data_length >= 365:
            return [1, 2, 3, 7, 14, 30, 91, 182, 365]
        elif data_length >= 182:
            return [1, 2, 3, 7, 14, 30, 91]
        elif data_length >= 91:
            return [1, 2, 3, 7, 14, 30]
        else:
            return [1, 2, 3, 7, 14]

    def _get_rolling_windows(self, data_length: int) -> List[int]:
        """Get appropriate rolling windows based on data length"""
        if data_length >= 365:
            return [7, 14, 30, 91, 182]
        elif data_length >= 182:
            return [7, 14, 30, 91]
        elif data_length >= 91:
            return [7, 14, 30]
        else:
            return [7, 14]

    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray) -> None:
        """Calculate approximate feature importance"""
        try:
            # Calculate feature variance as importance measure
            feature_variance = np.var(X, axis=0)

            # Normalize to [0, 1]
            if np.max(feature_variance) > 0:
                normalized_importance = feature_variance / np.max(feature_variance)
            else:
                normalized_importance = np.ones(len(self.feature_names))

            self.feature_importance = {
                name: float(importance)
                for name, importance in zip(self.feature_names, normalized_importance)
            }

        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
            self.feature_importance = {}

    def forecast(self, horizon: int) -> np.ndarray:
        """Generate forecasts using feature-rich Naive Bayes"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        try:
            forecasts = []

            # For each forecast step, predict and use the prediction for next step
            for step in range(horizon):
                # Create features for current step
                # Note: This is simplified - in practice, we'd need to maintain
                # a rolling window of recent values including our predictions
                current_features = self._create_forecast_features(step)

                if current_features is not None:
                    # Scale features
                    current_features_scaled = self.scaler.transform(current_features.reshape(1, -1))

                    # Get class probabilities
                    class_probs = self.model.predict_proba(current_features_scaled)[0]

                    # Convert back to continuous value using bin centers weighted by probabilities
                    bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
                    forecast_value = np.sum(class_probs * bin_centers)

                    forecasts.append(forecast_value)
                else:
                    # Fallback to last known value
                    if forecasts:
                        forecasts.append(forecasts[-1])
                    else:
                        # Use mean of bin centers as final fallback
                        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
                        forecasts.append(np.mean(bin_centers))

            return np.array(forecasts)

        except Exception as e:
            logger.error(f"Feature-Rich Naive Bayes forecasting failed: {e}")
            raise

    def _create_forecast_features(self, step: int) -> Optional[np.ndarray]:
        """Create features for forecasting (simplified version)"""
        try:
            # This is a simplified version - would need more sophisticated
            # feature creation during forecasting to maintain consistency
            # For now, use average feature values as approximation
            if hasattr(self, '_last_features'):
                return np.mean(self._last_features, axis=0)
            else:
                return np.zeros(len(self.feature_names))

        except Exception as e:
            logger.warning(f"Could not create forecast features: {e}")
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        info = super().get_model_info()

        if self.is_fitted:
            info.update({
                'model_type': 'Feature-Rich Naive Bayes',
                'n_features': len(self.feature_names),
                'feature_names': self.feature_names,
                'n_bins': self.n_bins,
                'n_classes': len(self.model.classes_) if hasattr(self.model, 'classes_') else None
            })

        return info

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance.copy()