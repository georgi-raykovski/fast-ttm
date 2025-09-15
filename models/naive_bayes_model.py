"""
Naive Bayes Forecasting Model
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.naive_bayes import GaussianNB
from .base_model import BaseModel


class NaiveBayesModel(BaseModel):
    """Naive Bayes model with lag features for time series forecasting"""

    def __init__(self, lags: List[int] = [1, 2, 3, 7, 14, 30]):
        super().__init__("NaiveBayes")
        self.lags = lags
        self.model = None
        self.bins = None
        self.train_data = None
        self.feature_names = None

    @staticmethod
    def create_lag_features(series: pd.Series, lags: List[int]) -> pd.DataFrame:
        """Create lag features for the time series"""
        df = pd.DataFrame({'y': series})
        for lag in lags:
            df[f'lag_{lag}'] = df['y'].shift(lag)
        df = df.dropna()
        return df

    def fit(self, data: pd.Series) -> None:
        """Fit the Naive Bayes model"""
        self.train_data = data.values

        # Create lag features
        df = self.create_lag_features(data, self.lags)

        if len(df) == 0:
            raise ValueError("Not enough data to create lag features")

        # Create bins for target variable
        self.bins = np.linspace(0, 100, 11)
        df['y_bin'] = pd.cut(df['y'], bins=self.bins, labels=False)

        # Prepare features and target
        X = df.drop(columns=['y', 'y_bin'])
        y = df['y_bin']

        # Store feature names for consistent prediction
        self.feature_names = X.columns.tolist()

        # Fit Gaussian Naive Bayes
        self.model = GaussianNB()
        self.model.fit(X, y)
        self.is_fitted = True

    def forecast(self, horizon: int) -> np.ndarray:
        """Generate forecasts using Naive Bayes model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        # Handle 30-day lags properly
        max_lag = max(self.lags)
        if len(self.train_data) < max_lag:
            # If we don't have enough history, repeat the series
            repeats = max_lag // len(self.train_data) + 1
            extended_train = np.tile(self.train_data, repeats)
            last_values = extended_train[-max_lag:].tolist()
        else:
            last_values = self.train_data[-max_lag:].tolist()

        forecasts = []
        for _ in range(horizon):
            # Handle cases where we don't have enough values for all lags
            features = []
            for lag in self.lags:
                if len(last_values) >= lag:
                    features.append(last_values[-lag])
                else:
                    # Use the earliest available value if lag is too large
                    features.append(last_values[0])

            # Create DataFrame with proper feature names for prediction
            features_df = pd.DataFrame([features], columns=self.feature_names)

            # Predict bin
            pred_bin = self.model.predict(features_df)[0]

            # Convert bin back to value (use bin midpoint)
            bin_mid = (self.bins[pred_bin] + self.bins[pred_bin + 1]) / 2
            forecasts.append(bin_mid)

            # Update history with new prediction
            last_values.append(bin_mid)

        return np.array(forecasts)