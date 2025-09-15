"""
Enhanced TTM (Tiny Time Mixer) Models with multiple approaches for short time series
"""

import numpy as np
import pandas as pd
import torch
import warnings
from typing import Dict
from torch.utils.data import Dataset, DataLoader
from .base_model import BaseModel
from utils.metrics import clip_cpu_forecasts
from utils.constants import TTM_DEFAULT_CONTEXT_LENGTH
from utils.exceptions import TTMLibraryError, ModelFittingError, ForecastingError

try:
    from tsfm_public import TinyTimeMixerForPrediction
    from transformers import Trainer, TrainingArguments
    TTM_AVAILABLE = True
except ImportError:
    TTM_AVAILABLE = False


class ShortTimeSeriesDataset(Dataset):
    """Custom dataset for fine-tuning TTM on short time series"""

    def __init__(self, data, context_len=60, pred_len=24):
        self.data = data
        self.context_len = context_len
        self.pred_len = pred_len

    def __len__(self):
        return max(1, len(self.data) - self.context_len - self.pred_len + 1)

    def __getitem__(self, idx):
        # Create sliding windows even with limited data
        start = idx
        end = start + self.context_len
        target_end = end + self.pred_len

        if end > len(self.data):
            # Pad with last value if needed
            context = np.pad(self.data[start:],
                           (0, end - len(self.data)),
                           mode='edge')
        else:
            context = self.data[start:end]

        if target_end > len(self.data):
            target = np.pad(self.data[end:],
                          (0, target_end - len(self.data)),
                          mode='edge')
        else:
            target = self.data[end:target_end]

        return {
            'past_values': torch.tensor(context.reshape(-1, 1), dtype=torch.float32),
            'future_values': torch.tensor(target.reshape(-1, 1), dtype=torch.float32)
        }


class TTMFineTunedModel(BaseModel):
    """TTM model with fine-tuning for short time series"""

    def __init__(self):
        super().__init__("TTM_FineTuned")
        self.model = None
        self.train_data = None

        if not TTM_AVAILABLE:
            raise TTMLibraryError()

    def fit(self, data: pd.Series) -> None:
        """Fine-tune TTM model on the short time series"""
        try:
            print("Fine-tuning TTM model for short time series...")

            # Load pre-trained model
            model_name = "ibm-granite/granite-timeseries-ttm-r1"
            self.model = TinyTimeMixerForPrediction.from_pretrained(model_name)

            # Prepare dataset
            dataset = ShortTimeSeriesDataset(
                data.values,
                context_len=min(60, len(data)),
                pred_len=min(24, len(data) // 3)
            )

            # Fine-tuning arguments
            import tempfile
            temp_dir = tempfile.mkdtemp()
            training_args = TrainingArguments(
                output_dir=temp_dir,  # Use temp directory that gets cleaned up
                num_train_epochs=50,  # More epochs for small data
                per_device_train_batch_size=1,
                learning_rate=1e-4,
                logging_steps=10000,  # Set very high to avoid logging
                save_strategy="no",
                evaluation_strategy="no",
                warmup_ratio=0.1,
                report_to=[],  # Empty list to disable all logging
                remove_unused_columns=False,
                logging_dir=None,  # Disable TensorBoard logging
                disable_tqdm=True,  # Disable progress bars
                log_level="error"  # Only show errors
            )

            # Fine-tune the model
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
            )

            trainer.train()

            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

            self.train_data = data.values
            self.is_fitted = True

            print("TTM fine-tuning completed")

        except Exception as e:
            print(f"TTM fine-tuning failed: {e}")
            # Fallback to loading pre-trained model
            self.model = TinyTimeMixerForPrediction.from_pretrained(
                "ibm-granite/granite-timeseries-ttm-r1"
            )
            self.train_data = data.values
            self.is_fitted = True

    def forecast(self, horizon: int) -> np.ndarray:
        """Generate forecasts using fine-tuned TTM model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        try:
            self.model.eval()

            # Use available data as context
            context_len = min(60, len(self.train_data))
            input_data = self.train_data[-context_len:].reshape(1, -1, 1)
            input_tensor = torch.tensor(input_data, dtype=torch.float32)

            with torch.no_grad():
                output = self.model(input_tensor)
                predictions = output.prediction_outputs.squeeze().numpy()

            # Handle prediction length
            if len(predictions) >= horizon:
                forecast = predictions[:horizon]
            else:
                # Extend with trend from predictions
                trend = np.mean(np.diff(predictions)) if len(predictions) > 1 else 0
                extension = [predictions[-1] + trend * (i + 1)
                           for i in range(horizon - len(predictions))]
                forecast = np.concatenate([predictions, extension])

            return clip_cpu_forecasts(forecast)

        except Exception as e:
            print(f"Fine-tuned TTM forecasting failed: {e}")
            return np.full(horizon, self.train_data[-1])


class TTMAugmentedModel(BaseModel):
    """TTM model with data augmentation for short time series"""

    def __init__(self):
        super().__init__("TTM_Augmented")
        self.model = None
        self.train_data = None

        if not TTM_AVAILABLE:
            raise TTMLibraryError()

    def fit(self, data: pd.Series) -> None:
        """Fit TTM with data augmentation"""
        try:
            # Load pre-trained model
            self.model = TinyTimeMixerForPrediction.from_pretrained(
                "ibm-granite/granite-timeseries-ttm-r1"
            )
            self.train_data = data.values
            self.is_fitted = True

            print("TTM loaded with data augmentation strategy")

        except Exception as e:
            print(f"Failed to load TTM model: {e}")
            raise

    def _create_augmented_data(self) -> np.ndarray:
        """Create augmented data using multiple strategies"""
        original_data = self.train_data
        augmented_series = []

        # 1. Add slight noise (3 variants)
        for i in range(3):
            noise = np.random.normal(0, 0.01 * np.std(original_data), len(original_data))
            augmented_series.append(original_data + noise)

        # 2. Slight scaling
        for scale in [0.95, 1.05]:
            augmented_series.append(original_data * scale)

        # 3. Slight shifting
        for shift in [-1, 1]:
            augmented_series.append(original_data + shift)

        # Combine all augmented series
        combined_data = np.concatenate(augmented_series)

        # If still less than 512, use seasonal patterns
        if len(combined_data) < 512:
            try:
                # Try to detect seasonal patterns
                from scipy import signal

                # Simple seasonal decomposition approach
                if len(original_data) >= 14:  # Need minimum data for seasonal detection
                    # Detect weekly pattern
                    weekly_pattern = original_data[-7:] if len(original_data) >= 7 else original_data

                    # Calculate how many repetitions needed
                    repeats = (512 - len(combined_data)) // len(weekly_pattern) + 1
                    seasonal_extension = np.tile(weekly_pattern, repeats)[:(512 - len(combined_data))]

                    # Blend the extension with some trend
                    trend = np.polyfit(range(len(original_data)), original_data, 1)[0]
                    trend_component = np.arange(len(seasonal_extension)) * trend * 0.5

                    final_extension = seasonal_extension + trend_component
                    combined_data = np.concatenate([combined_data, final_extension])

                # Final padding if still needed
                if len(combined_data) < 512:
                    remaining = 512 - len(combined_data)
                    final_pad = np.full(remaining, combined_data[-1])
                    combined_data = np.concatenate([combined_data, final_pad])

            except Exception as e:
                # Fallback to simple repetition
                print(f"Seasonal augmentation failed, using repetition: {e}")
                repeats = 512 // len(combined_data) + 1
                combined_data = np.tile(combined_data, repeats)

        return combined_data[-512:]  # Take last 512 points

    def forecast(self, horizon: int) -> np.ndarray:
        """Generate forecasts using augmented data approach"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        try:
            # Create augmented input data
            augmented_data = self._create_augmented_data()

            # Make prediction
            input_data = augmented_data.reshape(1, 512, 1)
            input_tensor = torch.tensor(input_data, dtype=torch.float32)

            with torch.no_grad():
                output = self.model(input_tensor)
                predictions = output.prediction_outputs.squeeze().numpy()

            # Post-process predictions
            test_forecast = clip_cpu_forecasts(predictions[:horizon])

            print(f"TTM with augmentation: created {len(augmented_data)} augmented points")
            return test_forecast

        except Exception as e:
            print(f"Augmented TTM failed: {e}")
            return np.full(horizon, self.train_data[-1])


class TTMEnsembleModel(BaseModel):
    """TTM ensemble with traditional methods for short time series"""

    def __init__(self):
        super().__init__("TTM_Ensemble")
        self.ttm_model = None
        self.train_data = None

        if not TTM_AVAILABLE:
            raise TTMLibraryError()

    def fit(self, data: pd.Series) -> None:
        """Fit ensemble of TTM and traditional methods"""
        try:
            # Load TTM model
            self.ttm_model = TinyTimeMixerForPrediction.from_pretrained(
                "ibm-granite/granite-timeseries-ttm-r1"
            )
            self.train_data = data.values
            self.is_fitted = True

            print("TTM ensemble model loaded")

        except Exception as e:
            print(f"Failed to load TTM ensemble: {e}")
            raise

    def forecast(self, horizon: int) -> np.ndarray:
        """Generate forecasts using ensemble approach"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        try:
            # Get TTM prediction (with simple padding)
            if len(self.train_data) < 512:
                repeats_needed = 512 // len(self.train_data) + 1
                padded_data = np.tile(self.train_data, repeats_needed)[-512:]
            else:
                padded_data = self.train_data[-512:]

            input_data = padded_data.reshape(1, 512, 1)
            input_tensor = torch.tensor(input_data, dtype=torch.float32)

            with torch.no_grad():
                output = self.ttm_model(input_tensor)
                ttm_forecast = output.prediction_outputs.squeeze().numpy()[:horizon]

            # Exponential Smoothing
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                es_model = ExponentialSmoothing(
                    self.train_data,
                    seasonal_periods=7,
                    seasonal='add'
                ).fit()
                es_forecast = es_model.forecast(horizon)
            except:
                # Simple exponential smoothing fallback
                alpha = 0.3
                es_forecast = []
                level = self.train_data[-1]
                for _ in range(horizon):
                    es_forecast.append(level)
                es_forecast = np.array(es_forecast)

            # ARIMA
            try:
                from statsmodels.tsa.arima.model import ARIMA
                arima_model = ARIMA(self.train_data, order=(1,1,1)).fit()
                arima_forecast = arima_model.forecast(horizon)
            except:
                # Linear trend fallback
                if len(self.train_data) > 1:
                    trend = (self.train_data[-1] - self.train_data[-min(7, len(self.train_data))]) / min(7, len(self.train_data))
                    arima_forecast = np.array([self.train_data[-1] + trend * (i + 1) for i in range(horizon)])
                else:
                    arima_forecast = np.full(horizon, self.train_data[-1])

            # Weighted ensemble (less weight to TTM due to data limitations)
            weights = [0.2, 0.4, 0.4]  # TTM, ES, ARIMA
            ensemble_forecast = (
                weights[0] * ttm_forecast +
                weights[1] * es_forecast +
                weights[2] * arima_forecast
            )

            # Clip to reasonable bounds
            ensemble_forecast = clip_cpu_forecasts(ensemble_forecast)

            print(f"TTM ensemble: combined TTM + ExponentialSmoothing + ARIMA")
            return ensemble_forecast

        except Exception as e:
            print(f"Ensemble forecast failed: {e}")
            return np.full(horizon, self.train_data[-1])