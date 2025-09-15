"""
Data loading utilities for fetching data from various sources
"""

import json
import pandas as pd
import requests
from typing import Union, Dict
from urllib.parse import urlparse
from utils.logging_config import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Utility class for loading data from files or URLs"""

    @staticmethod
    def load_from_file(file_path: str) -> pd.Series:
        """Load data from a local JSON file"""
        try:
            with open(file_path, 'r') as f:
                raw = json.load(f)

            return DataLoader._process_raw_data(raw, source=file_path)

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {file_path}")

    @staticmethod
    def load_from_url(url: str, timeout: int = 30, headers: Dict[str, str] = None) -> pd.Series:
        """
        Load data from a URL endpoint

        Args:
            url: URL to fetch data from
            timeout: Request timeout in seconds
            headers: Optional HTTP headers

        Returns:
            pd.Series: Time series data
        """
        try:
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL format: {url}")

            # Set default headers
            if headers is None:
                headers = {
                    'User-Agent': 'TTM-Forecaster/1.0',
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                }

            logger.info(f"Fetching data from: {url}")

            # Make request
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()  # Raise exception for bad status codes

            # Parse JSON
            try:
                raw = response.json()
            except json.JSONDecodeError:
                raise ValueError(f"Response from {url} is not valid JSON")

            logger.info(f"Successfully fetched {len(raw)} items from URL")
            return DataLoader._process_raw_data(raw, source=url)

        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to {url} timed out after {timeout} seconds")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Failed to connect to {url}")
        except requests.exceptions.HTTPError as e:
            raise ValueError(f"HTTP error {e.response.status_code} when fetching {url}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Request failed: {str(e)}")

    @staticmethod
    def _process_raw_data(raw: Union[list, dict], source: str) -> pd.Series:
        """
        Process raw JSON data into a pandas Series

        Expected formats:
        1. List of objects: [{"date": "2024-01-01", "value": 45.2}, ...]
        2. Object with data array: {"data": [{"date": "2024-01-01", "value": 45.2}, ...]}
        3. Object with separate arrays: {"dates": [...], "values": [...]}
        """
        try:
            # Handle different JSON structures
            if isinstance(raw, list):
                data_list = raw
            elif isinstance(raw, dict):
                if 'data' in raw:
                    data_list = raw['data']
                elif 'dates' in raw and 'values' in raw:
                    # Convert separate arrays to list of objects
                    dates = raw['dates']
                    values = raw['values']
                    if len(dates) != len(values):
                        raise ValueError("Dates and values arrays have different lengths")
                    data_list = [{'date': d, 'value': v} for d, v in zip(dates, values)]
                else:
                    raise ValueError("Unknown JSON structure - expected 'data' field or 'dates'/'values' fields")
            else:
                raise ValueError("JSON data must be a list or object")

            if not data_list:
                raise ValueError("No data found in JSON")

            # Convert to DataFrame
            df = pd.DataFrame(data_list)

            # Validate required columns
            if 'date' not in df.columns:
                raise ValueError("Missing 'date' column in data")
            if 'value' not in df.columns:
                raise ValueError("Missing 'value' column in data")

            # Process dates
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            # Create series
            series = pd.Series(df['value'].values, index=df['date'], name='value')

            logger.info(f"Processed {len(series)} data points from {series.index.min()} to {series.index.max()}")
            return series

        except Exception as e:
            raise ValueError(f"Failed to process data from {source}: {str(e)}")

    @staticmethod
    def load_data(source: str, **kwargs) -> pd.Series:
        """
        Load data from file or URL (auto-detect)

        Args:
            source: File path or URL
            **kwargs: Additional arguments for URL loading (timeout, headers)

        Returns:
            pd.Series: Time series data
        """
        if source.startswith(('http://', 'https://')):
            return DataLoader.load_from_url(source, **kwargs)
        else:
            return DataLoader.load_from_file(source)