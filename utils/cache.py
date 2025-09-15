"""
Caching utilities for TTM Forecasting System
"""

import hashlib
import json
import pickle
import time
import os
from pathlib import Path
from typing import Any, Optional, Dict, Union
from functools import wraps
import pandas as pd
from utils.logging_config import get_logger

logger = get_logger(__name__)


class SimpleCache:
    """Simple file-based cache for model results and data"""

    def __init__(self, cache_dir: str = "./cache", max_age_seconds: int = 3600):
        """
        Initialize cache

        Args:
            cache_dir: Directory to store cache files
            max_age_seconds: Maximum age before cache expires (default: 1 hour)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_age = max_age_seconds

    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments"""
        # Create a stable hash from all arguments
        content = {
            'args': args,
            'kwargs': kwargs
        }
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.md5(content_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key"""
        return self.cache_dir / f"{cache_key}.pkl"

    def _is_expired(self, cache_path: Path) -> bool:
        """Check if cache file is expired"""
        if not cache_path.exists():
            return True

        file_age = time.time() - cache_path.stat().st_mtime
        return file_age > self.max_age

    def get(self, cache_key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            cache_path = self._get_cache_path(cache_key)

            if self._is_expired(cache_path):
                return None

            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_data

        except Exception as e:
            logger.warning(f"Cache read failed for key {cache_key}: {e}")
            return None

    def set(self, cache_key: str, value: Any) -> bool:
        """Set value in cache"""
        try:
            cache_path = self._get_cache_path(cache_key)

            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
                logger.debug(f"Cache set for key: {cache_key}")
                return True

        except Exception as e:
            logger.error(f"Cache write failed for key {cache_key}: {e}")
            return False

    def clear(self) -> None:
        """Clear all cache files"""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")

    def cache_key_for_data(self, data_source: str, **kwargs) -> str:
        """Generate cache key for data loading"""
        return self._get_cache_key("data", data_source, **kwargs)

    def cache_key_for_forecast(self, model_name: str, data_hash: str,
                              horizon: int, **model_params) -> str:
        """Generate cache key for forecast results"""
        return self._get_cache_key("forecast", model_name, data_hash, horizon, **model_params)


class SecureCache(SimpleCache):
    """
    Secure cache implementation with encryption for sensitive data.

    Uses cryptography library to encrypt cached data and stores encryption keys
    with proper file permissions.
    """

    def __init__(self, cache_dir: str = "./cache", max_age_seconds: int = 3600):
        super().__init__(cache_dir, max_age_seconds)
        self.cipher = self._get_or_create_cipher()

    def _get_or_create_cipher(self):
        """Get or create encryption cipher with secure key storage"""
        try:
            from cryptography.fernet import Fernet

            key_file = self.cache_dir / '.cache_key'

            if key_file.exists():
                # Load existing key
                with open(key_file, 'rb') as f:
                    key = f.read()
                logger.debug("Loaded existing cache encryption key")
            else:
                # Generate new key
                key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(key)
                # Set secure permissions (owner read/write only)
                os.chmod(key_file, 0o600)
                logger.info("Generated new cache encryption key")

            return Fernet(key)

        except ImportError:
            logger.warning("cryptography library not available, falling back to unencrypted cache")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize cache encryption: {e}")
            return None

    def set(self, cache_key: str, value: Any) -> bool:
        """Set encrypted value in cache"""
        try:
            cache_path = self._get_cache_path(cache_key)

            # Serialize data
            data = pickle.dumps(value)

            # Encrypt if cipher available
            if self.cipher:
                data = self.cipher.encrypt(data)

            with open(cache_path, 'wb') as f:
                f.write(data)

            # Set secure file permissions
            os.chmod(cache_path, 0o600)

            logger.debug(f"Secure cache set for key: {cache_key}")
            return True

        except Exception as e:
            logger.error(f"Secure cache write failed for key {cache_key}: {e}")
            return False

    def get(self, cache_key: str) -> Optional[Any]:
        """Get decrypted value from cache"""
        try:
            cache_path = self._get_cache_path(cache_key)

            if self._is_expired(cache_path):
                return None

            with open(cache_path, 'rb') as f:
                data = f.read()

            # Decrypt if cipher available
            if self.cipher:
                data = self.cipher.decrypt(data)

            cached_data = pickle.loads(data)
            logger.debug(f"Secure cache hit for key: {cache_key}")
            return cached_data

        except Exception as e:
            logger.warning(f"Secure cache read failed for key {cache_key}: {e}")
            return None


# Global cache instance
_cache_instance = None

def get_cache() -> SimpleCache:
    """Get the global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        from utils.constants import DEFAULT_CACHE_MAX_AGE
        _cache_instance = SimpleCache(max_age_seconds=DEFAULT_CACHE_MAX_AGE)
    return _cache_instance


def cached_function(max_age_seconds: Optional[int] = None):
    """Decorator to cache function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            if max_age_seconds:
                cache.max_age = max_age_seconds

            cache_key = cache._get_cache_key(func.__name__, *args, **kwargs)

            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result)

            return result
        return wrapper
    return decorator


def hash_dataframe(df: Union[pd.DataFrame, pd.Series]) -> str:
    """Create a hash of a DataFrame/Series for caching"""
    try:
        # Use data values and index for hash
        content = pd.util.hash_pandas_object(df).sum()
        return str(content)
    except Exception:
        # Fallback: convert to string and hash
        return hashlib.md5(str(df.values).encode()).hexdigest()


def hash_dataframe_fast(df: Union[pd.DataFrame, pd.Series]) -> str:
    """
    Fast hashing using xxhash for large datasets with fallback to standard method.

    Performance improvement: xxhash is 3-5x faster than md5/sha1 for large data.
    """
    try:
        # Try to use xxhash for better performance
        try:
            import xxhash
            hasher = xxhash.xxh64()

            # Hash index and values separately for better performance
            if hasattr(df, 'index'):
                hasher.update(df.index.values.tobytes())
            hasher.update(df.values.tobytes())

            return hasher.hexdigest()
        except ImportError:
            # xxhash not available, use optimized pandas hashing
            return hash_dataframe(df)
    except Exception:
        # Final fallback to original method
        return hash_dataframe(df)


def optimize_array_memory(arr, dtype=None):
    """Optimize array memory usage by downcasting"""
    import numpy as np

    if isinstance(arr, (list, tuple)):
        arr = np.array(arr)

    if not isinstance(arr, np.ndarray):
        return arr

    # If dtype is specified, use it
    if dtype:
        return arr.astype(dtype)

    # Auto-optimize based on data
    if np.issubdtype(arr.dtype, np.integer):
        # Optimize integer arrays
        if arr.min() >= 0:
            # Unsigned integers
            if arr.max() <= np.iinfo(np.uint8).max:
                return arr.astype(np.uint8)
            elif arr.max() <= np.iinfo(np.uint16).max:
                return arr.astype(np.uint16)
            elif arr.max() <= np.iinfo(np.uint32).max:
                return arr.astype(np.uint32)
        else:
            # Signed integers
            if np.iinfo(np.int8).min <= arr.min() and arr.max() <= np.iinfo(np.int8).max:
                return arr.astype(np.int8)
            elif np.iinfo(np.int16).min <= arr.min() and arr.max() <= np.iinfo(np.int16).max:
                return arr.astype(np.int16)
            elif np.iinfo(np.int32).min <= arr.min() and arr.max() <= np.iinfo(np.int32).max:
                return arr.astype(np.int32)

    elif np.issubdtype(arr.dtype, np.floating):
        # Optimize float arrays
        # Check if float32 precision is sufficient
        if np.allclose(arr.astype(np.float32), arr, rtol=1e-6):
            return arr.astype(np.float32)

    return arr  # Return original if no optimization possible


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast memory optimization using pandas built-ins and categorical data.

    Performance improvements:
    - Use pandas categorical for string columns with low cardinality
    - Automatic numeric downcasting
    - Memory-efficient data types

    Returns optimized copy of DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        return df

    df_optimized = df.copy()

    # Convert object columns to categorical if beneficial
    for col in df_optimized.select_dtypes(include=['object']):
        # Use categorical if less than 50% unique values
        if df_optimized[col].nunique() / len(df_optimized) < 0.5:
            df_optimized[col] = df_optimized[col].astype('category')

    # Downcast numeric types automatically
    for col in df_optimized.select_dtypes(include=[np.number]):
        col_type = df_optimized[col].dtype

        # Downcast integers
        if 'int' in str(col_type):
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')

        # Downcast floats
        elif 'float' in str(col_type):
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')

    return df_optimized