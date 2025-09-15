from decouple import config

class Settings:
    """Application settings loaded from environment variables or .env"""

    # Server configuration
    API_HOST: str
    API_PORT: int
    API_DEBUG: bool

    # Data fetching
    DATA_BASE_URL: str
    REQUEST_TIMEOUT: int
    AUTH_TOKEN: str

    # Forecasting defaults
    DEFAULT_FORECAST_HORIZON: int
    DEFAULT_USE_ENHANCED_TTM: bool

    # Plotting
    SAVE_PLOTS: bool
    SHOW_PLOTS: bool

    def __init__(self):
        # Server
        self.API_HOST = config("API_HOST", default="0.0.0.0")
        self.API_PORT = config("API_PORT", default=8000, cast=int)
        self.API_DEBUG = config("API_DEBUG", default=False, cast=bool)

        # Data fetching
        self.DATA_BASE_URL = config("DATA_BASE_URL", default="http://localhost:8080")
        self.REQUEST_TIMEOUT = config("REQUEST_TIMEOUT", default=30, cast=int)
        self.AUTH_TOKEN = config("AUTH_TOKEN", default="", cast=str)

        # Forecasting
        self.DEFAULT_FORECAST_HORIZON = config("DEFAULT_FORECAST_HORIZON", default=30, cast=int)
        self.DEFAULT_USE_ENHANCED_TTM = config("DEFAULT_USE_ENHANCED_TTM", default=False, cast=bool)

        # Plotting
        self.SAVE_PLOTS = config("SAVE_PLOTS", default=False, cast=bool)
        self.SHOW_PLOTS = config("SHOW_PLOTS", default=False, cast=bool)

    def get_data_url(self, instance_name: str, metric: str) -> str:
        """Build the full URL for data requests"""
        return f"{self.DATA_BASE_URL}/api/metrics?instance={instance_name}&metric={metric}"

# Singleton instance for global use
settings = Settings()
