"""
Gunicorn configuration for production deployment
Uses settings.py for all environment variables
"""

from settings import settings

# Helper to return None when env var is missing/empty
def none_if_empty(value):
    return value if value not in ("", None) else None

# Server socket
bind = f"{settings.API_HOST}:{settings.API_PORT}"
backlog = 2048

# Worker processes
workers = max(1, settings.DEFAULT_FORECAST_HORIZON // 2)  # example, adjust as needed
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Timeouts
timeout = 120
keepalive = 5
graceful_timeout = 30

# Logging
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'ttm-forecasting-api'

# Server mechanics
daemon = False
pidfile = "/tmp/gunicorn.pid"
user = none_if_empty(settings.AUTH_TOKEN)  # example, adjust if needed
group = None
tmp_upload_dir = None

# SSL (if using HTTPS)
keyfile = none_if_empty(settings.AUTH_TOKEN)  # replace with SSL_KEYFILE if used
certfile = none_if_empty(settings.AUTH_TOKEN)  # replace with SSL_CERTFILE if used

# Preload application for better memory usage
preload_app = True

# Worker process management
worker_tmp_dir = "/dev/shm"

# Security
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190
