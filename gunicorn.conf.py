"""
Gunicorn configuration for production deployment
"""

import os
from decouple import config

# Server socket
bind = f"{config('API_HOST', default='0.0.0.0')}:{config('API_PORT', default=8000, cast=int)}"
backlog = 2048

# Worker processes
workers = config('WORKERS', default=4, cast=int)  # CPU cores * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Timeouts
timeout = 120
keepalive = 5
graceful_timeout = 30

# Logging
accesslog = config('ACCESS_LOG', default='-')  # stdout
errorlog = config('ERROR_LOG', default='-')   # stderr
loglevel = config('LOG_LEVEL', default='info')
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'ttm-forecasting-api'

# Server mechanics
daemon = config('DAEMON_MODE', default=False, cast=bool)
pidfile = config('PID_FILE', default='/tmp/gunicorn.pid')
user = config('USER', default=None)
group = config('GROUP', default=None)
tmp_upload_dir = None

# SSL (if using HTTPS)
keyfile = config('SSL_KEYFILE', default=None)
certfile = config('SSL_CERTFILE', default=None)

# Preload application for better memory usage
preload_app = True

# Worker process management
worker_tmp_dir = "/dev/shm"  # Use RAM for worker tmp directory

# Security
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190