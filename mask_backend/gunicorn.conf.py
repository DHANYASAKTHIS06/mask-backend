import multiprocessing

# Render free tier: 512MB RAM — keep workers low
workers = 2
worker_class = "sync"
worker_connections = 100
timeout = 120
keepalive = 5
bind = "0.0.0.0:10000"   # Render default port
preload_app = True

# Logging
accesslog = "-"
errorlog  = "-"
loglevel  = "info"
