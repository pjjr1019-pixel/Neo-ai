# Prometheus Metrics Exporter for NEO Hybrid AI
from prometheus_client import start_http_server, Gauge
import time
import psutil

CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percent')
MEMORY_USAGE = Gauge('memory_usage_percent', 'Memory usage percent')


def collect_metrics():
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)

if __name__ == "__main__":
    start_http_server(8000)
    while True:
        collect_metrics()
        time.sleep(5)
