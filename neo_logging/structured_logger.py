"""
Structured Logger for NEO Hybrid AI Trading System
Provides JSON-formatted logs for all Python services.
"""

import logging
import json
from datetime import datetime, timezone

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(self.JsonFormatter())
        self.logger.handlers = [handler]

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "funcName": record.funcName,
                "lineno": record.lineno,
            }
            return json.dumps(log_record)

    def info(self, msg, **kwargs):
        self.logger.info(msg, extra=kwargs)
    def warning(self, msg, **kwargs):
        self.logger.warning(msg, extra=kwargs)
    def error(self, msg, **kwargs):
        self.logger.error(msg, extra=kwargs)
    def debug(self, msg, **kwargs):
        self.logger.debug(msg, extra=kwargs)

if __name__ == "__main__":
    log = StructuredLogger("test")
    log.info("Structured logging started.")
    log.error("This is an error log.")
