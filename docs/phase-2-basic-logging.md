# Basic Logging Setup for NEO Hybrid AI

## Database Logs
- PostgreSQL: Logs available via Docker container (`docker logs neo-postgres`).
- Redis: Logs available via Docker container (`docker logs neo-redis`).

## Python Logging Example
import logging
logging.basicConfig(filename='neoai.log', level=logging.INFO)
logging.info('Service started')

## Java Logging Example
// Use java.util.logging or Log4j
Logger logger = Logger.getLogger("neoai");
logger.info("Service started");

## Service Logs
- All services should log startup, shutdown, errors, and key actions.
- Log files should be stored in a central /logs directory.

---
Update this file as logging evolves and new services are added.