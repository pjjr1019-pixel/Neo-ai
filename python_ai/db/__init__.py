"""
Database package for NEO Hybrid AI.

Provides async PostgreSQL connection pooling, session management,
and ORM models for data persistence.
"""

from python_ai.db.config import DatabaseConfig, get_database_url
from python_ai.db.connection import (
    DatabaseManager,
    close_db,
    get_async_db,
    get_db,
    init_db,
)
from python_ai.db.models import (
    Base,
    FeatureImportance,
    ModelMetrics,
    Prediction,
    TrainingSession,
)

__all__ = [
    "get_db",
    "get_async_db",
    "init_db",
    "close_db",
    "DatabaseManager",
    "Base",
    "Prediction",
    "TrainingSession",
    "ModelMetrics",
    "FeatureImportance",
    "DatabaseConfig",
    "get_database_url",
]
