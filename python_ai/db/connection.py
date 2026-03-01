"""
Database connection management for NEO Hybrid AI.

Provides synchronous and asynchronous database connections
with connection pooling, health checks, and session management.
"""

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from python_ai.db.config import get_database_config, get_database_url
from python_ai.db.models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections with pooling and health checks.

    Attributes:
        _sync_engine: Synchronous SQLAlchemy engine.
        _async_engine: Asynchronous SQLAlchemy engine.
        _sync_session_factory: Factory for sync sessions.
        _async_session_factory: Factory for async sessions.
        _initialized: Whether the manager has been initialized.
    """

    _instance: Optional["DatabaseManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "DatabaseManager":
        """Singleton pattern - ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize database manager (only once due to singleton)."""
        if not DatabaseManager._initialized:
            self._sync_engine: Optional[object] = None
            self._async_engine: Optional[AsyncEngine] = None
            self._sync_session_factory: Optional[sessionmaker] = None
            self._async_session_factory: Optional[async_sessionmaker] = None
            DatabaseManager._initialized = True

    def init_sync_engine(self, testing: bool = False) -> None:
        """Initialize synchronous database engine with connection pooling.

        Args:
            testing: If True, use SQLite test database.
        """
        if self._sync_engine is not None:
            return

        config = get_database_config()
        url = get_database_url(async_mode=False, testing=testing)

        # SQLite doesn't support pooling the same way
        if testing:
            self._sync_engine = create_engine(
                url,
                echo=config.echo,
                connect_args={"check_same_thread": False},
            )
        else:
            self._sync_engine = create_engine(
                url,
                poolclass=QueuePool,
                pool_size=config.pool_size,
                max_overflow=config.max_overflow,
                pool_timeout=config.pool_timeout,
                pool_recycle=config.pool_recycle,
                echo=config.echo,
            )

        self._sync_session_factory = sessionmaker(
            bind=self._sync_engine,
            autocommit=False,
            autoflush=False,
        )
        logger.info("Synchronous database engine initialized")

    def init_async_engine(self, testing: bool = False) -> None:
        """Initialize asynchronous database engine.

        Args:
            testing: If True, use SQLite test database.
        """
        if self._async_engine is not None:
            return

        config = get_database_config()
        url = get_database_url(async_mode=True, testing=testing)

        if testing:
            self._async_engine = create_async_engine(
                url,
                echo=config.echo,
            )
        else:
            self._async_engine = create_async_engine(
                url,
                pool_size=config.pool_size,
                max_overflow=config.max_overflow,
                pool_timeout=config.pool_timeout,
                pool_recycle=config.pool_recycle,
                echo=config.echo,
            )

        self._async_session_factory = async_sessionmaker(
            bind=self._async_engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
        logger.info("Asynchronous database engine initialized")

    def create_tables(self, testing: bool = False) -> None:
        """Create all database tables.

        Args:
            testing: If True, use sync engine for table creation.
        """
        if self._sync_engine is None:
            self.init_sync_engine(testing=testing)
        Base.metadata.create_all(bind=self._sync_engine)
        logger.info("Database tables created")

    async def create_tables_async(self) -> None:
        """Create all database tables asynchronously."""
        if self._async_engine is None:
            raise RuntimeError("Async engine not initialized")
        async with self._async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created (async)")

    @contextmanager
    def get_sync_session(self) -> Generator[Session, None, None]:
        """Get a synchronous database session.

        Yields:
            Session: SQLAlchemy session for database operations.

        Raises:
            RuntimeError: If sync engine not initialized.
        """
        if self._sync_session_factory is None:
            raise RuntimeError(
                "Sync engine not initialized. Call init_sync_engine()."
            )

        session = self._sync_session_factory()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an asynchronous database session.

        Yields:
            AsyncSession: Async SQLAlchemy session for database operations.

        Raises:
            RuntimeError: If async engine not initialized.
        """
        if self._async_session_factory is None:
            raise RuntimeError(
                "Async engine not initialized. Call init_async_engine()."
            )

        session = self._async_session_factory()
        try:
            yield session
            await session.commit()
        except SQLAlchemyError as e:
            await session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            await session.close()

    def check_health(self) -> dict:
        """Check database connection health.

        Returns:
            dict: Health status with connection and latency info.
        """
        import time

        result = {
            "status": "unknown",
            "connected": False,
            "latency_ms": None,
            "error": None,
        }

        if self._sync_engine is None:
            result["error"] = "Engine not initialized"
            result["status"] = "unhealthy"
            return result

        try:
            start = time.perf_counter()
            with self._sync_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            latency = (time.perf_counter() - start) * 1000

            result["status"] = "healthy"
            result["connected"] = True
            result["latency_ms"] = round(latency, 2)
        except SQLAlchemyError as e:
            result["status"] = "unhealthy"
            result["error"] = str(e)
            logger.error(f"Health check failed: {e}")

        return result

    async def check_health_async(self) -> dict:
        """Check database connection health asynchronously.

        Returns:
            dict: Health status with connection and latency info.
        """
        import time

        result = {
            "status": "unknown",
            "connected": False,
            "latency_ms": None,
            "error": None,
        }

        if self._async_engine is None:
            result["error"] = "Async engine not initialized"
            result["status"] = "unhealthy"
            return result

        try:
            start = time.perf_counter()
            async with self._async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            latency = (time.perf_counter() - start) * 1000

            result["status"] = "healthy"
            result["connected"] = True
            result["latency_ms"] = round(latency, 2)
        except SQLAlchemyError as e:
            result["status"] = "unhealthy"
            result["error"] = str(e)
            logger.error(f"Async health check failed: {e}")

        return result

    def close(self) -> None:
        """Close all database connections."""
        if self._sync_engine:
            self._sync_engine.dispose()
            self._sync_engine = None
            self._sync_session_factory = None
            logger.info("Sync engine disposed")

    async def close_async(self) -> None:
        """Close async database connections."""
        if self._async_engine:
            await self._async_engine.dispose()
            self._async_engine = None
            self._async_session_factory = None
            logger.info("Async engine disposed")

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        if cls._instance:
            cls._instance.close()
        cls._instance = None
        cls._initialized = False


# Convenience functions for FastAPI dependency injection


def get_db_manager() -> DatabaseManager:
    """Get the database manager singleton.

    Returns:
        DatabaseManager: The singleton database manager instance.
    """
    return DatabaseManager()


def init_db(testing: bool = False) -> DatabaseManager:
    """Initialize database with both sync and async engines.

    Args:
        testing: If True, use SQLite test database.

    Returns:
        DatabaseManager: Initialized database manager.
    """
    manager = get_db_manager()
    manager.init_sync_engine(testing=testing)
    manager.create_tables(testing=testing)
    return manager


async def init_db_async(testing: bool = False) -> DatabaseManager:
    """Initialize database asynchronously.

    Args:
        testing: If True, use SQLite test database.

    Returns:
        DatabaseManager: Initialized database manager.
    """
    manager = get_db_manager()
    manager.init_async_engine(testing=testing)
    await manager.create_tables_async()
    return manager


def close_db() -> None:
    """Close database connections and reset manager."""
    DatabaseManager.reset()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for sync database session.

    Yields:
        Session: Database session.
    """
    manager = get_db_manager()
    with manager.get_sync_session() as session:
        yield session


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for async database session.

    Yields:
        AsyncSession: Async database session.
    """
    manager = get_db_manager()
    async with manager.get_async_session() as session:
        yield session
