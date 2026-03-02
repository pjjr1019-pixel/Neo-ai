"""
Tests for database layer.

Covers connection management, models, and repository operations.
Uses SQLite in-memory database for isolation.
"""

import os
from datetime import datetime
from unittest.mock import patch

import pytest

# Set testing environment before imports
os.environ["DB_ECHO"] = "false"


class TestDatabaseConfig:
    """Tests for database configuration."""

    def test_config_default_values(self):
        """Test default configuration values."""
        from python_ai.db.config import DatabaseConfig, reset_config

        reset_config()
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.user == "neoai"
        assert config.database == "neoai_db"
        assert config.pool_size == 5
        assert config.max_overflow == 10

    def test_config_sync_url(self):
        """Test synchronous URL generation."""
        from python_ai.db.config import DatabaseConfig

        config = DatabaseConfig()
        url = config.sync_url
        assert url.startswith("postgresql://")
        assert "neoai" in url
        assert "5432" in url

    def test_config_async_url(self):
        """Test async URL generation."""
        from python_ai.db.config import DatabaseConfig

        config = DatabaseConfig()
        url = config.async_url
        assert "asyncpg" in url

    def test_config_test_url(self):
        """Test SQLite test URL generation."""
        from python_ai.db.config import DatabaseConfig

        config = DatabaseConfig()
        assert "sqlite" in config.test_url
        assert "sqlite" in config.async_test_url

    def test_get_database_url_sync(self):
        """Test get_database_url for sync mode."""
        from python_ai.db.config import get_database_url, reset_config

        reset_config()
        url = get_database_url(async_mode=False, testing=True)
        assert "sqlite" in url
        assert "aiosqlite" not in url

    def test_get_database_url_async(self):
        """Test get_database_url for async mode."""
        from python_ai.db.config import get_database_url, reset_config

        reset_config()
        url = get_database_url(async_mode=True, testing=True)
        assert "aiosqlite" in url

    def test_config_from_environment(self):
        """Test configuration from environment variables."""
        from python_ai.db.config import DatabaseConfig, reset_config

        reset_config()
        with patch.dict(
            os.environ,
            {
                "DB_HOST": "custom-host",
                "DB_PORT": "5433",
                "DB_USER": "custom-user",
            },
        ):
            config = DatabaseConfig()
            assert config.host == "custom-host"
            assert config.port == 5433
            assert config.user == "custom-user"


class TestDatabaseManager:
    """Tests for DatabaseManager class."""

    def test_singleton_pattern(self):
        """Test that DatabaseManager is a singleton."""
        from python_ai.db.connection import DatabaseManager

        DatabaseManager.reset()
        manager1 = DatabaseManager()
        manager2 = DatabaseManager()
        assert manager1 is manager2

    def test_init_sync_engine_testing(self):
        """Test initializing sync engine for testing."""
        from python_ai.db.connection import DatabaseManager

        DatabaseManager.reset()
        manager = DatabaseManager()
        manager.init_sync_engine(testing=True)
        assert manager._sync_engine is not None
        assert manager._sync_session_factory is not None
        manager.close()

    def test_create_tables(self):
        """Test table creation."""
        from python_ai.db.connection import DatabaseManager

        DatabaseManager.reset()
        manager = DatabaseManager()
        manager.init_sync_engine(testing=True)
        manager.create_tables(testing=True)
        # Should not raise
        manager.close()

    def test_get_sync_session(self):
        """Test getting a sync session."""
        from python_ai.db.connection import DatabaseManager

        DatabaseManager.reset()
        manager = DatabaseManager()
        manager.init_sync_engine(testing=True)
        manager.create_tables(testing=True)

        with manager.get_sync_session() as session:
            assert session is not None
            # Can execute queries
            result = session.execute(__import__("sqlalchemy").text("SELECT 1"))
            assert result is not None

        manager.close()

    def test_get_sync_session_not_initialized(self):
        """Test error when session requested without init."""
        from python_ai.db.connection import DatabaseManager

        DatabaseManager.reset()
        manager = DatabaseManager()

        with pytest.raises(RuntimeError, match="not initialized"):
            with manager.get_sync_session():
                pass

    def test_health_check_healthy(self):
        """Test health check when connection is healthy."""
        from python_ai.db.connection import DatabaseManager

        DatabaseManager.reset()
        manager = DatabaseManager()
        manager.init_sync_engine(testing=True)

        health = manager.check_health()
        assert health["status"] == "healthy"
        assert health["connected"] is True
        assert health["latency_ms"] is not None
        assert health["error"] is None

        manager.close()

    def test_health_check_not_initialized(self):
        """Test health check when not initialized."""
        from python_ai.db.connection import DatabaseManager

        DatabaseManager.reset()
        manager = DatabaseManager()

        health = manager.check_health()
        assert health["status"] == "unhealthy"
        assert health["connected"] is False
        assert "not initialized" in health["error"]

    def test_close_and_reset(self):
        """Test closing and resetting the manager."""
        from python_ai.db.connection import DatabaseManager

        DatabaseManager.reset()
        manager = DatabaseManager()
        manager.init_sync_engine(testing=True)
        manager.close()
        assert manager._sync_engine is None

        DatabaseManager.reset()
        assert DatabaseManager._instance is None


class TestModels:
    """Tests for SQLAlchemy models."""

    def test_prediction_to_dict(self):
        """Test Prediction.to_dict() method."""
        from python_ai.db.models import Prediction

        pred = Prediction(
            id=1,
            input_data={"feature": 1},
            output_data={"prediction": 0.5},
            model_version="v1.0.0",
            confidence=0.95,
            created_at=datetime(2026, 3, 1, 12, 0, 0),
            latency_ms=10.5,
        )
        d = pred.to_dict()
        assert d["id"] == 1
        assert d["input_data"] == {"feature": 1}
        assert d["model_version"] == "v1.0.0"
        assert d["confidence"] == 0.95
        assert "2026-03-01" in d["created_at"]

    def test_training_session_to_dict(self):
        """Test TrainingSession.to_dict() method."""
        from python_ai.db.models import TrainingSession

        session = TrainingSession(
            id=1,
            model_name="test_model",
            model_version="v1.0.0",
            hyperparameters={"lr": 0.01},
            status="completed",
        )
        d = session.to_dict()
        assert d["model_name"] == "test_model"
        assert d["hyperparameters"] == {"lr": 0.01}

    def test_model_metrics_to_dict(self):
        """Test ModelMetrics.to_dict() method."""
        from python_ai.db.models import ModelMetrics

        metric = ModelMetrics(
            id=1,
            training_session_id=1,
            metric_name="accuracy",
            metric_value=0.95,
            epoch=10,
        )
        d = metric.to_dict()
        assert d["metric_name"] == "accuracy"
        assert d["metric_value"] == 0.95
        assert d["epoch"] == 10

    def test_feature_importance_to_dict(self):
        """Test FeatureImportance.to_dict() method."""
        from python_ai.db.models import FeatureImportance

        importance = FeatureImportance(
            id=1,
            training_session_id=1,
            feature_name="price",
            importance_score=0.85,
            importance_method="shap",
        )
        d = importance.to_dict()
        assert d["feature_name"] == "price"
        assert d["importance_score"] == 0.85

    def test_evolution_history_to_dict(self):
        """Test EvolutionHistory.to_dict() method."""
        from python_ai.db.models import EvolutionHistory

        history = EvolutionHistory(
            id=1,
            generation=5,
            strategy_params={"param": 1},
            performance_score=0.92,
            is_elite=1,
        )
        d = history.to_dict()
        assert d["generation"] == 5
        assert d["is_elite"] is True


class TestRepositories:
    """Tests for repository operations."""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        """Set up test database before each test."""
        from python_ai.db.connection import DatabaseManager
        from python_ai.db.models import Base

        DatabaseManager.reset()
        manager = DatabaseManager()
        manager.init_sync_engine(testing=True)
        # Drop all tables and recreate for clean state
        Base.metadata.drop_all(bind=manager._sync_engine)
        Base.metadata.create_all(bind=manager._sync_engine)
        yield manager
        manager.close()
        DatabaseManager.reset()

    def test_prediction_repository_create(self, setup_db):
        """Test creating a prediction."""
        from python_ai.db.repository import PredictionRepository

        with setup_db.get_sync_session() as session:
            pred = PredictionRepository.create(
                session,
                input_data={"x": 1},
                output_data={"y": 0.5},
                model_version="v1.0.0",
                confidence=0.9,
                latency_ms=5.0,
            )
            assert pred.id is not None
            assert pred.input_data == {"x": 1}

    def test_prediction_repository_get_by_id(self, setup_db):
        """Test getting prediction by ID."""
        from python_ai.db.repository import PredictionRepository

        with setup_db.get_sync_session() as session:
            pred = PredictionRepository.create(
                session,
                input_data={"x": 2},
                output_data={"y": 0.7},
            )
            retrieved = PredictionRepository.get_by_id(session, pred.id)
            assert retrieved is not None
            assert retrieved.input_data == {"x": 2}

    def test_prediction_repository_get_recent(self, setup_db):
        """Test getting recent predictions."""
        from python_ai.db.repository import PredictionRepository

        with setup_db.get_sync_session() as session:
            for i in range(5):
                PredictionRepository.create(
                    session,
                    input_data={"x": i},
                    output_data={"y": i * 0.1},
                )
            recent = PredictionRepository.get_recent(session, limit=3)
            assert len(recent) == 3

    def test_prediction_repository_get_count(self, setup_db):
        """Test getting prediction count."""
        from python_ai.db.repository import PredictionRepository

        with setup_db.get_sync_session() as session:
            for i in range(3):
                PredictionRepository.create(
                    session,
                    input_data={"x": i},
                    output_data={"y": i},
                )
            count = PredictionRepository.get_count(session)
            assert count == 3

    def test_training_session_repository(self, setup_db):
        """Test training session repository operations."""
        from python_ai.db.repository import TrainingSessionRepository

        with setup_db.get_sync_session() as session:
            ts = TrainingSessionRepository.create(
                session,
                model_name="test_model",
                model_version="v1.0.0",
                hyperparameters={"lr": 0.01},
            )
            assert ts.id is not None
            assert ts.status == "running"

            TrainingSessionRepository.complete(session, ts, status="completed")
            assert ts.status == "completed"
            assert ts.completed_at is not None

    def test_model_metrics_repository(self, setup_db):
        """Test model metrics repository operations."""
        from python_ai.db.repository import (
            ModelMetricsRepository,
            TrainingSessionRepository,
        )

        with setup_db.get_sync_session() as session:
            ts = TrainingSessionRepository.create(
                session, model_name="test", model_version="v1"
            )
            metric = ModelMetricsRepository.record_metric(
                session,
                training_session_id=ts.id,
                metric_name="accuracy",
                metric_value=0.95,
                epoch=10,
            )
            assert metric.id is not None

            metrics = ModelMetricsRepository.get_metrics_for_session(
                session, ts.id
            )
            assert len(metrics) == 1

    def test_feature_importance_repository(self, setup_db):
        """Test feature importance repository operations."""
        from python_ai.db.repository import (
            FeatureImportanceRepository,
            TrainingSessionRepository,
        )

        with setup_db.get_sync_session() as session:
            ts = TrainingSessionRepository.create(
                session, model_name="test", model_version="v1"
            )
            importance = FeatureImportanceRepository.record_importance(
                session,
                training_session_id=ts.id,
                feature_name="price",
                importance_score=0.85,
            )
            assert importance.id is not None

            importances = FeatureImportanceRepository.get_for_session(
                session, ts.id
            )
            assert len(importances) == 1

    def test_evolution_history_repository(self, setup_db):
        """Test evolution history repository operations."""
        from python_ai.db.repository import EvolutionHistoryRepository

        with setup_db.get_sync_session() as session:
            history = EvolutionHistoryRepository.record_generation(
                session,
                generation=1,
                strategy_params={"x": 1},
                performance_score=0.9,
                is_elite=True,
            )
            assert history.id is not None

            best = EvolutionHistoryRepository.get_best_strategies(
                session, limit=5
            )
            assert len(best) == 1
            assert best[0].is_elite == 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_init_db(self):
        """Test init_db convenience function."""
        from python_ai.db.connection import DatabaseManager, close_db, init_db

        DatabaseManager.reset()
        manager = init_db(testing=True)
        assert manager._sync_engine is not None
        close_db()

    def test_get_db_manager(self):
        """Test get_db_manager function."""
        from python_ai.db.connection import (
            DatabaseManager,
            close_db,
            get_db_manager,
        )

        DatabaseManager.reset()
        manager = get_db_manager()
        assert manager is not None
        assert isinstance(manager, DatabaseManager)
        close_db()


class TestPackageImports:
    """Tests for package-level imports."""

    def test_package_imports(self):
        """Test that all exports are available from package."""
        from python_ai.db import (
            Base,
            DatabaseConfig,
            DatabaseManager,
            FeatureImportance,
            ModelMetrics,
            Prediction,
            TrainingSession,
            close_db,
            get_async_db,
            get_database_url,
            get_db,
            init_db,
        )

        assert Base is not None
        assert DatabaseConfig is not None
        assert DatabaseManager is not None
        assert Prediction is not None
        assert TrainingSession is not None
        assert ModelMetrics is not None
        assert FeatureImportance is not None
        assert get_db is not None
        assert get_async_db is not None
        assert init_db is not None
        assert close_db is not None
        assert get_database_url is not None
