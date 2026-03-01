"""
SQLAlchemy ORM models for NEO Hybrid AI.

Defines database tables for predictions, training sessions,
model metrics, and feature importance tracking.
"""

from datetime import datetime
from typing import Any, Dict

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class Prediction(Base):
    """Model for storing prediction requests and results.

    Attributes:
        id: Unique identifier for the prediction.
        input_data: JSON representation of input features.
        output_data: JSON representation of prediction output.
        model_version: Version of the model used for prediction.
        confidence: Prediction confidence score.
        created_at: Timestamp of prediction creation.
        latency_ms: Prediction latency in milliseconds.
    """

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    input_data = Column(JSON, nullable=False)
    output_data = Column(JSON, nullable=False)
    model_version = Column(String(50), nullable=False, default="v1.0.0")
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    latency_ms = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_predictions_created_at", "created_at"),
        Index("ix_predictions_model_version", "model_version"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to dictionary representation."""
        return {
            "id": self.id,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "model_version": self.model_version,
            "confidence": self.confidence,
            "created_at": (
                self.created_at.isoformat() if self.created_at else None
            ),
            "latency_ms": self.latency_ms,
        }


class TrainingSession(Base):
    """Model for tracking training sessions.

    Attributes:
        id: Unique identifier for the training session.
        model_name: Name of the model being trained.
        model_version: Version string for the trained model.
        hyperparameters: JSON of hyperparameters used.
        training_data_hash: Hash of training data for reproducibility.
        started_at: When training started.
        completed_at: When training completed.
        status: Training status (running, completed, failed).
        error_message: Error details if training failed.
    """

    __tablename__ = "training_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    hyperparameters = Column(JSON, nullable=True)
    training_data_hash = Column(String(64), nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(20), default="running", nullable=False)
    error_message = Column(Text, nullable=True)

    # Relationship to metrics
    metrics = relationship(
        "ModelMetrics",
        back_populates="training_session",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_training_sessions_model_name", "model_name"),
        Index("ix_training_sessions_status", "status"),
        Index("ix_training_sessions_started_at", "started_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert training session to dictionary representation."""
        return {
            "id": self.id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "hyperparameters": self.hyperparameters,
            "training_data_hash": self.training_data_hash,
            "started_at": (
                self.started_at.isoformat() if self.started_at else None
            ),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "status": self.status,
            "error_message": self.error_message,
        }


class ModelMetrics(Base):
    """Model for storing evaluation metrics.

    Attributes:
        id: Unique identifier for the metrics record.
        training_session_id: Foreign key to training session.
        metric_name: Name of the metric (accuracy, f1, loss, etc.).
        metric_value: Numeric value of the metric.
        epoch: Training epoch (if applicable).
        recorded_at: When the metric was recorded.
    """

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    training_session_id = Column(
        Integer, ForeignKey("training_sessions.id"), nullable=False
    )
    metric_name = Column(String(50), nullable=False)
    metric_value = Column(Float, nullable=False)
    epoch = Column(Integer, nullable=True)
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship back to training session
    training_session = relationship(
        "TrainingSession", back_populates="metrics"
    )

    __table_args__ = (
        Index("ix_model_metrics_training_session_id", "training_session_id"),
        Index("ix_model_metrics_metric_name", "metric_name"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return {
            "id": self.id,
            "training_session_id": self.training_session_id,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "epoch": self.epoch,
            "recorded_at": (
                self.recorded_at.isoformat() if self.recorded_at else None
            ),
        }


class FeatureImportance(Base):
    """Model for storing feature importance scores.

    Attributes:
        id: Unique identifier.
        training_session_id: Foreign key to training session.
        feature_name: Name of the feature.
        importance_score: Calculated importance score.
        importance_method: Method used to calculate importance.
    """

    __tablename__ = "feature_importances"

    id = Column(Integer, primary_key=True, autoincrement=True)
    training_session_id = Column(
        Integer, ForeignKey("training_sessions.id"), nullable=False
    )
    feature_name = Column(String(100), nullable=False)
    importance_score = Column(Float, nullable=False)
    importance_method = Column(String(50), default="shap", nullable=False)

    __table_args__ = (
        Index(
            "ix_feature_importances_training_session_id",
            "training_session_id",
        ),
        Index("ix_feature_importances_feature_name", "feature_name"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert feature importance to dictionary representation."""
        return {
            "id": self.id,
            "training_session_id": self.training_session_id,
            "feature_name": self.feature_name,
            "importance_score": self.importance_score,
            "importance_method": self.importance_method,
        }


class EvolutionHistory(Base):
    """Model for tracking evolution engine history.

    Attributes:
        id: Unique identifier.
        generation: Generation number.
        strategy_params: JSON of strategy parameters.
        performance_score: Performance score achieved.
        is_elite: Whether this was an elite strategy.
        created_at: When this record was created.
    """

    __tablename__ = "evolution_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    generation = Column(Integer, nullable=False)
    strategy_params = Column(JSON, nullable=False)
    performance_score = Column(Float, nullable=False)
    # SQLite compat: use int
    is_elite = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_evolution_history_generation", "generation"),
        Index("ix_evolution_history_performance", "performance_score"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert evolution history to dictionary representation."""
        return {
            "id": self.id,
            "generation": self.generation,
            "strategy_params": self.strategy_params,
            "performance_score": self.performance_score,
            "is_elite": bool(self.is_elite),
            "created_at": (
                self.created_at.isoformat() if self.created_at else None
            ),
        }
