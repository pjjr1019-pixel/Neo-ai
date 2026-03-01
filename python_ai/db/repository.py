"""
Repository layer for database operations.

Provides CRUD operations and query methods for all models.
Follows repository pattern for clean separation of concerns.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from python_ai.db.models import (
    EvolutionHistory,
    FeatureImportance,
    ModelMetrics,
    Prediction,
    TrainingSession,
)


class PredictionRepository:
    """Repository for Prediction model operations."""

    @staticmethod
    def create(
        session: Session,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        model_version: str = "v1.0.0",
        confidence: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ) -> Prediction:
        """Create a new prediction record.

        Args:
            session: Database session.
            input_data: Input features as dict.
            output_data: Prediction output as dict.
            model_version: Model version string.
            confidence: Prediction confidence score.
            latency_ms: Prediction latency in milliseconds.

        Returns:
            Prediction: Created prediction record.
        """
        prediction = Prediction(
            input_data=input_data,
            output_data=output_data,
            model_version=model_version,
            confidence=confidence,
            latency_ms=latency_ms,
        )
        session.add(prediction)
        session.flush()
        return prediction

    @staticmethod
    async def create_async(
        session: AsyncSession,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        model_version: str = "v1.0.0",
        confidence: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ) -> Prediction:
        """Create a new prediction record asynchronously.

        Args:
            session: Async database session.
            input_data: Input features as dict.
            output_data: Prediction output as dict.
            model_version: Model version string.
            confidence: Prediction confidence score.
            latency_ms: Prediction latency in milliseconds.

        Returns:
            Prediction: Created prediction record.
        """
        prediction = Prediction(
            input_data=input_data,
            output_data=output_data,
            model_version=model_version,
            confidence=confidence,
            latency_ms=latency_ms,
        )
        session.add(prediction)
        await session.flush()
        return prediction

    @staticmethod
    def get_by_id(
        session: Session, prediction_id: int
    ) -> Optional[Prediction]:
        """Get prediction by ID.

        Args:
            session: Database session.
            prediction_id: Prediction ID to retrieve.

        Returns:
            Optional[Prediction]: Prediction if found, None otherwise.
        """
        return session.get(Prediction, prediction_id)

    @staticmethod
    def get_recent(
        session: Session, limit: int = 100, model_version: Optional[str] = None
    ) -> List[Prediction]:
        """Get recent predictions.

        Args:
            session: Database session.
            limit: Maximum number of predictions to return.
            model_version: Optional filter by model version.

        Returns:
            List[Prediction]: List of recent predictions.
        """
        query = (
            select(Prediction)
            .order_by(desc(Prediction.created_at))
            .limit(limit)
        )
        if model_version:
            query = query.where(Prediction.model_version == model_version)
        result = session.execute(query)
        return list(result.scalars().all())

    @staticmethod
    def get_count(session: Session) -> int:
        """Get total prediction count.

        Args:
            session: Database session.

        Returns:
            int: Total number of predictions.
        """
        result = session.execute(select(func.count(Prediction.id)))
        return result.scalar() or 0


class TrainingSessionRepository:
    """Repository for TrainingSession model operations."""

    @staticmethod
    def create(
        session: Session,
        model_name: str,
        model_version: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_data_hash: Optional[str] = None,
    ) -> TrainingSession:
        """Create a new training session.

        Args:
            session: Database session.
            model_name: Name of the model.
            model_version: Version of the model.
            hyperparameters: Hyperparameters used for training.
            training_data_hash: Hash of training data.

        Returns:
            TrainingSession: Created training session.
        """
        training_session = TrainingSession(
            model_name=model_name,
            model_version=model_version,
            hyperparameters=hyperparameters,
            training_data_hash=training_data_hash,
        )
        session.add(training_session)
        session.flush()
        return training_session

    @staticmethod
    def complete(
        session: Session,
        training_session: TrainingSession,
        status: str = "completed",
        error_message: Optional[str] = None,
    ) -> TrainingSession:
        """Mark training session as completed.

        Args:
            session: Database session.
            training_session: Training session to update.
            status: Final status (completed, failed).
            error_message: Error message if failed.

        Returns:
            TrainingSession: Updated training session.
        """
        now = datetime.utcnow()
        training_session.completed_at = now  # type: ignore
        training_session.status = status  # type: ignore
        training_session.error_message = error_message  # type: ignore
        session.flush()
        return training_session

    @staticmethod
    def get_by_id(
        session: Session, session_id: int
    ) -> Optional[TrainingSession]:
        """Get training session by ID.

        Args:
            session: Database session.
            session_id: Training session ID.

        Returns:
            Optional[TrainingSession]: Training session if found.
        """
        return session.get(TrainingSession, session_id)

    @staticmethod
    def get_latest(
        session: Session, model_name: Optional[str] = None
    ) -> Optional[TrainingSession]:
        """Get latest completed training session.

        Args:
            session: Database session.
            model_name: Optional filter by model name.

        Returns:
            Optional[TrainingSession]: Latest training session.
        """
        query = (
            select(TrainingSession)
            .where(TrainingSession.status == "completed")
            .order_by(desc(TrainingSession.completed_at))
            .limit(1)
        )
        if model_name:
            query = query.where(TrainingSession.model_name == model_name)
        result = session.execute(query)
        return result.scalar_one_or_none()


class ModelMetricsRepository:
    """Repository for ModelMetrics operations."""

    @staticmethod
    def record_metric(
        session: Session,
        training_session_id: int,
        metric_name: str,
        metric_value: float,
        epoch: Optional[int] = None,
    ) -> ModelMetrics:
        """Record a training metric.

        Args:
            session: Database session.
            training_session_id: Training session ID.
            metric_name: Name of the metric.
            metric_value: Value of the metric.
            epoch: Training epoch number.

        Returns:
            ModelMetrics: Created metric record.
        """
        metric = ModelMetrics(
            training_session_id=training_session_id,
            metric_name=metric_name,
            metric_value=metric_value,
            epoch=epoch,
        )
        session.add(metric)
        session.flush()
        return metric

    @staticmethod
    def get_metrics_for_session(
        session: Session, training_session_id: int
    ) -> List[ModelMetrics]:
        """Get all metrics for a training session.

        Args:
            session: Database session.
            training_session_id: Training session ID.

        Returns:
            List[ModelMetrics]: List of metrics.
        """
        query = select(ModelMetrics).where(
            ModelMetrics.training_session_id == training_session_id
        )
        result = session.execute(query)
        return list(result.scalars().all())


class FeatureImportanceRepository:
    """Repository for FeatureImportance operations."""

    @staticmethod
    def record_importance(
        session: Session,
        training_session_id: int,
        feature_name: str,
        importance_score: float,
        importance_method: str = "shap",
    ) -> FeatureImportance:
        """Record feature importance score.

        Args:
            session: Database session.
            training_session_id: Training session ID.
            feature_name: Name of the feature.
            importance_score: Importance score.
            importance_method: Method used for calculation.

        Returns:
            FeatureImportance: Created feature importance record.
        """
        importance = FeatureImportance(
            training_session_id=training_session_id,
            feature_name=feature_name,
            importance_score=importance_score,
            importance_method=importance_method,
        )
        session.add(importance)
        session.flush()
        return importance

    @staticmethod
    def get_for_session(
        session: Session, training_session_id: int
    ) -> List[FeatureImportance]:
        """Get feature importances for a training session.

        Args:
            session: Database session.
            training_session_id: Training session ID.

        Returns:
            List[FeatureImportance]: List of feature importances.
        """
        query = (
            select(FeatureImportance)
            .where(
                FeatureImportance.training_session_id == training_session_id
            )
            .order_by(desc(FeatureImportance.importance_score))
        )
        result = session.execute(query)
        return list(result.scalars().all())


class EvolutionHistoryRepository:
    """Repository for EvolutionHistory operations."""

    @staticmethod
    def record_generation(
        session: Session,
        generation: int,
        strategy_params: Dict[str, Any],
        performance_score: float,
        is_elite: bool = False,
    ) -> EvolutionHistory:
        """Record evolution generation data.

        Args:
            session: Database session.
            generation: Generation number.
            strategy_params: Strategy parameters.
            performance_score: Performance score.
            is_elite: Whether this is an elite strategy.

        Returns:
            EvolutionHistory: Created evolution history record.
        """
        history = EvolutionHistory(
            generation=generation,
            strategy_params=strategy_params,
            performance_score=performance_score,
            is_elite=1 if is_elite else 0,
        )
        session.add(history)
        session.flush()
        return history

    @staticmethod
    def get_best_strategies(
        session: Session, limit: int = 10
    ) -> List[EvolutionHistory]:
        """Get best performing strategies.

        Args:
            session: Database session.
            limit: Maximum number of strategies.

        Returns:
            List[EvolutionHistory]: Best strategies by performance.
        """
        query = (
            select(EvolutionHistory)
            .order_by(desc(EvolutionHistory.performance_score))
            .limit(limit)
        )
        result = session.execute(query)
        return list(result.scalars().all())

    @staticmethod
    def get_generation(
        session: Session, generation: int
    ) -> List[EvolutionHistory]:
        """Get all strategies from a generation.

        Args:
            session: Database session.
            generation: Generation number.

        Returns:
            List[EvolutionHistory]: Strategies from the generation.
        """
        query = select(EvolutionHistory).where(
            EvolutionHistory.generation == generation
        )
        result = session.execute(query)
        return list(result.scalars().all())
