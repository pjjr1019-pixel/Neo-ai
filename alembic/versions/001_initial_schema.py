"""Initial schema — create all NEO tables.

Revision ID: 001
Revises: None
Create Date: 2026-03-01
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create predictions, training_sessions, model_metrics,
    feature_importance, and evolution_history tables."""
    op.create_table(
        "predictions",
        sa.Column(
            "id", sa.Integer(), autoincrement=True, nullable=False
        ),
        sa.Column("input_data", sa.JSON(), nullable=False),
        sa.Column("output_data", sa.JSON(), nullable=False),
        sa.Column(
            "model_version",
            sa.String(50),
            nullable=False,
            server_default="v1.0.0",
        ),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_predictions_created_at",
        "predictions",
        ["created_at"],
    )
    op.create_index(
        "ix_predictions_model_version",
        "predictions",
        ["model_version"],
    )

    op.create_table(
        "training_sessions",
        sa.Column(
            "id", sa.Integer(), autoincrement=True, nullable=False
        ),
        sa.Column(
            "model_name",
            sa.String(100),
            nullable=False,
        ),
        sa.Column(
            "model_version",
            sa.String(50),
            nullable=False,
        ),
        sa.Column(
            "hyperparameters", sa.JSON(), nullable=True
        ),
        sa.Column(
            "training_data_hash",
            sa.String(64),
            nullable=True,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "completed_at", sa.DateTime(), nullable=True
        ),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "error_message", sa.Text(), nullable=True
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "model_metrics",
        sa.Column(
            "id", sa.Integer(), autoincrement=True, nullable=False
        ),
        sa.Column(
            "training_session_id",
            sa.Integer(),
            sa.ForeignKey("training_sessions.id"),
            nullable=False,
        ),
        sa.Column(
            "metric_name",
            sa.String(100),
            nullable=False,
        ),
        sa.Column(
            "metric_value", sa.Float(), nullable=False
        ),
        sa.Column(
            "recorded_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "feature_importance",
        sa.Column(
            "id", sa.Integer(), autoincrement=True, nullable=False
        ),
        sa.Column(
            "training_session_id",
            sa.Integer(),
            sa.ForeignKey("training_sessions.id"),
            nullable=False,
        ),
        sa.Column(
            "feature_name",
            sa.String(100),
            nullable=False,
        ),
        sa.Column(
            "importance_score",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "recorded_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "evolution_history",
        sa.Column(
            "id", sa.Integer(), autoincrement=True, nullable=False
        ),
        sa.Column(
            "generation", sa.Integer(), nullable=False
        ),
        sa.Column(
            "strategy_params", sa.JSON(), nullable=False
        ),
        sa.Column(
            "fitness_score", sa.Float(), nullable=False
        ),
        sa.Column(
            "sharpe_ratio", sa.Float(), nullable=True
        ),
        sa.Column(
            "max_drawdown", sa.Float(), nullable=True
        ),
        sa.Column(
            "win_rate", sa.Float(), nullable=True
        ),
        sa.Column(
            "total_return", sa.Float(), nullable=True
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Drop all tables in reverse order."""
    op.drop_table("feature_importance")
    op.drop_table("model_metrics")
    op.drop_table("evolution_history")
    op.drop_table("training_sessions")
    op.drop_table("predictions")
