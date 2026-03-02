"""
Domain-specific exception hierarchy for NEO Hybrid AI.

All NEO exceptions inherit from ``NeoBaseError`` so callers can
catch a single base class for any NEO-specific failure.
"""

__all__ = [
    "NeoAPIError",
    "NeoAuthError",
    "NeoBaseError",
    "NeoConfigError",
    "NeoDataError",
    "NeoFeatureError",
    "NeoModelError",
    "NeoModelIntegrityError",
    "NeoModelNotTrainedError",
    "NeoModelTrainingError",
    "NeoPipelineError",
    "NeoPermissionError",
    "NeoRateLimitError",
    "NeoTokenError",
]


class NeoBaseError(Exception):
    """Root exception for all NEO Hybrid AI errors.

    Attributes:
        detail: Human-readable error description.
        context: Optional dict of extra context for logging.
    """

    def __init__(
        self,
        detail: str = "An internal error occurred",
        *,
        context: dict | None = None,
    ) -> None:
        """Initialise with a human-readable detail message.

        Args:
            detail: Error description.
            context: Optional dict of extra debugging context.
        """
        self.detail = detail
        self.context = context or {}
        super().__init__(detail)


# ── Data / Pipeline errors ────────────────────────────────────


class NeoDataError(NeoBaseError):
    """Raised when data ingestion, validation, or processing fails."""


class NeoFeatureError(NeoDataError):
    """Raised when feature engineering or indicator calculation fails."""


class NeoPipelineError(NeoDataError):
    """Raised when the data pipeline encounters an error."""


# ── Model / ML errors ─────────────────────────────────────────


class NeoModelError(NeoBaseError):
    """Raised for model training, prediction, or persistence errors."""


class NeoModelNotTrainedError(NeoModelError):
    """Raised when prediction is attempted on an untrained model."""


class NeoModelIntegrityError(NeoModelError):
    """Raised when a saved model file fails integrity verification."""


class NeoModelTrainingError(NeoModelError):
    """Raised when model training fails (insufficient data, etc.)."""


# ── API / Endpoint errors ─────────────────────────────────────


class NeoAPIError(NeoBaseError):
    """Raised for API-layer errors (validation, routing, etc.)."""


class NeoRateLimitError(NeoAPIError):
    """Raised when a client exceeds the rate limit."""


# ── Auth errors ────────────────────────────────────────────────


class NeoAuthError(NeoBaseError):
    """Raised for authentication / authorization failures."""


class NeoTokenError(NeoAuthError):
    """Raised when JWT creation, decoding, or refresh fails."""


class NeoPermissionError(NeoAuthError):
    """Raised when the user lacks required permissions."""


# ── Configuration errors ──────────────────────────────────────


class NeoConfigError(NeoBaseError):
    """Raised for configuration or environment setup errors."""
