"""
Retry with exponential backoff and jitter for NEO Hybrid AI.

Provides a decorator and a callable wrapper that retries
failed operations with configurable backoff, jitter,
and per-exception filtering.
"""

import functools
import logging
import random
import time
from typing import (
    Any,
    Callable,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default set of transient / retryable exceptions.
DEFAULT_RETRYABLE: Sequence[Type[BaseException]] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[Sequence[Type[BaseException]]] = None,
) -> Callable[..., Any]:
    """Decorator for retrying a function on transient errors.

    Uses exponential backoff with optional full jitter
    (as recommended by AWS architecture blog).

    Args:
        max_attempts: Total number of tries (including first).
        base_delay: Initial delay in seconds.
        max_delay: Cap on computed delay in seconds.
        exponential_base: Multiplier per attempt.
        jitter: Whether to add random jitter.
        retryable_exceptions: Exception types to retry on.
            Defaults to ``ConnectionError``,
            ``TimeoutError``, ``OSError``.

    Returns:
        Decorated function with retry logic.

    Example::

        @retry(max_attempts=5, base_delay=0.5)
        def fetch_price(symbol: str) -> float:
            ...
    """
    if retryable_exceptions is None:
        retryable_exceptions = DEFAULT_RETRYABLE

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        """Wrap *func* with retry logic."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            """Execute with retries on failure."""
            last_exc: Optional[BaseException] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(retryable_exceptions) as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__name__,
                            max_attempts,
                            exc,
                        )
                        raise
                    delay = min(
                        base_delay * (exponential_base ** (attempt - 1)),
                        max_delay,
                    )
                    if jitter:
                        delay = random.uniform(0, delay)
                    logger.warning(
                        "%s attempt %d/%d failed (%s), " "retrying in %.2fs",
                        func.__name__,
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            # Should not reach here, but satisfy mypy.
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


def retry_call(
    func: Callable[..., T],
    args: Optional[Sequence[Any]] = None,
    kwargs: Optional[dict[str, Any]] = None,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[Sequence[Type[BaseException]]] = None,
) -> T:
    """Call *func* with retry semantics (non-decorator API).

    Useful when the function cannot be decorated (e.g.
    third-party code or lambdas).

    Args:
        func: Callable to execute.
        args: Positional arguments for *func*.
        kwargs: Keyword arguments for *func*.
        max_attempts: Total tries.
        base_delay: Initial wait (seconds).
        max_delay: Maximum wait (seconds).
        exponential_base: Backoff multiplier.
        jitter: Add randomness to delay.
        retryable_exceptions: Which exceptions to retry.

    Returns:
        The return value of *func* on success.
    """

    @retry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions,
    )
    def _inner() -> T:
        """Invoke the wrapped callable."""
        return func(*(args or ()), **(kwargs or {}))

    return _inner()  # type: ignore[no-any-return]
