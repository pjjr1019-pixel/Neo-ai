"""
I/O Timeout Wrapper for NEO Hybrid AI.

Provides a clean interface for enforcing timeouts on any
callable — useful for database queries, HTTP requests,
and exchange API calls that may hang.
"""

import functools
import logging
import threading
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TimeoutError(Exception):
    """Raised when an operation exceeds its time limit."""


def _timeout_thread(
    func: Callable[..., T],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    timeout: float,
) -> T:
    """Run *func* in a daemon thread with a deadline.

    Works on all platforms (Windows, Linux, macOS).

    Args:
        func: Callable to execute.
        args: Positional arguments.
        kwargs: Keyword arguments.
        timeout: Maximum execution time in seconds.

    Returns:
        The return value of *func*.

    Raises:
        TimeoutError: If the function doesn't finish
            within *timeout* seconds.
    """
    result: list[Any] = []
    exception: list[Optional[BaseException]] = [None]

    def _target() -> None:
        """Execute the callable in a thread."""
        try:
            result.append(func(*args, **kwargs))
        except BaseException as exc:
            exception[0] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError(f"{func.__name__} exceeded {timeout}s timeout")
    if exception[0] is not None:
        raise exception[0]
    return result[0]  # type: ignore[no-any-return]


def with_timeout(
    timeout: float = 10.0,
) -> Callable[..., Any]:
    """Decorator that enforces a timeout on a function.

    Uses a daemon-thread approach so it works on all
    platforms (including Windows where ``signal.SIGALRM``
    is unavailable).

    Args:
        timeout: Maximum seconds the function may run.

    Returns:
        Decorated function.

    Example::

        @with_timeout(5.0)
        def query_db() -> list:
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        """Wrap *func* with a timeout guard."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            """Run with timeout enforcement."""
            return _timeout_thread(func, args, kwargs, timeout)

        return wrapper

    return decorator


def timeout_call(
    func: Callable[..., T],
    args: Optional[tuple[Any, ...]] = None,
    kwargs: Optional[dict[str, Any]] = None,
    timeout: float = 10.0,
) -> T:
    """Call *func* with a timeout (non-decorator API).

    Args:
        func: Callable to execute.
        args: Positional arguments.
        kwargs: Keyword arguments.
        timeout: Maximum seconds.

    Returns:
        The return value of *func*.

    Raises:
        TimeoutError: If the deadline is exceeded.
    """
    return _timeout_thread(func, args or (), kwargs or {}, timeout)
