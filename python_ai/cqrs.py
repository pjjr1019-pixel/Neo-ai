"""
CQRS (Command Query Responsibility Segregation) for NEO Hybrid AI.

Separates write (command) and read (query) paths to
allow independent scaling and optimisation.  Commands
mutate state; queries are side-effect-free reads.
"""

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
)

logger = logging.getLogger(__name__)

R = TypeVar("R")  # Result type


# ------------------------------------------------------------------
# Base messages
# ------------------------------------------------------------------


@dataclass
class Command:
    """Base class for all commands (write operations).

    Attributes:
        command_type: Auto-populated with the class name.
    """

    command_type: str = field(init=False)

    def __post_init__(self) -> None:
        """Set command_type from the class name."""
        self.command_type = self.__class__.__name__


@dataclass
class Query:
    """Base class for all queries (read operations).

    Attributes:
        query_type: Auto-populated with the class name.
    """

    query_type: str = field(init=False)

    def __post_init__(self) -> None:
        """Set query_type from the class name."""
        self.query_type = self.__class__.__name__


@dataclass
class CommandResult:
    """Result returned by a command handler.

    Attributes:
        success: Whether the command succeeded.
        data: Optional result payload.
        error: Error message on failure.
    """

    success: bool = True
    data: Optional[Any] = None
    error: Optional[str] = None


# ------------------------------------------------------------------
# Handler protocols
# ------------------------------------------------------------------


class CommandHandler(ABC):
    """Abstract handler for a specific command type."""

    @abstractmethod
    def handle(self, command: Command) -> CommandResult:
        """Execute the command.

        Args:
            command: The command to process.

        Returns:
            A :class:`CommandResult`.
        """


class QueryHandler(ABC, Generic[R]):
    """Abstract handler for a specific query type."""

    @abstractmethod
    def handle(self, query: Query) -> R:
        """Execute the query.

        Args:
            query: The query to process.

        Returns:
            The query result.
        """


# ------------------------------------------------------------------
# Buses
# ------------------------------------------------------------------


class CommandBus:
    """Dispatches commands to their registered handlers.

    Each command type string maps to exactly one handler.
    Middleware functions can wrap the handler invocation
    for logging, validation, etc.

    Example::

        bus = CommandBus()
        bus.register("PlaceOrder", PlaceOrderHandler())
        result = bus.dispatch(PlaceOrder(symbol="BTC"))
    """

    def __init__(self) -> None:
        """Initialise with empty handler registry."""
        self._handlers: Dict[str, CommandHandler] = {}
        self._middleware: List[
            Callable[
                [Command, Callable[[Command], CommandResult]],
                CommandResult,
            ]
        ] = []
        self._lock = threading.Lock()
        self._stats: Dict[str, int] = {
            "dispatched": 0,
            "succeeded": 0,
            "failed": 0,
        }

    def register(self, command_type: str, handler: CommandHandler) -> None:
        """Register a handler for *command_type*.

        Args:
            command_type: The ``command_type`` string.
            handler: Handler instance.
        """
        with self._lock:
            self._handlers[command_type] = handler
        logger.debug(
            "Registered command handler for %s",
            command_type,
        )

    def add_middleware(
        self,
        mw: Callable[
            [Command, Callable[[Command], CommandResult]],
            CommandResult,
        ],
    ) -> None:
        """Add middleware that wraps handler execution.

        Middleware signature::

            def mw(cmd, next_fn) -> CommandResult:
                # pre-processing
                result = next_fn(cmd)
                # post-processing
                return result

        Args:
            mw: Middleware function.
        """
        self._middleware.append(mw)

    def dispatch(self, command: Command) -> CommandResult:
        """Dispatch a command to its handler.

        Args:
            command: The command to dispatch.

        Returns:
            :class:`CommandResult` from the handler.

        Raises:
            KeyError: If no handler is registered.
        """
        ct = command.command_type
        with self._lock:
            handler = self._handlers.get(ct)
            self._stats["dispatched"] += 1
        if handler is None:
            with self._lock:
                self._stats["failed"] += 1
            raise KeyError(f"No handler for command '{ct}'")

        def _invoke(cmd: Command) -> CommandResult:
            """Invoke the handler directly."""
            return handler.handle(cmd)

        chain = _invoke
        for mw in reversed(self._middleware):

            def _wrap(
                cmd: Command,
                _mw: Any = mw,
                _next: Any = chain,
            ) -> CommandResult:
                """Apply middleware then next handler."""
                return _mw(cmd, _next)  # type: ignore[no-any-return]

            chain = _wrap

        result = chain(command)
        with self._lock:
            if result.success:
                self._stats["succeeded"] += 1
            else:
                self._stats["failed"] += 1
        return result

    @property
    def stats(self) -> Dict[str, int]:
        """Return dispatch statistics."""
        with self._lock:
            return dict(self._stats)


class QueryBus:
    """Dispatches queries to their registered handlers.

    Example::

        bus = QueryBus()
        bus.register("GetPortfolio", GetPortfolioHandler())
        portfolio = bus.dispatch(GetPortfolio(user="alice"))
    """

    def __init__(self) -> None:
        """Initialise with empty handler registry."""
        self._handlers: Dict[str, QueryHandler[Any]] = {}
        self._lock = threading.Lock()
        self._stats: Dict[str, int] = {
            "dispatched": 0,
            "succeeded": 0,
            "failed": 0,
        }

    def register(
        self,
        query_type: str,
        handler: QueryHandler[Any],
    ) -> None:
        """Register a handler for *query_type*.

        Args:
            query_type: The ``query_type`` string.
            handler: Handler instance.
        """
        with self._lock:
            self._handlers[query_type] = handler
        logger.debug("Registered query handler for %s", query_type)

    def dispatch(self, query: Query) -> Any:
        """Dispatch a query to its handler.

        Args:
            query: The query to dispatch.

        Returns:
            The handler's result.

        Raises:
            KeyError: If no handler is registered.
        """
        qt = query.query_type
        with self._lock:
            handler = self._handlers.get(qt)
            self._stats["dispatched"] += 1
        if handler is None:
            with self._lock:
                self._stats["failed"] += 1
            raise KeyError(f"No handler for query '{qt}'")
        try:
            result = handler.handle(query)
            with self._lock:
                self._stats["succeeded"] += 1
            return result
        except Exception:
            with self._lock:
                self._stats["failed"] += 1
            raise

    @property
    def stats(self) -> Dict[str, int]:
        """Return dispatch statistics."""
        with self._lock:
            return dict(self._stats)
