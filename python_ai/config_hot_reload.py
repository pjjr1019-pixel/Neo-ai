"""
Hot-Reload Strategy Configs for NEO Hybrid AI.

Watches a strategy configuration file on disk and
triggers callbacks when the file changes, allowing
live reconfiguration without restarts.
"""

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConfigHotReloader:
    """File-watcher that reloads JSON config on change.

    Polls the config file at a configurable interval
    and invokes registered callbacks when the content
    hash changes.

    Args:
        config_path: Path to the JSON config file.
        poll_interval: Seconds between checks.
    """

    def __init__(
        self,
        config_path: str = "strategy_config.json",
        poll_interval: float = 5.0,
    ) -> None:
        """Initialise the hot reloader."""
        self._path = config_path
        self._interval = poll_interval
        self._last_hash: Optional[str] = None
        self._config: Dict[str, Any] = {}
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ── config access ─────────────────────────────────

    @property
    def config(self) -> Dict[str, Any]:
        """Return the current config snapshot."""
        with self._lock:
            return dict(self._config)

    def _file_hash(self) -> Optional[str]:
        """Compute SHA-256 of the config file."""
        if not os.path.exists(self._path):
            return None
        with open(self._path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    def _load(self) -> bool:
        """Load config from disk if changed.

        Returns:
            ``True`` if config was reloaded.
        """
        current_hash = self._file_hash()
        if current_hash is None:
            return False
        if current_hash == self._last_hash:
            return False

        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                new_cfg = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(
                "Failed to parse config %s: %s",
                self._path,
                exc,
            )
            return False

        with self._lock:
            self._config = new_cfg
        self._last_hash = current_hash
        logger.info("Config reloaded from %s", self._path)
        return True

    # ── callbacks ─────────────────────────────────────

    def on_change(
        self,
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Register a callback for config changes.

        Args:
            callback: Receives the new config dict.
        """
        self._callbacks.append(callback)

    def _notify(self) -> None:
        """Invoke all registered callbacks."""
        cfg = self.config
        for cb in self._callbacks:
            try:
                cb(cfg)
            except Exception as exc:
                logger.error("Callback error: %s", exc)

    # ── polling loop ──────────────────────────────────

    def _poll_loop(self) -> None:
        """Background polling loop."""
        while self._running:
            if self._load():
                self._notify()
            time.sleep(self._interval)

    def start(self) -> None:
        """Start the background file watcher."""
        if self._running:
            return
        self._load()
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Config hot-reloader started (%.1fs interval)",
            self._interval,
        )

    def stop(self) -> None:
        """Stop the background watcher."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1)
        logger.info("Config hot-reloader stopped")

    def reload_now(self) -> bool:
        """Force an immediate reload check.

        Returns:
            ``True`` if config was updated.
        """
        changed = self._load()
        if changed:
            self._notify()
        return changed
