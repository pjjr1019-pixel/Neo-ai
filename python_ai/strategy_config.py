"""
Strategy Configuration loader for NEO Hybrid AI.

Loads trading strategy parameters from YAML or JSON files so
strategies can be iterated without code changes.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Default strategy parameters (used when no config file exists)
DEFAULT_STRATEGY: Dict[str, Any] = {
    "name": "default",
    "version": "1.0",
    "symbols": ["BTC/USDT", "ETH/USDT"],
    "timeframe": "1h",
    "risk": {
        "max_position_pct": 0.25,
        "stop_loss_atr_mult": 2.0,
        "take_profit_atr_mult": 3.0,
        "max_drawdown_pct": 0.20,
        "kelly_cap": 0.25,
    },
    "model": {
        "retrain_threshold": 50,
        "drift_z_threshold": 2.0,
        "accuracy_floor": 0.3,
        "cooldown_seconds": 3600,
    },
    "execution": {
        "fee_pct": 0.001,
        "slippage_pct": 0.0005,
        "paper_mode": True,
    },
    "indicators": {
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "bb_period": 20,
        "bb_std": 2.0,
        "atr_period": 14,
        "sma_period": 20,
    },
}


class StrategyConfig:
    """Parsed and validated strategy configuration.

    Provides attribute-style access to commonly used parameters
    and nested dictionaries for everything else.

    Attributes:
        raw: The full configuration dictionary.
        name: Strategy name.
        symbols: List of trading symbols.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise from a config dict.

        Args:
            config: Strategy configuration dictionary.
        """
        self.raw = config
        self.name: str = config.get("name", "unnamed")
        self.version: str = config.get("version", "0.0")
        self.symbols: List[str] = config.get(
            "symbols", DEFAULT_STRATEGY["symbols"]
        )
        self.timeframe: str = config.get("timeframe", "1h")
        self.risk: Dict[str, Any] = config.get(
            "risk", DEFAULT_STRATEGY["risk"]
        )
        self.model_cfg: Dict[str, Any] = config.get(
            "model", DEFAULT_STRATEGY["model"]
        )
        self.execution: Dict[str, Any] = config.get(
            "execution", DEFAULT_STRATEGY["execution"]
        )
        self.indicators: Dict[str, Any] = config.get(
            "indicators", DEFAULT_STRATEGY["indicators"]
        )

    # ── Convenience accessors ─────────────────────────────────

    @property
    def max_position_pct(self) -> float:
        """Max percentage of equity per position."""
        return float(self.risk.get("max_position_pct", 0.25))

    @property
    def stop_loss_atr_mult(self) -> float:
        """ATR multiplier for stop-loss."""
        return float(self.risk.get("stop_loss_atr_mult", 2.0))

    @property
    def take_profit_atr_mult(self) -> float:
        """ATR multiplier for take-profit."""
        return float(self.risk.get("take_profit_atr_mult", 3.0))

    @property
    def is_paper(self) -> bool:
        """True if running in paper-trading mode."""
        return bool(self.execution.get("paper_mode", True))

    def to_dict(self) -> Dict[str, Any]:
        """Full config as a dictionary."""
        return dict(self.raw)


def load_strategy(
    path: Optional[str] = None,
) -> StrategyConfig:
    """Load strategy configuration from a file.

    Supports ``.yaml``, ``.yml``, and ``.json``.
    Falls back to ``DEFAULT_STRATEGY`` when no file exists.

    Args:
        path: Path to config file. If None, looks for
              ``strategy.yaml`` or ``strategy.json`` in cwd.

    Returns:
        Parsed StrategyConfig.
    """
    if path is None:
        for candidate in ("strategy.yaml", "strategy.yml", "strategy.json"):
            if os.path.isfile(candidate):
                path = candidate
                break

    if path is None or not os.path.isfile(path):
        logger.info("No strategy file found — using DEFAULT_STRATEGY")
        return StrategyConfig(dict(DEFAULT_STRATEGY))

    logger.info("Loading strategy from %s", path)
    ext = os.path.splitext(path)[1].lower()

    if ext in (".yaml", ".yml"):
        return _load_yaml(path)
    if ext == ".json":
        return _load_json(path)

    raise ValueError(f"Unsupported config format: {ext}")


def _load_yaml(path: str) -> StrategyConfig:
    """Load YAML strategy file."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "PyYAML not installed — falling back to default strategy"
        )
        return StrategyConfig(dict(DEFAULT_STRATEGY))

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    merged = {**DEFAULT_STRATEGY, **data}
    return StrategyConfig(merged)


def _load_json(path: str) -> StrategyConfig:
    """Load JSON strategy file."""
    with open(path, "r") as f:
        data = json.load(f)
    merged = {**DEFAULT_STRATEGY, **data}
    return StrategyConfig(merged)


def save_strategy(
    config: StrategyConfig,
    path: str = "strategy.json",
) -> None:
    """Save strategy configuration to disk.

    Args:
        config: StrategyConfig to persist.
        path: Output file path.
    """
    with open(path, "w") as f:
        json.dump(config.raw, f, indent=2)
    logger.info("Strategy saved to %s", path)
