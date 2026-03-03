# Changelog

All notable changes to the NEO Hybrid AI project.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `.env.example` template for environment configuration
- `Makefile` with common development commands
- `CHANGELOG.md` (this file)
- Pre-commit hooks: added isort, mypy alongside existing black/flake8/bandit
- `_torch_optim_available()` guard for CI environments with broken triton
- `tests/conftest.py` with shared project-root path setup
- `python_ai/strategy_evaluation.py` for novelty, speciation, Pareto front,
  and composite top-N strategy selection
- `python_ai/strategy_lifecycle.py` for lineage, retirement archive, warm-start,
  complexity penalties, and age-adjusted fitness
- Phase 7 docs:
  `docs/phase-7-evolution-engine.md`,
  `docs/phase-7-strategy-evaluation.md`,
  `docs/phase-7-strategy-lifecycle.md`
- Phase 7 tests:
  `tests/test_evolution_engine_phase7.py`,
  `tests/test_strategy_evaluation.py`,
  `tests/test_strategy_lifecycle.py`
- Phase 0 governance/compliance modules:
  `ci/model_validation.py`,
  `ci/deployment_strategies.py`,
  `security/compliance_audit.py`,
  `security/regulation_monitor.py`,
  `security/user_consent.py`
- Phase 4 engineering modules:
  `python_ai/schemas.py`,
  `python_ai/feature_factory.py`,
  `data/feature_cache.py`,
  `data/storage.py`,
  `data/io.py`
- Phase 5 modules:
  `python_ai/model_selection.py`,
  `python_ai/model_export.py`,
  `python_ai/distributed_hpo.py`,
  `python_ai/robustness.py`,
  `python_ai/synthetic_data.py`
- GUI/deployment modules:
  `python_ai/gui/*`,
  `deployment/installer.py`,
  `deployment/updater.py`
- New docs:
  `docs/phase-4-python-ai-engine.md`,
  `docs/phase-5-*.md`,
  `docs/phase-8-gui.md`,
  `docs/phase-10-user-experience.md`,
  `docs/phase-11-release-notes.md`

### Changed
- Replaced `datetime.utcnow()` with `datetime.now(timezone.utc)` across
  all modules (deprecated since Python 3.12)
- Upgraded password hashing from `sha256_crypt` to `bcrypt`
- Moved all 46 test files from `python_ai/` to `tests/` directory
- Renamed `integration_test.py` → `test_integration.py`
- Updated `pyproject.toml`, `pytest.ini`, and CI workflow for
  new `tests/` layout
- Pinned `bcrypt<5` for passlib 1.7.4 compatibility
- RSI calculation now uses Wilder's exponential smoothing
- Bollinger Bands use per-bar rolling standard deviation
- MACD signal line uses 9-period EMA (was uniform average)
- Replaced `pickle` with `joblib` + SHA-256 integrity in `ml_model.py`
- FastAPI endpoints use `Depends()` DI for model/pipeline injection
- Learn endpoint uses `asyncio.Lock` for buffer safety
- Secret key validator auto-generates runtime key in dev/test,
  rejects insecure values in production
- `EvolutionEngine` now supports parallel population evaluation and
  generator-based mutation/population iteration to reduce memory overhead
- `EvolutionEngine` self-play now avoids duplicate pair matches, adds
  `elo_tournament_selection` for faster competitive ranking, and improves
  `meta_learn` cross-validation efficiency/correctness for uneven fold counts
- `strategy_evaluation.novelty_scores` now precomputes pairwise distances
  once per population instead of recomputing each strategy pair twice
- `StrategyLifecycleManager.family_tree` now uses a deque-based BFS queue
  to remove O(n) front-pop operations on long lineages
- Bandit config migrated to YAML format for `bandit -c .bandit` compatibility
- Makefile test targets now execute `tests/` and include `test-parallel`
- MLModel optimizations: RF parallel training (`n_jobs=-1`), prediction cache,
  async save/load, optional ONNX runtime path
- Backtesting engine extended with vectorized parity, parallel jobs, and
  multi-timeframe evaluation APIs

### Security
- Removed hardcoded default secrets from `config/settings.py` and
  `auth/jwt_handler.py`
- Wired `get_optional_user` auth dependency to all data endpoints
- Fixed `alert_notifier.py` mypy `resp.status` operator type errors

## [0.3.0] - 2026-03-01

### Added
- Phase 11.05 critical code audit fixes (56a–56f)
- Auth dependency injection on FastAPI endpoints
- SHA-256 model file integrity verification

## [0.2.0] - 2026-02-28

### Added
- 19 modules across Phases 10–16 (Session 5)
- Request encryption, IP allowlist, regime detector
- Event sourcing, CQRS, dead letter queue, bulkhead
- Anomaly autoencoder, LSTM model, transformer model
- Retrain scheduler, RL agent, cross-exchange validator
- Statistical arbitrage, alert notifier, data archival
- Stream ingestion, audit logger, risk governance
- 682 tests passing, 5 skipped, full compliance

## [0.1.0] - 2026-02-27

### Added
- Initial project setup (Phases 0–9)
- FastAPI service with predict/learn/metrics/explain endpoints
- ML ensemble model (RandomForest + GradientBoosting)
- Data pipeline with technical indicators
- Auth system (JWT, API keys, RBAC)
- CI/CD pipeline (GitHub Actions)
- Docker support
- PostgreSQL/SQLite database layer
- Backtesting engine, evolution engine
- Strategy configuration loader
