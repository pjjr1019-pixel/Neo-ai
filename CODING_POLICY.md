# NEO Hybrid AI Coding Policy

# Always write all future code in full compliance with:
# - flake8 (PEP8) style
# - zero warnings
# - robust test coverage
# - best coding practices
# - maintainability and automation
# This policy is permanent and must be followed for every file, function, and test.

# If you see this file, you must enforce these rules for all future code.

# Bandit Security Policy

- Bandit warnings B101 (assert used) and B311 (random for non-crypto) are expected in test code and ML/AI simulation code.
- These warnings are not actionable unless code is used for security/cryptographic purposes.
- The .bandit config suppresses these warnings for tests and ML code.
- All other Bandit warnings must be reviewed and addressed as appropriate.
