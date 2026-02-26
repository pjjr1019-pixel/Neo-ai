# NEO Hybrid AI Coding Policy

## Style & Linting
- Follow PEP8 and flake8 for all Python code
- No unused imports, variables, or functions
- No lines longer than 79 characters (E501)
- Use 4 spaces per indentation (never tabs)
- Two blank lines before/after top-level functions/classes
- One blank line between methods in a class
- No trailing whitespace or blank lines at end of file
- Use meaningful variable, function, and class names
- Keep code DRY (Don't Repeat Yourself)
- Use docstrings for all public functions, classes, and modules
- Use type hints for all function signatures

## Testing
- 100% test coverage for all new code
- Use pytest for all tests
- Write unit, integration, and edge case tests
- Test for invalid inputs, exceptions, and boundary conditions
- Ensure all tests pass twice in a row before pushing
- Add a flake8 compliance test to the suite
- Never skip tests or leave TODOs in test files

## Automation & CI/CD
- Automate linting, testing, and coverage in CI pipeline
- Fail CI on any error, warning, or style violation
- Use pre-commit hooks for flake8 and pytest
- Keep requirements.txt and requirements-dev.txt up to date
- Use Docker for reproducible builds and deployments

## Documentation
- Write clear README.md with setup, usage, and API docs
- Document all endpoints, models, and business logic
- Keep docstrings and comments up to date
- Use Markdown for all documentation files

## Maintainability
- Modularize code: one responsibility per module/class/function
- Avoid global state and side effects
- Refactor legacy code to meet these standards
- Use dependency injection for testability
- Keep code and tests easy to read and modify
- Remove dead code and legacy artifacts

## Security & Reliability
- Validate all user inputs and API payloads
- Handle exceptions gracefully and log errors
- Never expose secrets or credentials in code
- Use environment variables for sensitive config
- Keep dependencies up to date and secure

## Collaboration
- Use clear, descriptive commit messages
- Push only when all tests and lint checks pass
- Review code for compliance before merging
- Encourage peer review and feedback

## Efficient Compliance & Debugging
- Run flake8 directly to see all style errors at once, not just through pytest.
- Use flake8's --format=default option for clear error locations.
- Search for all function definitions and docstrings in the codebase to batch-add missing docstrings.
- Review all blank line placements between functions and classes in each file.
- Adjust forbidden string checks in compliance tests to avoid matching empty strings or docstring delimiters.
- Automate these steps for batch fixing and validation.

## Agent Automation Policy
- Permit the agent to rewrite or batch-edit files for style, docstring, and forbidden string compliance.
- Allow the agent to run flake8 and pytest as often as needed to validate fixes.
- If a patch fails, allow the agent to use full file overwrite or insert_edit_into_file for stubborn cases.
- Grant permission for direct editing of files with formatting issues, even if patches fail due to context mismatches.
- Enable the agent to automate all compliance steps without manual intervention.

---

**This policy is permanent and must be followed for every file, function, and test. If you see this file, you must enforce these rules for all future code.**
