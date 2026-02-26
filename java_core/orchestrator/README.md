# Java Orchestrator Core

This module implements the core orchestrator logic for the NEO Hybrid AI system.

## Components
- **ApiClient**: Sends features to the Python AI `/predict` endpoint and receives action/confidence.
- **RiskManagementEngine**: Applies thresholds, confidence filters, and volatility checks.
- **AutonomousLoop**: Runs the fetch → features → AI → risk → execute → log → feedback loop.
- **Dashboard**: (Optional) Displays real-time signals, strategy visualization, and model version.
- **OrchestratorMain**: Entry point to wire up and run the orchestrator.

## Coding Policy
- Follows all project coding standards and best practices.
- Modular, testable, and well-documented code.
- Logging and error handling included.

## Usage
Compile and run `OrchestratorMain.java` after configuring the Python AI service endpoint.

---
