# Data Handling Policy

## Scope
- Applies to all ingestion, model training, inference, and logging workflows.
- Covers personal data, account data, API keys, and behavioral telemetry.

## Principles
- Data minimization: collect only fields required for execution and audit.
- Purpose limitation: no secondary use without explicit consent.
- Storage limitation: enforce retention windows and archival/deletion jobs.
- Integrity and confidentiality: encryption at rest/in transit, least privilege.

## Operational Controls
- Consent checks are enforced through `security/user_consent.py`.
- Compliance audits are scheduled via `security/compliance_audit.py`.
- Regulation change tracking is handled by `security/regulation_monitor.py`.
- CI compliance gates execute through `ci/compliance_checks.py`.

## Incident Response
- Suspected data exposure is escalated to security on-call.
- Access tokens are rotated immediately after confirmed compromise.
- Forensics logs are preserved for regulatory reporting.
