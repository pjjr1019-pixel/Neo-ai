"""
PII Redactor Module
- Technology: Custom regex-based redaction (can be replaced with Presidio or other tools)
- Acceptance: PII detected/redacted in this module
"""
import re
from typing import List

# Example patterns for PII (expand as needed)
PII_PATTERNS = {
    'email': re.compile(r'[\w\.-]+@[\w\.-]+'),
    'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
    'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
}

REDACTION_TEXT = '[REDACTED]'

def redact_pii(text: str, patterns: List[str] = None) -> str:
    """
    Redact PII from the input text using specified patterns.
    :param text: Input string
    :param patterns: List of PII types to redact (default: all)
    :return: Redacted string
    """
    if patterns is None:
        patterns = list(PII_PATTERNS.keys())
    for key in patterns:
        pattern = PII_PATTERNS.get(key)
        if pattern:
            text = pattern.sub(REDACTION_TEXT, text)
    return text

if __name__ == "__main__":
    sample = "Contact John Doe at john.doe@example.com or 555-123-4567. SSN: 123-45-6789."
    print(redact_pii(sample))
