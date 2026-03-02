import unittest
from security.pii_redactor import redact_pii

class TestPIIRedactor(unittest.TestCase):
    def test_email_redaction(self):
        text = "Contact: alice@example.com"
        redacted = redact_pii(text, patterns=["email"])
        self.assertNotIn("alice@example.com", redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_phone_redaction(self):
        text = "Call me at 555-123-4567."
        redacted = redact_pii(text, patterns=["phone"])
        self.assertNotIn("555-123-4567", redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_ssn_redaction(self):
        text = "SSN: 123-45-6789"
        redacted = redact_pii(text, patterns=["ssn"])
        self.assertNotIn("123-45-6789", redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_multiple_pii_redaction(self):
        text = "Email: bob@example.com, Phone: 555-987-6543, SSN: 987-65-4321"
        redacted = redact_pii(text)
        self.assertEqual(redacted.count("[REDACTED]"), 3)

    def test_no_pii(self):
        text = "No PII here!"
        redacted = redact_pii(text)
        self.assertEqual(text, redacted)

if __name__ == "__main__":
    unittest.main()
