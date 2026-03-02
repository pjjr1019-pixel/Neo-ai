"""
Tests for alert_notifier.py — URL-scheme safety and channel behaviour.

Regression suite that guards against B310 (CWE-22):
every notification channel must reject ``file://`` and custom
URL schemes so that ``urllib.request.urlopen`` is never called
with anything other than ``http`` or ``https``.

An AST-level test also ensures no new code introduces a raw
``urlopen`` call that bypasses the ``_safe_urlopen`` guard.
"""

import ast
import os
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from python_ai.alert_notifier import (
    _ALLOWED_SCHEMES,
    Alert,
    AlertDispatcher,
    DiscordChannel,
    SlackChannel,
    TelegramChannel,
    WebhookChannel,
    _safe_urlopen,
)

# ── Helper fixtures ───────────────────────────────────────────


@pytest.fixture()
def sample_alert() -> Alert:
    """Return a reusable test alert."""
    return Alert(
        title="Unit-Test",
        message="body",
        severity="info",
        source="test-suite",
    )


# ── 1. _safe_urlopen scheme validation ───────────────────────


class TestSafeUrlopen:
    """Direct tests for the ``_safe_urlopen`` helper."""

    @pytest.mark.parametrize("scheme", ["http", "https"])
    def test_allowed_schemes_pass(self, scheme: str) -> None:
        """Allowed schemes must not raise ValueError."""
        req = urllib.request.Request(f"{scheme}://example.com/hook")
        with patch(
            "python_ai.alert_notifier.urllib.request.urlopen"
        ) as mock_open:
            mock_open.return_value.__enter__ = MagicMock()
            mock_open.return_value.__exit__ = MagicMock(
                return_value=False,
            )
            # Should NOT raise
            _safe_urlopen(req, timeout=5)
            mock_open.assert_called_once()

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "file:///C:/Windows/System32/config/SAM",
            "ftp://evil.com/payload",
            "gopher://evil.com/data",
            "data:text/html,<h1>pwned</h1>",
            "javascript:alert(1)",
        ],
    )
    def test_forbidden_schemes_raise(self, url: str) -> None:
        """Non-http(s) schemes must raise ValueError (B310 guard)."""
        req = urllib.request.Request(url)
        with pytest.raises(ValueError, match="not allowed"):
            _safe_urlopen(req, timeout=5)

    def test_empty_scheme_rejected(self) -> None:
        """A URL with no scheme should be rejected.

        ``urllib.request.Request`` itself rejects bare
        scheme-less URLs, so there is no way for a
        no-scheme URL to reach ``_safe_urlopen``.
        We verify that ``Request`` raises early.
        """
        with pytest.raises(ValueError, match="unknown url type"):
            urllib.request.Request("//example.com/hook")

    def test_allowed_schemes_constant(self) -> None:
        """_ALLOWED_SCHEMES must contain exactly http and https."""
        assert _ALLOWED_SCHEMES == frozenset({"http", "https"})


# ── 2. Per-channel scheme rejection ──────────────────────────


class TestChannelSchemeRejection:
    """Each channel must reject disallowed URL schemes.

    These are regression tests that would have caught
    the original B310 finding.
    """

    def test_webhook_rejects_file_scheme(
        self,
        sample_alert: Alert,
    ) -> None:
        """WebhookChannel.send must return False for file:// URLs."""
        ch = WebhookChannel(url="file:///tmp/evil")
        assert ch.send(sample_alert) is False

    def test_slack_rejects_file_scheme(
        self,
        sample_alert: Alert,
    ) -> None:
        """SlackChannel.send must return False for file:// URLs."""
        ch = SlackChannel(webhook_url="file:///tmp/evil")
        assert ch.send(sample_alert) is False

    def test_telegram_rejects_file_scheme(
        self,
        sample_alert: Alert,
    ) -> None:
        """TelegramChannel.send must return False for file:// URLs."""
        ch = TelegramChannel(
            bot_token="fake",
            chat_id="123",
        )
        # Monkey-patch the URL constructed internally
        ch._token = "../../../../etc/passwd"
        assert ch.send(sample_alert) is False

    def test_discord_rejects_file_scheme(
        self,
        sample_alert: Alert,
    ) -> None:
        """DiscordChannel.send must return False for file:// URLs."""
        ch = DiscordChannel(webhook_url="file:///tmp/evil")
        assert ch.send(sample_alert) is False

    @pytest.mark.parametrize(
        "scheme",
        ["ftp", "gopher", "data", "javascript"],
    )
    def test_webhook_rejects_custom_schemes(
        self,
        scheme: str,
        sample_alert: Alert,
    ) -> None:
        """WebhookChannel must reject arbitrary bad schemes."""
        ch = WebhookChannel(url=f"{scheme}://evil.example.com")
        assert ch.send(sample_alert) is False


# ── 3. Channels succeed with valid https URLs ────────────────


class TestChannelHappyPath:
    """Channels work normally when given valid https URLs."""

    @patch("python_ai.alert_notifier.urllib.request.urlopen")
    def test_webhook_sends_on_https(
        self,
        mock_open: MagicMock,
        sample_alert: Alert,
    ) -> None:
        """WebhookChannel.send returns True on HTTP 200."""
        resp = MagicMock()
        resp.status = 200
        mock_open.return_value.__enter__ = MagicMock(
            return_value=resp,
        )
        mock_open.return_value.__exit__ = MagicMock(
            return_value=False,
        )
        ch = WebhookChannel(url="https://hooks.example.com/v1")
        assert ch.send(sample_alert) is True

    @patch("python_ai.alert_notifier.urllib.request.urlopen")
    def test_slack_sends_on_https(
        self,
        mock_open: MagicMock,
        sample_alert: Alert,
    ) -> None:
        """SlackChannel.send returns True on HTTP 200."""
        resp = MagicMock()
        resp.status = 200
        mock_open.return_value.__enter__ = MagicMock(
            return_value=resp,
        )
        mock_open.return_value.__exit__ = MagicMock(
            return_value=False,
        )
        ch = SlackChannel(
            webhook_url="https://hooks.slack.com/services/X",
        )
        assert ch.send(sample_alert) is True

    @patch("python_ai.alert_notifier.urllib.request.urlopen")
    def test_discord_sends_on_https(
        self,
        mock_open: MagicMock,
        sample_alert: Alert,
    ) -> None:
        """DiscordChannel.send returns True on HTTP 200."""
        resp = MagicMock()
        resp.status = 200
        mock_open.return_value.__enter__ = MagicMock(
            return_value=resp,
        )
        mock_open.return_value.__exit__ = MagicMock(
            return_value=False,
        )
        ch = DiscordChannel(
            webhook_url="https://discord.com/api/webhooks/X",
        )
        assert ch.send(sample_alert) is True


# ── 4. Dispatcher integration ────────────────────────────────


class TestDispatcherWithSchemeGuard:
    """Dispatcher gracefully handles channels that reject bad URLs."""

    def test_dispatch_counts_scheme_failure(
        self,
        sample_alert: Alert,
    ) -> None:
        """Bad-scheme channel should count as failed, not crash."""
        bad_ch = WebhookChannel(url="file:///tmp/evil")
        d = AlertDispatcher(channels=[bad_ch])
        result = d.dispatch(sample_alert)
        assert result["sent"] == 0
        assert result["failed"] == 1

    @patch("python_ai.alert_notifier.urllib.request.urlopen")
    def test_dispatch_mixed_channels(
        self,
        mock_urlopen: MagicMock,
        sample_alert: Alert,
    ) -> None:
        """Dispatcher handles a mix of valid and invalid channels.

        Uses different channel types so their class-name keys
        don't collide in the results dict.
        """
        resp = MagicMock()
        resp.status = 200
        mock_urlopen.return_value.__enter__ = MagicMock(
            return_value=resp,
        )
        mock_urlopen.return_value.__exit__ = MagicMock(
            return_value=False,
        )

        good = SlackChannel(
            webhook_url="https://hooks.slack.com/services/X",
        )
        bad = WebhookChannel(url="file:///etc/passwd")
        d = AlertDispatcher(channels=[good, bad])
        result = d.dispatch(sample_alert)
        assert result["sent"] == 1
        assert result["failed"] == 1
        assert result["results"]["SlackChannel"] is True
        assert result["results"]["WebhookChannel"] is False


# ── 5. AST-level guard: no raw urlopen in production code ────


class TestNoRawUrlopenCalls:
    """Scan production Python files for raw urlopen calls.

    Any future module that calls ``urllib.request.urlopen``
    directly (instead of going through ``_safe_urlopen``)
    will be caught here — preventing B310 regressions
    at the source-code level.
    """

    # Files that ARE allowed to call urlopen directly
    _EXEMPTED_FILES = frozenset(
        {
            "alert_notifier.py",  # contains the _safe_urlopen wrapper
        }
    )

    def _python_files(self) -> list:
        """Collect all .py files under python_ai/, skip tests."""
        root = os.path.join(
            os.path.dirname(__file__),
        )
        result = []
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                if fn.startswith("test_"):
                    continue
                result.append(os.path.join(dirpath, fn))
        return result

    def test_no_direct_urlopen_usage(self) -> None:
        """No production file should call urlopen directly.

        This AST scan detects ``urllib.request.urlopen(...)``
        attribute-call patterns.  The only exemption is
        ``alert_notifier.py`` itself (where ``_safe_urlopen``
        wraps the raw call).
        """
        violations: list = []
        for filepath in self._python_files():
            basename = os.path.basename(filepath)
            if basename in self._EXEMPTED_FILES:
                continue
            with open(filepath, encoding="utf-8") as fh:
                try:
                    tree = ast.parse(fh.read(), filename=filepath)
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                # Match: urllib.request.urlopen(...)
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "urlopen"
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "request"
                    and isinstance(func.value.value, ast.Name)
                    and func.value.value.id == "urllib"
                ):
                    violations.append(f"{filepath}:{node.lineno}")
        assert not violations, (
            "Raw urllib.request.urlopen() calls found — "
            "use _safe_urlopen() from alert_notifier instead:\n"
            + "\n".join(violations)
        )
