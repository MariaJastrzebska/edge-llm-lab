"""Unit tests for the Telegram notifier.

No network: urllib is faked. Covers the two things that actually bite —
the notifier must never raise into its caller, and the token must never
reach a log line (it lives in the request URL).
"""

import urllib.error
from unittest.mock import Mock

import pytest

from edge_llm_lab.utils import gcp_secrets, notify

TOKEN = "8725671825:AAExampleTokenValue"


class FakeResponse:
    def __init__(self, status=200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    """Token and chat ID present, cache clean, no real Secret Manager."""
    gcp_secrets.clear_cache()
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "5551234567")
    yield
    gcp_secrets.clear_cache()


def test_sends_and_reports_success(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["body"] = request.data
        return FakeResponse(200)

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)

    assert notify.send("hello") is True
    assert seen["url"] == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    assert b"5551234567" in seen["body"]
    assert b"hello" in seen["body"]


def test_missing_token_returns_false_without_calling(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    called = []
    monkeypatch.setattr(
        notify.urllib.request, "urlopen", lambda *a, **k: called.append(1)
    )
    # No secret backing it either.
    monkeypatch.setattr(gcp_secrets, "_get_client", lambda: None)

    assert notify.send("hello") is False
    assert called == []


def test_missing_chat_id_returns_false(monkeypatch):
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setattr(gcp_secrets, "_get_client", lambda: None)

    assert notify.send("hello") is False


def test_non_200_returns_false(monkeypatch):
    monkeypatch.setattr(
        notify.urllib.request, "urlopen", lambda *a, **k: FakeResponse(500)
    )

    assert notify.send("hello") is False


def test_http_error_returns_false(monkeypatch):
    """403 here means Start was never pressed on the bot — must not raise."""

    def raise_403(*args, **kwargs):
        raise urllib.error.HTTPError(
            url="https://api.telegram.org/", code=403, msg="Forbidden",
            hdrs=None, fp=Mock(read=lambda: b'{"description":"bot cannot initiate"}'),
        )

    monkeypatch.setattr(notify.urllib.request, "urlopen", raise_403)

    assert notify.send("hello") is False


def test_network_error_returns_false(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("connection reset")

    monkeypatch.setattr(notify.urllib.request, "urlopen", boom)

    assert notify.send("hello") is False


def test_token_never_reaches_the_log(monkeypatch, caplog):
    """urllib puts the full URL — token included — in its error strings."""

    def boom(*args, **kwargs):
        raise OSError(f"failed opening https://api.telegram.org/bot{TOKEN}/sendMessage")

    monkeypatch.setattr(notify.urllib.request, "urlopen", boom)

    with caplog.at_level("ERROR"):
        assert notify.send("hello") is False

    assert TOKEN not in caplog.text
    assert "<token>" in caplog.text


def test_redact_leaves_other_text_alone():
    assert notify._redact("plain message", TOKEN) == "plain message"
    assert notify._redact("", TOKEN) == ""
    # An empty token must not turn every string into a redaction.
    assert notify._redact("anything", "") == "anything"


def test_report_names_repo_and_branch():
    text = notify.report()

    assert text.startswith("edge-llm-lab · ")
    assert "commits" in text


def test_dry_run_sends_nothing(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(notify, "send", lambda *a, **k: called.append(1))

    assert notify.main(["--dry-run"]) == 0
    assert called == []
    assert "edge-llm-lab" in capsys.readouterr().out
