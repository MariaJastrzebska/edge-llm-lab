"""Unit tests for GCP Secret Manager resolution.

No network and no google-cloud-secret-manager install required — the client is
faked, so these run on any device the lab runs on.
"""

from unittest.mock import Mock

import pytest

from edge_llm_lab.utils import gcp_secrets
from edge_llm_lab.utils.gcp_secrets import MissingSecret


class FakeClient:
    """Stands in for SecretManagerServiceClient."""

    def __init__(self, payloads, fail_with=None):
        self.payloads = payloads
        self.fail_with = fail_with
        self.calls = []

    def access_secret_version(self, request):
        name = request["name"]
        self.calls.append(name)
        if self.fail_with:
            raise self.fail_with
        secret = name.split("/secrets/")[1].split("/versions/")[0]
        if secret not in self.payloads:
            raise KeyError(f"no such secret: {secret}")
        return Mock(payload=Mock(data=self.payloads[secret].encode("utf-8")))


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Every test starts with an empty cache and a known project."""
    gcp_secrets.clear_cache()
    monkeypatch.setenv("GCP_PROJECT_ID", "llm-apps-test")
    for var in list(gcp_secrets.SECRETS) + ["NEPTUNE_API_TOKEN_SECRET"]:
        monkeypatch.delenv(var, raising=False)
    yield
    gcp_secrets.clear_cache()


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeClient(
        {
            "NEPTUNE_API_TOKEN": "neptune-token-from-secret-manager",
            "OPENAI_API_KEY": "sk-judge-key",
        }
    )
    monkeypatch.setattr(gcp_secrets, "_get_client", lambda: client)
    return client


def test_env_var_wins_over_secret_manager(monkeypatch, fake_client):
    """A value already exported (CI, .env) is never overridden."""
    monkeypatch.setenv("NEPTUNE_API_TOKEN", "env-token")

    assert gcp_secrets.resolve("NEPTUNE_API_TOKEN") == "env-token"
    assert fake_client.calls == []  # Secret Manager not consulted at all


def test_falls_back_to_secret_manager(fake_client):
    assert gcp_secrets.resolve("NEPTUNE_API_TOKEN") == "neptune-token-from-secret-manager"
    assert fake_client.calls == [
        "projects/llm-apps-test/secrets/NEPTUNE_API_TOKEN/versions/latest"
    ]


def test_blank_env_var_is_treated_as_unset(monkeypatch, fake_client):
    monkeypatch.setenv("NEPTUNE_API_TOKEN", "   ")

    assert gcp_secrets.resolve("NEPTUNE_API_TOKEN") == "neptune-token-from-secret-manager"


def test_trailing_newline_is_stripped(monkeypatch):
    """Console-pasted secrets pick up a newline; it breaks auth headers."""
    client = FakeClient({"NEPTUNE_API_TOKEN": "neptune-token\n"})
    monkeypatch.setattr(gcp_secrets, "_get_client", lambda: client)

    assert gcp_secrets.resolve("NEPTUNE_API_TOKEN") == "neptune-token"


def test_result_is_cached(fake_client):
    gcp_secrets.resolve("NEPTUNE_API_TOKEN")
    gcp_secrets.resolve("NEPTUNE_API_TOKEN")

    assert len(fake_client.calls) == 1


def test_distinct_secrets_are_cached_separately(fake_client):
    assert gcp_secrets.resolve("NEPTUNE_API_TOKEN") == "neptune-token-from-secret-manager"
    assert gcp_secrets.resolve("OPENAI_API_KEY") == "sk-judge-key"
    assert len(fake_client.calls) == 2


def test_clear_cache_forces_refetch(monkeypatch, fake_client):
    gcp_secrets.resolve("NEPTUNE_API_TOKEN")
    gcp_secrets.clear_cache()
    # clear_cache() drops the client too, so re-patch before the second read.
    monkeypatch.setattr(gcp_secrets, "_get_client", lambda: fake_client)
    gcp_secrets.resolve("NEPTUNE_API_TOKEN")

    assert len(fake_client.calls) == 2


def test_secret_name_override(monkeypatch):
    """<ENV_VAR>_SECRET renames a secret without a code change."""
    client = FakeClient({"NEPTUNE_API_TOKEN_v2": "rotated-token"})
    monkeypatch.setattr(gcp_secrets, "_get_client", lambda: client)
    monkeypatch.setenv("NEPTUNE_API_TOKEN_SECRET", "NEPTUNE_API_TOKEN_v2")

    assert gcp_secrets.resolve("NEPTUNE_API_TOKEN") == "rotated-token"


def test_missing_library_degrades_to_default(monkeypatch):
    """No google-cloud-secret-manager installed is a normal edge-device state."""
    monkeypatch.setattr(gcp_secrets, "_get_client", lambda: None)

    assert gcp_secrets.resolve("NEPTUNE_API_TOKEN") is None


def test_lookup_failure_degrades_to_default(monkeypatch):
    """An overnight sweep must not die because a lookup hiccuped."""
    client = FakeClient({}, fail_with=RuntimeError("403 PermissionDenied"))
    monkeypatch.setattr(gcp_secrets, "_get_client", lambda: client)

    assert gcp_secrets.resolve("NEPTUNE_API_TOKEN", default="fallback") == "fallback"


def test_required_raises_when_unresolvable(monkeypatch):
    monkeypatch.setattr(gcp_secrets, "_get_client", lambda: None)

    with pytest.raises(MissingSecret, match="NEPTUNE_API_TOKEN"):
        gcp_secrets.resolve("NEPTUNE_API_TOKEN", required=True)


def test_missing_project_degrades_to_default(monkeypatch, fake_client):
    monkeypatch.setattr(gcp_secrets, "project_id", lambda: None)

    assert gcp_secrets.resolve("NEPTUNE_API_TOKEN") is None


def test_source_of_is_manifest_safe(monkeypatch, fake_client):
    """source_of names the origin, never the secret — safe for a run manifest."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    assert gcp_secrets.source_of("OPENAI_API_KEY") == "env"
    assert gcp_secrets.source_of("NEPTUNE_API_TOKEN") == "secret-manager"
    assert gcp_secrets.source_of("UNMAPPED_TOKEN") == "missing"


def test_status_reports_source_without_values(monkeypatch, fake_client):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    report = gcp_secrets.status()

    assert report["NEPTUNE_API_TOKEN"]["source"] == "secret-manager"
    assert report["OPENAI_API_KEY"]["source"] == "env"
    # Only lengths are exposed, never the secret itself.
    for info in report.values():
        assert set(info) == {"source", "secret", "length"}


def test_registry_matches_the_secrets_that_exist():
    """Guards against drifting back to invented names.

    Edge_lab_fed_bot is a Telegram bot, not the Neptune token — an earlier
    revision mapped it to NEPTUNE_API_TOKEN and would have handed a bot token
    to the Neptune client.
    """
    assert gcp_secrets.SECRETS == {
        "NEPTUNE_API_TOKEN": "NEPTUNE_API_TOKEN",
        "OPENAI_API_KEY": "OPENAI_API_KEY",
        "TELEGRAM_BOT_TOKEN": "Edge_lab_fed_bot",
        "TELEGRAM_CHAT_ID": "TELEGRAM_CHAT_ID",
    }
