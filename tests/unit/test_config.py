"""Unit tests for akl.config."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from akl.config import Settings
from akl.errors import ConfigError, ConfigFileError

pytestmark = pytest.mark.unit

REQUIRED_SECRETS = {
    "AKL_DB_PASSWORD": "db-pw",
    "AKL_S3_ACCESS_KEY": "ak",
    "AKL_S3_SECRET_KEY": "sk",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("AKL_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(name="secrets_env")
def _secrets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in REQUIRED_SECRETS.items():
        monkeypatch.setenv(key, value)


def _write_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_defaults_apply_without_yaml(secrets_env: None) -> None:
    settings = Settings.load(config_file=None, env_file=None)
    assert settings.db.host == "postgres"
    assert settings.s3.bucket == "akl-lakehouse"
    assert settings.qdrant.collection_alias == "kb_chunks"


def test_yaml_overrides_defaults(secrets_env: None, tmp_path: Path) -> None:
    settings = Settings.load(
        config_file=_write_yaml(tmp_path, "database:\n  host: db.internal\n  port: 6543\n"),
        env_file=None,
    )
    assert settings.db.host == "db.internal"
    assert settings.db.port == 6543


def test_env_overrides_yaml(
    secrets_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AKL_DB_HOST", "from-env")
    settings = Settings.load(
        config_file=_write_yaml(tmp_path, "database:\n  host: db.internal\n"), env_file=None
    )
    assert settings.db.host == "from-env"


def test_file_secret_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret_file = tmp_path / "db_pw"
    secret_file.write_text("from-file\n", encoding="utf-8")
    monkeypatch.setenv("AKL_DB_PASSWORD_FILE", str(secret_file))
    monkeypatch.setenv("AKL_S3_ACCESS_KEY", "ak")
    monkeypatch.setenv("AKL_S3_SECRET_KEY", "sk")
    settings = Settings.load(config_file=None, env_file=None)
    assert settings.db.password.get_secret_value() == "from-file"


def test_missing_secrets_raise_aggregated_config_error() -> None:
    with pytest.raises(ConfigError) as excinfo:
        Settings.load(config_file=None, env_file=None)
    variables = {problem["variable"] for problem in excinfo.value.details["problems"]}
    assert {"AKL_DB_PASSWORD", "AKL_S3_ACCESS_KEY", "AKL_S3_SECRET_KEY"} <= variables
    assert excinfo.value.code == "AKL-E0001"


def test_invalid_value_reports_variable(secrets_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AKL_DB_PORT", "70000")
    with pytest.raises(ConfigError) as excinfo:
        Settings.load(config_file=None, env_file=None)
    assert any(
        problem["variable"] == "AKL_DB_PORT" for problem in excinfo.value.details["problems"]
    )


def test_prod_forbids_sslmode_disable(secrets_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AKL_ENV", "prod")
    with pytest.raises(ConfigError, match="cross-section"):
        Settings.load(config_file=None, env_file=None)


def test_malformed_yaml_raises_config_file_error(secrets_env: None, tmp_path: Path) -> None:
    with pytest.raises(ConfigFileError):
        Settings.load(config_file=_write_yaml(tmp_path, "database: [unclosed\n"), env_file=None)


def test_secrets_are_redacted(secrets_env: None) -> None:
    settings = Settings.load(config_file=None, env_file=None)
    dumped = settings.redacted()
    assert dumped["db"]["password"] == "**********"
    assert "db-pw" not in str(dumped)
    assert "***" in settings.db.dsn()
    assert "db-pw" in settings.db.dsn(reveal=True)
