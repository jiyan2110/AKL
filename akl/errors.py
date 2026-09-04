"""AKL error hierarchy."""

from __future__ import annotations

from typing import Any


class AKLError(Exception):
    """Base class for all AKL errors."""

    code: str = "AKL-E9999"
    http_status: int = 500
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}
        if retryable is not None:
            self.retryable = retryable

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class ConfigError(AKLError):
    """Invalid or missing settings detected at startup."""

    code = "AKL-E0001"


class ConfigFileError(AKLError):
    """A configuration YAML file is unreadable or malformed."""

    code = "AKL-E0002"
