"""Unit tests: PII scanner, RBAC scopes, auth error semantics (Milestones 49-52)."""

from __future__ import annotations

import pytest

from akl.governance.pii import scan_text
from akl.security.auth import ROLE_SCOPES, scopes_for_roles

pytestmark = pytest.mark.unit


def test_scan_text_detects_all_types_and_ignores_versions() -> None:
    text = "Contact jane.doe@example.com or 415-555-0134. SSN 123-45-6789. Card 4111 1111 1111 1111. Server 10.0.0.5."
    findings = scan_text(text)
    by_type = {f.pii_type: f.value for f in findings}
    assert by_type["email"] == "jane.doe@example.com"
    assert by_type["phone"] == "415-555-0134"
    assert by_type["ssn"] == "123-45-6789"
    assert by_type["credit_card"] == "4111111111111111"
    assert by_type["ip_address"] == "10.0.0.5"
    # a version string must never be mistaken for a credit card or SSN
    assert scan_text("Upgrade to version 4.6.0 or 1.2.3-beta before deploying.") == []


def test_scan_text_rejects_luhn_invalid_card_numbers() -> None:
    # 16 digits, well-formed shape, but fails the Luhn checksum -> not a real card number
    findings = scan_text("Reference number 1234 5678 9012 3456 for this ticket.")
    assert not any(f.pii_type == "credit_card" for f in findings)


def test_scan_text_deduplicates_repeated_values() -> None:
    findings = scan_text("Email a@example.com, then email a@example.com again.")
    assert len([f for f in findings if f.pii_type == "email"]) == 1


def test_scan_text_respects_enabled_types_filter() -> None:
    text = "Email a@example.com and phone 415-555-0134."
    only_email = scan_text(text, enabled_types=frozenset({"email"}))
    assert {f.pii_type for f in only_email} == {"email"}
    none_enabled = scan_text(text, enabled_types=frozenset())
    assert none_enabled == []


def test_scan_text_empty_and_clean_input() -> None:
    assert scan_text("") == []
    assert scan_text("The quick brown fox jumps over the lazy dog.") == []


def test_role_scopes_curator_gains_governance_scopes() -> None:
    curator = ROLE_SCOPES["curator"]
    assert {"documents:permissions", "keys:manage", "audit:read"} <= curator
    assert "gdpr:manage" not in curator  # GDPR admin-on-behalf-of stays admin-only by design
    admin = scopes_for_roles(["admin"])
    assert "*" in admin


def test_role_scopes_reader_cannot_manage_keys_or_permissions() -> None:
    reader = ROLE_SCOPES["reader"]
    assert "keys:manage" not in reader
    assert "documents:permissions" not in reader
    assert "audit:read" not in reader
