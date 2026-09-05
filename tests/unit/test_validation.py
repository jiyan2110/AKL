"""Unit tests: validators, language, quality, dedup, quarantine schema (Milestone 15)."""

from __future__ import annotations

import random

import pyarrow as pa
import pytest

from akl.ingestion.dedup import find_near_duplicates, hamming, simhash
from akl.ingestion.language import detect_language, prose_sample
from akl.ingestion.models import FetchedObject, SourceItem
from akl.ingestion.parsers.markdown import MarkdownParser
from akl.ingestion.quality import document_quality
from akl.ingestion.quarantine import QUARANTINE_REASONS
from akl.ingestion.registry import ConnectorRegistry, ParserRegistry
from akl.ingestion.service import register_builtins
from akl.ingestion.validators import find_secrets, text_density, validate_bytes, validate_document
from akl.lakehouse.schemas import enforce

pytestmark = pytest.mark.unit


def _doc(text: str) -> object:
    item = SourceItem(
        uri="https://d.example.com/x.md",
        canonical_uri="https://d.example.com/x.md",
        source_type="markdown",
        filename="x.md",
    )
    return MarkdownParser().parse(
        FetchedObject.from_bytes(item, text.encode(), mime_type="text/markdown")
    )


GOOD = "# Title\n\n" + "\n\n".join(
    "This paragraph explains how the system stores documents in the lakehouse layers."
    for _ in range(6)
)


def test_validate_bytes_bounds() -> None:
    assert validate_bytes(10).rejected
    assert validate_bytes(10).reject_code == "AKL-E3001"
    assert not validate_bytes(1000).rejected


def test_validate_document_rules() -> None:
    ok = validate_document(_doc(GOOD))  # type: ignore[arg-type]
    assert not ok.rejected
    assert ok.flags == ()
    short = validate_document(_doc("# T\n\nshort"))  # type: ignore[arg-type]
    assert short.reject_code == "AKL-E3005"
    secret = validate_document(_doc(GOOD + "\n\nkey: AKIAABCDEFGHIJKLMNOP"))  # type: ignore[arg-type]
    assert secret.reject_code == "AKL-E3008"
    allowed = validate_document(
        _doc(GOOD + "\n\nkey: AKIAABCDEFGHIJKLMNOP"), allow_secret_like=True
    )  # type: ignore[arg-type]
    assert not allowed.rejected
    assert "akl_w3008" in allowed.flags


def test_find_secrets_and_density() -> None:
    # Assembled at runtime so the literal PEM marker never appears in the repo
    # (pre-commit's detect-private-key hook scans source text for it).
    pem_marker = "-----BEGIN " + "RSA PRIVATE " + "KEY-----"
    openssh_marker = "-----BEGIN " + "OPENSSH PRIVATE " + "KEY-----"
    assert find_secrets(pem_marker) == ["private_key"]
    assert find_secrets(openssh_marker) == ["private_key"]
    assert find_secrets("nothing here") == []
    assert text_density("abc def") == 1.0
    assert text_density("@@@@ ####") < 0.4


def test_language_detection() -> None:
    doc = _doc(GOOD)
    lang, conf = detect_language(prose_sample(doc.text, doc.blocks))  # type: ignore[attr-defined]
    assert lang == "en"
    assert conf > 0.9
    assert detect_language("x") == ("und", 0.0)


def test_document_quality_range_and_ordering() -> None:
    good = document_quality(_doc(GOOD), language_confidence=0.99)  # type: ignore[arg-type]
    poor = document_quality(
        _doc(
            "# T\n\n@@@ ### $$$ %%% ^^^ &&& *** ((( ))) ___ +++ === [[[ ]]] {{{ }}} ||| ~~~ ``` <<< >>> ??? ///"
        ),
        language_confidence=0.0,
    )  # type: ignore[arg-type]
    assert 0.0 <= poor < good <= 1.0
    assert good > 0.6


def test_simhash_near_duplicates() -> None:
    random.seed(1)
    words = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "system",
        "deploy",
        "cluster",
        "node",
        "table",
        "index",
        "query",
        "vector",
    ]
    a = " ".join(random.choice(words) for _ in range(2000))
    b = a.replace("alpha", "alpah", 3)
    c = " ".join(random.choice(words) for _ in range(2000))
    assert hamming(simhash(a), simhash(b)) <= 3
    assert hamming(simhash(a), simhash(c)) > 10
    decisions = find_near_duplicates(
        [("new", simhash(b), 0.8), ("other", simhash(c), 0.8)], [("old", simhash(a), 0.8)]
    )
    assert [(d.duplicate_document_id, d.canonical_document_id) for d in decisions] == [
        ("new", "old")
    ]
    # higher-quality newcomer becomes canonical
    flipped = find_near_duplicates([("new", simhash(b), 0.95)], [("old", simhash(a), 0.8)])
    assert [(d.duplicate_document_id, d.canonical_document_id) for d in flipped] == [("old", "new")]
    assert simhash("") == 0


def test_quarantine_schema_and_builtins() -> None:
    row = {
        "quarantine_id": "q",
        "error_code": "AKL-E3005",
        "stage": "validate",
        "run_id": "r",
        "created_at": pa.scalar(0, pa.timestamp("us", tz="UTC")).as_py(),
        "ingest_date": pa.scalar(0, pa.date32()).as_py(),
    }
    table = enforce(
        pa.Table.from_pylist([row], schema=QUARANTINE_REASONS.schema), QUARANTINE_REASONS
    )
    assert table.num_rows == 1
    connectors, parsers = ConnectorRegistry(), ParserRegistry()
    register_builtins(connectors, parsers)
    assert all(t in connectors for t in ("markdown", "pdf", "html", "github"))
    assert parsers.select(None, "md", "github").name == "markdown"
    assert parsers.select(None, "pdf", "pdf").name == "pdf"
    assert parsers.select("text/html", "", "html").name == "html"
    assert parsers.select(None, "rst", "github").name == "text"
    assert parsers.select(None, "py", "github").name == "code"
