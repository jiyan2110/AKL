"""Unit tests for the chunking engine (Milestones 16–20)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import pyarrow as pa
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from akl.chunking.code_splitter import split_code
from akl.chunking.engine import ChunkingEngine, config_hash
from akl.chunking.identity import chunk_checksum, chunk_key, render_context_prefix
from akl.chunking.incremental import diff_chunks
from akl.chunking.models import ChunkStatus
from akl.chunking.quality import chunk_quality, length_fit, repetition_ratio
from akl.chunking.semantic import semantic_boundaries
from akl.chunking.sentences import SentenceSplitter, Span, clause_split, window_split
from akl.chunking.structural import build_sections
from akl.chunking.table_splitter import parse_markdown_table, split_table, transpose_wide_table
from akl.chunking.tokenizer import TokenCounter
from akl.config import ChunkingSettings
from akl.ingestion.models import (
    FetchedObject,
    SourceItem,
    UnifiedDocument,
    document_from_silver_row,
)
from akl.ingestion.parsers.markdown import MarkdownParser
from akl.lakehouse.schemas import enforce
from akl.lakehouse.schemas.silver import CHUNKS, DOCUMENTS
from akl.lakehouse.silver import SilverStore

pytestmark = pytest.mark.unit

COUNT = TokenCounter().count
SMALL = ChunkingSettings(
    chunk_target_tokens=60,
    chunk_max_tokens=80,
    chunk_min_tokens=10,
    chunk_overlap_tokens=24,
    chunk_code_max_tokens=70,
    chunk_table_max_tokens=70,
)


def parse(md: str, uri: str = "https://x/d.md", language: str = "en") -> UnifiedDocument:
    item = SourceItem(uri=uri, canonical_uri=uri, source_type="markdown", filename="d.md")
    doc = MarkdownParser().parse(
        FetchedObject.from_bytes(item, md.encode(), mime_type="text/markdown")
    )
    return doc.model_copy(update={"language": language})


PARAS = [
    f"Paragraph {i} describes component {i} of the platform in a complete sentence with enough words to matter."
    for i in range(12)
]
LONG_DOC = (
    "# Guide\n\n## Alpha\n\n"
    + "\n\n".join(PARAS[:6])
    + "\n\n## Beta\n\n"
    + "\n\n".join(PARAS[6:])
    + "\n"
)


# --------------------------------------------------------------------------- sentences / tokens
def test_sentence_splitter_offsets() -> None:
    text = "Hello world. This is Dr. Smith; he works at ACME, Inc. Third sentence!"
    spans = SentenceSplitter().split(text, 100)
    assert [s.text for s in spans] == [
        "Hello world.",
        "This is Dr. Smith; he works at ACME, Inc.",
        "Third sentence!",
    ]
    assert all(text[s.start - 100 : s.end - 100] == s.text for s in spans)
    assert [p.text for p in clause_split(spans[1], min_chars=10)] == [
        "This is Dr. Smith;",
        "he works at ACME,",
        "Inc.",
    ]
    assert [p.text for p in window_split(Span("a b c d e f g", 0, 13), 3)] == [
        "a b c",
        "d e f",
        "g",
    ]


def test_token_counter_heuristic_is_monotone() -> None:
    assert COUNT("") == 0
    assert COUNT("hello") == 1
    assert COUNT("hello world") == 2
    assert COUNT("internationalization") > 1
    assert COUNT("a, b.") == 4


# --------------------------------------------------------------------------- structural
def test_build_sections_heading_paths_and_empty_headings_folded() -> None:
    doc = parse("# T\n\n## A\n\n### A1\n\npara a1\n\n## B\n\npara b\n")
    sections = build_sections(doc)
    assert [s.heading_path for s in sections] == [("T", "A", "A1"), ("T", "B")]
    assert all(s.blocks for s in sections)


# --------------------------------------------------------------------------- engine core
def test_engine_offsets_ids_linkage_and_prefix() -> None:
    doc = parse(LONG_DOC)
    eng = ChunkingEngine(SMALL)
    chunks = eng.chunk(doc)
    assert len(chunks) >= 4
    for c in chunks:
        assert doc.text[c.start_char : c.end_char] == c.text
        assert c.token_count <= SMALL.chunk_max_tokens
        assert c.chunk_key == chunk_key(doc.document_id, c.heading_path, c.section_ordinal)
        assert c.chunk_checksum == chunk_checksum(c.text)
        assert c.context_prefix.startswith("Guide")
        assert c.line_start is not None
        assert c.line_end is not None
        assert c.line_start <= c.line_end
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        assert prev.next_chunk_id == cur.chunk_id
        assert cur.prev_chunk_id == prev.chunk_id
    assert chunks[0].prev_chunk_id is None
    assert chunks[-1].next_chunk_id is None
    # deterministic
    assert [c.chunk_id for c in eng.chunk(doc)] == [c.chunk_id for c in chunks]


def test_overlap_is_sentence_aligned_and_within_section() -> None:
    doc = parse(LONG_DOC)
    chunks = ChunkingEngine(SMALL).chunk(doc)
    alpha = [c for c in chunks if c.heading_path[-1] == "Alpha"]
    assert len(alpha) >= 2
    assert alpha[0].overlap_prev_tokens == 0
    for prev, cur in zip(alpha, alpha[1:], strict=False):
        assert 0 < cur.overlap_prev_tokens <= SMALL.chunk_overlap_tokens
        assert cur.start_char < prev.end_char  # overlap region
        assert cur.text.startswith("Paragraph")  # starts at a sentence boundary
    beta_first = next(c for c in chunks if c.heading_path[-1] == "Beta")
    assert beta_first.overlap_prev_tokens == 0  # never crosses sections


def test_short_section_is_single_chunk_and_no_fragmentation() -> None:
    doc = parse(
        "# T\n\n## Small\n\nOne sentence here.\n\n```py\nx = 1\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    )
    chunks = ChunkingEngine(SMALL).chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "mixed"
    assert chunks[0].code_language == "py"


def test_code_block_lead_in_and_split() -> None:
    code = (
        "```python\n" + "\n".join(f"def f{i}(x):\n    return x * {i}\n" for i in range(40)) + "```"
    )
    doc = parse(
        "# T\n\n## Code\n\n" + "\n\n".join(PARAS[:3]) + "\n\nThe helpers follow.\n\n" + code + "\n"
    )
    chunks = ChunkingEngine(SMALL).chunk(doc)
    code_chunks = [c for c in chunks if c.chunk_type == "code"]
    assert len(code_chunks) > 1
    assert all(c.token_count <= SMALL.chunk_code_max_tokens for c in code_chunks)
    assert all(doc.text[c.start_char : c.end_char] == c.text for c in code_chunks)
    assert all(c.code_language == "python" for c in code_chunks)
    assert code_chunks[0].text.startswith("```python\ndef f0")
    assert any(f.startswith("code_split_2_of_") for f in code_chunks[1].quality_flags)
    small = parse("# T\n\n" + "\n\n".join(PARAS[:3]) + "\n\nRun this.\n\n```bash\nmake up\n```\n")
    small_code = [c for c in ChunkingEngine(SMALL).chunk(small) if c.chunk_type == "code"]
    assert small_code
    assert small_code[0].text.startswith("Run this.")


def test_table_split_repeats_header_and_transposes_wide() -> None:
    md = "| a | b |\n|---|---|\n" + "\n".join(f"| {i} | v{i} |" for i in range(60))
    pieces = split_table(md, COUNT, 50, caption="Ports")
    assert len(pieces) > 1
    assert all(p.markdown.startswith("Ports\n\n| a | b |") for p in pieces)
    assert sum(p.row_end - p.row_start for p in pieces) == 60
    header = [f"c{i}" for i in range(35)]
    wide = (
        "| "
        + " | ".join(header)
        + " |\n|"
        + "---|" * 35
        + "\n| "
        + " | ".join(str(i) for i in range(35))
        + " |"
    )
    h, rows = parse_markdown_table(wide)
    assert len(h) == 35
    assert transpose_wide_table(h, rows).startswith("Row 1: c0: 0; c1: 1")
    assert split_table(wide, COUNT, 400)[0].markdown.startswith("Row 1:")
    doc = parse("# T\n\n## Data\n\n" + "\n\n".join(PARAS[:3]) + "\n\nPorts table.\n\n" + md + "\n")
    tables = [c for c in ChunkingEngine(SMALL).chunk(doc) if c.chunk_type == "table"]
    assert len(tables) > 1
    assert all("| a | b |" in c.text for c in tables)
    assert {c.start_char for c in tables} == {tables[0].start_char}  # all cite the table block


def test_code_splitter_fallbacks() -> None:
    py = "\n".join(f"def f{i}():\n    pass\n" for i in range(30))
    pieces = split_code(py, "python", COUNT, 40)
    assert len(pieces) > 1
    assert all(COUNT(p.text) <= 40 for p in pieces)
    assert pieces[1].text.startswith("def f")
    blob = "\n".join(f"line {i} of an unknown language file" for i in range(120))
    pieces2 = split_code(blob, None, COUNT, 50)
    assert len(pieces2) > 1
    assert "".join(p.text for p in pieces2).replace("\n", "") == blob.replace("\n", "")
    assert split_code("tiny", "python", COUNT, 40)[0].total == 1


def test_semantic_boundaries_with_fake_embedder() -> None:
    def embed(sentences: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "cat" in s else [0.0, 1.0] for s in sentences]

    assert semantic_boundaries(["cat a", "cat b", "dog c", "dog d"], embed, 0.25) == {2}
    assert semantic_boundaries(["cat a", "cat b"], embed, 0.25) == set()
    assert semantic_boundaries(["a", "b", "c", "d"], None, 0.25) == set()
    doc = parse(
        "# T\n\n## S\n\n"
        + "\n\n".join(
            ["The cat sat on the mat and purred quietly for a while."] * 6
            + ["The dog ran across the field chasing a ball happily."] * 6
        )
    )
    with_sem = ChunkingEngine(
        ChunkingSettings(
            chunk_target_tokens=400,
            chunk_max_tokens=448,
            chunk_min_tokens=10,
            chunk_overlap_tokens=0,
        ),
        embedder=lambda s: embed(list(s)),
    ).chunk(doc)
    assert len(with_sem) == 1  # single-chunk-section rule wins when the section fits (PRD §4.3)
    big = ChunkingSettings(
        chunk_target_tokens=400, chunk_max_tokens=448, chunk_min_tokens=10, chunk_overlap_tokens=0
    )
    long_doc = parse(
        "# T\n\n## S\n\n"
        + "\n\n".join(
            ["The cat sat on the mat and purred quietly for a while."] * 30
            + ["The dog ran across the field chasing a ball happily."] * 30
        )
    )
    cut = ChunkingEngine(big, embedder=lambda s: embed(list(s))).chunk(long_doc)
    assert any(
        c.text.lstrip().startswith("The dog") for c in cut
    )  # a chunk starts exactly at the topic shift


def test_quality_and_flags() -> None:
    assert length_fit(320, 320, 64, 448) == 1.0
    assert length_fit(20, 320, 64, 448) == 0.0
    assert 0 < length_fit(500, 320, 64, 448) < 1
    assert repetition_ratio("a b c a b c a b c a b c") > 0.5
    score, flags = chunk_quality(
        "The system stores data.",
        chunk_type="prose",
        tokens=30,
        target=320,
        minimum=64,
        maximum=448,
        has_heading=True,
    )
    assert "short" in flags
    assert 0 < score < 1
    doc = parse("# T\n\n## S\n\n" + "\n\n".join(PARAS[:6]))
    chunks = ChunkingEngine(
        ChunkingSettings(
            chunk_target_tokens=60, chunk_max_tokens=80, chunk_min_tokens=10, chunk_overlap_tokens=0
        )
    ).chunk(doc)
    assert all(0 <= c.quality_score <= 1 for c in chunks)
    assert all("low_quality" not in c.quality_flags for c in chunks)


def test_context_prefix_and_config_hash() -> None:
    prefix = render_context_prefix(
        "Getting Started",
        ("Getting Started", "Install", "Docker", "Volumes", "Deep"),
        max_tokens=40,
        count=COUNT,
    )
    assert prefix == "Getting Started › Install › Docker › Volumes › Deep"
    assert render_context_prefix("T", ("A",) * 20, max_tokens=3, count=COUNT) in ("A › A", "A", "")
    h1 = config_hash(ChunkingSettings())
    assert len(h1) == 16
    assert h1 == config_hash(ChunkingSettings(chunker_version="9.9.9"))  # version excluded
    assert h1 != config_hash(ChunkingSettings(chunk_target_tokens=300))


def test_silver_rows_conform_and_roundtrip_document() -> None:
    doc = parse(LONG_DOC)
    chunks = ChunkingEngine(SMALL).chunk(doc)
    table = enforce(
        pa.Table.from_pylist([c.to_silver_row() for c in chunks], schema=CHUNKS.schema), CHUNKS
    )
    assert table.num_rows == len(chunks)
    row = SilverStore.prepare_document_row(doc.to_silver_row())
    enforce(pa.Table.from_pylist([row], schema=DOCUMENTS.schema), DOCUMENTS)
    row["metadata"] = dict(row["metadata"])
    rebuilt = document_from_silver_row(row)
    assert rebuilt.text == doc.text
    assert [b.kind for b in rebuilt.blocks] == [b.kind for b in doc.blocks]
    assert [c.chunk_id for c in ChunkingEngine(SMALL).chunk(rebuilt)] == [
        c.chunk_id for c in chunks
    ]


# --------------------------------------------------------------------------- incremental
@dataclass
class Old:
    chunk_id: uuid.UUID
    chunk_key: str
    chunk_checksum: str
    lineage_id: uuid.UUID


def _old(chunks: list) -> list[Old]:  # type: ignore[type-arg]
    return [Old(c.chunk_id, c.chunk_key, c.chunk_checksum, c.chunk_id) for c in chunks]


def test_incremental_no_change_writes_nothing() -> None:
    eng = ChunkingEngine(SMALL)
    v1 = eng.chunk(parse(LONG_DOC))
    diff = diff_chunks(v1[0].document_id, eng.chunk(parse(LONG_DOC)), _old(v1))
    assert (diff.unchanged, diff.modified, diff.moved, diff.added, diff.removed) == (
        len(v1),
        0,
        0,
        0,
        0,
    )
    assert not diff.changed


def test_incremental_modified_moved_added_removed() -> None:
    eng = ChunkingEngine(
        ChunkingSettings(
            chunk_target_tokens=60, chunk_max_tokens=80, chunk_min_tokens=10, chunk_overlap_tokens=0
        )
    )
    base = (
        "# T\n\n## Alpha\n\n"
        + "\n\n".join(PARAS[:6])
        + "\n\n## Beta\n\n"
        + "\n\n".join(PARAS[6:12])
    )
    v1 = eng.chunk(parse(base))
    old = _old(v1)
    edited = (
        base.replace("component 0", "component zero").replace("## Beta", "## Gamma")
        + "\n\n## New\n\n"
        + "\n\n".join(PARAS[:6])
    )
    v2 = eng.chunk(parse(edited))
    diff = diff_chunks(v1[0].document_id, v2, old)
    assert diff.modified == 1
    assert diff.moved == len(
        [c for c in v1 if c.heading_path[-1] == "Beta"]
    )  # heading renamed, content identical
    assert diff.added == len([c for c in v2 if c.heading_path[-1] == "New"])
    assert diff.removed == 0
    assert len(diff.superseded_ids) == diff.modified + diff.moved
    statuses = {c.status for c in diff.to_write}
    assert statuses == {ChunkStatus.MODIFIED, ChunkStatus.MOVED, ChunkStatus.ADDED}
    lineage = {o.lineage_id for o in old}
    assert all(c.lineage_id in lineage for c in diff.to_write if c.status is not ChunkStatus.ADDED)
    assert all(c.lineage_id == c.chunk_id for c in diff.to_write if c.status is ChunkStatus.ADDED)
    # removal: drop Beta entirely
    v3 = eng.chunk(parse("# T\n\n## Alpha\n\n" + "\n\n".join(PARAS[:6])))
    diff3 = diff_chunks(v1[0].document_id, v3, old)
    assert diff3.removed == len([c for c in v1 if c.heading_path[-1] == "Beta"])
    assert diff3.unchanged == len(v3)
    assert diff3.to_write == []
    assert diff3.changed  # removals must be written as tombstones


# --------------------------------------------------------------------------- properties
@settings(max_examples=25, deadline=None)
@given(
    n_paras=st.integers(min_value=1, max_value=14),
    words=st.integers(min_value=3, max_value=60),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_property_chunks_cover_text_and_respect_limits(n_paras: int, words: int, seed: int) -> None:
    import random

    rng = random.Random(seed)
    vocab = [
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
    paras = [
        " ".join(rng.choice(vocab) for _ in range(words)).capitalize() + "." for _ in range(n_paras)
    ]
    doc = parse("# H\n\n## S\n\n" + "\n\n".join(paras))
    cfg = ChunkingSettings(
        chunk_target_tokens=48, chunk_max_tokens=64, chunk_min_tokens=8, chunk_overlap_tokens=8
    )
    chunks = ChunkingEngine(cfg).chunk(doc)
    assert chunks
    for c in chunks:
        assert doc.text[c.start_char : c.end_char] == c.text
        assert (
            c.token_count <= cfg.chunk_max_tokens or c.text.count(" ") < 2
        )  # a single unsplittable word may exceed
    # every prose character is covered by at least one chunk
    covered = [False] * len(doc.text)
    for c in chunks:
        for i in range(c.start_char, c.end_char):
            covered[i] = True
    body_start = doc.blocks[-1].start_char if len(doc.blocks) <= 2 else doc.blocks[2].start_char
    assert all(covered[i] or doc.text[i].isspace() for i in range(body_start, len(doc.text)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
