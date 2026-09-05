"""ChunkingEngine: UnifiedDocument → chunks (PRD Chapter 4)."""

from __future__ import annotations

import bisect
import hashlib
import json
from collections.abc import Sequence

from akl.chunking import identity
from akl.chunking.code_splitter import split_code
from akl.chunking.merge_split import Candidate, apply_overlap, atomize, merge_undersized, pack
from akl.chunking.models import Chunk, ChunkStatus
from akl.chunking.quality import chunk_quality
from akl.chunking.semantic import Embedder, semantic_boundaries
from akl.chunking.sentences import SentenceSplitter, Span
from akl.chunking.structural import Section, build_sections
from akl.chunking.table_splitter import split_table
from akl.chunking.tokenizer import TokenCounter
from akl.config import ChunkingSettings
from akl.ingestion.models import (
    Block,
    CodeBlock,
    ImageBlock,
    ListBlock,
    ParagraphBlock,
    TableBlock,
    UnifiedDocument,
)

LEAD_IN_MAX_TOKENS = 80


def config_hash(settings: ChunkingSettings) -> str:
    """16-hex digest of every chunking parameter except ``chunker_version`` (PRD §4.1)."""
    payload = settings.model_dump(mode="json")
    payload.pop("chunker_version", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class ChunkingEngine:
    def __init__(
        self,
        settings: ChunkingSettings,
        counter: TokenCounter | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.cfg = settings
        self.counter = counter or TokenCounter()
        self.count = self.counter.count
        self.embedder = embedder if settings.chunk_semantic_enabled else None
        self.config_hash = config_hash(settings)
        self._splitters: dict[str, SentenceSplitter] = {}

    # -- public -----------------------------------------------------------------------
    def chunk(self, doc: UnifiedDocument) -> list[Chunk]:
        text = doc.text
        newline_positions = [i for i, ch in enumerate(text) if ch == "\n"]
        chunks: list[Chunk] = []
        for section in build_sections(doc):
            section_chunks = self._chunk_section(doc, section)
            for ordinal, chunk in enumerate(section_chunks):
                chunk.section_ordinal = ordinal
            chunks.extend(section_chunks)

        for index, chunk in enumerate(chunks):
            chunk.chunk_index = index
            chunk.language = doc.language
            chunk.security_level = doc.security_level
            chunk.allowed_groups = tuple(doc.allowed_groups)
            chunk.chunker_version = self.cfg.chunker_version
            chunk.chunk_config_hash = self.config_hash
            chunk.context_prefix = identity.render_context_prefix(
                doc.title,
                chunk.heading_path,
                max_tokens=self.cfg.chunk_context_prefix_tokens,
                count=self.count,
            )
            chunk.page_start, chunk.page_end = self._pages(doc, chunk.start_char, chunk.end_char)
            if doc.source_type in ("markdown", "github", "html"):
                chunk.line_start = bisect.bisect_right(newline_positions, chunk.start_char - 1) + 1
                chunk.line_end = (
                    bisect.bisect_right(
                        newline_positions, max(chunk.start_char, chunk.end_char - 1)
                    )
                    + 1
                )
            score, flags = chunk_quality(
                chunk.text,
                chunk_type=chunk.chunk_type,
                tokens=chunk.token_count,
                target=self.cfg.chunk_target_tokens,
                minimum=self.cfg.chunk_min_tokens,
                maximum=self.cfg.chunk_max_tokens,
                has_heading=bool(chunk.heading_path),
            )
            chunk.quality_score = score
            all_flags = list(chunk.quality_flags) + list(flags)
            if score < self.cfg.chunk_quality_min:
                all_flags.append("low_quality")
            elif score < 0.5:
                all_flags.append("marginal")
            chunk.quality_flags = tuple(dict.fromkeys(all_flags))
            chunk.chunk_key = identity.chunk_key(
                doc.document_id, chunk.heading_path, chunk.section_ordinal
            )
            chunk.chunk_checksum = identity.chunk_checksum(chunk.text)
            chunk.chunk_id = identity.chunk_id(
                doc.document_id, chunk.chunk_key, chunk.chunk_checksum
            )
            chunk.lineage_id = chunk.chunk_id
            chunk.embedded_text_sha256 = identity.embedded_text_sha256(
                chunk.context_prefix, chunk.text
            )
            chunk.status = ChunkStatus.ADDED
        for prev, cur in zip(chunks, chunks[1:], strict=False):
            prev.next_chunk_id = cur.chunk_id
            cur.prev_chunk_id = prev.chunk_id
        return chunks

    # -- sections -----------------------------------------------------------------------
    def _chunk_section(self, doc: UnifiedDocument, section: Section) -> list[Chunk]:
        text = doc.text
        blocks = section.blocks
        total_tokens = self.count(text[section.start_char : section.end_char])
        if total_tokens <= self.cfg.chunk_max_tokens:
            kinds = {b.kind for b in blocks}
            if kinds <= {"paragraph", "image"}:
                ctype = "prose"
            elif kinds == {"list"}:
                ctype = "list"
            elif kinds == {"code"}:
                ctype = "code"
            elif kinds == {"table"}:
                ctype = "table"
            else:
                ctype = "mixed"
            code_lang = next((b.language for b in blocks if isinstance(b, CodeBlock)), None)
            return [
                self._make(
                    doc,
                    section,
                    text[section.start_char : section.end_char],
                    section.start_char,
                    section.end_char,
                    ctype,
                    code_language=code_lang,
                )
            ]

        out: list[Chunk] = []
        prose_run: list[Block] = []
        last_paragraph: ParagraphBlock | None = None
        for block in blocks:
            if isinstance(block, CodeBlock | TableBlock):
                out.extend(self._chunk_prose(doc, section, prose_run))
                prose_run = []
                if isinstance(block, CodeBlock):
                    out.extend(self._chunk_code(doc, section, block, last_paragraph))
                else:
                    out.extend(self._chunk_table(doc, section, block, last_paragraph))
                last_paragraph = None
            else:
                prose_run.append(block)
                if isinstance(block, ParagraphBlock):
                    last_paragraph = block
        out.extend(self._chunk_prose(doc, section, prose_run))
        return out

    def _chunk_prose(
        self, doc: UnifiedDocument, section: Section, blocks: Sequence[Block]
    ) -> list[Chunk]:
        if not blocks:
            return []
        text = doc.text
        spans: list[Span] = []
        for block in blocks:
            if isinstance(block, ListBlock):
                # each list item is a sentence-like unit: split the rendered text per line
                cursor = block.start_char
                for line in text[block.start_char : block.end_char].split("\n"):
                    if line.strip():
                        idx = text.index(line, cursor)
                        spans.append(
                            Span(
                                line.strip(),
                                idx + (len(line) - len(line.lstrip())),
                                idx + len(line.rstrip()),
                            )
                        )
                        cursor = idx + len(line)
            elif isinstance(block, ImageBlock):
                spans.append(
                    Span(text[block.start_char : block.end_char], block.start_char, block.end_char)
                )
            else:
                for span in self._splitter(doc.language).split(
                    text[block.start_char : block.end_char], block.start_char
                ):
                    spans.extend(atomize(span, self.count, self.cfg.chunk_max_tokens))
        if not spans:
            return []
        boundaries = semantic_boundaries(
            [s.text for s in spans], self.embedder, self.cfg.chunk_semantic_threshold
        )
        cands = pack(
            spans, self.count, self.cfg.chunk_target_tokens, self.cfg.chunk_max_tokens, boundaries
        )
        cands = merge_undersized(
            cands, self.count, self.cfg.chunk_min_tokens, self.cfg.chunk_max_tokens
        )
        cands = apply_overlap(cands, self.count, self.cfg.chunk_overlap_tokens)
        ctype = "list" if all(isinstance(b, ListBlock) for b in blocks) else "prose"
        return [self._from_candidate(doc, section, c, ctype) for c in cands]

    def _chunk_code(
        self,
        doc: UnifiedDocument,
        section: Section,
        block: CodeBlock,
        lead_in: ParagraphBlock | None,
    ) -> list[Chunk]:
        text = doc.text
        block_text = text[block.start_char : block.end_char]
        if self.count(block_text) <= self.cfg.chunk_code_max_tokens:
            start = block.start_char
            if (
                lead_in is not None
                and self.count(text[lead_in.start_char : lead_in.end_char]) <= LEAD_IN_MAX_TOKENS
            ):
                start = lead_in.start_char
            return [
                self._make(
                    doc,
                    section,
                    text[start : block.end_char],
                    start,
                    block.end_char,
                    "code",
                    code_language=block.language,
                )
            ]
        pieces = split_code(block_text, block.language, self.count, self.cfg.chunk_code_max_tokens)
        chunks: list[Chunk] = []
        for piece in pieces:
            s, e = block.start_char + piece.start, block.start_char + piece.end
            chunk = self._make(doc, section, text[s:e], s, e, "code", code_language=block.language)
            if piece.index > 0:
                chunk.quality_flags = (
                    *chunk.quality_flags,
                    f"code_split_{piece.index + 1}_of_{piece.total}",
                )
            chunks.append(chunk)
        return chunks

    def _chunk_table(
        self,
        doc: UnifiedDocument,
        section: Section,
        block: TableBlock,
        lead_in: ParagraphBlock | None,
    ) -> list[Chunk]:
        text = doc.text
        caption = block.caption or (
            text[lead_in.start_char : lead_in.end_char]
            if lead_in
            and self.count(text[lead_in.start_char : lead_in.end_char]) <= LEAD_IN_MAX_TOKENS
            else None
        )
        pieces = split_table(
            block.markdown, self.count, self.cfg.chunk_table_max_tokens, caption=caption
        )
        chunks: list[Chunk] = []
        for piece in pieces:
            chunk = self._make(
                doc, section, piece.markdown, block.start_char, block.end_char, "table"
            )
            if piece.total > 1:
                chunk.quality_flags = (
                    *chunk.quality_flags,
                    f"table_split_{piece.index + 1}_of_{piece.total}",
                )
            chunks.append(chunk)
        return chunks

    # -- helpers ----------------------------------------------------------------------------
    def _from_candidate(
        self, doc: UnifiedDocument, section: Section, cand: Candidate, ctype: str
    ) -> Chunk:
        chunk = self._make(doc, section, cand.text(doc.text), cand.start, cand.end, ctype)
        chunk.overlap_prev_tokens = sum(self.count(s.text) for s in cand.overlap)
        return chunk

    def _make(
        self,
        doc: UnifiedDocument,
        section: Section,
        text: str,
        start: int,
        end: int,
        ctype: str,
        *,
        code_language: str | None = None,
    ) -> Chunk:
        return Chunk(
            document_id=doc.document_id,
            document_version_id=doc.document_version_id,
            source_type=doc.source_type,
            chunk_index=0,
            chunk_type=ctype,
            heading_path=section.heading_path,
            heading_level=section.heading_level,
            text=text,
            context_prefix="",
            start_char=start,
            end_char=end,
            token_count=self.count(text),
            code_language=code_language,
        )

    def _splitter(self, language: str | None) -> SentenceSplitter:
        lang = (language or "en") if language not in (None, "und") else "en"
        if lang not in self._splitters:
            self._splitters[lang] = SentenceSplitter(lang)
        return self._splitters[lang]

    @staticmethod
    def _pages(doc: UnifiedDocument, start: int, end: int) -> tuple[int | None, int | None]:
        if not doc.page_map:
            return None, None
        ps = pe = None
        for page in doc.page_map:
            if page["start_char"] <= start < max(page["end_char"], page["start_char"] + 1):
                ps = page["page"]
            if page["start_char"] < end <= page["end_char"] or (
                page["start_char"] <= end and end == page["end_char"]
            ):
                pe = page["page"]
        return ps, pe or ps
