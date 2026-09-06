"""Synthetic QA generation (PRD §12.2): from Gold active units, produce (question, expected
chunk/document) pairs into ``gold/eval/qa_pairs`` for the eval runner.

Two generation methods:
* ``template`` (default, no external dependency): turns a chunk's heading breadcrumb + first
  sentence into a question via a small set of deterministic rules. Deterministic, offline, always
  available — this is what CI runs.
* ``llm``: asks the configured LLM to write one question whose answer is contained in the chunk.
  Used when a real generation model is configured (``AKL_LLM_PROVIDER=openai_compat``); falls back
  to ``template`` per-chunk if the LLM call fails, so a flaky/unavailable model never empties the set.

A configurable fraction of the set is **distractor** questions (``expected_chunk_ids: []``) —
generic questions about topics unlikely to be in the corpus, used to measure refusal precision/recall.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from akl.chunking.sentences import SentenceSplitter
from akl.rag.llm.provider import LLMProvider, LLMUnavailableError

DISTRACTOR_QUESTIONS: tuple[str, ...] = (
    "What is the recommended sourdough starter hydration ratio for high-altitude baking?",
    "How many moons does the dwarf planet Eris have according to this documentation?",
    "What is the marketing budget for the company's regional office in Antarctica?",
    "Which constellation is referenced in the disaster-recovery runbook?",
    "What flavor of ice cream does the on-call rotation prefer?",
)


@dataclass(frozen=True)
class QaPair:
    qa_id: str
    question: str
    expected_chunk_ids: list[str]
    expected_document_id: str | None
    reference_answer: str | None
    generation_method: str
    difficulty: str | None
    version: str

    def as_row(self) -> dict[str, Any]:
        return {
            "qa_id": self.qa_id,
            "question": self.question,
            "expected_chunk_ids": self.expected_chunk_ids,
            "expected_document_id": self.expected_document_id,
            "reference_answer": self.reference_answer,
            "generation_method": self.generation_method,
            "difficulty": self.difficulty,
            "version": self.version,
        }


_TEMPLATES: tuple[str, ...] = (
    'According to the {source} titled "{title}", what does the section on {topic} say?',
    'What does "{title}" explain about {topic}?',
    'In the context of {topic}, what is described in "{title}"?',
)


def _topic_from_breadcrumb(breadcrumb: str | None, title: str | None) -> str:
    if not breadcrumb:
        return "this topic"
    parts = [p.strip() for p in breadcrumb.split("›") if p.strip() and p.strip() != (title or "")]
    return parts[-1] if parts else "this topic"


def _template_question(row: dict[str, Any], *, rng: random.Random) -> tuple[str, str | None]:
    """Returns ``(question, reference_answer)`` built from the chunk's heading and first sentence."""
    title = str(row.get("title") or "the document")
    topic = _topic_from_breadcrumb(row.get("heading_breadcrumb"), row.get("title"))
    source = str(row.get("source_type") or "document")
    template = rng.choice(_TEMPLATES)
    question = template.format(source=source, title=title, topic=topic)
    text = str(row.get("text") or "")
    sentences = SentenceSplitter().split(text)
    reference = sentences[0].text if sentences else (text[:200] or None)
    return question, reference


def _llm_question(llm: LLMProvider, row: dict[str, Any]) -> tuple[str, str | None] | None:
    text = str(row.get("text") or "")
    prompt = [
        {
            "role": "system",
            "content": "Write exactly one clear, specific question whose answer is fully contained in the passage below. Output only the question, nothing else.",
        },
        {"role": "user", "content": text[:1500]},
    ]
    try:
        result = llm.complete(prompt, max_tokens=64, temperature=0.3)
    except LLMUnavailableError:
        return None
    question = result.text.strip().splitlines()[0].strip() if result.text.strip() else ""
    if not question:
        return None
    return question, text[:200]


def generate_qa_pairs(
    rows: Sequence[dict[str, Any]],
    *,
    version: str,
    n: int = 50,
    distractor_ratio: float = 0.15,
    method: str = "template",
    llm: LLMProvider | None = None,
    seed: int = 0,
) -> list[QaPair]:
    """Sample ``n`` chunks from ``rows`` (Gold active units) and generate one question each,
    plus ``n * distractor_ratio`` distractor questions with no expected chunk.
    """
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling for reproducible eval sets, not security
    pool = list(rows)
    rng.shuffle(pool)
    n_distractor = round(n * distractor_ratio)
    if distractor_ratio > 0 and n >= 1:
        n_distractor = max(
            1, n_distractor
        )  # a caller who asks for distractors must get at least one
    n_answerable = max(0, n - n_distractor)
    sample = pool[:n_answerable]

    pairs: list[QaPair] = []
    for row in sample:
        chunk_id = str(row["chunk_id"])
        document_id = str(row.get("document_id")) if row.get("document_id") else None
        question: str | None = None
        reference: str | None = None
        used_method = "template"
        if method == "llm" and llm is not None:
            got = _llm_question(llm, row)
            if got is not None:
                question, reference = got
                used_method = "llm"
        if question is None:
            question, reference = _template_question(row, rng=rng)
            used_method = "template"
        difficulty = "easy" if len(str(row.get("text") or "")) < 400 else "medium"
        pairs.append(
            QaPair(
                qa_id=uuid.uuid4().hex,
                question=question,
                expected_chunk_ids=[chunk_id],
                expected_document_id=document_id,
                reference_answer=reference,
                generation_method=used_method,
                difficulty=difficulty,
                version=version,
            )
        )

    distractor_pool = list(DISTRACTOR_QUESTIONS)
    rng.shuffle(distractor_pool)
    for i in range(n_distractor):
        pairs.append(
            QaPair(
                qa_id=uuid.uuid4().hex,
                question=distractor_pool[i % len(distractor_pool)],
                expected_chunk_ids=[],
                expected_document_id=None,
                reference_answer=None,
                generation_method="distractor",
                difficulty="distractor",
                version=version,
            )
        )
    rng.shuffle(pairs)
    return pairs
