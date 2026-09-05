"""Unit tests: LLM provider, prompt builder, memory rewrite, generation guards (Milestone 34)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from tests.unit.test_retrieval import rows

from akl.config import LLMSettings, RetrievalSettings
from akl.rag.citations import attach_citations
from akl.rag.context_builder import ContextBuilder
from akl.rag.llm.provider import (
    LLMConfigError,
    LLMUnavailableError,
    OpenAICompatProvider,
    build_llm,
)
from akl.rag.memory import ConversationMemory, MemoryState, _rule_summary
from akl.rag.prompt import PromptBuilder
from akl.rag.retrieval.models import Candidate
from akl.rag.service import _split_safe, _unsupported_tokens

pytestmark = pytest.mark.unit


def _llm_transport(
    answer: str = "Backups run nightly [1]. Snapshots are kept [2].", *, fail: bool = False
) -> httpx.MockTransport:
    def handler(req: httpx.Request) -> httpx.Response:
        if fail:
            return httpx.Response(503, text="down")
        body = json.loads(req.content)
        if body.get("stream"):
            chunks = [answer[i : i + 7] for i in range(0, len(answer), 7)]
            lines = [
                f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}\n\n" for c in chunks
            ]
            lines.append("data: [DONE]\n\n")
            return httpx.Response(
                200, content="".join(lines).encode(), headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(
            200,
            json={
                "id": "x",
                "model": body["model"],
                "choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 20},
            },
        )

    return httpx.MockTransport(handler)


def test_openai_compat_complete_and_stream() -> None:
    settings = LLMSettings(
        llm_provider="openai_compat", llm_model="fake-1", llm_base_url="http://llm.local/v1"
    )
    llm = OpenAICompatProvider(settings, transport=_llm_transport())
    res = llm.complete([{"role": "user", "content": "q"}], max_tokens=50, temperature=0.0)
    assert res.text.startswith("Backups run nightly [1]")
    assert res.input_tokens == 120 and res.output_tokens == 20  # noqa: PT018
    assert res.cost_usd(0.5, 1.5) == pytest.approx(0.06 + 0.03)
    streamed = "".join(
        llm.stream([{"role": "user", "content": "q"}], max_tokens=50, temperature=0.0)
    )
    assert streamed == "Backups run nightly [1]. Snapshots are kept [2]."
    down = OpenAICompatProvider(settings, transport=_llm_transport(fail=True))
    with pytest.raises(LLMUnavailableError):
        down.complete([{"role": "user", "content": "q"}], max_tokens=5, temperature=0.0)
    with pytest.raises(LLMUnavailableError):
        list(down.stream([{"role": "user", "content": "q"}], max_tokens=5, temperature=0.0))


def test_build_llm_none_and_errors() -> None:
    assert build_llm(LLMSettings()) is None
    with pytest.raises(LLMConfigError):
        build_llm(LLMSettings(llm_provider="openai_compat"))  # model missing
    with pytest.raises(LLMConfigError):
        build_llm(LLMSettings(llm_provider="weird", llm_model="m"))


def _ctx() -> ContextBuilder:
    return ContextBuilder(lambda t: len(t.split()), budget_tokens=1000, top_k=8)


def _candidates(n: int = 3) -> list[Candidate]:
    out = []
    for i, r in enumerate(rows()[:n]):
        c = Candidate(r["chunk_id"], r, rrf_score=0.1 - i * 0.01)
        c.rerank_score = 0.9 - i * 0.1
        out.append(c)
    return out


def test_prompt_builder_templates_history_truncation(tmp_path: Path) -> None:
    (tmp_path / "answer_v1.md").write_text("System for {org_name}.", encoding="utf-8")
    pb = PromptBuilder(
        tmp_path, org_name="ACME", count_tokens=lambda t: len(t.split()), max_input_tokens=90
    )
    ctx = _ctx().build(_candidates(2))
    history = [
        {"role": "user", "content": "old " * 40},
        {"role": "assistant", "content": "older answer " * 5},
        {"role": "user", "content": "recent question"},
    ]
    built = pb.build_answer("What runs nightly?", ctx, summary="Earlier: backups.", history=history)
    assert built.messages[0]["content"] == "System for ACME."
    assert built.messages[1]["content"].startswith("Conversation summary")
    assert built.messages[-1]["content"].endswith("Question: What runs nightly?")
    assert "[1] source=" in built.messages[-1]["content"]
    assert built.truncated_history is True
    assert "recent question" in [
        m["content"] for m in built.messages
    ]  # newest turns survive truncation
    assert (
        built.input_tokens <= 120 or len([m for m in built.messages if m["role"] != "system"]) == 1
    )
    assert pb.template("summarize_v1")  # builtin fallback when file is absent
    rewrite = pb.build_rewrite(
        "sum", [{"role": "user", "content": "how is AKL_S3_BUCKET set"}], "and its default?"
    )
    assert "Latest message: and its default?" in rewrite[-1]["content"]


def test_memory_rewrite_rule_based_and_summary() -> None:
    memory = ConversationMemory(
        None,
        LLMSettings(),
        PromptBuilder(
            Path("configs/prompts"), org_name="x", count_tokens=len, max_input_tokens=100
        ),
        None,
        len,
    )  # type: ignore[arg-type]
    state = MemoryState(
        conversation_id=__import__("uuid").uuid4(),
        summary=None,
        history=[
            {"role": "user", "content": "How is AKL_S3_BUCKET configured?"},
            {"role": "assistant", "content": "It is set in .env [1]."},
        ],
    )
    assert memory.rewrite("What is its default?", state) == "What is its default? AKL_S3_BUCKET"
    assert (
        memory.rewrite("How are backups scheduled for postgres nightly?", state)
        == "How are backups scheduled for postgres nightly?"
    )
    assert (
        memory.rewrite(
            "why", MemoryState(conversation_id=state.conversation_id, summary=None, history=[])
        )
        == "why"
    )
    summary = _rule_summary(
        "Earlier: setup.",
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ],
    )
    assert summary.startswith("Earlier: setup.")
    assert "q1 | q2" in summary


def test_generation_guards() -> None:
    assert _split_safe("Backups run nightly [1") == ("Backups run nightly ", "[1")
    assert _split_safe("Backups [1][2] and more") == ("Backups [1][2] and more", "")
    ctx = _ctx().build(_candidates(2))
    assert _unsupported_tokens("Set AKL_NOT_REAL_VAR in settings.yaml [1].", ctx) == [
        "AKL_NOT_REAL_VAR",
        "settings.yaml",
    ]
    assert _unsupported_tokens("Snapshot Qdrant after sync [2].", ctx) == []
    cited = attach_citations(
        "Backups run nightly [1]. Snapshots are kept [2].", ctx, max_uncited_ratio=0.2
    )
    assert [c.index for c in cited.citations] == [1, 2]
    assert "low_faithfulness" not in cited.flags
    assert RetrievalSettings().rag_strong_confidence == 0.6
