# Benchmarks

Two different tools answer two different questions:

| Tool | Question it answers | Command |
|---|---|---|
| `akl-cli bench run` (`akl/eval/benchmark.py`) | How fast is retrieval/generation itself, in-process, isolated from HTTP overhead? | `make bench` |
| Locust (`tests/load/locustfile.py`) | How does the whole API behave under concurrent traffic (connection handling, ASGI overhead, rate limiting)? | `make load-test` |

Neither number substitutes for the other — a fast in-process benchmark with a slow load-tested API
usually points at ASGI/networking/rate-limit configuration, not retrieval quality.

## Latency benchmark
`make bench` runs a fixed query set through the real `RAGService` (dense + BM25 + rerank, and
optionally generation) `repeats` times per query, in-process, and reports p50/p95/p99 per stage.
Reports are written to this directory as `<timestamp>.md` and are not committed automatically —
commit one when you want to record a baseline for comparison across a release.

```bash
uv run akl-cli bench run --repeats 10 --include-answer --max-p95-ms 2000
```

## Retrieval quality
`make eval-run` (`akl/eval/`) generates or reuses a synthetic QA set and reports Recall@k, MRR,
nDCG@k, and refusal precision/recall against real hybrid retrieval — see `.github/workflows/nightly.yml`
in the repository root for the exact thresholds gating CI (`recall_at_10 >= 0.85`, `mrr >= 0.70`,
`refusal_precision >= 0.80`).

```bash
uv run akl-cli eval generate-qa --n 50
uv run akl-cli eval run --check-answers --min-recall 0.85 --min-mrr 0.70
uv run akl-cli eval calibrate   # sweep AKL_RAG_MIN_CONFIDENCE and get a recommendation
```

## Load test
```bash
make load-test   # 20 users, 2 minutes, against http://localhost:8000 by default
```
Override `AKL_LOAD_TARGET`, `AKL_LOAD_USERS`, `AKL_LOAD_SPAWN_RATE`, `AKL_LOAD_DURATION` as needed,
and set `AKL_LOAD_API_KEY` or `AKL_LOAD_JWT` if the target has authentication enabled.

*No committed baseline numbers exist yet — this page describes the tooling; the first real
numbers should come from running the above against your own hardware/corpus, not from a number
copied out of this documentation.*
