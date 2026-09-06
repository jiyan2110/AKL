# Release Checklist

## Before tagging
1. `make lint && make test` — clean.
2. `make up && make wait && make test-component` — clean against a fresh stack.
3. `make dags-test` — all 5 DAGs import and `akl_ingestion` runs end to end.
4. `make eval-run` against a representative corpus (not just the tiny example docs) — thresholds
   pass, or you understand and accept why they don't.
5. `make bench` — no unexplained regression against the last recorded baseline in `docs/benchmarks/`.
6. Update `CHANGELOG.md`: move `[Unreleased]` entries under a new `## [x.y.z] — <date>` heading.
7. Bump `version` in `pyproject.toml` to match.
8. `scripts/generate_docs_reference.py` — commit any changes to `docs/reference/errors.md` /
   `metrics.md` (a diff here usually means new error codes or metrics were added without a
   matching docs update — that's fine, just make sure it's reviewed, not accidental).

## Tagging
```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```
This triggers `.github/workflows/release.yml`: multi-arch (`amd64`/`arm64`) images for `akl-api`
and `akl-airflow` are built, pushed to GHCR, SBOM'd, and signed (keyless, via GitHub OIDC — no
secret key material needed). A GitHub Release is cut with `CHANGELOG.md`'s matching section as the
body.

## After release
1. Verify the GitHub Release page shows both signed image digests and attached SBOMs.
2. Verify pull: `docker pull ghcr.io/<org>/ai-knowledge-lakehouse/api:X.Y.Z`.
3. If deploying: `.github/workflows/deploy.yml` (`workflow_dispatch`) — needs `SSH_HOST`,
   `SSH_USER`, `SSH_PRIVATE_KEY` secrets and a GitHub Environment matching the chosen target
   already configured. This is a template; review it against your actual target before the first
   real run.
4. Watch `.github/workflows/nightly.yml`'s next scheduled run (03:17 UTC) for the new version —
   it's your first automated confirmation that the release behaves correctly under the full
   integration + eval + benchmark suite, not just at build time.

## Rollback
Images are tagged both `X.Y.Z` and `X.Y` (rolling) — redeploy the previous `X.Y.Z` tag via
`deploy.yml`'s `ref` input, or `docker compose ... pull && up -d` with the previous tag pinned in
`.env.prod`. Database migrations in this project are additive-only within a major version (see
`akl/db/migrations/`), so a rollback should not require a matching down-migration — verify this
holds for whatever changed in the release you're rolling back before assuming it's safe.
