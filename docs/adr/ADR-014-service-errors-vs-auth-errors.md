# ADR-014 — Service-unavailability errors must never reuse 401/403

**Status:** Accepted (Batch H, Milestones 49–52)

## Context
While building API-key management endpoints, `Authenticator.list_api_keys()` (and
`create_api_key`/`revoke_api_key`/`verify_api_key`) raised `AuthError` (HTTP 401) whenever
`self.db is None` — i.e. when the API-key store simply isn't configured. A 401 tells the calling
client "your credentials are wrong, try different ones"; that is the wrong instruction when the
actual problem is that the server has no database to check credentials against. The same
confusion existed for `AKL_JWT_SECRET not configured`, and separately, a hard document delete with
a missing `X-Confirm` header was reported as a generic 500 (server fault) instead of a client input
error.

## Decision
- Introduced `ApiKeyStoreUnavailableError` (`AKL-E1007`, 503, retryable) for "no database
  configured for API keys", and `AuthConfigError` (`AKL-E1008`, 503, retryable) for "no JWT secret
  configured". Both replace the previous `AuthError` (401) use in these code paths.
- Introduced `HardDeleteConfirmationRequiredError` (`AKL-E3060`, 400) for a missing/incorrect
  `X-Confirm: hard-delete` header, replacing a generic base `AKLError` (500).
- `AuthError` (401) is now reserved for its one correct meaning: **the request presented no
  credentials, or credentials that do not check out** (`missing credentials`, invalid JWT, unknown
  or revoked API key). `ForbiddenError` (403) stays for **valid credentials, insufficient scope**.
  Anything else — server misconfiguration, unavailable dependency, malformed client input — gets
  its own status/code.

## Consequences
- + Clients (and alerting) can distinguish "fix your credentials" (401) from "retry later, this is
  us" (503) from "fix your request" (400) — conflating these was the direct cause of a confusing
  debugging session during Batch H (a test asserting 401 for a "no DB" condition passed by
  accident, masking the real behavior).
- + Establishes the pattern for any future error: pick the status by *what the client should do
  next*, not by which subsystem raised it.
- − Existing external clients or dashboards keyed on the old (incorrect) 401/500 codes for these
  specific conditions need updating; there are none yet (Batch H is the first release with these
  endpoints), so no migration is required.
