# Reference: API Design & Security

**Middleware pipeline:** requests flow through ordered middleware (auth, logging, exception handling, etc.). Order matters — authentication must run before authorization.

**Status codes:** use them honestly — 400 for bad input, 401 vs 403, 404, 409 for conflicts, 5xx for server faults. Validate input at the boundary.

**Auth:** short-lived JWT access tokens plus a refresh-token flow; store refresh tokens securely and rotate them. Never trust client-supplied identity.

**Resilience:** rate limiting, idempotency keys for unsafe retries, and timeouts on downstream calls to avoid cascading failures.
