# Senior Backend (.NET) — Answer Scoring Rubric

Use this rubric to evaluate a candidate's spoken answer to a technical question.
Score each relevant competency, then give an overall 1–5 with a one-line justification.
Only score competencies the question actually touches.

## Levels
- **1 — Well below:** vague, guesses, or actively wrong.
- **2 — Below:** names a concept but can't apply it; misses the systematic approach.
- **3 — Meets (mid):** correct on the common cases; some gaps under load or at the edges.
- **4 — Strong (senior):** systematic, measures before acting, covers DB + async + caching, discusses tradeoffs.
- **5 — Exceptional:** all of 4, plus quantifies, anticipates failure modes, and reasons about cost/latency budgets.

## Competency: Diagnostic method
Does the candidate **measure before changing**? Reproduce, gather evidence (APM, logs, profiler, DB query stats), form a hypothesis, then verify the fix moved the metric. Guessing and "just add a server / restart" is a 1–2.

## Competency: Data access (EF Core)
Recognizes N+1 queries, missing indexes, over-fetching, and change-tracking overhead. Knows `AsNoTracking` for read paths, eager vs lazy loading, and that `IQueryable` composes server-side while `IEnumerable` pulls into memory. Senior answers inspect the generated SQL.

## Competency: Async & concurrency
Understands that async frees threads rather than making one call faster; identifies sync-over-async (`.Result`/`.Wait()`) and thread-pool starvation as latency-under-load causes. Bonus: cancellation tokens, `ConfigureAwait`, avoiding blocking calls in request paths.

## Competency: Caching & performance
Proposes caching at the right layer (in-memory vs distributed/Redis) with invalidation awareness; distinguishes cacheable read-heavy data from volatile data. Mentions allocation/GC pressure only when relevant.

## Competency: Tradeoff & communication
States assumptions, weighs options out loud, and doesn't over-engineer. A senior answer names what they'd do *first* and why.
