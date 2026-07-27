# Reference: Data Access with EF Core

**N+1 queries** are the classic hotspot: loading a list, then lazily loading a related entity per row = 1 + N round trips. Fix with eager loading (`Include`) or a projection that fetches only what's needed.

**Read paths:** `AsNoTracking()` skips change-tracking overhead for queries that don't update — meaningful on large result sets.

**IQueryable vs IEnumerable:** operations on `IQueryable` translate to SQL and run in the database; once you hit `IEnumerable` (e.g. `.ToList()` too early), the rest runs in memory, often pulling far more rows than needed.

**Other levers:** missing indexes, over-fetching columns, and connection-pool exhaustion. Senior engineers read the generated SQL (logging or a profiler) rather than guessing.
