# Reference: Async & Concurrency in .NET

`async`/`await` is about **throughput, not raw speed**. Awaiting an I/O operation returns the thread to the pool so it can serve other requests; the single call is not faster.

**Common failure under load:** sync-over-async — calling `.Result` or `.Wait()` on a Task blocks a pool thread. Enough of these and the thread pool starves, so latency climbs even though CPU looks idle. Fix by making the path async end-to-end.

**Parallelism vs concurrency:** parallelism runs work on multiple cores at once; concurrency interleaves many in-flight operations. Web backends are usually concurrency-bound (waiting on DB/network), not CPU-bound.

**Good practice:** pass `CancellationToken` through the request pipeline; avoid blocking calls inside controllers; use `ConfigureAwait(false)` in library code with no sync-context needs.
