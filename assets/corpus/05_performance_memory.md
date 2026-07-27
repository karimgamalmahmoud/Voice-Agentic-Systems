# Reference: Performance & Memory

**GC generations:** short-lived objects (Gen 0) are cheap to collect; objects promoted to Gen 2 are expensive. High allocation rates in hot paths cause GC pressure and latency spikes.

**Disposal:** wrap `IDisposable` resources (connections, streams) in `using` so they're released promptly; leaked handles look like slow memory growth.

**Caching layers:** in-memory cache is fastest but per-instance; distributed cache (Redis) is shared across instances at the cost of a network hop. Always have an invalidation story.

**Profiling:** measure with a profiler or APM before optimizing. `Span<T>`/`Memory<T>` reduce allocations in hot paths, but only reach for them once a profile shows allocation is the bottleneck.
