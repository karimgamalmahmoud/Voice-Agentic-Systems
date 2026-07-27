# Reference: Dependency Injection & Service Lifetimes

.NET's built-in container supports three lifetimes:
- **Singleton** — one instance for the app lifetime.
- **Scoped** — one instance per request (e.g. `DbContext`).
- **Transient** — a new instance every time it's resolved.

**Captive dependency bug:** injecting a Scoped service (like `DbContext`) into a Singleton captures it for the whole app lifetime, which breaks per-request isolation and thread safety. This is a frequent source of subtle production bugs.

Prefer **constructor injection**; treat the *service locator* pattern (resolving from the container inside methods) as an anti-pattern that hides dependencies. Register interfaces, not concrete types, to keep code testable.
