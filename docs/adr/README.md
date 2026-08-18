# Architecture Decision Records

Architecture Decision Records (ADRs) preserve decisions that materially affect the system's data,
reliability, security, evaluation, or deployment behavior.

| ADR | Decision | Status |
| --- | --- | --- |
| [0001](0001-python-core.md) | Use Python for the core data and ML pipeline | Accepted |
| [0002](0002-capture-level-partitions.md) | Use capture-level partitions for primary evaluation | Accepted |
| [0003](0003-replay-before-live-capture.md) | Build deterministic replay before live capture | Accepted |

Each new ADR should describe the context, decision, consequences, and alternatives considered.
Accepted ADRs are not silently rewritten; superseding decisions should reference the earlier ADR.
