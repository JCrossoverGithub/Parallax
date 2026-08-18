# ADR-0003: Build deterministic replay before live capture

- Status: Accepted
- Date: 2026-08-18

## Context

Live packet capture introduces operating-system permissions, interface selection, changing traffic,
and nondeterministic timing. Those concerns can hide defects in flow construction, feature
generation, inference, persistence, and event delivery.

## Decision

Implement controlled PCAP replay before live packet capture. Replay must support preserved timing,
configurable time scaling, stable session and window identifiers, bounded buffering, and explicit
pause, resume, cancellation, completion, and failure states.

## Consequences

- The complete runtime can be tested with repeatable inputs.
- CI can use small fixtures without packet-capture privileges.
- The dashboard can be built and evaluated before a privileged sensor exists.
- Live capture remains a separate milestone and will use a narrow least-privilege process.
