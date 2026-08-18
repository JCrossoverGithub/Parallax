# ADR-0002: Use capture-level partitions for primary evaluation

- Status: Accepted
- Date: 2026-08-18

## Context

The VNAT data can produce many observation windows from one source capture. Randomly distributing
those windows across training and test sets can allow session-specific patterns to appear on both
sides of the evaluation and inflate performance.

## Decision

The primary evaluation will group records by source-capture identity. A capture may belong to only
one of the training, validation, calibration, or test partitions. Automated tests will reject a
manifest containing cross-partition capture overlap.

A randomized-window result may be produced for comparison with prior work, but it must be labeled
separately and cannot replace the capture-held-out result.

## Consequences

- Reported results may be lower than a random split.
- The evaluation better represents generalization to unseen sessions.
- Source-capture identity becomes a required part of the dataset contract.
- If the supplied feature data lacks reliable capture identity, that limitation must be resolved or
  documented before primary metrics are reported.
