# ADR-0001: Use Python for the core data and ML pipeline

- Status: Accepted
- Date: 2026-08-18

## Context

Parallax needs packet parsing, HDF5 access, statistical processing, wavelet features, conventional
machine learning, neural modeling, calibration, and an inference API. Keeping these capabilities in
one language reduces the risk of training-serving feature differences during the first release.

## Decision

Use Python 3.12 for dataset validation, flow processing, feature construction, model training,
evaluation, replay, and inference. Use a packaged `src/` layout managed by `uv`.

## Consequences

- Training and runtime code can share one versioned feature package.
- The project can use the scientific Python and PyTorch ecosystems directly.
- CPU-only CI can exercise the core pipeline.
- A higher-performance sensor or data plane may be introduced later only if measurement shows that
  Python cannot meet a defined requirement.

## Alternatives considered

- Node.js would align with other projects but provides a weaker scientific and ML ecosystem.
- Rust could provide stronger throughput and memory guarantees but would increase initial
  implementation and integration cost before a bottleneck is demonstrated.
