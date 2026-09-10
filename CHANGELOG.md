# Changelog

All notable public releases of Parallax are recorded here.

## 0.1.0 - 2026-09-09

First portfolio release.

### Data and reproducibility

- Added a versioned MIT Lincoln Laboratory VNAT release contract with
  fail-closed source checksum verification.
- Implemented deterministic capture-aligned observation-window extraction.
- Reproduced the accepted 129-feature statistical and wavelet representation.
- Added deterministic capture-grouped training, validation, calibration, and
  test partitions.
- Recorded complete artifact lineage and SHA-256 identities.
- Added clean-clone and external-artifact reconstruction documentation.

### Modeling and uncertainty

- Added deterministic reference baselines.
- Added a frozen prototypical traffic-classification model.
- Added calibration-only relative-Mahalanobis OOD scoring.
- Preserved the one-shot capture-held-out final test as immutable historical
  evidence.
- Documented confidence, calibration, OOD, and generalization limitations.

### Runtime

- Added metadata-only classic PCAP ingestion.
- Added deterministic bidirectional flow reconstruction.
- Added shared offline/runtime feature construction.
- Added controlled replay and runtime prediction events.
- Added checksum-bound model and OOD activation.
- Added bounded live-flow state and explicit capacity failures.

### Live sensing and security

- Added Linux AF_PACKET live capture.
- Isolated raw capture in a dedicated `parallax-sensor` systemd service.
- Restricted capture privilege to CAP_NET_RAW.
- Added bounded metadata-only AF_UNIX IPC between the sensor and operator.
- Kept FastAPI, inference, SQLite, replay, and the browser-facing API
  unprivileged.
- Added structured sensor, protocol, transport, runtime, and capacity failures.
- Added a documented threat model and privacy review.

### Operator experience

- Added FastAPI REST controls and ordered SSE prediction delivery.
- Added the Angular Monitor, Replay Lab, History, and prediction-investigation
  experiences.
- Added restart-safe SQLite replay and live-session history.
- Added active-session browser recovery.
- Added real portfolio screenshots and a reproducible demonstration runbook.

### Validation

- Demonstrated exact selected SSH and VoIP offline/runtime feature parity with
  maximum absolute difference `0.0`.
- Added deterministic synthetic steady-state, stale-churn, and capacity soak
  testing.
- Completed a three-minute privilege-separated live workload acceptance run.
- Passed 1,120 Python tests with 100% statement and branch coverage.
- Passed 79 Angular tests and the Angular production build.
- Added full-stack GitHub Actions CI.

### Important limitations

- The accepted VNAT test partition contains no true OOD examples.
- OOD detection power, OOD recall, and OOD AUROC are therefore not claimed.
- Raw confidence is not a calibrated probability of factual correctness.
- VNAT `C2` represents benign SSH/RDP traffic and is not malicious
  command-and-control ground truth.
- Live traffic is operational evidence, not new accuracy evidence.
- Parallax is not a production IDS, malware detector, enforcement system, or
  universal Internet application classifier.
