# Parallax Roadmap

## Current status

Milestones 1 through 4 are complete. Milestone 5 is next.

| Milestone | Status | Outcome |
| --- | --- | --- |
| 1. Data foundation | Complete | Trusted VNAT contracts, checksums, inspection, windows, features, and capture-grouped partitions |
| 2. Modeling and uncertainty | Complete | Baselines, frozen prototype candidate, calibration artifact, and one accepted final test evaluation |
| 3. Raw-PCAP parity | Complete | Metadata-only parsing, bidirectional flows, PCAP windows, and exact selected-capture feature parity |
| 4. Replayable runtime | Complete | Controlled replay through incremental features, frozen inference, and runtime prediction events |
| 5. Operator layer | Next | API/event service, replay controls, persistence, health, and operations dashboard |
| 6. Live sensor and hardening | Later | Least-privilege live capture, bounded-resource validation, observability, and operational hardening |

## Milestone 4 completion evidence

The replayable runtime is considered functionally complete because the repository now provides:

- Replay session lifecycle contracts with created, running, paused, completed, failed, and cancelled states.
- Configurable replay timing plus maximum-speed execution.
- Interruptible pause, resume, and cancellation behavior.
- Lazy PCAP replay entries carrying parsed packet metadata.
- Incremental bidirectional flow tracking.
- Incremental observation-window construction.
- Shared offline/runtime 129-feature calculation.
- Checksum- and provenance-bound loading of the accepted frozen prototype model.
- Checksum- and provenance-bound loading of the accepted OOD calibration artifact.
- Runtime classification and OOD scoring.
- Stable `RuntimePredictionEvent` serialization.
- Packet-to-prediction orchestration.
- Controlled replay integration that flushes final windows only after successful completion.

Selected real-data acceptance evidence:

| Capture | Parsed packets | Eligible windows | Incremental/batch parity | Maximum feature difference |
| --- | ---: | ---: | --- | ---: |
| `nonvpn_ssh_capture4.pcap` | 626 | 5 | 5 / 5 exact | 0.0 |
| `nonvpn_voip_capture2.pcap` | 119,103 | 45 | 45 / 45 exact | 0.0 |

The accepted model and OOD calibration also completed a runtime smoke test over all 50 of those
incrementally generated windows. That smoke test verified mechanical inference, finite normalized
class probabilities, bounded OOD scores, and exact artifact provenance. It intentionally did not
perform model selection, threshold tuning, or a new accepted-test evaluation.

At the Milestone 4 repository gate:

- 729 tests passed.
- Statement coverage was 100%.
- Branch coverage was 100%.
- Ruff passed.
- mypy passed.
- The locked dependency graph passed.
- Source distribution and wheel builds succeeded.

## Milestone 5 - Operator layer

### Goal

Turn the verified runtime into a usable operator-facing application without changing the accepted
model or feature semantics.

### First vertical slice

The first Milestone 5 slice is deliberately narrow:

> Start one PCAP replay from the operator interface and watch real
> `RuntimePredictionEvent` records appear as the replay progresses.

This slice should prove the boundary from the existing runtime into the service and browser before
additional dashboard functionality is added.

### Planned sequence

1. **Runtime service boundary**
   - Host replay sessions in a long-running application process.
   - Load the accepted model/calibration once at startup.
   - Expose health and active-model identity.
   - Create and control replay sessions.

2. **Versioned API and event transport**
   - Start, inspect, pause, resume, and cancel a replay.
   - Stream ordered runtime prediction and session-state events.
   - Provide a snapshot/reconnect mechanism so clients can recover missed state.
   - Keep browser clients isolated from raw model files and internal control objects.

3. **Initial dashboard**
   - Replay selection and start controls.
   - Running/paused/completed/failed/cancelled session state.
   - Time-ordered prediction feed.
   - Category probabilities.
   - Raw predictive confidence.
   - OOD score shown independently from confidence.
   - Capture, window, model, calibration, and feature provenance.

4. **Operational persistence and history**
   - Persist replay sessions, predictions, failures, and active artifact identity.
   - Support session history and replay-result inspection.
   - Preserve the no-payload-retention boundary.

5. **Observability and operator polish**
   - Processing counters and latency measurements.
   - Explicit invalid/dropped-window reporting.
   - Service-health display.
   - Clear research/demo limitations in the UI.

### Milestone 5 completion boundary

Milestone 5 is complete when an operator can start and control a replay through the UI, observe
ordered predictions and uncertainty in real time, reconnect without losing session state, inspect
completed session history, and verify the active artifact provenance without accessing packet
payloads.

## Milestone 6 - Live sensor and hardening

Milestone 6 introduces live local-interface capture only after the replay-backed operator path is
stable. It will focus on least privilege, bounded active-flow/resource behavior, measured runtime
performance, failure isolation, and clear separation between replayed and live traffic.

Live capture does not change the accepted experimental evidence or authorize stronger claims about
model accuracy or OOD detection.
