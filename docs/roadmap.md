# Parallax Roadmap

## Current status

Milestones 1 through 5 are complete. Milestone 6 has completed its operational
security/privacy review and is awaiting final acceptance closure.

The complete live packet-to-browser path has been validated on JPCMAIN. Live
capture now runs behind a dedicated AF_UNIX metadata boundary in a separate
systemd service with CAP_NET_RAW, while the FastAPI/operator process remains
unprivileged.

| Milestone | Status | Outcome |
| --- | --- | --- |
| 1. Data foundation | Complete | Trusted VNAT contracts, checksums, inspection, windows, features, and capture-grouped partitions |
| 2. Modeling and uncertainty | Complete | Baselines, frozen prototype candidate, calibration artifact, and one accepted final test evaluation |
| 3. Raw-PCAP parity | Complete | Metadata-only parsing, bidirectional flows, PCAP windows, and exact selected-capture feature parity |
| 4. Replayable runtime | Complete | Controlled replay through incremental features, frozen inference, and runtime prediction events |
| 5. Operator layer | Complete | REST controls, SSE prediction streaming, Angular dashboard, durable replay history, and restart-safe history inspection |
| 6. Live sensor and hardening | Complete | Live capture, durable history, structured failures, overload behavior, least-privilege isolation, sustained-load validation, security/privacy review, and acceptance closure complete |

## Milestone 4 - Replayable runtime

Milestone 4 established the deterministic runtime that Milestone 5 now hosts.

Implemented behavior includes:

- Replay lifecycle states: created, running, paused, completed, failed, and cancelled.
- Configured replay speed and maximum-speed execution.
- Interruptible pause, resume, and cancellation.
- Lazy PCAP replay entries containing metadata rather than packet payloads.
- Incremental bidirectional flow tracking.
- Incremental observation-window construction.
- Shared offline/runtime 129-feature calculation.
- Checksum- and provenance-bound loading of the accepted frozen prototype model.
- Checksum- and provenance-bound loading of the accepted OOD calibration artifact.
- Runtime classification and OOD scoring.
- Stable RuntimePredictionEvent serialization.
- Successful-completion-only final-window flushing.

Selected real-data parity evidence:

| Capture | Parsed packets | Eligible windows | Incremental/batch parity | Maximum feature difference |
| --- | ---: | ---: | --- | ---: |
| `nonvpn_ssh_capture4.pcap` | 626 | 5 | 5 / 5 exact | 0.0 |
| `nonvpn_voip_capture2.pcap` | 119,103 | 45 | 45 / 45 exact | 0.0 |

The accepted model and OOD calibration completed a mechanical runtime smoke test over all 50
incrementally generated windows. This did not perform model selection, threshold tuning, or a new
accepted-test evaluation.

## Milestone 5 - Operator layer

### Status

Complete.

### Implemented architecture

The operator path is now:

VNAT PCAP
-> controlled packet replay
-> incremental flow tracking
-> incremental observation windows
-> shared 129-feature calculation
-> accepted frozen classifier and OOD calibration
-> ordered RuntimePredictionEvent
-> server-sent event stream
-> Angular operator dashboard

Operational state also follows:

Replay/session state and prediction events
-> local SQLite history
-> process restart boundary
-> history REST API
-> Angular Replay History view

### Runtime service and API

The operator service:

- Loads the accepted model and calibration artifacts once at application startup.
- Verifies their expected checksums and provenance.
- Restricts replay selection to capture filenames under the configured capture root.
- Hosts replay execution in a long-running process.
- Reports service health and active model identity.
- Starts and inspects replay sessions.
- Supports pause, resume, and cancel controls.
- Maintains bounded in-memory live-event history.
- Preserves stable event sequence numbers.
- Rejects expired or invalid event cursors.
- Exposes ordered runtime prediction events using SSE.
- Supports browser reconnection through SSE event identifiers and Last-Event-ID semantics.
- Does not expose packet payloads or model files to browser clients.

SSE was selected instead of WebSockets because the prediction transport is server-to-browser while
replay controls remain ordinary REST operations.

### Angular operator dashboard

The dashboard provides:

- Capture selection.
- Configured-speed and maximum-speed replay.
- Start, pause, resume, and cancel controls.
- Running, paused, completed, failed, and cancelled state display.
- Live prediction-event count.
- Current predicted category.
- Raw predictive confidence.
- OOD score displayed independently from predictive confidence.
- Per-category probability bars.
- Time-ordered prediction history.
- Packet count and pseudonymous flow identity per prediction window.
- Service-health display.
- Capture, model, and calibration provenance.
- Persisted Replay History.
- Read-only reopening of completed historical sessions.

### Durable history

Milestone 5 uses a local SQLite database for operator replay history.

Persisted information includes:

- Replay identity.
- Source capture identity and checksum.
- Replay state.
- Replay timing configuration.
- Prediction-event count.
- Structured failure information.
- Ordered runtime prediction events.
- Per-prediction model, calibration, feature-artifact, and split-manifest provenance.

Packet payloads are not persisted.

Historical sessions are intentionally read-only. An old session cannot issue pause, resume, or
cancel commands against an unrelated or nonexistent in-memory replay.

### Acceptance evidence

The operator path was exercised using both selected real VNAT captures.

SSH operator demonstration:

- `nonvpn_ssh_capture4.pcap`
- 626 parsed packets.
- 5 eligible runtime prediction windows.
- 5 operator-visible prediction events.
- Successful terminal completion.

VoIP operator demonstration:

- `nonvpn_voip_capture2.pcap`
- 119,103 parsed packets.
- 45 eligible runtime prediction windows.
- 45 operator-visible prediction events.
- Successful terminal completion.

The VoIP session was also used to verify durable history:

1. A new replay completed with 45 prediction events.
2. The replay and events were written to SQLite.
3. The operator API process was terminated.
4. A new API process was started.
5. The prior session remained available through the history API.
6. The Angular dashboard reopened the historical session.
7. Its stored prediction events and historical prediction provenance were restored.

This persistence test crosses a real process-restart boundary rather than merely constructing a
second service object in the same process.

### Quality boundary

Milestone 5 is covered by:

- Strict Python typing with mypy.
- Ruff formatting and linting.
- 100% statement and branch coverage for the Python operator package.
- Angular component and API-client behavior tests.
- Angular production builds.
- Repository-wide locked dependency, lint, type, test, coverage, and build gates.

Processing-latency instrumentation, bounded-resource measurements,
live-sensor observability, and explicit capacity/failure behavior were completed
as Milestone 6 work rather than retrofitted into Milestone 5.

## Milestone 6 - Live sensor and hardening

### Goal

Replace the prerecorded PCAP source with live packet metadata while retaining
the verified flow, window, feature, inference, event, and operator paths, then
harden the live runtime for bounded and least-privilege operation.

### Implemented architecture

The current live path is:

```text
local interface
-> dedicated CAP_NET_RAW sensor service
-> Ethernet / IPv4 decoding
-> PacketMetadata
-> metadata-only AF_UNIX IPC
-> unprivileged operator process
-> incremental flow tracking
-> incremental observation windows
-> shared 129-feature calculation
-> accepted frozen classifier + OOD calibration
-> RuntimePredictionEvent
-> SSE
-> Angular operations console
```

Raw frames remain on the sensor side of the privilege boundary.

### Completed work

The following Milestone 6 capabilities are implemented and validated:

- live interface discovery and selection;
- shared Ethernet/IPv4 packet decoding;
- Linux AF_PACKET capture with bounded polling and clean lifecycle;
- metadata-only live packet sources;
- label-free runtime observation windows and features;
- live execution through the existing frozen prediction pipeline;
- packet-rate and processing-latency instrumentation;
- bounded live flow tracking with stale eviction and capacity accounting;
- coordinated flow/window expiration invariants;
- operator-owned live-session lifecycle;
- explicit live start, status, stop, event snapshot, and SSE APIs;
- bounded retained live event sequences and cursor validation;
- Angular Live Sensor workspace;
- real browser-to-prediction end-to-end validation;
- active live-session discovery and page-refresh recovery;
- versioned bounded metadata-only sensor IPC;
- dedicated Unix-domain sensor server;
- unprivileged IPC RuntimePacketSource client;
- operator switchover away from direct raw capture;
- dedicated `parallax-sensor` executable;
- hardened systemd service identity and runtime socket permissions;
- CAP_NET_RAW-only sensor execution;
- successful live operation with an unprivileged FastAPI process;
- durable SQLite persistence and read-only inspection of completed live sessions;
- restart-safe live-history recovery without reviving completed sessions as active;
- structured sensor, IPC, capture, and execution failure propagation;
- explicit terminal `flow_capacity_exceeded` behavior rather than silent active-flow eviction;
- per-session containment of capture-cleanup and client transport failures so one bad sensor
  session does not terminate the shared listener;
- deterministic synthetic steady, stale-churn, and capacity soak profiles;
- three-minute real privilege-separated live soak with SSE, persistence, process-memory sampling,
  clean finalization, and operator-restart history validation.

### Acceptance closure

Milestone 6 is complete.

Durable live history, structured sensor/capture failures, explicit capacity
behavior, sensor-session containment, sustained-load validation, and the final
security/privacy review are complete.

The security review reconfirmed the metadata-only boundary, privilege
isolation, failure containment, persistence exclusions, and absence of packet
payloads from ordinary logs and operational history. It also identified and
corrected world-readable SQLite history permissions; operator history is now
owner-only.

The final acceptance record is
[Milestone 6 Acceptance](milestone-6-acceptance.md).

Live capture does not change the accepted VNAT experiment, authorize model
retuning, or establish accuracy/OOD performance on arbitrary real-world
traffic.

Detailed load and memory evidence is recorded in
[Performance and Soak Validation](performance-report.md).

## Experimental integrity

The following boundaries remain unchanged:

- The accepted final VNAT test evaluation is not rerun for tuning or inspection.
- The accepted model and OOD calibration artifacts are not modified to accommodate runtime data.
- Validation remains candidate-selection data.
- Calibration remains OOD-density fitting data.
- Accepted test data remains a one-time frozen final evaluation.
- Raw class probabilities are not described as calibrated probabilities.
- Accepted test results do not establish true OOD detection performance because that test contained
  no true OOD examples.
- VNAT C2 is benign SSH/RDP traffic and must not be described as malicious command-and-control.
- Raw captures, generated datasets, models, and operational databases remain outside Git.
- Packet payloads are not logged or persisted.
