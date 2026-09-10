# Milestone 6 Acceptance Record

## Status

**Milestone 6 - Live Sensor and Hardening: COMPLETE**

Milestone 6 established and hardened Parallax's live packet-to-prediction
runtime while preserving the accepted scientific boundary of the frozen VNAT
experiment.

The milestone is accepted based on implemented code, automated tests, real
privilege-separated live validation, sustained-load measurements,
restart-persistence validation, and an as-built security/privacy review.

No model, calibration, threshold, feature, or split artifact was modified to
obtain live-runtime results.

## Accepted architecture

```text
local interface
-> dedicated CAP_NET_RAW sensor service
-> Ethernet / IPv4 decoding
-> PacketMetadata
-> metadata-only AF_UNIX IPC
-> unprivileged operator process
-> incremental bidirectional flow tracking
-> incremental observation windows
-> shared 129-feature calculation
-> accepted frozen classifier
-> accepted frozen OOD calibration
-> RuntimePredictionEvent
-> REST / SSE
-> Angular operations console
-> owner-only SQLite operational history
```

Raw Ethernet frames remain within the privileged sensor process.

Packet payload bytes are not passed through sensor IPC, exposed through the
operator prediction contract, or persisted in ordinary operational history.

## Frozen scientific identity

The accepted runtime remained bound to:

| Artifact | SHA-256 |
| --- | --- |
| Prototype model bundle | `1c61678611043a7f70f02836ae23bbc7c1abf683b015e23ebafed4d941ecdef7` |
| OOD calibration artifact | `af1d066ea2d96943c873b75e897ca4c7d910c79813a61b605eec8c7ec3c1fd9d` |
| Feature artifact | `611dcb63c66e04f461fb7500f96762c1722a06fbdde700dc5aa42ae021e39f16` |
| Capture split manifest | `a1aeee5f118c1e3bd57aa7232eb009634c098b9025571786614798953ebd8a0f` |

Live observations did not authorize retraining, recalibration, threshold
selection, feature changes, or post-test model selection.

## Deliverable acceptance

| Milestone 6 deliverable | Result | Primary evidence |
| --- | --- | --- |
| Least-privilege local sensor | Accepted | `docs/sensor-service.md`, `docs/threat-model.md` |
| Live monitoring mode | Accepted | `docs/live-sensor-validation.md` |
| Performance characterization | Accepted | `docs/performance-report.md` |
| Threat/privacy review | Accepted | `docs/threat-model.md` |
| Failure and restart testing | Accepted | automated tests, live validation, performance report |
| Final documentation reconciliation | Accepted | this acceptance record |

## Live capture acceptance

Parallax successfully captured live traffic from the JPCMAIN WSL `eth0`
interface through Linux AF_PACKET.

The privileged capture service performs Ethernet/IPv4 decoding and emits only
validated PacketMetadata through its Unix-domain IPC protocol.

The operator does not directly open AF_PACKET or SOCK_RAW sockets.

The general Python interpreter has no persistent file capability.

## Privilege acceptance

The deployed sensor runs under the dedicated `parallax-sensor` account.

Its capability bounding, inherited, permitted, effective, and ambient sets
contain only the CAP_NET_RAW bit required for packet capture.

The operator was validated with:

```text
CapInh: 0
CapPrm: 0
CapEff: 0
CapAmb: 0
```

The sensor service is restricted to AF_PACKET and AF_UNIX socket families and
uses the documented systemd filesystem and kernel hardening controls.

Access to `/run/parallax/sensor.sock` is restricted to the sensor account and
members of the trusted `parallax` group.

Membership in that group is therefore part of the local capture-authorization
boundary.

## Live runtime acceptance

The live path reuses the existing runtime contracts rather than implementing a
second feature or inference pipeline.

Accepted live behavior includes:

- label-free runtime flow tracking;
- capture-aligned incremental observation windows;
- shared 129-feature construction;
- frozen runtime classification;
- frozen OOD scoring;
- RuntimePredictionEvent generation;
- operator-owned live-session lifecycle;
- REST status and control APIs;
- SSE prediction delivery;
- Angular live monitoring;
- active-session recovery after browser refresh.

The live runtime maintains bounded flow state, stale-flow cleanup, bounded
retained event history, and stable event sequence cursors.

## Structured failure acceptance

Stable live failure handling includes sensor, IPC, capture, configuration,
state, execution, and flow-capacity failures.

The implemented local taxonomy includes:

```text
sensor_unavailable
sensor_timeout
sensor_disconnected
sensor_protocol_error
sensor_ipc_error
sensor_configuration_error
sensor_state_error
capture_error
interface_error
protocol_error
invalid_request
live_execution_error
flow_capacity_exceeded
operator_restart
```

Per-client transport and capture-cleanup failures are contained at the sensor
session boundary so one failed client session does not terminate the shared
long-running sensor listener.

Listener lifecycle and unexpected programming failures are not silently
suppressed.

## Capacity acceptance

The live runtime does not silently evict a valid active flow merely to admit a
new one when configured capacity is reached.

The deterministic capacity profile configured 4,096 maximum active flows and
attempted a 4,097th distinct flow.

Observed behavior was:

```text
accepted flows:       4,096
capacity rejections:      1
terminal outcome: capacity_exceeded
```

The operator maps this condition to the stable
`flow_capacity_exceeded` failure code.

## Synthetic soak acceptance

The deterministic synthetic soak harness exercised runtime mechanics without
performing model scoring.

The accepted profiles were:

```text
steady
stale-churn
capacity
```

The 100,000-packet steady profile retained 1,024 active flows without stale
eviction or capacity rejection.

The 100,000-packet stale-churn profile created 100,000 cumulative flows,
performed 99,328 stale evictions, and retained bounded simultaneous state.

The capacity profile failed explicitly at its configured active-flow limit.

These measurements characterize implementation mechanics only. They are not
classification-accuracy, OOD-generalization, or maximum-throughput evidence.

## Real sustained-load acceptance

The real privilege-separated live acceptance run was:

```text
run_id:
02376e82-bf63-42b9-a456-491c8179ba7d

controlled workload duration:
180.094985 seconds

generated UDP datagrams:
28,704

configured generator flows:
16
```

Observed runtime results were:

```text
live predictions before stop: 64
final predictions:            80
SSE prediction events:        80
SSE terminal events:           1
persisted prediction events:  80
final session state:   completed
structured failure:          none
```

The 28,704 count is the traffic generator's datagram count. It is not asserted
to be the exact number of packets processed by the sensor because unrelated
interface traffic may also have been present.

Observed process-memory behavior during this workload was:

```text
sensor sampled RSS growth:       0 KiB
operator sampled RSS growth: 6,036 KiB
```

These are measurements from the accepted reference workload, not hard resource
ceilings or production capacity guarantees.

Full measurements are recorded in `docs/performance-report.md` and
`docs/evidence/live-soak-2026-09-09.json`.

## Durable history acceptance

Completed live sessions and RuntimePredictionEvent records are persisted in
SQLite.

Historical sessions are read-only through the operator history interface.

The accepted sustained run persisted all 80 final prediction events.

After the operator process was terminated and restarted against the same
database:

```text
active session:                 none
historical state:          completed
historical failure:             none
historical event count:           80
events successfully read:         80
sensor restarted:                 no
```

The restarted operator remained unprivileged and the privileged sensor
continued running with the same process identity.

Nonterminal sessions interrupted by an operator restart are marked failed with
`operator_restart` instead of being silently resumed.

## Persistence privacy acceptance

The security review scanned 192 persisted live prediction events.

Observed findings were:

```text
IPv4-looking persisted strings: 0
MAC-looking persisted strings:  0
```

No reviewed persisted event contained raw:

- source addresses;
- destination addresses;
- source ports;
- destination ports;
- Ethernet frames;
- packet buffers;
- application payload bytes;
- runtime feature vectors.

Persisted prediction events contain only intended prediction/window identity,
classification output, uncertainty output, and artifact provenance.

Flow identifiers are pseudonymous hashes derived from session capture identity
and connection state; the original connection tuple is not persisted in the
prediction event.

## Filesystem privacy acceptance

The final security review identified one implementation defect: operator SQLite
history was originally created as mode 0644.

That defect was corrected before milestone closure.

Current history behavior is:

```text
new database:
0600

existing database opened by Parallax:
tightened to 0600

new parent directory created by SqliteOperatorHistory:
0700

existing caller-owned parent:
unchanged
```

The migration preserves existing database contents.

The default Parallax-owned `data/operator` state directory is owner-only.

## Security review acceptance

The as-built threat and privacy review is recorded in
`docs/threat-model.md`.

It confirms:

- least-privilege raw capture;
- metadata-only IPC;
- unprivileged operator execution;
- owner-only operational history;
- no packet-payload persistence;
- no reviewed raw endpoint persistence;
- structured failure containment;
- bounded flow and event state;
- explicit capacity failure;
- restart-safe history.

The systemd security assessment was interpreted according to the sensor's
actual purpose rather than optimized mechanically. CAP_NET_RAW, AF_PACKET,
AF_UNIX, and host-network visibility are intentional requirements of the local
sensor.

No implementation-blocking security/privacy finding remains for the accepted
local research/demo scope.

## Scientific interpretation boundary

Milestone 6 validates system mechanics and operational behavior.

It does not establish:

- improved VNAT classification accuracy;
- OOD recall;
- OOD AUROC;
- reliable unseen-application detection;
- maliciousness detection;
- arbitrary Internet application classification;
- maximum sustainable network throughput;
- production NDR effectiveness.

The accepted classifier is closed-set over exactly:

```text
STREAMING
VOIP
CHAT
C2
FILE_TRANSFER
```

`C2` in the VNAT label mapping represents benign SSH/RDP traffic and must not
be presented as evidence of malicious command-and-control detection.

Raw softmax confidence remains a closed-set model preference, not a calibrated
probability of factual correctness.

Any new OOD or robustness claim requires a separately designed and
preregistered experiment.

## Accepted residual limitations

Milestone closure intentionally accepts several limitations for the current
local research/demo scope:

- endpoint metadata exists transiently in operator memory;
- the trusted `parallax` group can request local capture;
- the sensor requires CAP_NET_RAW and AF_PACKET;
- the sensor currently serves one capture client at a time;
- operational prediction metadata is retained locally;
- local root/host compromise is outside the Parallax application boundary;
- the current local acceptance does not establish a secure Internet-facing
  deployment.

Remote deployment requires a separate deployment-specific security review,
authentication, TLS, authorization, ingress controls, secrets management, and
retention policy.

## Quality gate

Milestone 6 closure requires the repository-wide authoritative quality gate to
remain green:

- `uv lock --check`;
- Ruff lint;
- Ruff formatting check;
- mypy over `src` and `tests`;
- pytest with 100% Parallax statement/branch coverage;
- Python package build;
- Angular test suite;
- Angular production build;
- soak-script shell syntax;
- acceptance-evidence JSON validation;
- `git diff --check`.

The known FastAPI/Starlette TestClient httpx deprecation warning and the
Angular component-style budget warning are non-blocking known warnings.

## Closure decision

Milestone 6 is complete when this acceptance record and the reconciled project
documentation pass the authoritative repository gate and are committed.

No additional Milestone 6 implementation work is planned.

The next project milestone is:

**Milestone 7 - Portfolio Release**

Milestone 7 will focus on release packaging, public documentation,
architecture/evaluation presentation, demonstration material, and final
release-level reproducibility rather than changing the accepted scientific
result.
