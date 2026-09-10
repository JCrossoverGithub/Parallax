# Parallax Performance and Soak Validation

## Status

Milestone 6 sustained-load validation was completed on 9 September 2026 using
JPCMAIN under Windows 11 with WSL 2 Ubuntu 24.04.

The validation covers two complementary paths:

1. deterministic synthetic runtime/resource stress using the
   `parallax-live-soak` harness; and
2. sustained operation through the real privilege-separated live architecture,
   frozen model and OOD calibration, operator API, SSE delivery, and SQLite
   persistence.

The compact machine-readable acceptance record is stored at
`docs/evidence/live-soak-2026-09-09.json`.

The repeatable real-system procedure is stored at
`scripts/live-soak-acceptance.sh`.

These results characterize implementation behavior on the reference
development system. They are not accuracy evidence, an OOD-performance
experiment, a maximum-throughput benchmark, or a production capacity claim.

## Measurement boundary

The live architecture under test was:

```text
eth0
-> Linux AF_PACKET
-> CAP_NET_RAW-only parallax-sensor service
-> Ethernet / IPv4 decoding
-> PacketMetadata
-> AF_UNIX /run/parallax/sensor.sock
-> unprivileged FastAPI/operator process
-> incremental flows and observation windows
-> shared 129-feature calculation
-> accepted frozen classifier
-> accepted frozen OOD calibration
-> RuntimePredictionEvent
-> SSE
-> SQLite live-session history
```

Raw Ethernet frames and packet payload bytes remain on the privileged sensor
side of the IPC boundary.

The real workload generator count records how many UDP datagrams the generator
sent. It must not be interpreted as an exact count of packets processed by
Parallax because the live interface may contain additional traffic.

## Frozen runtime identity

The real sustained run loaded and verified the accepted artifact chain:

| Artifact | SHA-256 |
| --- | --- |
| Prototype model bundle | `1c61678611043a7f70f02836ae23bbc7c1abf683b015e23ebafed4d941ecdef7` |
| OOD calibration artifact | `af1d066ea2d96943c873b75e897ca4c7d910c79813a61b605eec8c7ec3c1fd9d` |
| Feature artifact | `611dcb63c66e04f461fb7500f96762c1722a06fbdde700dc5aa42ae021e39f16` |
| Capture split manifest | `a1aeee5f118c1e3bd57aa7232eb009634c098b9025571786614798953ebd8a0f` |

No accepted model, calibration, threshold, feature, or split artifact was
changed for performance validation.

## Deterministic synthetic soak harness

Commit `e0c01bb` added the `parallax-live-soak` command and three deterministic
profiles:

- `steady`;
- `stale-churn`;
- `capacity`.

The harness exercises the real runtime flow and observation-window machinery.
For the synthetic mechanics profiles, model scoring is intentionally disabled:
the minimum packet requirement is configured above the generated packet count
and a non-scoring placeholder satisfies the scorer contract.

Synthetic results therefore measure runtime/resource mechanics rather than
classifier or OOD behavior.

### Steady profile

The steady workload processed 100,000 packets across 1,024 flows while keeping
all configured flows active.

| Measurement | Observed value |
| --- | ---: |
| Packets processed | 100,000 |
| Flows created | 1,024 |
| Final tracked flows | 1,024 |
| Peak tracked flows | 1,024 |
| Stale evictions | 0 |
| Capacity rejections | 0 |
| Wall time | 2.391498 s |
| Observed runtime rate | 41,815.01 packets/s |
| Mean processing latency | 0.022606 ms |
| Maximum processing latency | 0.736028 ms |
| Finalization latency | 0.323556 ms |
| Peak RSS growth | 1,344 KiB |

This is an observed synthetic execution rate, not a maximum-throughput claim.

### Stale-churn profile

The stale-churn workload created 100,000 flows over repeated generations while
allowing stale state to expire.

| Measurement | Observed value |
| --- | ---: |
| Packets processed | 100,000 |
| Flows created | 100,000 |
| Final tracked flows | 672 |
| Peak tracked flows | 1,024 |
| Stale evictions | 99,328 |
| Capacity rejections | 0 |
| Wall time | 1.560013 s |
| Observed runtime rate | 64,102.52 packets/s |
| Mean processing latency | 0.014212 ms |
| Maximum processing latency | 67.655142 ms |
| Peak RSS growth | 768 KiB |

The maximum-latency value is an observed outlier from this run and is retained
as evidence rather than generalized into a latency guarantee.

The workload demonstrated that high cumulative flow churn does not imply
unbounded simultaneously retained flow state when stale eviction is active.

### Capacity profile

The capacity profile configured a maximum of 4,096 simultaneously tracked
flows and attempted to introduce a 4,097th distinct active flow.

| Measurement | Observed value |
| --- | ---: |
| Configured maximum tracked flows | 4,096 |
| Source packets emitted | 4,097 |
| Packets accepted before rejection | 4,096 |
| Flows created | 4,096 |
| Final tracked flows | 4,096 |
| Capacity rejections | 1 |
| Stale evictions | 0 |
| Outcome | `capacity_exceeded` |
| Wall time | 0.192145 s |
| Peak RSS growth | 2,688 KiB |

The runtime does not silently evict an otherwise valid active flow to make room
for a new one. At capacity, a genuinely new flow is rejected explicitly. The
operator path translates the corresponding runtime capacity failure into the
stable structured live-session failure code `flow_capacity_exceeded`.

## Real privilege-separated sustained run

The primary real-system acceptance used run:

```text
02376e82-bf63-42b9-a456-491c8179ba7d
```

The operator was launched as the normal development account with access to the
`parallax` Unix-socket group. It held no inherited, permitted, effective, or
ambient Linux capabilities.

The sensor ran as the dedicated `parallax-sensor` account and retained only
`CAP_NET_RAW`.

The controlled generator used 16 persistent UDP flows for approximately three
minutes.

| Measurement | Observed value |
| --- | ---: |
| Elapsed generator time | 180.094985 s |
| Generated UDP datagrams | 28,704 |
| Generator rate | 159.38 datagrams/s |
| Configured generator flows | 16 |
| Session state during workload | `running` |
| Live predictions before stop | 64 |
| Final prediction count | 80 |
| SSE prediction events | 80 |
| SSE terminal events | 1 |
| Persisted prediction events | 80 |
| Final session state | `completed` |
| Structured failure | none |

The normal stop/finalization path emitted 16 additional prediction events after
the 64 events already retained during the sustained workload. SSE delivered
all 80 predictions, durable history contained all 80 predictions, and exactly
one terminal SSE event was observed.

This agreement provides acceptance evidence for final-window flushing,
ordered event delivery, terminal lifecycle handling, and live-history
persistence.

## Process memory

Memory sampling was performed every ten seconds during the sustained real run.

| Process | Baseline RSS | Maximum sampled RSS | Sampled growth |
| --- | ---: | ---: | ---: |
| `parallax-sensor` | 158,272 KiB | 158,272 KiB | 0 KiB |
| FastAPI/operator | 422,992 KiB | 429,028 KiB | 6,036 KiB |

The operator therefore showed approximately 5.89 MiB of sampled RSS growth
during this workload.

These values include the existing Python, scientific-runtime, model, API,
history, and SSE process state. They are measurements from this reference run,
not hard memory ceilings.

The sensor's sampled RSS did not grow during the three-minute workload. A
separate systemd process peak may include import/startup transients and is not
substituted for the run-relative sampled measurement above.

## Privilege isolation during load

During the real soak, the sensor capability boundary remained:

```text
sensor effective capability: CAP_NET_RAW only
sensor ambient capability:   CAP_NET_RAW only
```

The operator remained:

```text
CapInh: 0
CapPrm: 0
CapEff: 0
CapAmb: 0
```

The broader operator capability bounding set did not grant the process an
effective capability.

Raw packet capture therefore remained isolated from FastAPI, model inference,
SQLite persistence, SSE delivery, and Angular-facing APIs throughout the
sustained workload.

## Restart-safe history acceptance

After the completed live run, the operator process was terminated normally and
started again against the same SQLite history database.

The privileged sensor was not restarted.

Acceptance observations were:

| Check | Result |
| --- | --- |
| Operator process identity changed | yes |
| Restarted operator effective capabilities | none |
| Restarted operator ambient capabilities | none |
| Sensor remained active | yes |
| Sensor process identity remained unchanged | yes |
| Active live session after restart | none |
| Historical live-session state | `completed` |
| Historical failure | none |
| Historical event count | 80 |
| Historical events readable after restart | 80 |

This establishes that a completed live session is not incorrectly recovered as
active and that its persisted predictions remain available after an API-process
restart.

## Reproduction

The real sustained acceptance runner is:

```text
scripts/live-soak-acceptance.sh
```

It expects the normal operator API and `parallax-sensor.service` to already be
running. It refuses to start if another live session is active.

Its default workload is:

```text
duration:         180 seconds
generator flows:  16
traffic interval: 0.1 seconds
sample interval:  10 seconds
interface:        eth0
```

For a short mechanics smoke run:

```bash
DURATION_SECONDS=20 scripts/live-soak-acceptance.sh
```

For the recorded three-minute acceptance profile:

```bash
scripts/live-soak-acceptance.sh
```

Each execution preserves JSON, SSE, traffic, memory-sampling, and session
evidence in a timestamped temporary output directory.

Raw operational evidence and SQLite databases remain outside Git. The reviewed
compact result is stored in
`docs/evidence/live-soak-2026-09-09.json`.

## Interpretation and limitations

The sustained-load work establishes bounded runtime behavior and successful
operation of the implemented live path under the recorded reference workloads.

It does not establish:

- model accuracy on the generated UDP traffic;
- OOD recall, AUROC, or unseen-application detection quality;
- a maximum sustainable packet rate;
- production network capacity;
- universal latency or memory guarantees.

The accepted VNAT scientific experiment remains frozen. Any new accuracy,
robustness, or OOD-generalization claim requires a separately designed and
preregistered experiment.
