# Live Sensor Validation

## Status

Parallax completed its initial live-interface validation, operator live-mode
integration, browser end-to-end validation, restart-safe active-session
recovery, least-privilege packet-capture isolation, durable live history,
structured failure hardening, capacity behavior, and sustained-load validation
on 9 September 2026 using JPCMAIN under WSL 2 Ubuntu 24.04.

The live system now operates with a dedicated packet-metadata sensor process.
The FastAPI/operator process does not require raw-socket privileges.

These validations establish runtime plumbing, operational integration,
resource controls, and privilege separation. They are not an accuracy
benchmark, an OOD-performance experiment, or evidence that the accepted VNAT
model generalizes to arbitrary contemporary network traffic.

## Final live architecture

The validated production boundary is:

```text
UNPRIVILEGED OPERATOR PROCESS

Angular dashboard
-> FastAPI operator API
-> OperatorLiveRuntimeExecutor
-> SensorIpcPacketSource
-> AF_UNIX /run/parallax/sensor.sock

PRIVILEGED SENSOR PROCESS

UnixSensorServer
-> SensorIpcSession
-> LivePacketSource
-> LiveEthernetCapture
-> Linux AF_PACKET
-> Ethernet / IPv4 decoding
-> PacketMetadata
-> metadata-only AF_UNIX IPC

UNPRIVILEGED OPERATOR PROCESS

PacketMetadata
-> incremental bidirectional flow tracking
-> incremental observation windows
-> shared 129-feature calculation
-> accepted frozen classifier
-> accepted OOD calibration
-> RuntimePredictionEvent
-> SSE
-> Angular operations dashboard
```

Raw Ethernet frames remain inside the sensor process. Packet payload bytes do
not cross the IPC boundary.

## Initial live capture validation

The first controlled capture path was:

```text
JPCMAIN network traffic
-> WSL eth0
-> Linux AF_PACKET capture
-> Ethernet frame adaptation
-> shared Raw-IPv4 decoder
-> PacketMetadata
```

The observed WSL interface was:

```text
eth0  172.27.223.59/20
```

Controlled validation traffic included bidirectional UDP and ICMP traffic.

A bounded ICMP capture demonstrated:

```text
Raw Ethernet frames observed:   12
Supported IPv4 packets decoded: 12
```

The capture socket closed cleanly.

## Frozen artifact verification

Before live scoring, the runtime successfully verified the complete accepted
artifact chain:

| Artifact | SHA-256 |
| --- | --- |
| Feature artifact | `611dcb63c66e04f461fb7500f96762c1722a06fbdde700dc5aa42ae021e39f16` |
| Capture split manifest | `a1aeee5f118c1e3bd57aa7232eb009634c098b9025571786614798953ebd8a0f` |
| Prototype model bundle | `1c61678611043a7f70f02836ae23bbc7c1abf683b015e23ebafed4d941ecdef7` |
| OOD calibration artifact | `af1d066ea2d96943c873b75e897ca4c7d910c79813a61b605eec8c7ec3c1fd9d` |

No model, calibration, threshold, feature, or split artifact was modified for
live traffic.

## First frozen-model live prediction

The first controlled frozen-runtime live run used:

```text
Interface:       eth0
Packet limit:    60
Traffic:         controlled bidirectional ICMP
Packets decoded: 60
Events emitted:  1
```

The runtime emitted:

```text
prediction:   VOIP
confidence:   1.000000
relative MD:  -325.404388
OOD score:    0.513798
```

All 60 observed Ethernet frames in this bounded run decoded as supported IPv4
traffic. No non-IPv4 or invalid frames were counted.

## Live runtime observability

A subsequent 120-packet controlled ICMP run measured the live runtime rather
than merely proving functionality:

```text
Packets processed:                  120
Prediction events:                    1
Elapsed wall time:              1.222234 s
Observed packet rate:           98.18 packets/s
Observed event rate:             0.8182 events/s
Mean packet processing latency:  0.012019 ms
Maximum packet latency:           0.044145 ms
Finalization latency:            30.211949 ms
Total pipeline compute:          31.654191 ms
```

At this observed rate the runtime spent only a small fraction of wall time in
packet processing. Final model/OOD evaluation dominated the measured compute
cost. The measurement is an observed workload result, not a maximum-throughput
claim.

## Bounded runtime state

Live runtime state was separated from the deterministic offline batch flow
tracker.

The live tracker supports:

- stale-flow eviction;
- an explicit maximum tracked-flow count;
- peak tracked-flow measurement;
- capacity-rejection accounting;
- explicit coordination between flow expiry and observation-window lifetime.

A sustained synthetic churn test exercised 4,096 distinct flows over 32
generations with a configured maximum of 128 simultaneously tracked flows.

The final state remained bounded at 128 flows, with stale flows evicted as
expected and no arbitrary active-flow eviction.

## Operator live-mode integration

The operator layer now has a lifecycle independent from replay:

```text
STARTING
-> RUNNING
-> STOPPING
-> COMPLETED

or

STARTING / RUNNING / STOPPING
-> FAILED
```

The operator service provides:

- interface discovery;
- live-session start;
- live-session inspection;
- explicit stop;
- bounded retained live prediction events;
- ordered event sequence numbers;
- REST snapshots;
- SSE delivery;
- cursor validation;
- active-session discovery.

The browser dashboard supports live-mode operation separately from Replay Lab
and History.

## Browser end-to-end validation

A real browser acceptance test established:

```text
Browser
-> Angular
-> FastAPI
-> operator live session
-> live runtime executor
-> packet source
-> Linux live capture
-> PacketMetadata
-> flow/window/features
-> accepted frozen model + OOD calibration
-> RuntimePredictionEvent
-> SSE
-> browser prediction feed
```

The first version of this validation temporarily ran the API process with
elevated privilege because raw AF_PACKET capture still lived in the operator
process at that point.

That architecture was subsequently replaced by the dedicated sensor service
described below.

## Active-session recovery

Live browser recovery was validated across a page refresh while capture was
still running.

The browser:

1. discovered the active live session;
2. restored retained prediction events;
3. recovered the global event cursor;
4. reconnected SSE after the last known sequence;
5. avoided duplicate predictions;
6. continued receiving new live events;
7. stopped the original live session successfully.

The operator's short startup polling also terminates after the session reaches
RUNNING rather than remaining as unnecessary background polling.

## Least-privilege sensor isolation

The raw capture boundary now runs as the dedicated systemd service:

```text
parallax-sensor.service
```

The deployed service identity is:

```text
user:  parallax-sensor
group: parallax
```

The runtime directory and socket were validated as:

```text
/run/parallax
drwxr-x---  parallax-sensor parallax

/run/parallax/sensor.sock
srw-rw----  parallax-sensor parallax
```

The sensor process received only the required Linux raw-socket capability.

Observed capability state:

```text
CapInh: 0000000000002000
CapPrm: 0000000000002000
CapEff: 0000000000002000
CapBnd: 0000000000002000
CapAmb: 0000000000002000
```

`getpcaps` reported:

```text
cap_net_raw=eip
```

No persistent capability was applied to the Python interpreter.

The FastAPI operator was then launched as the normal development account,
using membership in the `parallax` IPC group rather than sudo. After an older
root-run validation server occupying port 8000 was stopped, the
privilege-separated live path operated successfully.

The resulting boundary is:

```text
operator API:      unprivileged
model/OOD runtime: unprivileged
Angular dashboard: unprivileged

sensor service:    CAP_NET_RAW only
raw AF_PACKET:     sensor process only
```

## Metadata-only IPC contract

The sensor protocol is versioned and uses bounded newline-delimited JSON over
an AF_UNIX stream socket.

The privileged process accepts only a capture-interface start request and
emits:

- a ready response;
- validated PacketMetadata messages;
- structured sensor error messages.

Packet IPC contains:

- timestamp;
- source and destination addresses;
- source and destination ports;
- IP protocol;
- packet size.

It does not contain Ethernet-frame bytes or application payload data.

The client implements bounded stream framing and correctly handles Unix-stream
fragmentation and message coalescing.

## Sustained-load and soak validation

Milestone 6 now includes both deterministic synthetic soak validation and a
real privilege-separated sustained run.

The synthetic harness exercises steady state, high stale-flow churn, and a
hard capacity boundary without treating those mechanics measurements as model
or OOD evidence.

Recorded synthetic results include:

```text
steady
  packets processed:        100,000
  peak tracked flows:         1,024
  observed rate:          41,815.01 packets/s
  mean latency:            0.022606 ms
  RSS growth:                  1,344 KiB

stale-churn
  packets processed:        100,000
  flows created:            100,000
  stale evictions:           99,328
  peak tracked flows:         1,024
  observed rate:          64,102.52 packets/s
  RSS growth:                    768 KiB

capacity
  configured maximum:         4,096 flows
  accepted before failure:    4,096 packets
  capacity rejections:            1
  outcome:        capacity_exceeded
```

The primary real-system run exercised the deployed sensor service, AF_UNIX
metadata IPC, unprivileged operator, frozen scorer, SSE, and SQLite persistence
for approximately three minutes.

```text
generated UDP datagrams:       28,704
generator rate:                 159.38 datagrams/s
configured generator flows:         16
live events before stop:            64
final prediction events:            80
SSE prediction events:              80
SSE terminal events:                 1
persisted prediction events:        80
final state:                 completed
failure:                          none

sensor sampled RSS growth:          0 KiB
operator sampled RSS growth:    6,036 KiB
```

The normal finalization path emitted 16 additional prediction events after the
64 retained during the sustained portion. SSE and durable history both ended
with all 80 events.

The 28,704 value is the number of UDP datagrams generated by the controlled
workload. It is not an assertion that Parallax processed exactly 28,704
packets because other traffic may have been present on the live interface.

After completion, the operator process was restarted against the same SQLite
history database while the sensor process remained running. No live session
was incorrectly recovered as active, and the completed historical session and
all 80 persisted prediction events remained readable.

The restarted operator still had no inherited, permitted, effective, or
ambient Linux capabilities. The sensor retained only CAP_NET_RAW.

Full measurements and reproduction instructions are recorded in
[Performance and Soak Validation](performance-report.md), with a reviewed
machine-readable record in
`evidence/live-soak-2026-09-09.json`.

## Interpretation of live classifications

Live predictions must not be interpreted as ground-truth identification of
the observed traffic.

The classifier is closed-set and must choose among:

- Streaming
- VoIP
- Chat
- C2
- File Transfer

The reported confidence is a raw model softmax output, not a calibrated
probability.

The accepted model is also known to overpredict VoIP on its frozen VNAT test
partition.

The first live ICMP window's OOD score of `0.513798` was below both frozen OOD
thresholds:

```text
0.95
0.99
```

Therefore the accepted OOD calibrator did not flag that particular unseen
traffic as high-OOD.

This result is a limitation of the frozen experiment, not a reason to modify
the accepted artifact.

The accepted VNAT evaluation contained no true OOD examples and therefore did
not establish OOD recall, AUROC, or generalization to arbitrary real-world
traffic.

A separate preregistered OOD/robustness experiment is required before making
claims about unseen applications, protocols, or categories.

## Established evidence

The completed live work establishes that Parallax can:

1. capture live WSL interface traffic;
2. decode supported Ethernet/IPv4 traffic into the same PacketMetadata
   representation used by replay;
3. keep raw frames and payload bytes out of the operator process;
4. process live metadata through the existing flow/window/feature path;
5. score live windows with the checksum-bound accepted classifier and OOD
   calibration;
6. emit normal RuntimePredictionEvent objects;
7. deliver those predictions through the operator API and SSE to Angular;
8. recover an active browser session without duplicating retained events;
9. bound live flow state and measure processing behavior;
10. isolate AF_PACKET access in a dedicated CAP_NET_RAW-only service;
11. preserve completed live sessions and prediction events in restart-safe,
    read-only SQLite history;
12. preserve structured sensor/capture/capacity failures through the operator
    lifecycle and user-visible API state;
13. contain per-session sensor cleanup/transport failures without terminating
    the shared sensor listener;
14. sustain the recorded real and synthetic workloads without observed
    unbounded flow or process-memory growth.

## Remaining Milestone 6 hardening

The packet-to-prediction path, durable live history, structured failure
handling, explicit capacity behavior, sensor-session containment, and
sustained-load validation are complete.

The final operational security/privacy review is complete.

Milestone 6 remains open only for acceptance/documentation closure. The
as-built review is recorded in
[Threat Model and Security Review](threat-model.md).

The accepted VNAT experiment remains frozen throughout this work.
