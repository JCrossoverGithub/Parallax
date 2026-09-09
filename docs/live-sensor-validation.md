# Live Sensor Validation

## Status

Parallax completed its initial live-interface validation, operator live-mode
integration, browser end-to-end validation, restart-safe active-session
recovery, and least-privilege packet-capture isolation on 9 September 2026
using JPCMAIN under WSL 2 Ubuntu 24.04.

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
10. isolate AF_PACKET access in a dedicated CAP_NET_RAW-only service.

## Remaining Milestone 6 hardening

The core live architecture is complete. Remaining Milestone 6 work is focused
on operational hardening rather than establishing the packet-to-prediction
path.

Remaining work includes:

- durable persistence and read-only inspection of completed live sessions;
- preserving structured sensor/capture failures through the operator session
  and API instead of collapsing them into a generic execution failure;
- explicit operator-visible overload/capacity failure behavior;
- sustained-load and soak validation including packet rate, processing
  latency, tracked-flow state, capacity behavior, and process memory;
- final operational security/privacy review and milestone acceptance
  documentation.

The accepted VNAT experiment remains frozen throughout this work.
