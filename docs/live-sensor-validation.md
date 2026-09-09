# Live Sensor Validation

## Status

Parallax completed its first controlled live-interface and frozen-runtime
validation on 9 September 2026 using JPCMAIN under WSL 2 Ubuntu 24.04.

This validation establishes runtime plumbing and artifact activation. It is not
an accuracy benchmark, an OOD-performance experiment, or evidence that the
accepted VNAT model generalizes to current network traffic.

## Live capture boundary

The validated path was:

```text
JPCMAIN network traffic
-> WSL eth0
-> Linux AF_PACKET capture
-> Ethernet frame adaptation
-> shared Raw-IPv4 decoder
-> PacketMetadata
-> bidirectional flow tracking
-> incremental runtime windowing
-> shared 129-feature calculation
-> accepted frozen classifier
-> accepted OOD calibration
-> RuntimePredictionEvent
```

The live capture path retained packet metadata only. Packet payloads were not
logged or persisted.

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

The first controlled live frozen-runtime run used:

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

## Interpretation

The prediction must not be interpreted as evidence that the ICMP traffic was
actually VoIP.

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

The live ICMP window's OOD score of `0.513798` is below both frozen OOD
thresholds:

```text
0.95
0.99
```

Therefore the accepted OOD calibrator did not flag this particular unseen live
traffic as high-OOD.

This is a useful limitation, not a reason to alter the frozen experiment. The
accepted VNAT evaluation contained no true OOD examples and therefore never
established OOD detection recall, AUROC, or generalization to arbitrary
real-world traffic.

A separate preregistered OOD/robustness experiment is required before making
claims about detection of unseen applications, protocols, or categories.

## Established evidence

This validation establishes that:

1. JPCMAIN live traffic can be captured through the WSL `eth0` boundary.
2. Linux packet capture produces ephemeral Ethernet frames without requiring
   payload persistence.
3. Live Ethernet frames reach the same shared Raw-IPv4 `PacketMetadata`
   representation used by replay.
4. Live packets can enter the existing incremental flow/window pipeline.
5. The existing 129-feature runtime representation can be calculated from live
   traffic.
6. The accepted checksum-bound prototype model and OOD calibration can score
   those live features.
7. A normal `RuntimePredictionEvent` can be produced from real live traffic.
8. Live runtime provenance does not fabricate VNAT application, category, or
   VPN-status labels.
9. The capture source closes cleanly after bounded execution.
10. The first live ICMP result demonstrates that the current OOD score must not
    be assumed to detect arbitrary unseen network traffic.

## Remaining Milestone 6 work

The validated live sensor is not yet the completed operational live mode.

Remaining work includes:

- operator-service live-session lifecycle;
- explicit live start/stop controls;
- SSE delivery of live prediction events;
- durable live-session history;
- clear replay-versus-live dashboard state;
- packet-rate and processing-latency instrumentation;
- active-flow and resource-use measurements;
- explicit bounded-resource policies;
- privileged-sensor isolation;
- structured capture and overload failures;
- operational privacy/security documentation;
- sustained-load validation.

The accepted VNAT experiment remains frozen throughout this work.
