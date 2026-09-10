# Parallax Threat Model and Security Review

## Status

This document records the as-built Milestone 6 security and privacy review for
Parallax.

The review covers:

- live packet capture;
- the privileged sensor service;
- AF_UNIX sensor IPC;
- the unprivileged operator process;
- runtime flow/window state;
- prediction events;
- SSE and REST exposure;
- SQLite operational history;
- failure containment;
- local filesystem permissions.

The review was performed against the implemented live architecture on JPCMAIN
under Windows 11 with WSL 2 Ubuntu 24.04.

Parallax remains a research and demonstration system. This review does not
claim that it is hardened for arbitrary Internet-facing production deployment.

## Security objectives

The principal security and privacy objectives are:

1. Packet payload bytes must not cross from privileged capture into the
   operator process.
2. Raw Ethernet frames must remain inside the privileged sensor process.
3. The FastAPI/operator process must not require raw-socket privileges.
4. Persistent history must not contain raw packet payloads, raw endpoint
   addresses, raw ports, or feature vectors.
5. Operational history must not be world-readable.
6. A malformed or disconnected sensor client must not terminate the shared
   sensor service.
7. Live flow state and event retention must remain bounded.
8. The browser must receive only intended operator-facing prediction data.
9. Accepted model, feature, calibration, and split artifacts must remain
   checksum-bound and unchanged by live traffic.

## Assets

Assets protected by this design include:

- packet payload contents;
- packet endpoint metadata;
- capture-interface access;
- operational prediction history;
- model and calibration artifact integrity;
- service availability;
- local operator state;
- prediction provenance.

Packet metadata remains potentially sensitive even when payloads are absent.
Timing, endpoint tuples, packet sizes, and traffic patterns may reveal user or
application behavior.

## Trust boundaries

### Boundary 1 - network interface to sensor

```text
Local network interface
-> Linux AF_PACKET
-> privileged parallax-sensor process
```

The sensor must possess CAP_NET_RAW to open its AF_PACKET capture socket.

The sensor process is the only production component that receives raw Ethernet
frames.

### Boundary 2 - sensor to operator

```text
PRIVILEGED

AF_PACKET
-> Ethernet / IPv4 decoding
-> PacketMetadata
-> AF_UNIX /run/parallax/sensor.sock

UNPRIVILEGED

SensorIpcPacketSource
-> runtime flow/window pipeline
-> model + OOD scoring
```

Only validated PacketMetadata crosses this boundary.

The IPC packet contract contains:

- timestamp;
- source address;
- source port;
- destination address;
- destination port;
- IP protocol;
- packet size.

It does not contain:

- raw Ethernet frames;
- raw IP packet buffers;
- transport payload bytes;
- application payload bytes.

IPC messages use a versioned, exact-field, newline-delimited JSON contract and
are bounded to 4,096 bytes.

### Boundary 3 - operator to browser

```text
runtime prediction
-> operator service
-> REST / SSE
-> Angular dashboard
```

The browser receives RuntimePredictionEvent objects rather than PacketMetadata
or raw flow records.

The prediction event contains:

- run identity;
- pseudonymous flow/window identity;
- window offsets and packet count;
- closed-set classification output;
- raw model confidence;
- relative-Mahalanobis distance;
- OOD score;
- model, calibration, feature, and split provenance.

Raw source/destination addresses and ports are not part of the prediction
event contract.

### Boundary 4 - operator to persistent history

```text
OperatorReplayService
-> RuntimePredictionEvent.as_dict()
-> SQLite
```

Live-session history retains operational configuration, state, structured
failure information, event counts, and serialized prediction events.

Raw PacketMetadata and raw runtime feature vectors are not persisted.

## Privileged sensor design

The deployed sensor runs as:

```text
user:  parallax-sensor
group: parallax
```

The service receives only:

```text
CAP_NET_RAW
```

The observed process capability state was:

```text
CapInh: 0000000000002000
CapPrm: 0000000000002000
CapEff: 0000000000002000
CapBnd: 0000000000002000
CapAmb: 0000000000002000
```

The Python interpreter itself has no persistent file capabilities.

The service is restricted to:

```text
AF_UNIX
AF_PACKET
```

It cannot create ordinary AF_INET or AF_INET6 sockets under the deployed
systemd policy.

The service also uses:

- `NoNewPrivileges=yes`;
- `ProtectSystem=strict`;
- `ProtectHome=yes`;
- `PrivateTmp=yes`;
- `ProtectKernelTunables=yes`;
- `ProtectKernelModules=yes`;
- `ProtectKernelLogs=yes`;
- `ProtectControlGroups=yes`;
- `RestrictSUIDSGID=yes`;
- `LockPersonality=yes`;
- `ReadWritePaths=/run/parallax`.

The sensor does not own:

- FastAPI;
- model inference;
- OOD calibration;
- SQLite history;
- replay execution;
- Angular;
- HTTP listeners.

## Sensor socket authorization

The sensor socket is deployed as:

```text
/run/parallax
drwxr-x---  parallax-sensor parallax

/run/parallax/sensor.sock
srw-rw----  parallax-sensor parallax
```

Membership in the local `parallax` group therefore grants access to the sensor
IPC endpoint.

The `parallax` group must be treated as a trusted capture-authorization group,
not merely as a convenience group.

An account added to this group can request local interface capture through the
sensor protocol.

## Sensor failure containment

The sensor listener serves one IPC capture session at a time.

Per-session failures are contained at the client-session boundary.

The implementation specifically contains:

- client transport write failures;
- capture cleanup failures;
- malformed client requests;
- client disconnects;
- interface-resolution failures;
- supported structured capture failures.

A capture-session cleanup failure does not terminate the long-running shared
sensor listener.

Listener bind, accept, and listener-lifecycle failures remain service-level
errors rather than being silently suppressed.

Unexpected programming exceptions are not intentionally hidden.

## Operator privilege boundary

The FastAPI/operator process was validated while running as the normal
development user.

Observed held capabilities were:

```text
CapInh: 0
CapPrm: 0
CapEff: 0
CapAmb: 0
```

The operator imports SensorIpcPacketSource for live capture and does not open
AF_PACKET or SOCK_RAW sockets itself.

A broader Linux capability bounding set on the ordinary shell process does not
mean those capabilities are held. The measured effective and ambient sets
were empty.

## Persistence privacy review

The implemented SQLite schema contains:

```text
replay_sessions
prediction_events
live_sessions
live_prediction_events
```

Live prediction events are persisted as serialized RuntimePredictionEvent
objects.

A review of 192 persisted live prediction events found:

```text
IPv4-looking stored string values: 0
MAC-looking stored string values:  0
```

No persisted event field contained:

- `source_address`;
- `destination_address`;
- `source_port`;
- `destination_port`;
- raw frame data;
- packet byte buffers;
- payload bytes;
- raw feature vectors.

The persisted event shape contained only:

```text
classification
provenance
run_id
schema_version
uncertainty
window
```

The nested `window` object contained:

```text
capture_id
end_offset_seconds
flow_id
packet_count
start_offset_seconds
window_id
window_index
```

For live runs, `flow_id` is derived as a truncated SHA-256 value from the
session capture identity and connection tuple. The raw connection tuple itself
is not persisted in prediction history.

## History filesystem permissions

The security review identified one privacy defect: SQLite operator-history
files were originally created with mode 0644 and were therefore readable by
other local users.

The history implementation was hardened so that:

```text
new database file:
0600

existing database reopened by Parallax:
tightened to 0600

new parent directory created by SqliteOperatorHistory:
0700

existing caller-owned parent directory:
permissions unchanged
```

This avoids changing arbitrary shared directories such as `/tmp`.

The default Parallax-owned history directory was also tightened to owner-only
access.

The permission change preserves existing SQLite history rather than replacing
the database.

## Runtime memory-only sensitive state

Live processing necessarily holds packet-derived metadata temporarily in
memory while constructing bidirectional flows and observation windows.

Transient runtime structures can contain:

- source and destination addresses;
- ports;
- packet timestamps;
- packet sizes;
- directions;
- connection tuples;
- calculated feature vectors.

These values are required for feature calculation but are not part of the
persistent prediction-event contract.

This distinction is important:

```text
transient processing of metadata != persistent retention of metadata
```

Parallax does not claim that endpoint metadata never enters operator memory.
It claims that raw packet frames and payload bytes do not enter the operator
and that raw endpoint tuples and feature vectors are not written to ordinary
operational history.

## Browser exposure

The operator-facing event contract deliberately excludes raw endpoint tuples.

The dashboard receives:

- pseudonymous flow identity;
- window timing;
- packet count;
- model classification;
- raw confidence;
- OOD score;
- artifact provenance.

The browser does not receive PacketMetadata objects through the implemented
prediction API.

## Logging review

The sensor, live operator, and runtime paths do not contain ordinary logging
or print statements that dump PacketMetadata, raw frames, packet buffers, or
feature vectors.

The synthetic soak CLI prints its explicit report structure only.

Failure messages may contain operational error descriptions such as interface
or socket failures. They must not be extended to include packet buffers,
authentication secrets, or raw payload contents.

## Resource and denial-of-service controls

The live runtime uses bounded flow state.

Configured behavior includes:

- stale-flow eviction;
- maximum tracked-flow capacity;
- explicit capacity rejection;
- bounded retained event history;
- bounded sensor IPC messages;
- finite sensor poll timeouts.

When a new flow would exceed the configured active-flow capacity, Parallax
fails explicitly with:

```text
flow_capacity_exceeded
```

It does not silently evict an otherwise valid active flow merely to admit a
new one.

Synthetic capacity validation demonstrated rejection of the 4,097th active
flow when the configured maximum was 4,096.

## Sustained-load evidence

The real three-minute Milestone 6 soak exercised:

```text
AF_PACKET
-> privileged sensor
-> metadata IPC
-> unprivileged operator
-> runtime inference
-> SSE
-> SQLite persistence
```

Observed results included:

```text
generated UDP datagrams:     28,704
configured generator flows:      16
final predictions:                80
SSE prediction events:            80
persisted predictions:            80
final state:               completed
structured failure:             none
```

The sensor showed zero sampled RSS growth during the workload.

The operator showed approximately 5.89 MiB sampled RSS growth.

These are reference-workload observations, not maximum-capacity guarantees.

## Restart behavior

A completed live session was persisted, the unprivileged operator process was
terminated, and a new operator process was launched against the same SQLite
database.

The sensor process was not restarted.

After restart:

- no completed session was incorrectly recovered as active;
- the historical session remained `completed`;
- all 80 historical prediction events remained readable;
- the restarted operator still held no effective or ambient capabilities;
- the sensor remained active with the same process identity.

Interrupted nonterminal live sessions are marked failed with the stable
`operator_restart` failure code rather than being silently resumed.

## systemd-analyze interpretation

`systemd-analyze security` reported an overall exposure rating of:

```text
4.6 OK
```

The numeric score is not treated as a security acceptance target.

Several reported exposures are required by the sensor's function:

- `CAP_NET_RAW` is required for AF_PACKET capture;
- AF_PACKET must remain permitted;
- `PrivateNetwork=yes` would defeat observation of the host network;
- AF_UNIX must remain available for local operator IPC;
- group write access to the sensor socket is intentional for the trusted
  `parallax` authorization group.

The review therefore evaluates the appropriateness of each privilege rather
than attempting to minimize the systemd score mechanically.

## Threats and controls

| Threat | Primary control | Residual risk |
| --- | --- | --- |
| Packet payload leakage into operator | Decode to PacketMetadata inside sensor; metadata-only IPC contract | Sensor process itself necessarily receives raw frames |
| Packet endpoint persistence | Prediction contract excludes raw connection tuple; persistence review verifies absence | Endpoint metadata exists transiently in operator memory |
| Unauthorized raw capture | CAP_NET_RAW isolated to dedicated service; socket restricted to trusted group | Any account intentionally granted `parallax` membership can request capture |
| Privilege escalation through operator | Operator holds no raw-capture capability; No setcap on Python | Normal local OS compromise remains outside Parallax's control |
| Malformed IPC client | Exact schema, message-size bound, single start request, structured protocol errors | Local authorized client can still consume one sequential sensor session |
| Sensor client disconnect | Disconnect detection and session cleanup | Repeated authorized connections may still create local resource load |
| Active-flow exhaustion | Hard maximum and explicit `flow_capacity_exceeded` failure | Session terminates rather than degrading silently |
| Memory growth from stale flows | Stale eviction and bounded tracked-flow state | Workload-specific memory behavior still requires operational monitoring |
| Prediction-history disclosure | Database mode 0600; default history directory 0700 | Process owner can read its own history |
| Browser disclosure of raw endpoints | RuntimePredictionEvent omits addresses and ports | Prediction timing/category information can itself be sensitive |
| Model artifact substitution | Frozen expected SHA-256 identities verified at startup | Host-level compromise could replace code and trusted constants together |
| Remote API exposure | Current acceptance is local-only | Authentication/TLS are required before remote deployment |

## Explicitly accepted residual risks

The following are accepted for the current local research/demo scope:

1. Packet metadata is processed in memory by the operator.
2. The trusted `parallax` group can authorize local capture access.
3. The sensor requires CAP_NET_RAW and AF_PACKET.
4. The sensor uses a sequential single-client IPC listener.
5. Prediction timing, pseudonymous flow identifiers, classifications, and OOD
   values are retained as operational history.
6. The local FastAPI acceptance environment does not itself establish a secure
   Internet-facing deployment.
7. Local root or host compromise is outside the application's trust boundary.

## Requirements for remote deployment

Parallax must not be presented as safely Internet-facing based only on this
local acceptance.

Before any remote deployment, at minimum:

- authenticate operator API access;
- terminate transport using authenticated TLS;
- define authorization roles;
- protect the history database and model artifacts under a dedicated service
  identity;
- limit network ingress to intended clients;
- review CORS and browser-origin policy;
- establish secrets-management policy;
- define log retention;
- define metadata retention/deletion policy;
- repeat the threat review against the actual deployment topology.

## Scientific integrity boundary

Security hardening does not alter the accepted VNAT scientific result.

The accepted model, feature artifact, split manifest, OOD calibration, and
thresholds remain frozen.

Live traffic must not be used to claim:

- new classification accuracy;
- true OOD detection performance;
- maliciousness;
- arbitrary application identification;
- production NDR effectiveness.

## Milestone 6 security review conclusion

The implemented local live architecture satisfies the Milestone 6 security
and privacy objectives after remediation of the SQLite filesystem-permission
finding.

Validated properties include:

- raw capture isolated to a dedicated CAP_NET_RAW-only process;
- unprivileged model/operator/API execution;
- metadata-only bounded sensor IPC;
- no raw packet frames or payload bytes crossing into the operator;
- no raw endpoints or feature vectors persisted in reviewed live history;
- owner-only operational SQLite history;
- restricted trusted-group sensor socket;
- per-session sensor failure containment;
- explicit resource-capacity failure;
- restart-safe persisted live history.

No additional implementation-blocking finding remains from this review.

Milestone 6 remains open only for final acceptance/documentation closure.
