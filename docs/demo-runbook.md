# Parallax Portfolio Demo Runbook

## Purpose

This runbook demonstrates the accepted Parallax packet-to-prediction system
using the completed Milestone 6 architecture.

The walkthrough demonstrates:

- least-privilege live packet capture;
- metadata-only sensor IPC;
- unprivileged runtime inference;
- real-time REST/SSE operator integration;
- prediction investigation;
- durable live-session history.

It is an operational demonstration, not an accuracy or OOD-generalization
experiment.

## Demonstrated architecture

```text
local interface
-> CAP_NET_RAW-only parallax-sensor
-> Ethernet / IPv4 decoding
-> PacketMetadata
-> AF_UNIX metadata IPC
-> unprivileged FastAPI operator
-> bidirectional flows
-> observation windows
-> 129-feature runtime
-> frozen classifier + OOD calibration
-> RuntimePredictionEvent
-> REST / SSE
-> Angular operations console
-> SQLite history
```

Raw Ethernet frames remain inside the privileged sensor process.

## Prerequisites

Reference environment:

```text
Linux / WSL 2 Ubuntu 24.04
Python 3.12
uv 0.12.5
Node.js 24.19.0
npm 11.17.0
```

The accepted model and OOD calibration artifacts must already exist at their
documented local paths.

The operator account must have access to the trusted `parallax` Unix group.

The reference sensor service must be installed and configured from:

```text
deploy/systemd/parallax-sensor.service
```

See `docs/sensor-service.md` for deployment details.

## 1. Verify the sensor boundary

Check the sensor:

```bash
systemctl is-active parallax-sensor.service

systemctl --no-pager --full status \
  parallax-sensor.service
```

Verify its runtime directory and socket:

```bash
stat -c '%A %a %U %G %n' \
  /run/parallax \
  /run/parallax/sensor.sock
```

The accepted deployment uses:

```text
/run/parallax
0750 parallax-sensor:parallax

/run/parallax/sensor.sock
0660 parallax-sensor:parallax
```

The sensor is the only Parallax component that requires CAP_NET_RAW.

Do not start the operator API with sudo.

## 2. Start the operator API

From the repository root:

```bash
uv run --locked uvicorn \
  parallax.operator.application:create_operator_application \
  --factory \
  --host 127.0.0.1 \
  --port 8000
```

Verify health from another terminal:

```bash
curl -s \
  http://127.0.0.1:8000/health \
  | uv run python -m json.tool
```

Inspect available live interfaces:

```bash
curl -s \
  http://127.0.0.1:8000/api/v1/live/interfaces \
  | uv run python -m json.tool
```

The JPCMAIN WSL reference environment exposes `eth0`.

## 3. Start the Angular operator console

From the repository:

```bash
cd web

npm ci

npm start -- \
  --host 127.0.0.1 \
  --port 4200
```

Open:

```text
http://127.0.0.1:4200
```

The Angular development proxy routes `/api` and `/health` to the local
operator API at `127.0.0.1:8000`.

## 4. Start a live capture

Open the **Monitor** workspace.

Select:

```text
eth0
```

Start the sensor session.

The console should transition to a live/running state and establish the SSE
event channel.

## 5. Generate controlled metadata traffic

The following generator creates 16 persistent local UDP flows for two minutes:

```bash
uv run python - <<'PY'
import socket
import time

DESTINATION = ("192.0.2.1", 443)
FLOW_COUNT = 16
INTERVAL_SECONDS = 0.10
DURATION_SECONDS = 120

sockets = []

for index in range(FLOW_COUNT):
    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )
    sock.bind(
        (
            "0.0.0.0",
            40_000 + index,
        )
    )
    sockets.append(sock)

payload = b"parallax-demo"

started = time.monotonic()
sent = 0

try:
    while time.monotonic() - started < DURATION_SECONDS:
        for sock in sockets:
            sock.sendto(
                payload,
                DESTINATION,
            )
            sent += 1

        time.sleep(INTERVAL_SECONDS)
finally:
    for sock in sockets:
        sock.close()

elapsed = time.monotonic() - started

print(
    f"generated_datagrams={sent} "
    f"elapsed_seconds={elapsed:.3f}"
)
PY
```

`192.0.2.1` is from the documentation-only TEST-NET-1 address range.

The generated traffic is used only to exercise runtime mechanics.

Its resulting classifications must not be treated as ground-truth accuracy
evidence.

## 6. Observe live predictions

As observation windows complete, the Monitor workspace displays prediction
events.

The main view exposes:

- session state;
- selected capture interface;
- prediction count;
- current predicted category;
- raw confidence;
- OOD score;
- packet count;
- pseudonymous flow identity;
- ordered runtime prediction events.

Raw confidence is the frozen classifier's closed-set preference.

OOD is an independent distribution-shift score.

Neither value is a threat verdict.

## 7. Investigate a prediction

Select a row from the prediction feed.

The Event Investigation view exposes:

- predicted category;
- raw confidence;
- OOD score;
- relative-Mahalanobis distance;
- five-category class distribution;
- run identity;
- window identity;
- flow identity;
- capture identity;
- packet count;
- window timing;
- frozen model provenance;
- frozen OOD-calibration provenance.

This view demonstrates that operator-facing predictions remain traceable to
their runtime window and accepted frozen artifacts.

## 8. Complete the live session

Allow the controlled traffic generator to finish.

Use **Stop Sensor** in the operator console.

Wait for the live session to reach:

```text
COMPLETED
```

The terminal outcome should not contain a structured failure.

## 9. Inspect durable history

Open the **History** workspace.

The completed live session should be listed with:

- interface;
- run identity;
- completed state;
- persisted event count;
- live-session source type.

Select the completed session.

Its persisted RuntimePredictionEvent records should be restored into the
prediction feed.

The historical view can be inspected without restarting or reviving the
original capture session.

## 10. Suggested portfolio walkthrough

A concise demonstration can be presented in roughly three minutes:

```text
0:00 - Explain the architecture and privacy boundary
0:30 - Start eth0 live monitoring
0:45 - Generate controlled traffic
1:10 - Show predictions arriving through SSE
1:30 - Open Event Investigation
2:00 - Explain confidence versus OOD
2:20 - Stop the session
2:35 - Open Session History
2:50 - Reopen the completed run and show persisted predictions
```

## Reference screenshots

### Live Sensor

![Live Sensor](assets/screenshots/parallax-live-sensor.png)

### Prediction Investigation

![Prediction Investigation](assets/screenshots/parallax-prediction-investigation.png)

### Session History

![Session History](assets/screenshots/parallax-session-history.png)

## Interpretation boundaries

The demonstration establishes system behavior.

It does not establish:

- classification accuracy on arbitrary live traffic;
- OOD recall;
- OOD AUROC;
- reliable unseen-application identification;
- maliciousness detection;
- production IDS/NDR effectiveness;
- maximum sustainable packet rate.

The accepted VNAT model, feature artifact, split manifest, OOD calibration,
and thresholds remain frozen.

See:

- `docs/milestone-6-acceptance.md`
- `docs/performance-report.md`
- `docs/live-sensor-validation.md`
- `docs/threat-model.md`
- `docs/model-card.md`
