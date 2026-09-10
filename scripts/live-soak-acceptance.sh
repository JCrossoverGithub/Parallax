#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/projects/Parallax}"
API="${API:-http://127.0.0.1:8000}"
DURATION_SECONDS="${DURATION_SECONDS:-180}"
FLOW_COUNT="${FLOW_COUNT:-16}"
TRAFFIC_INTERVAL="${TRAFFIC_INTERVAL:-0.1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-10}"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="/tmp/parallax-6.14b-$STAMP"

RUN_ID=""
SSE_PID=""
SAMPLE_PID=""
TRAFFIC_PID=""
STOP_ON_EXIT=0

mkdir -p "$OUT"
cd "$ROOT"

json_field() {
    local file="$1"
    local key="$2"

    uv run python -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]])' \
        "$file" "$key"
}

cleanup() {
    local status=$?

    if [[ -n "$TRAFFIC_PID" ]] && kill -0 "$TRAFFIC_PID" 2>/dev/null; then
        kill "$TRAFFIC_PID" 2>/dev/null || true
    fi

    if [[ -n "$SAMPLE_PID" ]] && kill -0 "$SAMPLE_PID" 2>/dev/null; then
        kill "$SAMPLE_PID" 2>/dev/null || true
    fi

    if [[ -n "$SSE_PID" ]] && kill -0 "$SSE_PID" 2>/dev/null; then
        kill "$SSE_PID" 2>/dev/null || true
    fi

    if [[ "$STOP_ON_EXIT" -eq 1 && -n "$RUN_ID" ]]; then
        curl -sS \
            -X POST \
            "$API/api/v1/live/$RUN_ID/stop" \
            >/dev/null 2>&1 || true
    fi

    exit "$status"
}

trap cleanup INT TERM EXIT

echo "============================================================"
echo "Parallax Milestone 6.14b sustained live acceptance"
echo "============================================================"
echo "Output directory: $OUT"
echo "Duration:         ${DURATION_SECONDS}s"
echo "Flows:            $FLOW_COUNT"
echo

echo "=== PREFLIGHT: OPERATOR ==="

curl -fsS "$API/health" > "$OUT/health.json"
uv run python -m json.tool "$OUT/health.json"

OPERATOR_PID="$(
    ps -u "$(id -u)" -o pid=,args= \
    | awk '
        /[u]vicorn .*parallax\.operator\.application:create_operator_application/ {
            print $1
            exit
        }
    '
)"

if [[ -z "$OPERATOR_PID" ]]; then
    echo "ERROR: Could not locate operator Uvicorn process." >&2
    exit 1
fi

echo
echo "Operator PID: $OPERATOR_PID"

echo
echo "=== PREFLIGHT: SENSOR ==="

if ! systemctl is-active --quiet parallax-sensor.service; then
    echo "ERROR: parallax-sensor.service is not active." >&2
    exit 1
fi

SENSOR_PID="$(
    systemctl show \
        -p MainPID \
        --value \
        parallax-sensor.service
)"

echo "Sensor PID:   $SENSOR_PID"

if ! ss -xl | grep -q '/run/parallax/sensor.sock'; then
    echo "ERROR: sensor Unix socket is not listening." >&2
    exit 1
fi

sg parallax -c '
    stat -c "%A %a %U %G %n" /run/parallax/sensor.sock
    test -S /run/parallax/sensor.sock
'

echo
echo "=== PREFLIGHT: SECURITY BOUNDARY ==="

{
    echo "[sensor]"
    grep -E \
        '^(Name|Uid|Gid|CapInh|CapPrm|CapEff|CapBnd|CapAmb|VmRSS|VmHWM|Threads):' \
        "/proc/$SENSOR_PID/status"

    echo
    echo "[operator]"
    grep -E \
        '^(Name|Uid|Gid|CapInh|CapPrm|CapEff|CapBnd|CapAmb|VmRSS|VmHWM|Threads):' \
        "/proc/$OPERATOR_PID/status"
} | tee "$OUT/security-baseline.txt"

uv run python - "$SENSOR_PID" "$OPERATOR_PID" "$OUT/baseline.json" <<'PY'
import json
import sys
from pathlib import Path

sensor_pid = int(sys.argv[1])
operator_pid = int(sys.argv[2])
output = Path(sys.argv[3])


def status(pid: int) -> dict[str, int | str]:
    result: dict[str, int | str] = {}

    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        value = value.strip()

        if key in {"VmRSS", "VmHWM"}:
            result[key] = int(value.split()[0])
        elif key in {
            "Name",
            "Uid",
            "Gid",
            "CapInh",
            "CapPrm",
            "CapEff",
            "CapBnd",
            "CapAmb",
        }:
            result[key] = value

    return result


payload = {
    "sensor": status(sensor_pid),
    "operator": status(operator_pid),
}

output.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

echo
echo "=== PREFLIGHT: EXISTING LIVE SESSION ==="

curl -fsS "$API/api/v1/live/active" > "$OUT/active-before.json"
cat "$OUT/active-before.json" | uv run python -m json.tool

HAS_ACTIVE="$(
    uv run python - "$OUT/active-before.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print("yes" if json.load(handle) is not None else "no")
PY
)"

if [[ "$HAS_ACTIVE" != "no" ]]; then
    echo "ERROR: An active live session already exists." >&2
    exit 1
fi

GATEWAY="$(
    ip route show default \
    | awk '{print $3; exit}'
)"

if [[ -z "$GATEWAY" ]]; then
    echo "ERROR: Could not determine default gateway." >&2
    exit 1
fi

echo
echo "Gateway: $GATEWAY"

echo
echo "=== START LIVE SESSION ==="

curl -fsS \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{"interface":"eth0"}' \
    "$API/api/v1/live" \
    > "$OUT/start.json"

uv run python -m json.tool "$OUT/start.json"

RUN_ID="$(json_field "$OUT/start.json" run_id)"
STOP_ON_EXIT=1

echo
echo "RUN_ID=$RUN_ID"

echo
echo "=== WAIT FOR RUNNING ==="

STATE=""

for _ in $(seq 1 40); do
    curl -fsS \
        "$API/api/v1/live/$RUN_ID" \
        > "$OUT/session-current.json"

    STATE="$(json_field "$OUT/session-current.json" state)"
    echo "state=$STATE"

    case "$STATE" in
        running)
            break
            ;;
        failed)
            echo "ERROR: live session failed during startup." >&2
            uv run python -m json.tool "$OUT/session-current.json"
            exit 1
            ;;
    esac

    sleep 0.25
done

if [[ "$STATE" != "running" ]]; then
    echo "ERROR: live session never reached running state." >&2
    exit 1
fi

echo
echo "=== ATTACH SSE OBSERVER ==="

curl \
    -NsS \
    --max-time "$((DURATION_SECONDS + 90))" \
    "$API/api/v1/live/$RUN_ID/stream?after=0" \
    > "$OUT/sse.txt" \
    2> "$OUT/sse.stderr" &

SSE_PID=$!
echo "SSE PID: $SSE_PID"

echo
echo "=== START RESOURCE SAMPLER ==="

uv run python - \
    "$RUN_ID" \
    "$SENSOR_PID" \
    "$OPERATOR_PID" \
    "$DURATION_SECONDS" \
    "$SAMPLE_INTERVAL" \
    "$OUT/samples.csv" \
    "$API" <<'PY' &
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

run_id = sys.argv[1]
sensor_pid = int(sys.argv[2])
operator_pid = int(sys.argv[3])
duration = float(sys.argv[4])
interval = float(sys.argv[5])
output = Path(sys.argv[6])
api = sys.argv[7]


def process_values(pid: int) -> tuple[int, int]:
    rss = None
    hwm = None

    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1])
        elif line.startswith("VmHWM:"):
            hwm = int(line.split()[1])

    if rss is None or hwm is None:
        raise RuntimeError(f"missing memory values for PID {pid}")

    return rss, hwm


def json_get(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=3) as response:
        return json.load(response)


with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)

    writer.writerow(
        [
            "elapsed_seconds",
            "sensor_rss_kib",
            "sensor_hwm_kib",
            "operator_rss_kib",
            "operator_hwm_kib",
            "state",
            "event_count",
        ]
    )

    started = time.monotonic()

    while True:
        elapsed = time.monotonic() - started

        sensor_rss, sensor_hwm = process_values(sensor_pid)
        operator_rss, operator_hwm = process_values(operator_pid)

        session = json_get(
            f"{api}/api/v1/live/{run_id}"
        )

        events = json_get(
            f"{api}/api/v1/live/{run_id}/events"
        )

        writer.writerow(
            [
                f"{elapsed:.3f}",
                sensor_rss,
                sensor_hwm,
                operator_rss,
                operator_hwm,
                session["state"],
                events["last_sequence"],
            ]
        )

        handle.flush()

        if session["state"] in {"completed", "failed"}:
            break

        if elapsed >= duration:
            break

        time.sleep(interval)
PY

SAMPLE_PID=$!
echo "Sampler PID: $SAMPLE_PID"

echo
echo "=== START CONTROLLED TRAFFIC ==="

uv run python - \
    "$GATEWAY" \
    "$DURATION_SECONDS" \
    "$FLOW_COUNT" \
    "$TRAFFIC_INTERVAL" <<'PY' > "$OUT/traffic.txt" &
import socket
import sys
import time

gateway = sys.argv[1]
duration = float(sys.argv[2])
flow_count = int(sys.argv[3])
interval = float(sys.argv[4])

sockets = []

for index in range(flow_count):
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

    sockets.append(
        (
            sock,
            (
                gateway,
                50_000 + index,
            ),
        )
    )

started = time.monotonic()
deadline = started + duration
packets_sent = 0

try:
    while time.monotonic() < deadline:
        cycle_started = time.monotonic()

        for sock, destination in sockets:
            sock.sendto(b"x", destination)
            packets_sent += 1

        remaining = interval - (
            time.monotonic() - cycle_started
        )

        if remaining > 0:
            time.sleep(remaining)
finally:
    for sock, _ in sockets:
        sock.close()

elapsed = time.monotonic() - started

print(f"packets_sent={packets_sent}")
print(f"elapsed_seconds={elapsed:.6f}")
print(
    "send_rate_packets_per_second="
    f"{packets_sent / elapsed:.2f}"
)
PY

TRAFFIC_PID=$!
echo "Traffic PID: $TRAFFIC_PID"

echo
echo "Running sustained workload for approximately ${DURATION_SECONDS}s..."

if wait "$TRAFFIC_PID"; then
    TRAFFIC_STATUS=0
else
    TRAFFIC_STATUS=$?
fi
TRAFFIC_PID=""

if wait "$SAMPLE_PID"; then
    SAMPLE_STATUS=0
else
    SAMPLE_STATUS=$?
fi
SAMPLE_PID=""

echo
echo "Traffic exit: $TRAFFIC_STATUS"
echo "Sampler exit: $SAMPLE_STATUS"

if [[ "$TRAFFIC_STATUS" -ne 0 || "$SAMPLE_STATUS" -ne 0 ]]; then
    echo "ERROR: workload or sampler failed." >&2
    exit 1
fi

echo
echo "=== TRAFFIC RESULT ==="
cat "$OUT/traffic.txt"

echo
echo "=== RESOURCE SAMPLES ==="

if command -v column >/dev/null 2>&1; then
    column -s, -t < "$OUT/samples.csv"
else
    cat "$OUT/samples.csv"
fi

echo
echo "=== LIVE SESSION AFTER SUSTAINED RUN ==="

curl -fsS \
    "$API/api/v1/live/$RUN_ID" \
    > "$OUT/session-before-stop.json"

uv run python -m json.tool "$OUT/session-before-stop.json"

STATE="$(json_field "$OUT/session-before-stop.json" state)"

echo
echo "=== LIVE EVENT SNAPSHOT ==="

curl -fsS \
    "$API/api/v1/live/$RUN_ID/events" \
    > "$OUT/events-before-stop.json"

uv run python - "$OUT/events-before-stop.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)

events = payload.get("events", [])

print("state:", payload.get("state"))
print("last_sequence:", payload.get("last_sequence"))
print("retained_events:", len(events))

if events:
    print("first_run_id:", events[0].get("run_id"))
    print("last_run_id:", events[-1].get("run_id"))
PY

echo
echo "=== SSE BEFORE STOP ==="

echo -n "prediction events: "
grep -c '^event: prediction' "$OUT/sse.txt" || true

echo -n "terminal events:   "
grep -c '^event: live-terminal' "$OUT/sse.txt" || true

if [[ "$STATE" == "running" || "$STATE" == "starting" ]]; then
    echo
    echo "=== REQUEST CLEAN STOP ==="

    curl -fsS \
        -X POST \
        "$API/api/v1/live/$RUN_ID/stop" \
        > "$OUT/stop.json"

    uv run python -m json.tool "$OUT/stop.json"
fi

echo
echo "=== WAIT FOR TERMINAL STATE ==="

for _ in $(seq 1 80); do
    curl -fsS \
        "$API/api/v1/live/$RUN_ID" \
        > "$OUT/final-session.json"

    STATE="$(json_field "$OUT/final-session.json" state)"
    echo "state=$STATE"

    case "$STATE" in
        completed|failed)
            break
            ;;
    esac

    sleep 0.25
done

if [[ "$STATE" != "completed" && "$STATE" != "failed" ]]; then
    echo "ERROR: session did not reach terminal state." >&2
    exit 1
fi

STOP_ON_EXIT=0

echo
echo "=== FINAL SESSION ==="
uv run python -m json.tool "$OUT/final-session.json"

for _ in $(seq 1 20); do
    if ! kill -0 "$SSE_PID" 2>/dev/null; then
        break
    fi

    sleep 0.25
done

if kill -0 "$SSE_PID" 2>/dev/null; then
    kill "$SSE_PID" 2>/dev/null || true
fi

wait "$SSE_PID" 2>/dev/null || true
SSE_PID=""

echo
echo "=== FINAL SSE ==="

echo -n "prediction events: "
grep -c '^event: prediction' "$OUT/sse.txt" || true

echo -n "terminal events:   "
grep -c '^event: live-terminal' "$OUT/sse.txt" || true

echo
echo "=== FINAL PROCESS STATE ==="

{
    echo "[sensor]"
    grep -E \
        '^(VmRSS|VmHWM|Threads|CapEff|CapAmb):' \
        "/proc/$SENSOR_PID/status"

    echo
    echo "[operator]"
    grep -E \
        '^(VmRSS|VmHWM|Threads|CapEff|CapAmb):' \
        "/proc/$OPERATOR_PID/status"
} | tee "$OUT/security-final.txt"

systemctl status \
    parallax-sensor.service \
    --no-pager \
    > "$OUT/sensor-final-status.txt"

echo
echo "=== DURABLE HISTORY ==="

for _ in $(seq 1 40); do
    if curl -fsS \
        "$API/api/v1/history/live/$RUN_ID" \
        > "$OUT/history.json" 2>/dev/null
    then
        break
    fi

    sleep 0.25
done

if [[ ! -s "$OUT/history.json" ]]; then
    echo "ERROR: persisted live history could not be read." >&2
    exit 1
fi

uv run python -m json.tool "$OUT/history.json"

for _ in $(seq 1 40); do
    if curl -fsS \
        "$API/api/v1/history/live/$RUN_ID/events" \
        > "$OUT/history-events.json" 2>/dev/null
    then
        break
    fi

    sleep 0.25
done

echo
echo "=== ACCEPTANCE SUMMARY ==="

uv run python - \
    "$OUT" \
    "$RUN_ID" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_id = sys.argv[2]

baseline = json.loads(
    (root / "baseline.json").read_text(encoding="utf-8")
)

final_session = json.loads(
    (root / "final-session.json").read_text(encoding="utf-8")
)

events = json.loads(
    (root / "events-before-stop.json").read_text(encoding="utf-8")
)

history = json.loads(
    (root / "history.json").read_text(encoding="utf-8")
)

history_events_payload = json.loads(
    (root / "history-events.json").read_text(encoding="utf-8")
)

with (root / "samples.csv").open(
    newline="",
    encoding="utf-8",
) as handle:
    samples = list(csv.DictReader(handle))

traffic_text = (
    root / "traffic.txt"
).read_text(encoding="utf-8")

traffic = dict(
    re.findall(
        r"^([^=\n]+)=([^\n]+)$",
        traffic_text,
        flags=re.MULTILINE,
    )
)

sse_text = (
    root / "sse.txt"
).read_text(
    encoding="utf-8",
    errors="replace",
)

sensor_baseline = int(
    baseline["sensor"]["VmRSS"]
)
operator_baseline = int(
    baseline["operator"]["VmRSS"]
)

sensor_rss = [
    int(row["sensor_rss_kib"])
    for row in samples
]

operator_rss = [
    int(row["operator_rss_kib"])
    for row in samples
]

event_list = events.get("events", [])

if isinstance(history_events_payload, dict):
    history_event_list = history_events_payload.get(
        "events",
        [],
    )
elif isinstance(history_events_payload, list):
    history_event_list = history_events_payload
else:
    history_event_list = []

print(f"run_id: {run_id}")
print(f"final_state: {final_session.get('state')}")
print(f"failure: {final_session.get('failure')}")
print()

print(
    "traffic_packets_sent:",
    traffic.get("packets_sent"),
)
print(
    "traffic_send_rate_pps:",
    traffic.get(
        "send_rate_packets_per_second"
    ),
)
print()

print(
    "live_last_sequence:",
    events.get("last_sequence"),
)
print(
    "live_retained_events:",
    len(event_list),
)
print(
    "persisted_events:",
    len(history_event_list),
)
print(
    "sse_prediction_events:",
    sse_text.count(
        "event: prediction"
    ),
)
print(
    "sse_terminal_events:",
    sse_text.count(
        "event: live-terminal"
    ),
)
print()

print(
    "sensor_baseline_rss_kib:",
    sensor_baseline,
)
print(
    "sensor_max_sample_rss_kib:",
    max(sensor_rss),
)
print(
    "sensor_sample_rss_growth_kib:",
    max(sensor_rss) - sensor_baseline,
)
print()

print(
    "operator_baseline_rss_kib:",
    operator_baseline,
)
print(
    "operator_max_sample_rss_kib:",
    max(operator_rss),
)
print(
    "operator_sample_rss_growth_kib:",
    max(operator_rss) - operator_baseline,
)
print()

print(
    "history_state:",
    history.get("state"),
)
print(
    "history_failure:",
    history.get("failure"),
)
print()
print("artifacts:", root)
PY

echo
echo "6.14b run finished."
echo "Artifacts: $OUT"

trap - INT TERM EXIT
