# Parallax Least-Privilege Sensor Service

## Purpose

Live packet capture is isolated from the Parallax operator API.

The operator process does not open Linux AF_PACKET sockets and does not
require CAP_NET_RAW. It connects to the local sensor through:

`/run/parallax/sensor.sock`

The sensor sends PacketMetadata only. Raw Ethernet frames and packet
payload bytes never cross the IPC boundary.

## Privilege Boundary

Operator process:

- no raw-socket capability
- owns model inference and OOD scoring
- owns flow/window state
- owns operator API and SSE
- owns replay/history state

Sensor process:

- CAP_NET_RAW only
- AF_PACKET capture
- Ethernet/IPv4 decoding
- emits metadata-only PacketMetadata
- no model access
- no database access
- no HTTP listener

The Python interpreter itself must not receive persistent file
capabilities with setcap. Capability assignment belongs to the dedicated
service process.

## Service Identity

Recommended production identities:

- service user: `parallax-sensor`
- shared IPC group: `parallax`

The operator account must be a member of the `parallax` group to connect
to the Unix-domain socket.

Example provisioning:

    sudo groupadd --system parallax
    sudo useradd \
      --system \
      --gid parallax \
      --no-create-home \
      --shell /usr/sbin/nologin \
      parallax-sensor

Add the operator account to the group separately.

A new login/session is normally required before supplementary group
membership is visible.

## systemd Capability Model

The reference unit grants only:

`CAP_NET_RAW`

The capability is bounded to the sensor process with:

- AmbientCapabilities
- CapabilityBoundingSet
- NoNewPrivileges

The operator API remains unprivileged.

The sensor service restricts socket families to:

- AF_UNIX
- AF_PACKET

It therefore cannot create ordinary AF_INET or AF_INET6 network client
or server sockets.

## Runtime Socket

systemd owns `/run/parallax` through RuntimeDirectory.

The reference permissions are designed so:

- the sensor user owns the runtime directory
- the shared `parallax` group can traverse it
- group members cannot create arbitrary files there
- the sensor socket is not world-accessible

The sensor uses:

`/run/parallax/sensor.sock`

The operator default and sensor-service default intentionally match.

## Installation Layout

The reference unit expects the installed application environment under:

`/opt/parallax/.venv`

The PATH entry may be adjusted for another deployment layout.

Do not run the FastAPI/Uvicorn operator process with sudo merely to gain
packet-capture privileges.

## Acceptance Evidence

The reference deployment was validated on JPCMAIN under WSL 2 Ubuntu 24.04.

The service ran as:

```text
user:  parallax-sensor
group: parallax
```

The runtime boundary was observed as:

```text
/run/parallax
drwxr-x--- parallax-sensor parallax

/run/parallax/sensor.sock
srw-rw---- parallax-sensor parallax
```

The running sensor capability state was:

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

The operator application was then launched under the normal development
account with access to the shared `parallax` group. It connected to the
sensor socket and completed the live packet-to-prediction path without running
FastAPI under sudo.

No persistent file capability was applied to the Python interpreter.
