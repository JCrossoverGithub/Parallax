# Milestone 7 Acceptance — Portfolio Release

## Status

**Accepted for v0.1.0 release.**

Milestone 7 reconciles the engineering, scientific, operational, security, and
presentation work completed through Milestone 6 into the first public
portfolio release of Parallax.

The release candidate preserves the accepted model, calibration, feature,
split, and scientific interpretation boundaries. No model selection,
calibration change, threshold change, feature change, or new test-set
evaluation was performed during portfolio-release preparation.

## Release identity

Package version:

```text
0.1.0
```

License:

```text
MIT
```

Release branch:

```text
feature/portfolio-release
```

The final annotated Git tag is created only after this closure commit is
merged to `main` and the remote CI result is green.

## Portfolio deliverables

The release includes:

- a portfolio-facing README;
- as-built architecture documentation;
- real Angular operator-console screenshots;
- a reproducible portfolio demo runbook;
- a release reproducibility guide;
- dataset and model cards;
- uncertainty-modeling evidence;
- performance and soak evidence;
- live sensor validation evidence;
- a threat model and security/privacy review;
- full-stack GitHub Actions CI;
- MIT licensing;
- a public changelog.

### Demo video

A narrated portfolio demo video is intentionally deferred and is not included
in the initial `v0.1.0` repository release.

The repository includes the complete demonstration procedure in:

```text
docs/demo-runbook.md
```

A video may be recorded later and attached to the existing GitHub release
without changing the accepted source release or scientific evidence.

Its absence is therefore treated as a presentation follow-up rather than a
release blocker.

## Real product presentation

Three real screenshots from the completed operator application are included:

```text
docs/assets/screenshots/parallax-live-sensor.png
docs/assets/screenshots/parallax-prediction-investigation.png
docs/assets/screenshots/parallax-session-history.png
```

They demonstrate:

- active live sensing;
- real-time prediction delivery;
- confidence and OOD separation;
- prediction investigation;
- model/calibration provenance;
- completed-session persistence and historical reopening.

The screenshots are operational demonstrations. Their classifications are not
used as live accuracy evidence.

## Clean-clone acceptance

A clean clone was created from the remote portfolio-release branch at:

```text
136d0562555caebd4b3313271f3bd4ca50499e2d
```

The clean environment reconstructed its Python and Angular dependency trees
from committed lockfiles.

It successfully completed:

```text
Python tests:              1,120
Statement coverage:         100%
Branch coverage:            100%
Angular tests:                79
Python package build:      green
Angular production build: green
Ruff:                      clean
mypy:                      clean
```

The clean clone also verified:

- the Parallax CLI starts;
- release evidence JSON is valid;
- the soak runner has valid shell syntax;
- all local Markdown targets resolve;
- all portfolio screenshots exist;
- no tracked PCAP, HDF5, Parquet, SQLite, checkpoint, model-weight, or
  environment files were discovered by the release artifact audit;
- the resulting Git worktree remains clean.

Full reconstruction guidance is recorded in:

```text
docs/reproducibility.md
```

## Full-stack CI

GitHub Actions validates both major implementation surfaces.

### Python

CI performs:

- locked dependency synchronization;
- lock verification;
- Ruff linting;
- Ruff formatting verification;
- strict mypy checking;
- pytest with the required 100% coverage threshold;
- Python package build;
- live-soak script syntax validation;
- acceptance-evidence JSON validation.

### Angular

CI performs:

- Node/npm setup;
- `npm ci`;
- Angular/Vitest test execution;
- Angular production build.

The known Angular component-style budget warning remains non-blocking.

The known FastAPI/Starlette TestClient `httpx` deprecation warning also
remains non-blocking.

## Reproducible scientific artifact chain

The accepted artifact lineage remains:

```text
MIT LL VNAT raw HDF5
-> deterministic observation windows
-> 129-feature release-compatible artifact
-> capture-grouped split
-> frozen prototype model
-> calibration-only OOD artifact
-> checksum-bound replay/live runtime
```

Accepted identities:

| Artifact | SHA-256 |
| --- | --- |
| Feature artifact | `611dcb63c66e04f461fb7500f96762c1722a06fbdde700dc5aa42ae021e39f16` |
| Split manifest | `a1aeee5f118c1e3bd57aa7232eb009634c098b9025571786614798953ebd8a0f` |
| Prototype model | `1c61678611043a7f70f02836ae23bbc7c1abf683b015e23ebafed4d941ecdef7` |
| OOD calibration | `af1d066ea2d96943c873b75e897ca4c7d910c79813a61b605eec8c7ec3c1fd9d` |

The historical accepted test result remains immutable evidence.

It was not rerun as part of portfolio-release preparation.

## Runtime acceptance carried into the release

The accepted runtime includes:

- classic raw-PCAP metadata ingestion;
- deterministic bidirectional flows;
- incremental observation windows;
- shared 129-feature construction;
- frozen classifier inference;
- independent OOD scoring;
- deterministic replay;
- least-privilege live Linux sensing;
- metadata-only AF_UNIX sensor IPC;
- bounded live-flow state;
- structured runtime failures;
- REST controls;
- ordered SSE prediction delivery;
- Angular Monitor, Replay Lab, and History experiences;
- durable SQLite operational history;
- browser recovery;
- restart-safe historical inspection.

## Least-privilege boundary

Raw capture remains isolated in the dedicated sensor service.

The accepted reference boundary is:

```text
PRIVILEGED

network interface
-> AF_PACKET
-> Ethernet / IPv4 decoder
-> PacketMetadata

           |
           v

metadata-only AF_UNIX IPC

           |
           v

UNPRIVILEGED

operator
-> flow/window processing
-> feature extraction
-> model + OOD inference
-> SQLite
-> REST / SSE
-> Angular console
```

The sensor receives only CAP_NET_RAW.

The FastAPI/model/operator process was validated without inherited, permitted,
effective, or ambient Linux capabilities.

Raw Ethernet frames and payload bytes necessarily exist transiently inside the
privileged capture/decoder boundary, but they do not cross the sensor IPC
boundary or enter ordinary operational persistence.

Traffic metadata itself may still be sensitive.

## Operational validation carried into the release

The accepted three-minute real live run recorded:

| Measurement | Result |
| --- | ---: |
| Duration | 180.094985 s |
| Generated UDP datagrams | 28,704 |
| Generator flows | 16 |
| Final prediction events | 80 |
| SSE prediction events | 80 |
| Persisted prediction events | 80 |
| Terminal state | completed |
| Sensor sampled RSS growth | 0 KiB |
| Operator sampled RSS growth | 6,036 KiB |

The generated-datagram count describes the workload generator. It is not an
assertion that the capture path processed exactly that many packets.

This evidence establishes runtime mechanics and bounded reference behavior,
not maximum packet-processing throughput.

## Scientific boundaries

The accepted closed set remains exactly:

```text
STREAMING
VOIP
CHAT
C2
FILE_TRANSFER
```

Release claims remain bounded as follows:

- raw confidence is a closed-set softmax preference, not calibrated factual
  certainty;
- OOD is an independent distribution-shift score;
- the accepted VNAT test partition contains no true OOD examples;
- OOD AUROC, OOD recall, and OOD detection power are therefore not claimed;
- live traffic is operational evidence rather than new accuracy evidence;
- VNAT `C2` represents benign SSH/RDP traffic rather than malicious
  command-and-control ground truth;
- Parallax is not presented as a malware detector, production IDS,
  enforcement system, or universal Internet application classifier.

Any new OOD-generalization claim requires a separately designed and
preregistered experiment.

## Security and privacy disposition

The Milestone 6 security review remains the accepted release review.

Operational prediction history does not intentionally persist:

- packet payloads;
- raw Ethernet frames;
- raw endpoint tuples;
- raw feature vectors.

Operator-history databases created by Parallax use owner-only `0600`
permissions, and Parallax-created history directories use `0700`.

The `parallax` Unix group remains a trusted local capture-authorization
boundary.

The accepted local deployment is not presented as Internet-facing production
security. Remote deployment requires authentication, TLS, authorization,
ingress controls, retention policy, secrets management, and deployment-specific
security review.

## Licensing and third-party material

Parallax project-authored source and documentation are released under the MIT
License.

The MIT License does not relicense:

- the MIT Lincoln Laboratory VNAT dataset;
- third-party Python packages;
- third-party npm packages;
- other externally published material.

Those remain subject to their respective terms.

## Release decision

Milestone 7 is accepted for the first public portfolio release.

The repository satisfies the release goals for:

- implementation quality;
- reproducibility;
- scientific traceability;
- privilege separation;
- security/privacy documentation;
- real product presentation;
- full-stack automated validation;
- licensing;
- public limitations and claim boundaries.

The narrated demo video remains an explicitly deferred presentation
enhancement and does not block `v0.1.0`.

After this acceptance record is committed and the authoritative quality gate
passes, the release branch may be fast-forwarded into `main`.

The `v0.1.0` tag should be created from the resulting `main` commit only after
remote CI succeeds.
