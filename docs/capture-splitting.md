# Capture-Grouped Splitting

## Purpose

Parallax evaluates whether a model generalizes to unseen recording sessions, not whether it can
recognize another window from a capture it already observed. Windows from one VNAT source PCAP
therefore remain together. Random window splitting is not permitted for the primary result because
session-specific timing, hosts, applications, and collection conditions could cross the evaluation
boundary and inflate performance.

The versioned split contract is `vnat-capture-split-1`. Its immutable output contract is
`vnat-capture-split-manifest-1`.

## Partition roles

| Partition | Target windows | Role |
| --- | ---: | --- |
| Training | 60% | Fit model parameters |
| Validation | 15% | Select architectures and hyperparameters |
| Calibration | 10% | Fit probability and OOD calibration only |
| Test | 15% | Produce the final locked evaluation |

Test data must not influence model selection, preprocessing decisions, probability calibration,
OOD thresholds, or operating-point selection.

## Hard constraints

The optimizer may trade balance among partitions, but it may not trade away these requirements:

1. Every capture identifier is assigned to exactly one partition.
2. Every one of the five traffic categories appears in every partition.
3. Every category contributes at least 20 windows to every partition.
4. Every one of the ten applications appears in training and test.
5. VPN and non-VPN captures appear in every partition.
6. Every category/VPN-status combination appears in training.

Application coverage is not required in validation or calibration. Vimeo has only two eligible
captures, Netflix has three, and requiring all applications across four partitions would make the
problem infeasible.

## Deterministic optimization

The splitter formulates capture assignment as a mixed-integer linear program and solves it with
SciPy's HiGHS-backed `milp` interface. Absolute-deviation terms balance:

- Windows by category
- Capture groups by category
- Total windows
- Total capture groups
- Windows by VPN status

Captures are sorted by their stable identifiers before the optimization matrix is assembled, so
input iteration order does not affect the result. The default solve is bounded to 15 seconds with
a 10% relative MIP gap. In-memory callers may inspect a feasible time-limited result, but the
manifest exporter publishes only when the solver reports proven optimality.

## Artifact verification and publication

Create the accepted release-compatible manifest with:

```bash
uv run --locked parallax dataset split-features \
  data/processed/vnat-release-1/features-release-compatible.parquet \
  data/processed/vnat-release-1/capture-splits.json
```

The default trusted feature-artifact SHA-256 is
`611dcb63c66e04f461fb7500f96762c1722a06fbdde700dc5aa42ae021e39f16`.
The command computes and compares this checksum before opening Parquet, validates the exact
`vnat-feature-artifact-1` schema, and then scans only `capture_id`, `vpn_status`, `application`,
and `category` in bounded batches.

Publication is atomic and refuses an existing destination. The deterministic JSON records:

- Manifest, split, and feature schema versions
- Source filename, size, checksum, rows, columns, and row groups
- Complete partition and solver configuration
- Objective value, optimality result, and solver message
- Overall and per-partition capture/window distributions
- One auditable assignment for every capture

Timestamps and machine-specific absolute paths are intentionally omitted so identical inputs and
configuration produce identical bytes.

## Accepted VNAT release-compatible split

The accepted feature artifact contains 15,095 windows across 162 eligible captures. The optimal
assignment is:

| Partition | Captures | Windows | Actual windows | Target | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 95 | 9,046 | 59.9271% | 60% | -0.0729 pp |
| Validation | 25 | 2,098 | 13.8986% | 15% | -1.1014 pp |
| Calibration | 16 | 1,499 | 9.9304% | 10% | -0.0696 pp |
| Test | 26 | 2,452 | 16.2438% | 15% | +1.2438 pp |

The solver objective is `5.352902933`. Two independent API generations matched byte for byte;
the CLI output, CLI-written file, and accepted API manifest also matched. The resulting artifact
is 38,372 bytes with SHA-256
`a1aeee5f118c1e3bd57aa7232eb009634c098b9025571786614798953ebd8a0f`.

The target fractions are objectives rather than hard quotas because captures cannot be divided.
For example, one non-VPN Vimeo capture contains 858 windows, so forcing exact global fractions
would conflict with leakage prevention and label coverage.

## Use in modeling

Training code must join feature rows to this manifest by exact `capture_id`. It must reject an
unknown capture, a duplicate assignment, an incompatible schema version, or a feature-artifact
checksum mismatch. Calibration receives its own partition because probability calibration and OOD
threshold fitting are model-development operations and must not use the locked test set.

Any randomized-window experiment must use a separately named manifest, be labeled as a
paper-comparison result, and never replace the capture-held-out metrics.
