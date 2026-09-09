# Parallax

Parallax is an uncertainty-aware network traffic monitoring system. It is being built to
classify broad application activity from observable encrypted traffic metadata such as packet
timing, size, and direction without decrypting or persisting packet payloads.

## Project status

Parallax is in active development. The repository now includes a typed VNAT release contract,
fail-closed source checksum verification, structural inspection for the raw HDF5 dataset,
deterministic capture-aligned window extraction, reproduction of the published 129-feature
representation, versioned Parquet exports, deterministic capture-grouped partitioning, strict
quality gates, a manifest-bound modeling loader, deterministic reference baselines, and a frozen
prototypical embedding model with relative-Mahalanobis OOD calibration. The candidate completed
one checksum-bound test evaluation after its model, calibration, thresholds, and reporting policy
were frozen. Raw-PCAP ingestion, deterministic bidirectional flow reconstruction, incremental
capture-aligned windowing, shared runtime feature construction, controlled replay, checksum-bound
runtime inference, and an operator-facing prediction-event contract are implemented. Selected
VNAT SSH and VoIP captures have exact batch/runtime feature parity. Persistence, the external API
and event transport, the operations dashboard, and live monitoring are not yet implemented.

Milestone 4 completes the deterministic replay runtime through operator-facing prediction
events. The next operational target is the Milestone 5 operator layer:

```text
VNAT PCAP -> controlled replay -> flows -> windows -> features
          -> frozen model + OOD calibration -> RuntimePredictionEvent
          -> API/event service -> operations dashboard
```

The first Milestone 5 vertical slice will start one replay from the operator interface and display
real runtime prediction events as they are produced.

OOD means out of distribution. It is reported independently from ordinary predictive
confidence so unfamiliar traffic is not silently presented as a trustworthy known category.

## Initial scope

Parallax will initially:

- Use the MIT Lincoln Laboratory VPN/Non-VPN Network Application Traffic Dataset (VNAT).
- Establish leakage-resistant, capture-level training and evaluation partitions.
- Compare conventional baselines with an uncertainty-aware model.
- Reproduce statistical and wavelet-based traffic features.
- Replay recorded PCAP traffic through the same feature path used at inference time.
- Display category predictions, confidence, OOD scores, provenance, and service health.

Parallax is not currently a malware detector, production intrusion-detection system, traffic
enforcement tool, or general classifier for arbitrary Internet applications.

## Development setup

Requirements:

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

Install the locked development environment and run the command-line entry point:

```bash
uv sync --locked --all-groups
uv run --locked parallax
```

Inspect a verified VNAT release 1 raw dataframe:

```bash
uv run --locked parallax dataset inspect \
  data/raw/vnat/VNAT_Dataframe_release_1.h5
```

The command emits a JSON report containing source provenance, capture and connection counts,
packet-count statistics, and label distributions. The trusted release checksum is checked before
the HDF5 object data is deserialized.

Extract release-compatible observation windows into a versioned Parquet artifact:

```bash
uv run --locked parallax dataset extract-windows \
  data/raw/vnat/VNAT_Dataframe_release_1.h5 \
  data/processed/vnat-release-1/windows-release-compatible.parquet
```

The default policy retains windows containing more than 20 packets because that interpretation
most closely reproduces the released feature dataframe. Use `--threshold-policy paper-literal`
to also retain windows containing exactly 20 packets as implied by the paper's wording. The
command refuses to overwrite an existing artifact and writes a companion JSON manifest
containing the source checksum, extraction configuration, output checksum, and audited counts.

Calculate the versioned 129-feature representation from those windows:

```bash
uv run --locked parallax dataset extract-features \
  data/processed/vnat-release-1/windows-release-compatible.parquet \
  data/processed/vnat-release-1/features-release-compatible.parquet
```

Feature extraction processes one Parquet row group at a time and writes another immutable
artifact plus a provenance manifest. The default `release-compatible` policy preserves an
observed defect in the published feature dataframe: its directional byte-total columns duplicate
the packet-count columns. Use `--byte-total-policy corrected` for new experiments that should
calculate real directional byte totals.

Create the primary capture-grouped model-development split:

```bash
uv run --locked parallax dataset split-features \
  data/processed/vnat-release-1/features-release-compatible.parquet \
  data/processed/vnat-release-1/capture-splits.json
```

The command checks the trusted feature-artifact checksum before reading Parquet, keeps every
source capture wholly within one partition, and targets 60% training, 15% validation, 10%
calibration, and 15% test windows. It refuses to publish unless the mixed-integer solver proves
optimality and writes a deterministic JSON manifest containing every assignment, the complete
configuration, solver evidence, source provenance, and partition distributions.

Fit the initial training-only baselines and publish validation-only metrics:

```bash
uv run --locked parallax model validate-baselines \
  data/processed/vnat-release-1/features-release-compatible.parquet \
  data/processed/vnat-release-1/capture-splits.json \
  data/processed/vnat-release-1/baseline-validation.json
```

The command verifies both source artifacts, fits preprocessing and estimators using only the
training partition, and evaluates only validation data. Its deterministic JSON report records
provenance, configuration, convergence, aggregate metrics, per-category metrics, and confusion
matrices. That baseline command does not access calibration or test.

The accepted uncertainty-aware workflow is separated into validation, calibration, and final
test commands. Model selection uses validation only; OOD density fitting uses calibration only;
the frozen candidate was evaluated once on test. Exact commands, artifact checksums, metrics, and
interpretation boundaries are recorded in
[VNAT Prototype and Uncertainty Evaluation](docs/uncertainty-modeling.md).

Run the complete local quality gate:

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy src tests
uv run --locked pytest --cov=parallax --cov-report=term-missing --cov-fail-under=100
uv build --no-sources
```

## Documentation

- [Engineering design](docs/engineering-design.md)
- [Project roadmap](docs/roadmap.md)
- [Reproducible data pipeline](docs/data-pipeline.md)
- [Capture-grouped splitting](docs/capture-splitting.md)
- [Initial validation baselines](docs/baseline-modeling.md)
- [Prototype and uncertainty evaluation](docs/uncertainty-modeling.md)
- [Dataset card](docs/dataset-card.md)
- [VNAT release manifest](data/manifests/vnat-release-1.json)
- [Model card](docs/model-card.md)
- [Architecture decisions](docs/adr/README.md)
- [Contribution workflow](CONTRIBUTING.md)

## Data and privacy

Raw captures, HDF5 datasets, generated feature files, trained models, and experiment artifacts are
excluded from Git. Public demonstrations will use public VNAT captures. Packet payloads are not
required by the intended feature pipeline and will not be persisted by the operational system.

## Source material

- [MIT Lincoln Laboratory VNAT dataset](https://www.ll.mit.edu/r-d/datasets/vpnnonvpn-network-application-traffic-dataset-vnat)
- [Extensible Machine Learning for Encrypted Network Traffic Application Labeling via Uncertainty Quantification](https://doi.org/10.1109/TAI.2023.3244168)
