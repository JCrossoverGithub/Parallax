# Parallax

Parallax is an uncertainty-aware network traffic monitoring system. It is being built to
classify broad application activity from observable encrypted traffic metadata such as packet
timing, size, and direction without decrypting or persisting packet payloads.

## Project status

Parallax is in its initial engineering phase. The repository currently establishes the package,
quality gates, design record, and CI foundation. No trained model or live monitoring capability
is claimed yet.

The first operational target is a deterministic replay pipeline:

```text
VNAT PCAP -> bidirectional flows -> observation windows -> features
          -> category prediction + confidence + OOD score -> operations dashboard
```

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
- [Dataset card](docs/dataset-card.md)
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
