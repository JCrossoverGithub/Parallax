# Parallax Release Reproducibility

## Scope

Parallax intentionally does not commit:

- raw VNAT HDF5 data;
- the full VNAT PCAP archive;
- generated Parquet feature artifacts;
- trained model bundles;
- OOD calibration artifacts;
- operational SQLite databases.

The repository instead records the trusted source identities, deterministic
generation commands, artifact schemas, checksums, scientific policy, tests,
and compact acceptance evidence required to audit the project.

This document separates two reproducibility goals:

1. reproducing the repository software and quality gate from a clean clone;
2. reconstructing the accepted runtime artifacts from the public VNAT source.

The accepted one-shot final test is historical scientific evidence and should
not be rerun merely to prepare the live or replay demonstration.

## Clean-clone acceptance

Milestone 7 clean-clone acceptance was performed from the remote
`feature/portfolio-release` branch at commit:

```text
136d0562555caebd4b3313271f3bd4ca50499e2d
```

The acceptance clone began without a `.venv` or frontend `node_modules`
directory.

From committed lockfiles alone it successfully:

- created a fresh Python 3.12 environment;
- installed all locked Python dependencies;
- passed Ruff;
- passed mypy;
- passed 1,120 Python tests;
- achieved 100% Python statement and branch coverage;
- built the Python source distribution and wheel;
- executed the Parallax CLI smoke test;
- installed the Angular application with `npm ci`;
- passed all 79 Angular tests;
- built the Angular production bundle;
- validated the live-soak shell script;
- validated the compact live-soak JSON evidence;
- resolved every local Markdown link and image target;
- verified all three portfolio screenshots;
- finished with a clean Git working tree.

No tracked PCAP, HDF5, Parquet, SQLite, model-weight, checkpoint, or environment
file was found by the release artifact audit.

## Source dataset

Parallax release 1 uses the MIT Lincoln Laboratory VPN/Non-VPN Network
Application Traffic Dataset (VNAT).

Publisher page:

<https://www.ll.mit.edu/r-d/datasets/vpnnonvpn-network-application-traffic-dataset-vnat>

The repository source manifest is:

```text
data/manifests/vnat-release-1.json
```

The raw HDF5 input used by the accepted pipeline is:

```text
VNAT_Dataframe_release_1.h5
```

Recorded source URL:

<https://archive.ll.mit.edu/datasets/vnat/VNAT_Dataframe_release_1.h5>

Expected size:

```text
1,045,436,008 bytes
```

Expected SHA-256:

```text
5d0c3d76cd292f19e25b5229719264bc1ddd71920a20bb27a7dec6c7138914de
```

The checksum must be verified before the HDF5 object data is deserialized.

## Prepare the repository

From a clean clone:

```bash
uv sync --locked --all-groups

mkdir -p \
  data/raw/vnat \
  data/processed/vnat-release-1
```

Obtain `VNAT_Dataframe_release_1.h5` from the recorded MIT Lincoln Laboratory
source and place it at:

```text
data/raw/vnat/VNAT_Dataframe_release_1.h5
```

Verify its identity:

```bash
sha256sum \
  data/raw/vnat/VNAT_Dataframe_release_1.h5
```

The digest must equal:

```text
5d0c3d76cd292f19e25b5229719264bc1ddd71920a20bb27a7dec6c7138914de
```

Parallax also performs its own fail-closed checksum validation before reading
the trusted HDF5 representation.

## Reconstruct observation windows

Generate the accepted release-compatible observation-window artifact:

```bash
uv run --locked parallax dataset extract-windows \
  data/raw/vnat/VNAT_Dataframe_release_1.h5 \
  data/processed/vnat-release-1/windows-release-compatible.parquet
```

Accepted SHA-256:

```text
06f00af45cb635241575d251331e7ce96273212dba087e38b7610876ec9984d8
```

This policy preserves the accepted release-compatible interpretation used by
the frozen experiment.

## Reconstruct the 129-feature artifact

```bash
uv run --locked parallax dataset extract-features \
  data/processed/vnat-release-1/windows-release-compatible.parquet \
  data/processed/vnat-release-1/features-release-compatible.parquet
```

Accepted SHA-256:

```text
611dcb63c66e04f461fb7500f96762c1722a06fbdde700dc5aa42ae021e39f16
```

The release-compatible feature policy intentionally preserves the
publisher-observed directional byte-total behavior documented in the dataset
and model materials.

It must not be silently replaced with the corrected policy when reconstructing
the accepted experiment.

## Reconstruct the capture-grouped split

```bash
uv run --locked parallax dataset split-features \
  data/processed/vnat-release-1/features-release-compatible.parquet \
  data/processed/vnat-release-1/capture-splits.json
```

Accepted SHA-256:

```text
a1aeee5f118c1e3bd57aa7232eb009634c098b9025571786614798953ebd8a0f
```

Every source capture remains wholly within one of:

```text
training
validation
calibration
test
```

The accepted split contains 162 eligible captures.

## Reconstruct the frozen prototype candidate

Generate the deterministic training/validation candidate:

```bash
uv run --locked parallax model validate-prototype \
  data/processed/vnat-release-1/features-release-compatible.parquet \
  data/processed/vnat-release-1/capture-splits.json \
  data/processed/vnat-release-1/prototype-model.json \
  data/processed/vnat-release-1/prototype-validation.json
```

Accepted model-bundle SHA-256:

```text
1c61678611043a7f70f02836ae23bbc7c1abf683b015e23ebafed4d941ecdef7
```

Accepted validation-report SHA-256:

```text
79b28ac5f3fe6988161fa6df7e5fa8831e6db1b46c1f5c0ef66a2a1148a3322f
```

The accepted procedure uses deterministic CPU-only PyTorch execution.

The recorded experiment also produced byte-identical model and validation
artifacts during an independent deterministic replay.

## Reconstruct the OOD calibration artifact

After verifying the model-bundle checksum, generate the calibration-only OOD
artifact:

```bash
uv run --locked parallax model calibrate-prototype \
  data/processed/vnat-release-1/features-release-compatible.parquet \
  data/processed/vnat-release-1/capture-splits.json \
  data/processed/vnat-release-1/prototype-model.json \
  data/processed/vnat-release-1/prototype-ood-calibration.json \
  --expected-model-bundle-sha256 \
  1c61678611043a7f70f02836ae23bbc7c1abf683b015e23ebafed4d941ecdef7
```

Accepted calibration SHA-256:

```text
af1d066ea2d96943c873b75e897ca4c7d910c79813a61b605eec8c7ec3c1fd9d
```

Calibration uses only the frozen calibration partition and does not perform
model selection.

## Verify the runtime artifacts

Before starting the accepted operator runtime:

```bash
sha256sum \
  data/processed/vnat-release-1/prototype-model.json \
  data/processed/vnat-release-1/prototype-ood-calibration.json
```

The required identities are:

```text
model:
1c61678611043a7f70f02836ae23bbc7c1abf683b015e23ebafed4d941ecdef7

OOD calibration:
af1d066ea2d96943c873b75e897ca4c7d910c79813a61b605eec8c7ec3c1fd9d
```

The operator also verifies the feature-artifact and split-manifest identities
embedded in the accepted artifact chain.

A mismatch fails runtime activation rather than silently loading another
candidate.

## Do not rerun the accepted final test for demo preparation

The accepted candidate was evaluated once on the capture-held-out test
partition after its model, calibration, thresholds, and reporting policy were
frozen.

The historical accepted test-report SHA-256 is:

```text
5c3c95b040fae4de2ffc27ac8eb143b8deee4c43c7c55a621843d5190b2813b4
```

The documented `evaluate-prototype` command remains in
`docs/uncertainty-modeling.md` for auditability.

It should not be rerun merely to recreate the live/replay runtime because
doing so creates another opportunity to inspect the already-consumed test
partition.

Runtime preparation requires the accepted model and calibration artifacts,
not another final evaluation.

## PCAP replay data

The full VNAT PCAP archive is optional for reconstructing the accepted model
and for demonstrating live sensing.

It is required for demonstrations that use VNAT PCAP replay or for repeating
the selected raw-PCAP parity checks.

Recorded archive identity:

```text
filename:
VNAT_release_1.zip

size:
34,523,209,689 bytes

locally calculated SHA-256:
42388ec5821bd1d9c9d0cad437160476e9f521d2e49f7c5573377b085705c883
```

No independently authenticated publisher checksum was recorded for this
archive, so Parallax describes this value only as the local identity of the
archive used during development.

Obtain the archive through the MIT Lincoln Laboratory VNAT dataset page.

The selected acceptance-capture names and checksums are recorded in
`docs/dataset-card.md`.

## What can be reproduced without external data

A clean repository clone can execute the complete automated quality gate
without:

- the full VNAT dataset;
- trained model files;
- raw-capture privileges;
- GPU hardware;
- the original development workspace.

This is the same boundary enforced by CI.

The large external artifacts are required only when reconstructing the
scientific artifact chain or running data-dependent replay/live inference.

## Related documentation

- [Dataset Card](dataset-card.md)
- [Reproducible Data Pipeline](data-pipeline.md)
- [Capture-Grouped Splitting](capture-splitting.md)
- [VNAT Prototype and Uncertainty Evaluation](uncertainty-modeling.md)
- [Model Card](model-card.md)
- [Portfolio Demo Runbook](demo-runbook.md)
- [Milestone 6 Acceptance](milestone-6-acceptance.md)
