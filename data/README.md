# Parallax data directory

Raw datasets are not stored in Git.

```text
data/
  manifests/    Versioned source metadata, sizes, checksums, and observed schemas
  raw/          Downloaded source files; immutable after checksum verification
  processed/    Generated profiles, windows, features, and split outputs
```

VNAT release 1 metadata is tracked in
[`manifests/vnat-release-1.json`](manifests/vnat-release-1.json). Downloaded HDF5 and PCAP files
belong under `data/raw/vnat/` and remain excluded from version control.

Validate the raw dataframe with:

```bash
uv run --locked parallax dataset inspect \
  data/raw/vnat/VNAT_Dataframe_release_1.h5
```

The release checksum is verified before Pandas deserializes the fixed-format HDF5 object data.
Do not override `--expected-sha256` with a value obtained from the same untrusted download. A
trusted digest must come from the versioned manifest or another independently authenticated
source.

Create the default release-compatible window artifact with:

```bash
uv run --locked parallax dataset extract-windows \
  data/raw/vnat/VNAT_Dataframe_release_1.h5 \
  data/processed/vnat-release-1/windows-release-compatible.parquet
```

The exporter writes schema version `vnat-window-1` with Zstandard compression and creates a
companion `.manifest.json` file. Both outputs remain excluded from Git. Final-looking paths are
created only after extraction succeeds, and existing outputs are never overwritten.

Calculate the release-compatible feature artifact with:

```bash
uv run --locked parallax dataset extract-features \
  data/processed/vnat-release-1/windows-release-compatible.parquet \
  data/processed/vnat-release-1/features-release-compatible.parquet
```

This exporter writes schema version `vnat-feature-artifact-1`. Each row retains its window,
capture, flow, application, category, and VPN-status identity beside the ordered 129-feature
vector. The companion manifest records the source artifact checksum, feature configuration,
output checksum, and audited distributions. The accepted artifact has 15,095 rows in 236 row
groups, is 14,702,124 bytes, and has SHA-256
`611dcb63c66e04f461fb7500f96762c1722a06fbdde700dc5aa42ae021e39f16`.

The default byte-total policy intentionally reproduces the released dataframe, whose directional
byte-total columns duplicate its packet-count columns. Pass `--byte-total-policy corrected` when
creating an artifact for a new model rather than comparing against the publication.

Create the primary capture-grouped partition manifest with:

```bash
uv run --locked parallax dataset split-features \
  data/processed/vnat-release-1/features-release-compatible.parquet \
  data/processed/vnat-release-1/capture-splits.json
```

The splitter verifies the accepted feature artifact before reading it and scans only capture and
label columns in bounded batches. The `vnat-capture-split-manifest-1` output contains all 162
capture assignments across training, validation, calibration, and test partitions. It is 38,372
bytes and has SHA-256
`a1aeee5f118c1e3bd57aa7232eb009634c098b9025571786614798953ebd8a0f`. The output remains excluded
from Git with the other processed artifacts. Existing manifests are never overwritten, and a
feasible solution is not published unless solver optimality is proven.

The current Pandas loader still reads the complete fixed-format dataframe. Validation and export
used approximately 5.5 GiB of resident memory on the initial development machine. The accepted
release export completed in 30.58 seconds and produced a 99,585,954-byte Parquet file. Later
ingestion work may introduce a bounded conversion path for memory-constrained environments.
Feature extraction reads bounded Parquet row groups: the accepted run completed in 3 minutes
33.28 seconds with a peak resident set of 908,752 KiB. A second run produced the same Parquet
artifact byte for byte.

Small test fixtures may be added under `tests/fixtures/` only when their origin, license, purpose,
and expected result are documented.
