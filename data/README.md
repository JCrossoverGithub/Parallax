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

Generate the accepted baseline validation report with:

```bash
uv run --locked parallax model validate-baselines \
  data/processed/vnat-release-1/features-release-compatible.parquet \
  data/processed/vnat-release-1/capture-splits.json \
  data/processed/vnat-release-1/baseline-validation.json
```

The `vnat-baseline-validation-report-1` artifact binds its metrics to the accepted feature and
split checksums. It records training-only preprocessing and model fitting, validation-only
metrics, and solver convergence. The accepted report is 4,747 bytes and has SHA-256
`e4aece1d97cbd55892971bcac7897269d8976024746d3a7881009e4d75ed6c38`. Independent API and CLI
runs produced identical report bytes. The processed report remains excluded from Git, and
calibration and test are not evaluated by this command.

The accepted uncertainty-modeling artifacts are also stored under
`data/processed/vnat-release-1/` and remain excluded from Git:

| Artifact | Purpose | SHA-256 |
| --- | --- | --- |
| `prototype-model.json` | Frozen training-derived model and inference state | `1c61678611043a7f70f02836ae23bbc7c1abf683b015e23ebafed4d941ecdef7` |
| `prototype-validation.json` | Candidate-selection evidence from validation only | `79b28ac5f3fe6988161fa6df7e5fa8831e6db1b46c1f5c0ef66a2a1148a3322f` |
| `prototype-ood-calibration.json` | Training geometry and calibration-only class KDEs | `af1d066ea2d96943c873b75e897ca4c7d910c79813a61b605eec8c7ec3c1fd9d` |
| `prototype-test.json` | One-shot final test evidence | `5c3c95b040fae4de2ffc27ac8eb143b8deee4c43c7c55a621843d5190b2813b4` |

Every loader verifies the expected artifact checksum and its upstream feature, split, model, and
calibration bindings. Export commands refuse to overwrite existing paths. The model and
validation report reproduced byte for byte in an independent run; the calibration artifact also
reproduced byte for byte. The final test report was deliberately generated once and matched its
CLI standard output exactly. It must not be replayed as a source of additional tuning evidence.

See [VNAT Prototype and Uncertainty Evaluation](../docs/uncertainty-modeling.md) for the complete
partition policy, configuration, results, and limitations.

The current Pandas loader still reads the complete fixed-format dataframe. Validation and export
used approximately 5.5 GiB of resident memory on the initial development machine. The accepted
release export completed in 30.58 seconds and produced a 99,585,954-byte Parquet file. Later
ingestion work may introduce a bounded conversion path for memory-constrained environments.
Feature extraction reads bounded Parquet row groups: the accepted run completed in 3 minutes
33.28 seconds with a peak resident set of 908,752 KiB. A second run produced the same Parquet
artifact byte for byte.

Small test fixtures may be added under `tests/fixtures/` only when their origin, license, purpose,
and expected result are documented.
