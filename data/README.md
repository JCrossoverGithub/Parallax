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

The current Pandas loader still reads the complete fixed-format dataframe. Validation and export
used approximately 5.5 GiB of resident memory on the initial development machine. The accepted
release export completed in 30.58 seconds and produced a 99,585,954-byte Parquet file. Later
ingestion work may introduce a bounded conversion path for memory-constrained environments.

Small test fixtures may be added under `tests/fixtures/` only when their origin, license, purpose,
and expected result are documented.
