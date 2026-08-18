# Parallax data directory

Raw datasets are not stored in Git.

Planned local layout:

```text
data/
  raw/          Downloaded source files; immutable after verification
  processed/    Generated manifests, profiles, features, and split outputs
```

The first dataset is the MIT Lincoln Laboratory VNAT release. A later milestone will add a
versioned manifest containing authoritative source URLs, expected filenames, sizes, and SHA-256
checksums. Download and validation scripts must fail closed when a file does not match its
manifest.

Small test fixtures may be added under `tests/fixtures/` only when their origin, license, purpose,
and expected result are documented.
