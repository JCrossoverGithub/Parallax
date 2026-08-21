# Initial VNAT Validation Baselines

## Status

Parallax has implemented and accepted two deterministic reference classifiers for the five VNAT
traffic categories. Both models are fitted only on the capture-grouped training partition and
evaluated only on the validation partition. The calibration and test partitions have not been
evaluated.

These results establish a reference floor for later uncertainty-aware modeling. They are not a
final model-selection result, a deployment claim, or an estimate of performance on arbitrary
network traffic.

## Reproduce the report

Generate the immutable validation report with:

```bash
uv run --locked parallax model validate-baselines \
  data/processed/vnat-release-1/features-release-compatible.parquet \
  data/processed/vnat-release-1/capture-splits.json \
  data/processed/vnat-release-1/baseline-validation.json
```

The command verifies the trusted split-manifest checksum before parsing the manifest. The loader
then verifies the feature artifact's size, checksum, schema, Parquet metadata, capture assignments,
labels, window counts, feature dimensions, and finite values before returning read-only partition
arrays. Existing reports are never overwritten.

The API report, CLI report, and CLI standard output matched byte for byte in the accepted run.

## Experiment contract

| Item | Accepted value |
| --- | --- |
| Feature artifact | `vnat-feature-artifact-1` |
| Feature count | 129 `float32` values |
| Split manifest | `vnat-capture-split-manifest-1` |
| Training data | 9,046 windows from 95 captures |
| Validation data | 2,098 windows from 25 captures |
| Calibration data | Not evaluated |
| Test data | Not evaluated |
| Category order | Streaming, VoIP, Chat, C2, File Transfer |
| Scaler fit | Training partition only |
| Logistic regularization | `C=1.0` |
| Class weighting | Balanced |
| Solver | L-BFGS |
| Maximum iterations | 2,000 |
| Random seed | 17 |

The two reference models are:

1. A majority-class classifier using the observed training prior.
2. Multinomial logistic regression following a training-only `StandardScaler`.

No hyperparameter search was performed for this initial reference result.

## Validation results

| Model | Accuracy | Balanced accuracy | Micro F1 | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Majority class | 0.734509 | 0.200000 | 0.734509 | 0.169387 |
| Balanced logistic regression | 0.934223 | 0.733077 | 0.934223 | 0.745291 |

The majority classifier's 73.45% accuracy is not evidence of useful classification. Chat accounts
for 1,541 of the 2,098 validation windows, so predicting Chat for every example yields high raw
accuracy while producing 20% balanced accuracy and zero recall for four categories.

### Logistic regression by category

| Category | Precision | Recall | F1 | Validation support |
| --- | ---: | ---: | ---: | ---: |
| Streaming | 0.777778 | 0.950000 | 0.855305 | 280 |
| VoIP | 1.000000 | 0.190476 | 0.320000 | 42 |
| Chat | 0.994716 | 0.977287 | 0.985925 | 1,541 |
| C2 | 0.913978 | 0.833333 | 0.871795 | 102 |
| File Transfer | 0.673759 | 0.714286 | 0.693431 | 133 |

### Logistic regression confusion matrix

Rows are true categories and columns are predicted categories.

| True / predicted | Streaming | VoIP | Chat | C2 | File Transfer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Streaming | 266 | 0 | 4 | 3 | 7 |
| VoIP | 0 | 8 | 0 | 0 | 34 |
| Chat | 32 | 0 | 1,506 | 1 | 2 |
| C2 | 10 | 0 | 4 | 85 | 3 |
| File Transfer | 34 | 0 | 0 | 4 | 95 |

The most important observed failure is VoIP recall. The model classified 34 of 42 VoIP windows
as File Transfer. That result is based on one held-out VoIP capture and must not be generalized
beyond this validation partition. Streaming and File Transfer also show a bidirectional confusion
pattern.

## Accepted report

| Property | Value |
| --- | --- |
| Schema | `vnat-baseline-validation-report-1` |
| File | `baseline-validation.json` |
| Size | 4,747 bytes |
| SHA-256 | `e4aece1d97cbd55892971bcac7897269d8976024746d3a7881009e4d75ed6c38` |
| Logistic iterations | 488 of 2,000 |
| API runtime | 57.20 seconds |
| CLI replay runtime | 47.03 seconds |
| Peak resident memory | Approximately 300 MiB |

The report binds the result to these inputs:

| Artifact | SHA-256 |
| --- | --- |
| `features-release-compatible.parquet` | `611dcb63c66e04f461fb7500f96762c1722a06fbdde700dc5aa42ae021e39f16` |
| `capture-splits.json` | `a1aeee5f118c1e3bd57aa7232eb009634c098b9025571786614798953ebd8a0f` |

The exporter refuses to publish when the logistic solver reaches its configured iteration limit.
It records complete configuration, convergence evidence, provenance, aggregate metrics,
per-category metrics, and confusion matrices in deterministically encoded JSON.

## Interpretation boundaries

- Metrics are window-weighted; they are not averages over captures.
- Capture grouping prevents direct capture leakage, but several validation categories contain very
  few source captures.
- The release-compatible feature representation intentionally preserves the publisher's observed
  directional byte-total defect.
- The logistic baseline has not been probability-calibrated and exposes no OOD score.
- Application-level, VPN-status, robustness, and temporal-distribution slices are not yet reported.
- No serialized model bundle or runtime classifier has been approved.
- The test partition remains reserved for one final evaluation after model and threshold selection.

## Next modeling stage

The next stage will compare uncertainty-aware candidates against this validation reference. Model
selection remains restricted to training and validation data. Probability and OOD calibration will
use only the calibration partition after a candidate is selected; the final test partition will
remain untouched until the complete evaluation procedure is frozen.
