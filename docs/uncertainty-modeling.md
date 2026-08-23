# VNAT Prototype and Uncertainty Evaluation

## Status

Parallax completed its pre-registered uncertainty-modeling experiment on 23 August 2026. The
candidate was selected using the capture-held-out validation partition, frozen as an immutable
model bundle, fitted with relative-Mahalanobis OOD density estimates using the calibration
partition, and evaluated once on the test partition only after the evaluation policy and report
implementation were committed.

The final test result is evidence about this fixed VNAT split and implementation. It is not a
deployment claim, proof of performance on current network traffic, or measurement of OOD
detection power.

## Artifact chain

| Artifact | Schema | SHA-256 |
| --- | --- | --- |
| `features-release-compatible.parquet` | `vnat-feature-artifact-1` | `611dcb63c66e04f461fb7500f96762c1722a06fbdde700dc5aa42ae021e39f16` |
| `capture-splits.json` | `vnat-capture-split-manifest-1` | `a1aeee5f118c1e3bd57aa7232eb009634c098b9025571786614798953ebd8a0f` |
| `prototype-model.json` | `vnat-prototype-model-bundle-1` | `1c61678611043a7f70f02836ae23bbc7c1abf683b015e23ebafed4d941ecdef7` |
| `prototype-validation.json` | `vnat-prototype-validation-report-1` | `79b28ac5f3fe6988161fa6df7e5fa8831e6db1b46c1f5c0ef66a2a1148a3322f` |
| `prototype-ood-calibration.json` | `vnat-prototype-ood-calibration-artifact-1` | `af1d066ea2d96943c873b75e897ca4c7d910c79813a61b605eec8c7ec3c1fd9d` |
| `prototype-test.json` | `vnat-prototype-test-report-1` | `5c3c95b040fae4de2ffc27ac8eb143b8deee4c43c7c55a621843d5190b2813b4` |

Generated data and model artifacts remain excluded from Git. The repository records their
schemas, checksums, commands, policies, and measured results.

## Recorded commands

The model candidate and validation report were generated together:

```bash
uv run --locked parallax model validate-prototype \
  data/processed/vnat-release-1/features-release-compatible.parquet \
  data/processed/vnat-release-1/capture-splits.json \
  data/processed/vnat-release-1/prototype-model.json \
  data/processed/vnat-release-1/prototype-validation.json
```

The frozen model checksum was then required for calibration:

```bash
uv run --locked parallax model calibrate-prototype \
  data/processed/vnat-release-1/features-release-compatible.parquet \
  data/processed/vnat-release-1/capture-splits.json \
  data/processed/vnat-release-1/prototype-model.json \
  data/processed/vnat-release-1/prototype-ood-calibration.json \
  --expected-model-bundle-sha256 \
  1c61678611043a7f70f02836ae23bbc7c1abf683b015e23ebafed4d941ecdef7
```

The final test command required the frozen manifest, model, and calibration checksums:

```bash
uv run --locked parallax model evaluate-prototype \
  data/processed/vnat-release-1/features-release-compatible.parquet \
  data/processed/vnat-release-1/capture-splits.json \
  data/processed/vnat-release-1/prototype-model.json \
  data/processed/vnat-release-1/prototype-ood-calibration.json \
  data/processed/vnat-release-1/prototype-test.json \
  --expected-manifest-sha256 \
  a1aeee5f118c1e3bd57aa7232eb009634c098b9025571786614798953ebd8a0f \
  --expected-model-bundle-sha256 \
  1c61678611043a7f70f02836ae23bbc7c1abf683b015e23ebafed4d941ecdef7 \
  --expected-calibration-sha256 \
  af1d066ea2d96943c873b75e897ca4c7d910c79813a61b605eec8c7ec3c1fd9d
```

This final command is recorded for auditability and must not be replayed as another opportunity
to inspect or tune against test data. The accepted report is the immutable evidence.

## Partition discipline

| Partition | Windows | Captures | Permitted role |
| --- | ---: | ---: | --- |
| Training | 9,046 | 95 | Fit the standardizer, embedding network, class statistics, and OOD geometry |
| Validation | 2,098 | 25 | Select and characterize the candidate |
| Calibration | 1,499 | 16 | Fit class-conditional OOD density estimates only |
| Test | 2,452 | 26 | One final evaluation after the candidate and policy were frozen |

No capture crosses partitions. Calibration performed no model selection and did not evaluate
validation or test. The final test run performed no fitting or post-test model selection.

## Model contract

The model consumes the ordered 129-value release-compatible feature vector. A `StandardScaler`
is fitted on training data only. The embedding network contains four fully connected 64-unit
layers with ReLU activations, a 64-dimensional output, and 25% dropout before the third and fourth
linear layers.

Training uses deterministic CPU-only PyTorch execution with:

| Setting | Value |
| --- | --- |
| Support per class per episode | 5 |
| Query examples per episode | 512 |
| Episodes | 20,000 |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Weight decay | `0` |
| NumPy and PyTorch seeds | 17 |
| CPU threads | 1 |
| Deterministic algorithms | Enabled |

The accepted run reduced episodic loss from `1.538605` to `0.001864`, with a minimum observed
loss of `0.000002`. Model and report generation completed in 76.06 seconds with approximately
538 MiB peak resident memory. An independent replay produced byte-identical model and validation
artifacts.

Closed-set inference samples up to 100 training embeddings per class using seed 17. It calculates
class means and population diagonal covariances, then applies softmax to negative diagonal
Mahalanobis distances. Probabilities are not temperature-scaled.

## Validation selection evidence

| Model | Accuracy | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Balanced logistic regression | 0.934223 | 0.733077 | 0.745291 |
| Prototype candidate | 0.934700 | 0.878908 | 0.808133 |

The prototype preserved essentially the same raw accuracy while improving balanced accuracy by
`0.145831` and macro F1 by `0.062842`. Its 15-bin expected calibration error was `0.065399`.

The main validation tradeoff was a shift toward VoIP predictions. VoIP recall increased from
`0.190476` to `1.000000`, but precision decreased to `0.304348`. Streaming recall decreased from
`0.950000` to `0.746429`. This tradeoff was accepted before calibration or test access.

## OOD calibration

OOD scoring follows the paper-aligned relative-Mahalanobis procedure:

1. Reconstruct up to 100 frozen training support embeddings per class.
2. Fit full population class covariances and one global covariance.
3. Calculate class Mahalanobis distance minus global Mahalanobis distance.
4. Fit one univariate Gaussian KDE to each class's calibration scores.
5. Report one minus the fitted upper-tail probability, equivalently the KDE CDF.

The KDE implementation matches SciPy `gaussian_kde` with its default Scott bandwidth to numerical
precision. Calibration used all 1,499 calibration windows:

| Category | Samples | Scott bandwidth |
| --- | ---: | ---: |
| Streaming | 184 | 5,572.944345 |
| VoIP | 23 | 1,320.444293 |
| Chat | 1,075 | 2,318.366150 |
| C2 | 131 | 425.962485 |
| File Transfer | 86 | 4,604.975510 |

Several score distributions are highly skewed. Scott bandwidth is sensitive to the extreme
tails and compresses much of the fitted CDF near the middle for Streaming, Chat, and File
Transfer. A robust alternative was inspected on calibration data only but was not substituted
after the procedure was frozen. This behavior is a documented limitation of the accepted
calibration artifact.

## Final test result

The test report was generated once from commit `df785d4`. The command verified the feature,
manifest, model, and calibration checksum chain before accessing the test partition. The report
and CLI standard output matched byte for byte.

| Metric | Validation | Test | Test minus validation |
| --- | ---: | ---: | ---: |
| Accuracy | 0.934700 | 0.867047 | -0.067652 |
| Balanced accuracy | 0.878908 | 0.820828 | -0.058080 |
| Macro F1 | 0.808133 | 0.712857 | -0.095276 |
| Expected calibration error | 0.065399 | 0.132825 | +0.067426 |

The gap shows weaker capture-held-out generalization and substantially worse probability
calibration than validation suggested. No post-test change is permitted for this candidate.

### Per-category test metrics

| Category | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Streaming | 0.783898 | 0.725490 | 0.753564 | 255 |
| VoIP | 0.142857 | 0.977273 | 0.249275 | 44 |
| Chat | 0.998089 | 0.988644 | 0.993344 | 1,585 |
| C2 | 0.990741 | 0.484163 | 0.650456 | 442 |
| File Transfer | 0.906977 | 0.928571 | 0.917647 | 126 |

### Test confusion matrix

Rows are true categories and columns are predicted categories.

| True / predicted | Streaming | VoIP | Chat | C2 | File Transfer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Streaming | 185 | 63 | 2 | 1 | 4 |
| VoIP | 0 | 43 | 0 | 1 | 0 |
| Chat | 16 | 2 | 1,567 | 0 | 0 |
| C2 | 33 | 186 | 1 | 214 | 8 |
| File Transfer | 2 | 7 | 0 | 0 | 117 |

The dominant failure is overprediction of VoIP. The classifier captures 43 of 44 true VoIP
windows, but 186 C2 windows and 63 Streaming windows are also assigned to VoIP. This produces
high VoIP recall with very low precision and reduces C2 recall to 48.42%. Chat and File Transfer
remain strong on this partition.

### Known-traffic OOD behavior

The test partition contains only known VNAT categories. OOD thresholds therefore measure false
positive behavior, not detection power.

| Threshold | Flagged known windows | False-positive rate |
| --- | ---: | ---: |
| 0.95 | 11 of 2,452 | 0.004486 |
| 0.99 | 0 of 2,452 | 0.000000 |

All 11 flags at 0.95 are C2 windows, representing 2.49% of the 442 test C2 windows. The maximum
known-traffic OOD score is `0.989510`; the 95th and 99th percentiles are `0.595652` and
`0.711235`. These results show low false-positive rates on this test partition. They do not show
that the model detects unseen applications or categories.

## Interpretation boundaries

- Metrics are calculated per window, not per capture.
- The five categories and ten applications come from a controlled historical dataset.
- The C2 category represents benign SSH and RDP traffic, not malicious command-and-control
  ground truth.
- The release-compatible feature artifact preserves the publisher's directional byte-total
  defect for reproduction purposes.
- Class probabilities are raw model outputs; ECE worsened on test and they must not be described
  as calibrated probabilities.
- OOD calibration uses class labels from the calibration partition.
- No true OOD examples were present in the final test, so AUROC, OOD recall, and detection power
  are unavailable.
- Thresholds 0.95 and 0.99 were frozen before test evaluation and were not adjusted afterward.
- The final test result must not be used to tune or select another candidate under the same split.

## Next stage

Milestone 3 begins with verified raw PCAP ingestion, deterministic bidirectional flow and window
construction, and golden parity between offline and runtime feature generation. OOD detection
power requires a separately pre-registered application-held-out or external-dataset experiment;
it is not inferred from the known-traffic test result.
