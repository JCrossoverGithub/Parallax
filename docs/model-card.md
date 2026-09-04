# Parallax Model Card

## Status

Parallax has selected and evaluated one uncertainty-aware VNAT prototype candidate. The model was
trained on the capture-grouped training partition, selected on validation, frozen into a
checksum-bound bundle, fitted with relative-Mahalanobis OOD density estimates on calibration,
and evaluated once on test after the procedure was committed.

This is a research artifact and future runtime candidate. It is not approved as a production
classifier, intrusion-detection system, malware detector, or general Internet-traffic model.
Complete evidence is recorded in
[VNAT Prototype and Uncertainty Evaluation](uncertainty-modeling.md).

## Model identity

| Property | Value |
| --- | --- |
| Model schema | `vnat-prototype-model-1` |
| Bundle schema | `vnat-prototype-model-bundle-1` |
| Bundle SHA-256 | `1c61678611043a7f70f02836ae23bbc7c1abf683b015e23ebafed4d941ecdef7` |
| Calibration schema | `vnat-prototype-ood-calibration-artifact-1` |
| Calibration SHA-256 | `af1d066ea2d96943c873b75e897ca4c7d910c79813a61b605eec8c7ec3c1fd9d` |
| Feature count | 129 |
| Embedding dimension | 64 |
| Categories | Streaming, VoIP, Chat, C2, File Transfer |
| Execution target | Deterministic CPU-only PyTorch |

The network has four fully connected 64-unit ReLU layers and 25% dropout before layers three and
four. Training uses a training-only standardizer, 20,000 deterministic episodes, five support
examples per class, 512 queries per episode, Adam at `1e-3`, and seed 17.

Closed-set probabilities use negative diagonal-Mahalanobis distance to training-derived class
statistics. They are raw softmax outputs and are not temperature-scaled.

## Evaluation policy

| Partition | Role |
| --- | --- |
| Training | Fit preprocessing, network parameters, class statistics, and OOD geometry |
| Validation | Candidate selection and closed-set characterization |
| Calibration | Fit class-conditional OOD KDEs only |
| Test | One final evaluation after model and policy freeze |

Capture identifiers are disjoint across partitions. The final report was generated from commit
`df785d4`; its SHA-256 is
`5c3c95b040fae4de2ffc27ac8eb143b8deee4c43c7c55a621843d5190b2813b4`. No post-test model
selection or threshold change was performed.

## Closed-set performance

| Metric | Validation | Test |
| --- | ---: | ---: |
| Accuracy | 0.934700 | 0.867047 |
| Balanced accuracy | 0.878908 | 0.820828 |
| Macro F1 | 0.808133 | 0.712857 |
| Expected calibration error | 0.065399 | 0.132825 |

### Per-category test performance

| Category | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Streaming | 0.783898 | 0.725490 | 0.753564 | 255 |
| VoIP | 0.142857 | 0.977273 | 0.249275 | 44 |
| Chat | 0.998089 | 0.988644 | 0.993344 | 1,585 |
| C2 | 0.990741 | 0.484163 | 0.650456 | 442 |
| File Transfer | 0.906977 | 0.928571 | 0.917647 | 126 |

The primary failure is systematic overprediction of VoIP. Of the 301 VoIP predictions, only 43
are correct; 186 are C2 and 63 are Streaming. The model therefore achieves high VoIP recall but
low precision, while missing more than half of true C2 windows. Chat and File Transfer are much
more stable on the accepted test partition.

The validation-to-test declines in balanced accuracy and macro F1, together with ECE increasing
from 0.065 to 0.133, demonstrate material capture-held-out generalization and calibration risk.

## OOD behavior

The OOD artifact fits full-covariance relative-Mahalanobis geometry from frozen training support
and one SciPy-compatible Gaussian KDE per class from calibration scores. OOD score is one minus
the fitted upper-tail probability.

The known-traffic test set produces:

| Fixed threshold | Flagged windows | False-positive rate |
| --- | ---: | ---: |
| 0.95 | 11 of 2,452 | 0.004486 |
| 0.99 | 0 of 2,452 | 0.000000 |

All 11 flags at 0.95 are C2 windows. No true OOD examples are present, so these values measure
only in-distribution false-positive behavior. They provide no OOD AUROC, recall, or detection-power
claim. The Scott-bandwidth KDEs are also sensitive to extreme calibration-score tails and compress
several class score distributions near the middle.

## Intended use

- Reproduce and audit the VNAT uncertainty-aware modeling procedure.
- Provide a frozen model and calibration contract for deterministic PCAP replay development.
- Display category, raw confidence, and OOD score as separate outputs in a controlled demo.
- Serve as a reference for separately pre-registered OOD and robustness experiments.

## Excluded use and claims

- Do not use the model for automated blocking, enforcement, authorization, or safety decisions.
- Do not describe VNAT C2 labels as malicious command-and-control detection; they represent benign
  SSH and RDP traffic.
- Do not generalize results to arbitrary applications, networks, users, VPNs, or current traffic.
- Do not describe raw class probabilities as calibrated probabilities.
- Do not claim OOD detection performance from the known-traffic false-positive test.
- Do not use the final test report to tune or select another model under this split.
- Do not activate the model against an incompatible feature schema or unverified artifact chain.

## Data and representation limitations

- VNAT is a controlled, historical dataset containing ten applications and five categories.
- Metrics are window-weighted rather than capture-weighted.
- Some categories have few source captures, especially VoIP and Streaming.
- The release-compatible features intentionally preserve the publisher's observed directional
  byte-total defect.
- Filename-derived labels describe the source capture and may not perfectly describe every window.
- Application, VPN-status, robustness, and external-dataset slices remain unreported.

## Activation requirements

A runtime must verify the model bundle checksum, calibration checksum, upstream feature and split
bindings, schema versions, ordered feature names, feature dimension, finite values, and category
order before scoring. It must report confidence and OOD score independently and preserve the
active artifact identities with every prediction.

Selected-capture raw-PCAP feature parity is now satisfied: exact offline/runtime 129-feature
agreement was demonstrated on eligible SSH and VoIP VNAT windows. Exact raw flow reconstruction
across the acceptance captures also covered TCP, UDP, and ICMP metadata; the VoIP capture's 404
ICMP packets did not themselves produce eligible feature windows. Operational activation still
requires checksum-verified model and calibration artifacts at runtime, bounded inference
measurements, deterministic replay behavior, failure-mode testing, and clear UI communication of
the model's limitations.
