# Parallax Model Card

## Status

Parallax has trained and accepted two deterministic validation baselines: a majority-class
classifier and class-balanced multinomial logistic regression. Neither is selected or packaged
for deployment. No probability calibration, OOD calibration, or final test evaluation has been
performed.

The complete baseline evidence is recorded in
[Initial VNAT Validation Baselines](baseline-modeling.md).

## Initial validation evidence

| Model | Accuracy | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Majority class | 0.734509 | 0.200000 | 0.169387 |
| Balanced logistic regression | 0.934223 | 0.733077 | 0.745291 |

The models use 129 release-compatible features. Preprocessing and estimators are fitted on 9,046
training windows from 95 captures and evaluated on 2,098 validation windows from 25 disjoint
captures. Calibration and test remain unevaluated.

The logistic model performs strongly on Chat, Streaming, and C2 in this partition but recalls only
8 of 42 VoIP windows. Thirty-four VoIP windows are classified as File Transfer. The validation
set contains only one VoIP capture, so this is an observed failure mode rather than a stable
estimate of VoIP performance.

## Current intended use

The baselines provide a reproducible reference floor for model development and verify the
manifest-bound training and reporting pipeline. They must not be used as an operational traffic
classifier or cited as final VNAT test performance.

## Current exclusions

- No final model has been selected.
- No model bundle has been serialized or approved for runtime loading.
- Predicted probabilities have not been calibrated.
- No OOD score or rejection threshold exists.
- No calibration or test metrics are available.
- Results do not establish performance on current enterprise or public Internet traffic.
- The VNAT C2 label represents benign SSH and RDP behavior, not malware ground truth.

## Required release information

- Model architecture and semantic version
- Intended and excluded uses
- Training, validation, calibration, and test manifest checksums
- Feature-schema and preprocessing versions
- Class mapping
- Closed-set and per-category metrics
- Probability-calibration metrics
- OOD evaluation and threshold-selection method
- VPN and non-VPN evaluation slices
- Training and inference hardware
- CPU inference latency and model size
- Robustness and feature-ablation results
- Ethical, privacy, and operational limitations

## Activation requirements

The runtime must reject a model bundle when its checksum fails, required artifacts are absent, the
feature schema is incompatible, feature order or dimensions differ, or OOD calibration artifacts do
not match the model.
