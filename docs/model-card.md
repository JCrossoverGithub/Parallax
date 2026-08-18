# Parallax Model Card

## Status

No Parallax model has been trained or selected. This file defines the evidence that must accompany
a future model release and intentionally contains no performance claims.

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
