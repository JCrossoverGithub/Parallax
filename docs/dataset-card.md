# VNAT Dataset Card for Parallax

## Status

Initial source summary. Schema, release files, checksums, class counts, and partition suitability
have not yet been independently validated by the Parallax pipeline.

## Source

- Dataset: VPN/Non-VPN Network Application Traffic Dataset (VNAT)
- Publisher: MIT Lincoln Laboratory
- Dataset page: <https://www.ll.mit.edu/r-d/datasets/vpnnonvpn-network-application-traffic-dataset-vnat>
- Related publication: <https://doi.org/10.1109/TAI.2023.3244168>

## Reported contents

The publisher reports labeled VPN and non-VPN captures for ten applications grouped into five
categories: Streaming, Voice over IP, Chat, Command and Control, and File Transfer. Available
formats include raw PCAP, packet-level HDF5 data, and a derived-feature HDF5 file.

## Intended Parallax use

- Develop deterministic ingestion and feature contracts.
- Train five-category classification baselines.
- Evaluate probability calibration and OOD detection.
- Compare VPN and non-VPN behavior.
- Replay public captures through the runtime pipeline.

## Known limitations requiring explicit treatment

- Traffic was collected in controlled virtual subnetworks.
- The dataset covers a limited number of applications and broad categories.
- Some traffic was scripted or manually generated.
- Dataset composition is imbalanced by bytes, connections, and capture duration.
- The Command and Control category contains SSH and RDP, not labeled malware activity.
- Filename-derived labels may not be equivalent to per-window behavioral ground truth.
- Multiple concurrent applications within one VPN observation are not represented by a multilabel
  target.
- Application and protocol behavior may have changed since collection.

## Leakage control

Parallax will use source-capture identity as the grouping unit for its primary partitions. No
capture may cross training, validation, calibration, and test partitions.

## Pending validation

- Confirm authoritative release filenames, sizes, and checksums.
- Inspect HDF5 keys, columns, dtypes, and missing values.
- Confirm whether source-capture identity survives feature extraction.
- Reproduce publisher-reported category totals where possible.
- Determine redistribution rules for small automated-test fixtures.
