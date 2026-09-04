# VNAT Dataset Card for Parallax

## Status

VNAT release 1 was independently downloaded, checksummed, structurally inspected,
deterministically windowed, and exported by Parallax on 18 August 2026. Its 129-feature
representation was independently implemented and exported on 19 August 2026. The primary
capture-grouped model-development split was generated and accepted on 21 August 2026. Initial
majority-class and balanced logistic-regression baselines were subsequently evaluated on the
capture-held-out validation partition. A frozen prototype candidate was selected on validation,
fitted with OOD densities on calibration, and evaluated once on the capture-held-out test
partition on 23 August 2026. The full VNAT PCAP archive was subsequently downloaded and assigned
a local SHA-256 identity, and selected SSH and VoIP training captures passed exact raw-PCAP flow,
window, and 129-feature parity acceptance on 3 September 2026.

## Source and version

- Dataset: VPN/Non-VPN Network Application Traffic Dataset (VNAT)
- Publisher: MIT Lincoln Laboratory
- Dataset page: <https://www.ll.mit.edu/r-d/datasets/vpnnonvpn-network-application-traffic-dataset-vnat>
- Related publication: <https://doi.org/10.1109/TAI.2023.3244168>
- Release manifest: [`data/manifests/vnat-release-1.json`](../data/manifests/vnat-release-1.json)

| Artifact | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `VNAT_Dataframe_release_1.h5` | 1,045,436,008 | `5d0c3d76cd292f19e25b5229719264bc1ddd71920a20bb27a7dec6c7138914de` |
| `VNAT_Feature_Dataframe_release_1.h5` | 9,005,973 | `9e435b50743bec6eed288e9707878f624861dcdd7700165955e2509fa47d0f30` |
| `VNAT_release_1.zip` | 34,523,209,689 | `42388ec5821bd1d9c9d0cad437160476e9f521d2e49f7c5573377b085705c883` |

The PCAP archive digest above is the locally calculated identity of the archive used by Parallax.
No independently authenticated publisher checksum has been recorded for that archive.

### Selected raw-PCAP acceptance captures

| Capture | Partition | Size (bytes) | SHA-256 |
| --- | --- | ---: | --- |
| `nonvpn_ssh_capture4.pcap` | Training | 100,261 | `770e2eb1b17c717084e8bd96cc6a5c27a93918e7272db5fd253dee6282b73f91` |
| `nonvpn_voip_capture2.pcap` | Training | 10,742,119 | `46fc6a1b79bc7418183bfdbe88ba18b928e99a399121a2c0db3ce70c03796174` |

For `nonvpn_ssh_capture4.pcap`, Parallax parsed 626 PCAP records, reconstructed all 117 HDF5
connections exactly, produced five eligible windows, and reproduced all five offline 129-feature
vectors exactly with maximum feature difference `0.0`.

For `nonvpn_voip_capture2.pcap`, Parallax parsed 119,103 PCAP records comprising 118,699 UDP and
404 ICMP packets, reconstructed all 108 HDF5 connections exactly, and reproduced all 45 eligible
UDP-window feature vectors exactly with maximum feature difference `0.0`. VNAT represents these
ICMP connections with source port `0`, destination port `0`, and IP protocol `1`.

Release-compatible PCAP parsing preserves the publisher-observed UDP sizing convention: UDP uses
the UDP datagram length while TCP and ICMP retain the full IPv4 packet length. An explicit
corrected packet-size policy remains available for separately identified future experiments.

These results establish selected-capture parity for the batch PCAP-to-feature path. They do not
constitute timed replay, bounded incremental stream processing, runtime model inference, or live
capture.

## Parallax validation

Run the raw-data inspection with:

```bash
uv run --locked parallax dataset inspect \
  data/raw/vnat/VNAT_Dataframe_release_1.h5
```

The command verifies the trusted release checksum before Pandas reads the file. This ordering is
important because the fixed HDF5 representation contains serialized Python object data and must
not be loaded from an untrusted or unverified source.

On the initial development machine, inspection completed in 12.06 seconds with a peak resident
set of 5,774,964 KiB and no swap use. The current loader therefore works for development but is
not suitable for a memory-constrained runtime.

## Raw dataframe

The raw dataframe uses HDF5 key `/data`, Pandas fixed format, 33,711 rows, and five columns:

| Column | Observed content |
| --- | --- |
| `connection` | Biflow five-tuple: source IP, source port, destination IP, destination port, protocol |
| `timestamps` | Packet timestamp list |
| `sizes` | Packet-size list |
| `directions` | Packet-direction list |
| `file_names` | Source PCAP filename containing VPN status, application, and capture identity |

All 33,711 rows had aligned timestamp, size, and direction arrays. The file represents 38,103,270
packets across 165 source captures. Packet counts per connection ranged from 1 to 3,842,411, with
a median of 2.

| Category | Applications | Captures | Connections |
| --- | --- | ---: | ---: |
| Command and Control | RDP, SSH | 18 | 13,599 |
| Chat | Skype Chat | 110 | 1,301 |
| File Transfer | RSYNC, SCP, SFTP | 19 | 16,430 |
| Streaming | Netflix, Vimeo, YouTube | 12 | 1,764 |
| VoIP | Zoiper traffic labeled `voip` | 6 | 617 |

The capture set contains 83 non-VPN and 82 VPN files. At the connection level, it contains 33,332
non-VPN and 379 VPN rows. This difference is expected to affect sampling: traffic observed outside
the tunnel exposes many application biflows, while traffic observed at the VPN boundary can
collapse into a small number of outer tunnel connections.

All 165 filenames map unambiguously to one VPN status, application, category, and capture group
through the versioned Parallax contract.

## Feature dataframe

The feature dataframe uses HDF5 key `/features` and contains 15,093 rows. It has 129 `float32`
features plus one string label. The features include flow statistics and wavelet-derived values.

| Label | Rows |
| --- | ---: |
| `C2` | 1,675 |
| `CHAT` | 10,498 |
| `FILE_TRANSFER` | 851 |
| `STREAMING` | 1,826 |
| `VOIP` | 243 |

The inspected release contained no missing or infinite feature values. It contained 197 exact
duplicate rows and two constant features:

- `in_log_std_dev_detail_coeffs_0`
- `out_log_std_dev_detail_coeffs_0`

The feature file does not retain source filename, capture identity, application, or VPN status.
It can support a paper-comparison baseline, but it cannot support Parallax's primary
capture-grouped evaluation.

## Published preprocessing

The related paper groups packets into connections using biflow five-tuples, divides connections
into 40.96-second windows, discards windows with fewer than 20 packets, and discretizes packet
observations into 0.01-second bins. Each retained window becomes 129 statistical and
wavelet-derived features. The first observed packet determines the otherwise arbitrary forward
direction for a connection.

Parallax records these values in code as part of the release contract. The wording creates a
threshold ambiguity: discarding windows with fewer than 20 packets implies retaining exactly 20,
while the released feature counts are much closer to results obtained by retaining more than 20.

## Parallax window artifact

Parallax aligns windows to the earliest packet timestamp in each source capture while keeping
connections separate. Packet timestamps, sizes, and directions are stably sorted together before
window assignment. This reproduces the observed release substantially more closely than aligning
each connection independently or merging packets across connections.

Two named threshold policies preserve the ambiguity rather than silently choosing one:

- `release-compatible` retains a window when `packet_count > 20` and is the default.
- `paper-literal` retains a window when `packet_count >= 20`.

Full-release extraction produced the following comparison:

| Category | Released features | Release-compatible | Paper-literal |
| --- | ---: | ---: | ---: |
| `C2` | 1,675 | 1,675 | 1,688 |
| `CHAT` | 10,498 | 10,499 | 10,541 |
| `FILE_TRANSFER` | 851 | 851 | 852 |
| `STREAMING` | 1,826 | 1,827 | 1,925 |
| `VOIP` | 243 | 243 | 243 |
| **Total** | **15,093** | **15,095** | **15,249** |

The strict policy matches three categories exactly and differs by one row in both Chat and
Streaming. This is strong evidence that the released preprocessing used a strict threshold, but
the remaining two-row difference is unresolved. Parallax therefore does not claim exact feature
artifact reproduction.

Run the default extraction with:

```bash
uv run --locked parallax dataset extract-windows \
  data/raw/vnat/VNAT_Dataframe_release_1.h5 \
  data/processed/vnat-release-1/windows-release-compatible.parquet
```

The accepted `vnat-window-1` artifact contains 15,095 windows and 37,981,571 packets from 162 of
the 165 captures. These captures produced no eligible release-compatible windows:

- `nonvpn_rsync_newcapture1.pcap`
- `nonvpn_scp_newcapture1.pcap`
- `nonvpn_sftp_newcapture2.pcap`

The Zstandard-compressed Parquet file is 99,585,954 bytes and has SHA-256
`06f00af45cb635241575d251331e7ce96273212dba087e38b7610876ec9984d8`. A second extraction in the
same locked environment produced a byte-for-byte checksum match. The artifact contains 236 row
groups and uses non-nullable fields:

| Field group | Columns |
| --- | --- |
| Identity | `window_id`, `capture_id`, `flow_id`, `window_index` |
| Bounds | `start_offset_seconds`, `end_offset_seconds` |
| Labels | `vpn_status`, `application`, `category` |
| Packet data | `packet_count`, `timestamps`, `sizes`, `directions` |

Raw connection addresses and ports are intentionally omitted from the processed artifact. A
companion manifest records schema version, trusted source checksum, extraction parameters, output
checksum, omitted captures, and label distributions. The exporter refuses existing destinations
and builds both files under a temporary directory before moving them to their final paths.

## Parallax feature reproduction

Parallax independently calculates the complete ordered 129-feature vector from each
`vnat-window-1` row. The implementation uses the following release-compatible rules:

- Interarrival statistics use seconds and population standard deviation for outgoing, incoming,
  and combined packet timestamps.
- Active and idle statistics use a five-second activity timeout. Idle duration excludes that
  timeout, so an observed six-second gap contributes one second of idle time.
- Aggregate features use natural logarithms and the release's `1e-4` count offset.
- Packet sizes are accumulated into 4,096 directional 0.01-second bins.
- Thirteen energy-normalized stationary Haar wavelet bands provide relative energy, base-2
  Shannon entropy, log mean absolute coefficient, and log population-standard-deviation values.

The released feature dataframe contains a reproducible defect: each directional byte-total
column is exactly equal to the corresponding packet-count column. The default
`release-compatible` policy preserves this behavior for comparison. The separately named
`corrected` policy calculates actual directional byte totals for future modeling.

An empirical comparison used 500 uniquely fingerprint-aligned windows, with 100 examples from
each category:

| Comparison | Result |
| --- | ---: |
| Complete vectors matching bit for bit | 416 / 500 |
| Complete vectors within `1e-5` | 432 / 500 |
| Complete vectors within `1e-4` | 447 / 500 |
| Complete vectors within `1e-3` | 455 / 500 |
| Complete vectors within `1e-2` | 495 / 500 |

All 2,500 aggregate values matched exactly. Interarrival values had a maximum absolute error of
`9.53674316e-7`; active and idle values had a maximum of `0.000770568848`. Most remaining
differences were in fine-scale wavelet bands, where the maximum error was `0.0275537968`.
Together with the unresolved two-window count difference, these results support formula-level
reproduction but not a claim of exact equivalence to the publisher's feature artifact.

Create the accepted feature artifact with:

```bash
uv run --locked parallax dataset extract-features \
  data/processed/vnat-release-1/windows-release-compatible.parquet \
  data/processed/vnat-release-1/features-release-compatible.parquet
```

The resulting `vnat-feature-artifact-1` Parquet file contains 15,095 rows, 129 finite `float32`
feature columns, 236 row groups, and identities for 162 captures representing 37,981,571 packets.
It is 14,702,124 bytes and has SHA-256
`611dcb63c66e04f461fb7500f96762c1722a06fbdde700dc5aa42ae021e39f16`. A clean replay in the same
locked environment produced a byte-for-byte match. Row-group processing reduced peak resident
memory to 908,752 KiB and completed in 3 minutes 33.28 seconds on the initial development
machine.

## Capture-grouped model-development split

The publication reports randomized 80/20 train/test splits over derived examples. It does not
report capture-group isolation for that experiment. Multiple windows from one connection or
capture could therefore appear in both partitions.

Parallax uses the exact source PCAP filename as its primary grouping key. A deterministic
mixed-integer optimizer assigns every capture exactly once across training, validation,
calibration, and test. The versioned hard contract requires:

- Every category in every partition with at least 20 windows.
- Every application in both training and test.
- VPN and non-VPN captures in every partition.
- Every category/VPN-status combination in training.

Window totals, capture totals, per-category distributions, and VPN-status distributions influence
the balance objective. The default window-fraction targets are 60% training, 15% validation, 10%
calibration, and 15% test. A 15-second solver limit and 10% relative MIP gap bound optimization,
but the manifest exporter refuses to publish a merely feasible result: optimality must be proven.

The accepted assignment is:

| Partition | Captures | Windows | Window fraction |
| --- | ---: | ---: | ---: |
| Training | 95 | 9,046 | 59.9271% |
| Validation | 25 | 2,098 | 13.8986% |
| Calibration | 16 | 1,499 | 9.9304% |
| Test | 26 | 2,452 | 16.2438% |

All ten applications occur in training and test, and all five categories occur in every
partition. Validation and calibration are not required to contain every application because
Vimeo has only two source captures and Netflix has only three. The manifest is deterministic:
independent API and CLI runs matched byte for byte. The 38,372-byte artifact has SHA-256
`a1aeee5f118c1e3bd57aa7232eb009634c098b9025571786614798953ebd8a0f`.

A randomized window split may still be reported as a clearly labeled paper-comparison result, but
it cannot replace the capture-held-out primary evaluation.

Because only six VoIP captures and twelve Streaming captures are available, partition metrics
must be interpreted with their capture counts and not only their much larger window counts.

## Baseline validation context

The accepted initial baseline experiment uses 9,046 training windows from 95 captures and 2,098
validation windows from 25 different captures. The validation partition is strongly imbalanced:
1,541 windows, or 73.45%, are Chat. Consequently, the majority-class baseline reaches 73.45% raw
accuracy but only 20% balanced accuracy.

Balanced logistic regression reaches 93.42% validation accuracy, 73.31% balanced accuracy, and
0.745 macro F1. Its weakest category is VoIP, with 19.05% recall on 42 windows from one validation
capture; 34 of those windows are classified as File Transfer. These results must therefore be
read as capture-held-out validation evidence rather than population-level performance estimates.
That baseline workflow did not evaluate calibration or test. Full configuration and evidence are
documented in
[Initial VNAT Validation Baselines](baseline-modeling.md).

## Prototype evaluation context

The selected prototype reaches 93.47% validation accuracy, 87.89% balanced accuracy, and 0.808
macro F1. After calibration-only OOD density fitting and a complete policy freeze, the one-shot
test result is 86.70% accuracy, 82.08% balanced accuracy, and 0.713 macro F1. Expected calibration
error increases from 0.065 on validation to 0.133 on test.

The dominant test failure is category-specific rather than uniform: 186 of 442 C2 windows and 63
of 255 Streaming windows are predicted as VoIP. VoIP recall is 97.73%, but its precision is only
14.29%; C2 recall is 48.42%. Chat and File Transfer retain F1 scores of 0.993 and 0.918.

At the frozen 0.95 OOD threshold, 11 of 2,452 known test windows are flagged, all from C2. No true
OOD examples exist in this test partition, so this is a 0.45% known-traffic false-positive rate,
not a measurement of OOD detection power. The complete artifact chain and results are documented
in [VNAT Prototype and Uncertainty Evaluation](uncertainty-modeling.md).

## Intended use

- Develop deterministic ingestion, windowing, and feature contracts.
- Train five-category traffic-classification baselines.
- Measure probability calibration separately from out-of-distribution behavior.
- Compare VPN and non-VPN behavior without decrypting payloads.
- Replay public captures through the same feature path used at inference time.

## Limitations and prohibited claims

- Collection occurred in controlled virtual subnetworks and does not represent the full Internet.
- The release covers ten applications, five broad categories, and behavior from its collection era.
- Some traffic was scripted and some was manually generated.
- Category, application, connection, and window distributions are strongly imbalanced.
- The `C2` category contains benign SSH and RDP activity; it is not malware ground truth.
- Filename-derived labels describe the generating capture and may not describe every individual
  window equally well.
- Concurrent multilabel application behavior inside one VPN observation is not represented.
- A model trained on VNAT must not be presented as a general malware detector, intrusion-detection
  system, or universal application classifier.

## Remaining data work

- Investigate the two-window difference between release-compatible extraction and the released
  feature dataframe.
- Preserve a separate randomized-window manifest only for comparison with the publication.
- Add further raw-PCAP acceptance captures only when they answer a specific compatibility or
  robustness question rather than expanding the test matrix without a defined purpose.
