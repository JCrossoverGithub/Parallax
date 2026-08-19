# VNAT Dataset Card for Parallax

## Status

VNAT release 1 was independently downloaded, checksummed, structurally inspected,
deterministically windowed, and exported by Parallax on 18 August 2026. The PCAP archive has not
yet been downloaded or validated.

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

## Leakage control and evaluation

The publication reports randomized 80/20 train/test splits over derived examples. It does not
report capture-group isolation for that experiment. Multiple windows from one connection or
capture could therefore appear in both partitions.

Parallax will use the exact source PCAP filename as its primary grouping key. No capture may cross
training, validation, calibration, and test partitions. A randomized window split may be reported
only as a clearly labeled paper-comparison result.

Because only six VoIP captures and twelve Streaming captures are available, grouped partition
construction must verify per-category coverage rather than assuming a random group assignment is
valid.

## Intended use

- Develop deterministic ingestion, windowing, and feature contracts.
- Train five-category traffic-classification baselines.
- Evaluate probability calibration separately from out-of-distribution detection.
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

- Download and verify selected PCAP captures for replay acceptance tests.
- Reproduce the 129-feature schema from raw packet metadata.
- Investigate the two-window difference between release-compatible extraction and the released
  feature dataframe.
- Create versioned, capture-grouped split manifests with class-coverage checks.
- Preserve a separate randomized-window manifest only for comparison with the publication.
