# VNAT Dataset Card for Parallax

## Status

VNAT release 1 was independently downloaded, checksummed, structurally inspected, and exercised
through the Parallax CLI on 18 August 2026. The PCAP archive has not yet been downloaded or
validated.

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

Parallax records these values in code as part of the release contract. Feature reproduction will
be tested against independently implemented fixtures before it is trusted on the full dataset.

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
- Implement deterministic 40.96-second window extraction from the raw dataframe.
- Reproduce the 129-feature schema from raw packet metadata.
- Create versioned, capture-grouped split manifests with class-coverage checks.
- Preserve a separate randomized-window manifest only for comparison with the publication.
