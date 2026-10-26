# SSL-TGE-YOLO

Research code and evidence for **A label-efficient edge AI framework for underwater fish detection and tracking in smart aquaculture**.

SSL-TGE-YOLO uses an offline server-side teacher to generate frozen pseudo labels, a compact Ghost/ECA detector, and SORT for motion-based association. This repository accompanies controlled experiments on annotation budgets, model structure, tracking identity quality and separately measured device costs. It does not introduce Ghost, ECA, SORT or a new end-to-end tracking algorithm.

## Release contents

| Location | Contents |
| --- | --- |
| `native/` | Accepted YOLOv7-derived source snapshot and offline pseudo-label utilities |
| `configs/` | Compact/wider student, teacher and hyperparameter configurations |
| `experiments/` | Preserved internal/date-control and Jetson/Raspberry Pi experiment scripts |
| `tracking/` | SORT adapter and the actually used Ultralytics-derived ByteTrack implementation |
| `data/internal/` | 1486-image membership, image/label hashes, full YOLO labels, 80/144 manual subset IDs and validation/test COCO GT |
| `data/date_holdout/` | 436-image snapper date-held-out partition |
| `data/tracking/` | Visible identity boxes and frame mappings for R04/R02 |
| `results/manuscript_tables/` | All 19 tables from the 2026-10-26 manuscript, with original LaTeX cells and CSV exports |
| `models/artifacts.json` | Six locally retained final seed-101 deployment checkpoints and their hashes |
| `manifests/` | File integrity and source provenance |

Images, videos, checkpoint binaries and figure assets are packaged as separate GitHub Release attachments. **Check `data/asset_manifest.json`: an attachment with a null download URL has been prepared but is not yet published.** Do not assume that the existence of this repository means every binary attachment is already available.

## Verify this checkout

Python 3.9 or later is sufficient for these standard-library checks:

```bash
python verify_bundle.py
python tools/check_annotations.py
```

These checks do not start training, load model pickle files or contact a server. They verify integrity and annotations, not model performance.

## Data and evaluation scope

The mixed collection contains provider-described mangrove red snapper and locally purchased aquarium goldfish, modeled as a single `fish` class. The original pool has 1779 images. The audited retained set has 1486 unique image contents after excluding conflicting annotation groups and invalid records: 1188 training, 149 validation and 149 test images. Exact content deduplication does not establish recording/sequence independence.

The snapper-only date partition has 216 training, 146 validation and 74 test images. It is a within-collection date holdout, not an independent external site. Its 80-image manual subset and the mixed internal 80-image subset are different experimental budgets.

The complete label archive is provided for audit and evaluation. **Training with an 80- or 144-image label budget must use only the corresponding frozen manual subset and its teacher-generated pseudo corpus. Do not train on all archived GT labels and describe that as the low-label experiment.**

## Selected reported findings

- Compact 80-label offline SSL: internal AP50 **87.94 ± 1.99%**, AP50:95 **56.51 ± 1.06%**, over seeds 101/202/303.
- AP50 gain over the internal repetition control: **5.01 ± 2.07 percentage points**. The original SSL update budget is reconstructed; control counters are directly measured.
- Date-held-out AP50 gain: **5.77 ± 5.08 points**; AP50:95 paired differences change sign across seeds.
- Ghost stages reduce static parameter count by **43.38%** against a controlled standard-stage counterpart. This is not a latency-equivalence claim.
- SORT combined IDF1: **56.66%** on two same-source clips. Separate Jetson file-video measurements reach approximately **14.5 frames/s** for detection and tracking. These are separate accuracy and timing experiments.
- The tested YOLOv8n alternatives have lower native runtime costs. The study does not establish universal accuracy, energy or speed superiority.

See [reproducibility notes](docs/REPRODUCIBILITY.md), [data card](docs/DATA_CARD.md) and [environment notes](docs/ENVIRONMENTS.md) before running experiments. Runtime dependencies and full training have not been re-executed for this release layout. Frozen experiment orchestrators contain environment-specific resource guards; preserving their source is not a claim of a one-command installation.

## License and attribution

Read [LICENSE.md](LICENSE.md). YOLOv7/SORT, the used Ultralytics ByteTrack source and TrackEval retain their respective GPL-3.0, AGPL-3.0 and MIT notices. Original research data and documentation are CC BY 4.0. Model artifacts retain their recorded upstream provenance; no blanket MIT license is applied.

When using these materials, cite this repository URL and the exact commit/release, and cite the associated paper when its publication details are available. No publication DOI or author order is invented here.
