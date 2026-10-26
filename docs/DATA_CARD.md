# Data card

Collection location provided by the authors: a pomfret farm in Lingshui Li Autonomous County, Hainan, China (Base of Marine College, Hainan University). Snapper were recorded in underwater aquaculture net cages; locally purchased goldfish were recorded in an aquarium. The prior Gong study used golden pompano, a different fish from this collection. The detector uses one fish category.

| Partition | Training | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Audited mixed collection | 1188 | 149 | 149 | 1486 |
| Within-collection snapper date holdout | 216 | 146 | 74 | 436 |

The original mixed pool contains 439 snapper and 1340 goldfish images. Annotation-conflict exclusions account for 290 source files; three further invalid records are excluded. The mixed split retains 436 snapper and 1050 goldfish images. These exclusions are not relabeled or used as extra independent samples.

The date split uses 1/5 February 2019 for training, 11 February for validation and 23 February for testing, based on visible overlays and provider confirmation. Original recording PTS and complete recording identities are unavailable. Exact hash deduplication does not establish sequence independence. Near-duplicate images can remain.

Internal budgets: 80 manual training images (19 snapper/61 goldfish), and a nested 144-image budget (40/104). Date budget: 80 manual images, 40 from each training date. These budgets do not include the annotation work for held-out evaluation or diagnostic GT; measured total annotation-labor savings are not claimed.

R04/R02 contain 750 frames each at 3840×2160. There are 26/19 annotated identities and 1188/1227 visible boxes. Visible CSVs preserve occlusion flags and use zero-based frame indices. Do not drop empty frames or infer identity continuity from a few selected examples.

Original image/label hashes, manual IDs, canonical file names and frame mappings are distributed in `data/`. Raw media download status is explicit in `data/asset_manifest.json`. The original 1779-file pool with conflicting labels is not substituted for the audited 1486-image release.
