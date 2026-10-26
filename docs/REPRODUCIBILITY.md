# Reproducibility and interpretation

## What is preserved

Core native detector files match the accepted experiment snapshot. Source hashes are in `manifests/source_snapshot.json`. Files categorized as frozen source were copied byte-for-byte. JSON files prefixed `portable_` are explicitly marked derivatives with machine-specific paths replaced; they are not byte-identical runtime receipts.

The release contains full internal split membership, original label text, source-relative image/label paths and content hashes. `detection_dataset_1486.zip` has canonical images in `dataset/images/{train,val,test}/` and labels in `ground_truth/{train,val,test}/`. Its `manifest.json` binds canonical IDs to the retained source files. Date-held-out membership reuses these image contents; it does not create a second independent collection. Its staged IDs/filenames differ from the internal archive: use `data/date_holdout/image_mapping_to_release_asset.csv` to locate each image by SHA256, and use the date-specific label budgets and COCO GT. Do not resolve a date filename directly against the internal archive.

Original requirements are retained as a historical project file, not a validated environment lock. Model loading, GPU training and hardware measurements were not rerun during release curation. The teacher initialization used for training has a recorded hash but independently certified upstream provenance is unavailable; it is not silently replaced with another checkpoint.

## Training and controls

- 300 epochs, image size 640, seeds 101/202/303; teacher batch 4, student batch 8; final-epoch checkpoint is the main evaluation point.
- Manual-label budgets 80/144 apply to the internal training pool only. Held-out evaluation and diagnostic annotations are outside that budget.
- Pseudo generation: confidence 0.5, NMS IoU 0.45; exclude images with no retained prediction. An excluded prediction does not prove a true negative.
- Frozen pseudo corpora are teacher/seed-specific. Hidden training GT is diagnostic only; it does not tune thresholds or enter optimization.
- Equal epochs do not match training updates. Internal repetition controls match reconstructed SSL schedules; date-held-out control schedules and per-update learning rates are directly recorded.
- No EMA teacher update, feature distillation, iterative relabeling or separate unlabeled-loss coefficient is claimed.

## Evaluation

Detection: FP32, confidence floor 0.001, NMS IoU 0.65, at most 300 model predictions; COCO maxDets 1/10/100 and 101-point interpolation. AP50:95 is not AP50. Reported variability is sample SD across three training seeds, not a significance test.

Tracking: use every one of the 750 frames of each clip, including empty-GT frames. `cvat_frame_0` is zero-based. Identity numbers are independent between clips/methods. Use official TrackEval CLEAR and Identity combined-sequence aggregation rather than arithmetic averaging clip percentages. Common cached detections support the quality comparison; same-process timings are separate experiments.

Hardware: Jetson file-video throughput includes its specified decode/detection/postprocessing window. It is not a guaranteed live-camera rate. CUDA-event forward latency is not the reciprocal of end-to-end FPS. Raspberry Pi VDD_CORE rail readings are not whole-board input power and are not combined with separately measured FPS to invent energy/frame.

## Table exports

`results/manuscript_tables/printed_tables.json` preserves captions, full table blocks and footnotes from the current 19-table manuscript. CSV cells retain LaTeX expressions and merged-header syntax. They are printed summaries, not raw per-run telemetry. Any separately included initial-return summaries retain that narrower scope and do not supersede the final printed tables.

## Remaining boundaries

The release is an auditable source/data archive. Some original recordings are unavailable, complete recording-to-image maps are absent, and the original goldfish videos were lost. Only six deployment checkpoints are locally available in the prepared model asset, not every model/seed or the original teacher initialization. TensorRT engines must be built for a compatible target environment. No long-duration farm trial, feeding decision loop or synchronized water-quality fusion is established.

## Pseudo-label archive boundary

Pseudo-label generation code and thresholds are provided. The four binary assets do not contain all teacher/seed-specific frozen pseudo-label corpora or all raw run receipts. The initial-return summaries and printed table exports must not be treated as substitutes for those complete training artifacts.
