# Licensing by component

No single permissive license replaces the licenses below.

- `native/`: YOLOv7-derived source under the retained GPL-3.0 notice in `native/LICENSE`.
- SORT-derived code: retained GPL-3.0 notice in `tracking/SORT_LICENSE` and `third_party_licenses/SORT_GPL-3.0.txt`.
- `tracking/t1_vendor/`: the actually used Ultralytics-derived ByteTrack implementation, with its AGPL-3.0 notice. It is not the separate MIT-licensed ifzhang implementation.
- Original experiment orchestration and release tools: GPL-3.0, with existing third-party notices taking precedence for their own files.
- Original research data, annotations, tabulated results and documentation: CC BY 4.0; see `data/LICENSE.md`. Third-party software and pretrained/model-derived components are not relicensed by this data statement.
- TrackEval is separately licensed MIT; its notice is retained. The complete TrackEval dependency is not vendored here.

Retain original file headers and the notices under `third_party_licenses/`. Model artifacts retain the provenance and applicable upstream conditions listed in `models/artifacts.json`.
