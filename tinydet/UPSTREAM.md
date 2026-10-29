# Source provenance

This repository is a GitHub fork of https://github.com/ultralytics/ultralytics.
The project branch starts at official commit `3d75949c41f78a2c6763eb347ac7a94793098e89`
(Ultralytics 8.4.175), previously exercised by the project's YOLO11 export pipeline.
The upstream `ultralytics/` model, training loop and losses are unchanged by this migration.

The project adapters were migrated from `Jinlong-cs/tinydet-legacy` commit
`7e9a16edd4ae25c8301a6fba82e57b4ae518a4e1`, retaining their source and license notices.
That older repository was an independent adapter, not a GitHub fork.
The 2026-10-29 training plan uses YOLO11n first; earlier YOLO11s measurements remain historical results.

The raw-head and BPU Attention export adaptation derives from
[D-Robotics export_monkey_patch.py](https://github.com/D-Robotics/rdk_model_zoo/blob/rdk_x5/samples/vision/ultralytics_yolo/conversion/export_monkey_patch.py).
Its Apache-2.0 attribution and license are retained in `export.py` and `LICENSE-D-Robotics`.
The repository's upstream AGPL-3.0 license remains in the root `LICENSE`.

Training and export must load this checkout's Ultralytics package. Provenance records the actual fork commit
and dirty-tree status separately from the upstream base. A release should use a clean committed checkout.
Data, weights and experiment outputs are not distributed here. Each new checkpoint/ONNX/BIN requires its own validation.
