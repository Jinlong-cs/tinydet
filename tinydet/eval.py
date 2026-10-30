# SPDX-License-Identifier: AGPL-3.0-only
"""Evaluate PT or raw-head ONNX on the prepared validation split using official detection metrics."""

import argparse
import json
from pathlib import Path

from .common import HEIGHT, WIDTH, read_manifest, write_json
from .inference import Detector, add_arguments, latency_summary
from .train import validate_data

# isort: split

import cv2
import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.ops import xywh2xyxy


def evaluate(args):
    data = Path(args.data).resolve()
    identity = validate_data(data)
    names = json.loads((data.parent / "classes.json").read_text())
    detector = Detector(args)
    if detector.names != names:
        raise ValueError("Model class names and order must match the prepared dataset")
    rows = [row for row in read_manifest(data.parent / "manifest.jsonl") if row["split"] == "val"]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    validator = DetectionValidator(save_dir=out, args={"plots": False})
    metrics = validator.metrics
    metrics.names = dict(enumerate(names))
    prediction_counts = np.zeros(len(names), dtype=np.int64)
    false_positives = np.zeros(len(names), dtype=np.int64)
    timings = []
    with (out / "predictions.jsonl").open("w") as stream:
        for index, row in enumerate(rows):
            frame = cv2.imread(row["image"])
            if index == 0:
                detector.warmup(frame, args.warmup)
            boxes, elapsed = detector(frame)
            timings.append(elapsed)
            labels = torch.tensor(
                [
                    list(map(float, line.split()))
                    for line in Path(row["label"]).read_text().splitlines()
                    if line.strip()
                ],
                dtype=torch.float32,
            ).reshape(-1, 5)
            target = {
                "cls": labels[:, 0],
                "bboxes": xywh2xyxy(labels[:, 1:]) * torch.tensor([WIDTH, HEIGHT, WIDTH, HEIGHT]),
            }
            pred = {"bboxes": boxes[:, :4], "conf": boxes[:, 4], "cls": boxes[:, 5]}
            matched = validator._process_batch(pred, target)
            target_cls, pred_cls = labels[:, 0].numpy(), boxes[:, 5].numpy()
            metrics.update_stats(
                {
                    **matched,
                    "target_cls": target_cls,
                    "target_img": np.unique(target_cls),
                    "conf": boxes[:, 4].numpy(),
                    "pred_cls": pred_cls,
                    "im_name": row["id"],
                }
            )
            prediction_counts += np.bincount(pred_cls.astype(int), minlength=len(names))
            false_positives += np.bincount(pred_cls[~matched["tp"][:, 0]].astype(int), minlength=len(names))
            stream.write(
                json.dumps(
                    {"id": row["id"], "image": row["image"], "detections": boxes.tolist(), "latency_ms": elapsed}
                )
                + "\n"
            )
    metrics.process()
    fields = ("precision", "recall", "map50", "map50_95")
    measured = {
        int(cls): dict(zip(fields, map(float, metrics.class_result(index))))
        for index, cls in enumerate(metrics.ap_class_index)
    }
    overall = dict(zip(fields, map(float, metrics.mean_results()))) if measured else dict.fromkeys(fields)
    per_class = [
        {
            "id": index,
            "name": name,
            "images": int(metrics.nt_per_image[index]),
            "instances": int(metrics.nt_per_class[index]),
            "predictions": int(prediction_counts[index]),
            "false_positives_iou50": int(false_positives[index]),
            **measured.get(index, dict.fromkeys(fields)),
        }
        for index, name in enumerate(names)
    ]
    write_json(
        out / "metrics.json",
        {
            "model": detector.metadata,
            "data": {"path": str(data), "identity": identity, "split": "val", "images": len(rows)},
            "overall": overall,
            "per_class": per_class,
            "metric_policy": {
                "precision_recall": "Ultralytics maximum mean-F1 confidence point at IoU 0.5",
                "map": "IoU 0.5:0.95; averaged over classes with ground truth only",
                "unobserved_classes": "Precision, recall and AP are null when the class has no ground truth",
                "prediction_counts": "After configured confidence/NMS filtering; false positives use IoU 0.5",
                "training_comparison": "Uses prediction-time class selection and clipped boxes; training validation uses multi-label NMS and unclipped boxes",
            },
            "detections_format": "xyxy pixels in 640x352 input, score, class_id",
            "latency_ms": latency_summary(timings),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, conf=0.001)
    parser.add_argument("--data", required=True, help="data.yaml produced by prepare_data.py")
    evaluate(parser.parse_args())
