# SPDX-License-Identifier: AGPL-3.0-only
"""Shared batch-one CPU inference for TinyDet PT and raw-head ONNX models."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

from .common import HEIGHT, WIDTH, check_upstream, rgb_input, sha256, source_provenance
from .export import NAMES, decode_all

# isort: split

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision

from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import clip_boxes


class Detector:
    def __init__(self, args):
        """Load a local model and its class/preprocessing contract."""
        check_upstream()
        path = Path(args.model).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if args.threads < 1 or args.warmup < 0 or args.max_det < 1 or not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
            raise ValueError("Invalid thread count, warmup, detection limit or threshold")
        self.args = args
        torch.set_num_threads(args.threads)
        cv2.setNumThreads(args.threads)
        self.pt = path.suffix == ".pt"
        if self.pt:
            self.model = YOLO(path).model.float().cpu().eval()
            head = self.model.model[-1]
            if type(head) is not Detect or head.end2end or head.reg_max != 16 or head.stride.tolist() != [8, 16, 32]:
                raise ValueError("Expected a YOLO11 detection checkpoint with three strides and reg_max=16")
            self.names = [self.model.names[i] for i in range(head.nc)]
        elif path.suffix == ".onnx":
            classes = path.with_name("classes.json")
            receipt = json.loads(path.with_name("export.json").read_text())
            if (
                receipt["passed"] is not True
                or receipt["onnx_sha256"] != sha256(path)
                or receipt["classes_sha256"] != sha256(classes)
            ):
                raise ValueError("ONNX or classes do not match the successful export receipt")
            self.names = json.loads(classes.read_text())
            options = ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = args.threads, 1
            self.model = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            expected = [
                (name, [1, channels, HEIGHT // stride, WIDTH // stride], "tensor(float)")
                for stride, pair in zip((8, 16, 32), (NAMES[:2], NAMES[2:4], NAMES[4:]))
                for name, channels in zip(pair, (len(self.names), 64))
            ]
            if [(x.name, x.shape, x.type) for x in self.model.get_inputs()] != [
                ("images", [1, 3, HEIGHT, WIDTH], "tensor(float)")
            ] or [(x.name, x.shape, x.type) for x in self.model.get_outputs()] != expected:
                raise ValueError("Expected TinyDet FLOAT32 352x640 input and six raw class/DFL heads")
        else:
            raise ValueError("Use a local .pt checkpoint or the .onnx produced by tinydet.export")
        if (
            not isinstance(self.names, list)
            or not self.names
            or any(not isinstance(n, str) or not n.strip() for n in self.names)
            or len(set(self.names)) != len(self.names)
        ):
            raise ValueError("Expected unique, nonempty ordered model class names")
        self.metadata = {
            "path": str(path),
            "sha256": sha256(path),
            "backend": "pytorch_cpu" if self.pt else "onnxruntime_cpu",
            "classes": self.names,
            "class_order_sha256": hashlib.sha256(json.dumps(self.names, ensure_ascii=False).encode()).hexdigest(),
            "source": source_provenance(),
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "versions": {
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "onnxruntime": ort.__version__,
                "opencv": cv2.__version__,
            },
            "input": [1, 3, HEIGHT, WIDTH],
            "preprocessing": "whole-frame LINEAR resize; BGR to RGB; float32 NCHW /255",
            "postprocessing": {
                "conf": args.conf,
                "iou": args.iou,
                "max_det": args.max_det,
                "multi_label": args.multi_label,
                "class_agnostic": False,
                "clip_to_input": True,
                "nms": "torchvision.ops.nms",
                "max_nms": 30000,
            },
            "threads": args.threads,
            "warmup_iterations": args.warmup,
            "timing_scope": "CPU wall time, batch=1; excludes loading, warmup, file I/O, metrics, plotting and encoding",
            "inference_scope": "PT forward includes native DFL decode; ONNX forward returns raw heads, with DFL decode in postprocess",
        }

    @torch.inference_mode()
    def __call__(self, frame):
        start = perf_counter()
        image = rgb_input(frame)[None]
        if self.pt:
            image = torch.from_numpy(image)
        prepared = perf_counter()
        outputs = self.model(image)[0] if self.pt else self.model.run(NAMES, {"images": image})
        inferred = perf_counter()
        decoded = outputs if self.pt else torch.from_numpy(decode_all(outputs, len(self.names)))
        if not torch.isfinite(decoded).all():
            raise ValueError("Non-finite model output")
        boxes = non_max_suppression(
            decoded,
            conf_thres=self.args.conf,
            iou_thres=self.args.iou,
            max_det=self.args.max_det,
            multi_label=self.args.multi_label,
            nc=len(self.names),
        )[0]
        clip_boxes(boxes[:, :4], (HEIGHT, WIDTH))
        finished = perf_counter()
        return boxes, {
            "preprocess": (prepared - start) * 1000,
            "inference": (inferred - prepared) * 1000,
            "postprocess": (finished - inferred) * 1000,
            "total": (finished - start) * 1000,
        }

    def warmup(self, frame, count):
        for _ in range(count):
            self(frame)


def latency_summary(timings):
    summary = {"frames": len(timings), "stages": {}}
    for key in ("preprocess", "inference", "postprocess", "total"):
        values = [row[key] for row in timings]
        summary["stages"][key] = {
            "mean": float(np.mean(values)),
            "p50": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    return summary


def add_arguments(parser: argparse.ArgumentParser, conf):
    parser.add_argument(
        "--model", required=True, help="Local trusted PT, or raw ONNX beside its export.json/classes.json"
    )
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument("--conf", type=float, default=conf)
    parser.add_argument("--iou", type=float, default=0.7, help="Class-aware NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--multi-label", action="store_true", help="Allow multiple classes per candidate")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5, help="Unmeasured repetitions of the first frame")
