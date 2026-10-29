# SPDX-License-Identifier: AGPL-3.0-only
# raw_forward / bpu_attention adapted from D-Robotics export_monkey_patch.py.
# Original portions: Copyright (c) 2025, WuChao D-Robotics, Apache-2.0.
# See LICENSE-D-Robotics. Modified for this fixed YOLO11 raw-head/parity adapter.
"""YOLO11 raw cls/DFL heads, static opset11, with source-to-ONNX parity."""

import argparse
import copy
import types
from pathlib import Path

from .common import UPSTREAM, check_upstream, read_manifest, rgb_input, sha256, source_provenance, write_json

# isort: split

import numpy as np
import onnx
import onnxruntime as ort
import torch

from ultralytics import YOLO
from ultralytics.nn.modules.block import Attention
from ultralytics.nn.modules.head import Detect

NAMES = [f"{kind}_s{stride}" for stride in (8, 16, 32) for kind in ("cls", "box")]


def raw_forward(self, x):
    return tuple(t for i in range(self.nl) for t in (self.cv3[i](x[i]), self.cv2[i](x[i])))


def bpu_attention(self, x):
    # Algebraically equivalent softmax along BPU-supported axis; applied only to an export copy.
    b, c, h, w = x.shape
    q, k, v = (
        self.qkv(x)
        .view(b, self.num_heads, self.key_dim * 2 + self.head_dim, h * w)
        .split([self.key_dim, self.key_dim, self.head_dim], dim=2)
    )
    attention = ((q.transpose(-2, -1) @ k) * self.scale).permute(0, 3, 1, 2).contiguous()
    exponent = torch.exp(attention - attention.max(dim=1, keepdim=True).values)
    attention = (exponent / exponent.sum(dim=1, keepdim=True)).permute(0, 2, 3, 1).contiguous()
    result = (v @ attention.transpose(-2, -1)).reshape(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
    return self.proj(result)


def decode_all(outputs, nc):
    """Reference DFL decode before threshold/NMS, for numerical export checking only."""
    parts = []
    for j, stride in enumerate((8, 16, 32)):
        cls, box = outputs[2 * j : 2 * j + 2]
        _, _, h, w = box.shape
        d = box.transpose(0, 2, 3, 1).reshape(-1, 4, 16)
        e = np.exp(d - d.max(-1, keepdims=True))
        ltrb = (e / e.sum(-1, keepdims=True) * np.arange(16, dtype=np.float32)).sum(-1)
        yy, xx = np.mgrid[:h, :w]
        anchor = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], -1).astype(np.float32)
        lo, hi = (anchor - ltrb[:, :2]) * stride, (anchor + ltrb[:, 2:]) * stride
        xywh = np.concatenate([(lo + hi) / 2, hi - lo], -1)
        scores = 1 / (1 + np.exp(-cls.transpose(0, 2, 3, 1).reshape(-1, nc)))
        parts.append(np.concatenate([xywh, scores], -1))
    return np.concatenate(parts).T[None]


def export(args):
    check_upstream()
    source = source_provenance()
    torch.set_num_threads(4)
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    model = YOLO(args.checkpoint).model.float().cpu().eval()
    head = model.model[-1]
    if type(head) is not Detect or head.reg_max != 16 or head.end2end or head.nl != 3:
        raise ValueError("This adapter supports YOLO11 non-end2end detection with reg_max=16 and three strides")
    nc = head.nc
    classes = [model.names[i] for i in range(nc)]
    if not classes or any(not isinstance(name, str) or not name.strip() for name in classes):
        raise ValueError("Checkpoint must contain nonempty ordered class names")
    if len(set(classes)) != nc or head.stride.tolist() != [8, 16, 32]:
        raise ValueError("Expected unique classes and strides 8/16/32")
    shapes = [[1, channels, 352 // stride, 640 // stride] for stride in (8, 16, 32) for channels in (nc, 64)]
    patched = copy.deepcopy(model)
    for child in patched.modules():
        if type(child) is Detect:
            child.forward = types.MethodType(raw_forward, child)
        elif type(child) is Attention:
            child.forward = types.MethodType(bpu_attention, child)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    path = out / "model.onnx"
    x = np.random.default_rng(42).random((1, 3, 352, 640), dtype=np.float32)
    with torch.no_grad():
        if [list(t.shape) for t in patched(torch.from_numpy(x))] != shapes:
            raise ValueError("Unexpected raw output shapes")
    torch.onnx.export(
        patched,
        torch.from_numpy(x),
        str(path),
        input_names=["images"],
        output_names=NAMES,
        opset_version=11,
        dynamo=False,
        external_data=False,
        do_constant_folding=True,
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    if any(o.version != 11 for o in graph.opset_import if o.domain in ("", "ai.onnx")):
        raise ValueError("Expected opset11")
    if {"NonMaxSuppression", "ArgMax"} & {n.op_type for n in graph.graph.node}:
        raise ValueError("Unexpected postprocessing inside graph")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = 4, 1
    session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    if [(i.name, i.shape, i.type) for i in session.get_inputs()] != [("images", [1, 3, 352, 640], "tensor(float)")] or [
        (o.name, o.shape, o.type) for o in session.get_outputs()
    ] != [(name, shape, "tensor(float)") for name, shape in zip(NAMES, shapes)]:
        raise ValueError("ONNX interface mismatch")
    rows = [r for r in read_manifest(args.check_manifest) if r["split"] == "val"]
    if not rows or args.limit < 1:
        raise ValueError("At least one validation image required")
    cases = []
    for row in [None] + rows[: args.limit]:
        a = x if row is None else rgb_input(row["image"])[None]
        with torch.no_grad():
            decoded_reference, raw = model(torch.from_numpy(a))
        outputs = session.run(NAMES, {"images": a})
        offset, over = 0, 0
        for j, stride in enumerate((8, 16, 32)):
            h, w = 352 // stride, 640 // stride
            for k, key in enumerate(("scores", "boxes")):
                reference = raw[key][:, :, offset : offset + h * w].reshape(shapes[2 * j + k]).numpy()
                delta = np.abs(outputs[2 * j + k] - reference)
                over += int((delta > 5e-4 + 1e-4 * np.abs(reference)).sum())
            offset += h * w
        decoded = decode_all(outputs, nc)
        delta = np.abs(decoded - decoded_reference.numpy())
        cases.append(
            {
                "id": "random_seed42" if row is None else row["id"],
                "finite": bool(all(np.isfinite(o).all() for o in outputs)),
                "raw_over_tolerance": over,
                "boxes_max_abs_px": float(delta[:, :4].max()),
                "scores_max_abs": float(delta[:, 4:].max()),
            }
        )
    passed = all(
        c["finite"] and c["raw_over_tolerance"] == 0 and c["boxes_max_abs_px"] <= 0.05 and c["scores_max_abs"] <= 1e-4
        for c in cases
    )
    write_json(out / "classes.json", classes)
    write_json(
        out / "export.json",
        {
            "passed": passed,
            "checkpoint_sha256": sha256(args.checkpoint),
            "onnx_sha256": sha256(path),
            "classes_sha256": sha256(out / "classes.json"),
            "upstream_commit": UPSTREAM,
            "source": source,
            "opset": 11,
            "input_name": "images",
            "input_dtype": "float32",
            "input": [1, 3, 352, 640],
            "outputs": [{"name": n, "shape": s, "dtype": "float32"} for n, s in zip(NAMES, shapes)],
            "names": model.names,
            "normalization": "whole-frame LINEAR RGB/255",
            "reg_max": 16,
            "strides": [8, 16, 32],
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "check_manifest_sha256": sha256(args.check_manifest),
            "raw_atol": 5e-4,
            "raw_rtol": 1e-4,
            "decoded_box_atol_px": 0.05,
            "decoded_score_atol": 1e-4,
            "cases": cases,
        },
    )
    if not passed:
        raise RuntimeError("ONNX parity failed; do not compile this artifact")
    print(f"Export parity passed: {len(cases)} cases")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--check-manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=64)
    export(p.parse_args())
