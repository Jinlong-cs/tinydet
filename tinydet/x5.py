# SPDX-License-Identifier: AGPL-3.0-only
"""Train-only calibration and explicit Horizon X5 INT8 compilation."""

import argparse
import collections
import json
import re
import subprocess
from pathlib import Path

from .common import UPSTREAM, read_manifest, rgb_input, sha256, write_json

# isort: split


def calibrate(args):
    """Write train-only RGB calibration samples and their hashes."""
    import numpy as np

    rows = [r for r in read_manifest(args.manifest) if r["split"] == "train"]
    if not 0 < args.count <= len(rows):
        raise ValueError("Calibration count must not exceed unique train samples")
    # Round-robin domains without repeating rows, with seeded order inside each domain.
    rng = np.random.default_rng(42)
    groups = {}
    for row in rows:
        groups.setdefault(row.get("domain", "default"), []).append(row)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < args.count:
        for domain in sorted(groups):
            if groups[domain] and len(selected) < args.count:
                selected.append(groups[domain].pop())
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    data_dir = out / "rgb"
    data_dir.mkdir()
    receipt = []
    for i, row in enumerate(selected):
        target = data_dir / f"{i:05}.rgb"
        rgb_input(row["image"], normalize=False).tofile(target)
        receipt.append(
            {
                "file": target.name,
                "id": row["id"],
                "split": "train",
                "domain": row.get("domain", "default"),
                "image_sha256": sha256(row["image"]),
                "rgb_sha256": sha256(target),
            }
        )
    write_json(
        out / "manifest.json",
        {
            "source_manifest_sha256": sha256(args.manifest),
            "layout": "CHW RGB float32 raw 0..255; mapper alone applies 1/255",
            "shape": [3, 352, 640],
            "dtype": "float32",
            "samples": receipt,
        },
    )


def audit(out, name):
    """Read actual placement/precision and input range from compiler artifacts."""
    import onnx

    log = re.sub(r"\x1b\[[0-9;]*m", "", (out / "hb_mapper_makertbin.log").read_text())
    start = log.rfind("The main quantized node information:")
    end = log.find("The quantized model output:", start)
    if start < 0 or end < 0:
        raise ValueError("Compiler node table not found; precision cannot be certified")
    table = log[start:end]
    nodes = []
    for line in table.splitlines():
        match = re.match(r"^(\S+)\s+(BPU|CPU)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", line)
        if match:
            nodes.append(
                dict(zip(["name", "placement", "subgraph", "op", "cosine", "threshold", "dtype"], match.groups()))
            )
    eligible = [line for line in table.splitlines() if re.search(r"\s(?:BPU|CPU)\s", line)]
    if not nodes or len(nodes) != len(eligible):
        raise ValueError("Incomplete compiler node audit")
    graph = onnx.load(str(out / (name + "_original_float_model.onnx")))
    ranges = [
        onnx.helper.get_attribute_value(a)
        for n in graph.graph.node
        if n.op_type == "HzPreprocess"
        for a in n.attribute
        if a.name == "input_type"
    ]
    if ranges != [b"YUV_BT601_VIDEO_RANGE_128"]:
        raise ValueError(f"Unexpected mapper input range: {ranges}")
    binary = out / (name + ".bin")
    with (out / "hb_model_info.txt").open("w") as stream:
        subprocess.run(["hb_model_info", str(binary)], stdout=stream, stderr=subprocess.STDOUT, check=True)
    info = (out / "hb_model_info.txt").read_text()
    if "bt601_video" not in info or "InputDataType.NV12" not in info:
        raise ValueError("BIN input contract is not limited-range BT601 NV12")
    return {
        "node_count": len(nodes),
        "nodes": nodes,
        "placement_counts": dict(collections.Counter(n["placement"] for n in nodes)),
        "dtype_counts": dict(collections.Counter(n["dtype"] for n in nodes)),
        "exceptions": [n for n in nodes if n["placement"] != "BPU" or n["dtype"] != "int8"],
        "runtime_input": "NV12 BT601 video range",
        "precision_scope": "internal compute; public outputs may be FLOAT32",
    }


def compile_model(args):
    """Validate export inputs and optionally run the X5 compiler."""
    import onnx
    import yaml

    model, cal, out = Path(args.onnx).resolve(), Path(args.calibration).resolve(), Path(args.out).resolve()
    if not model.is_file() or not cal.is_dir():
        raise ValueError("ONNX file and calibration/rgb directory must exist inside the compiler environment")
    export_path = model.parent / "export.json"
    export_report = json.loads(export_path.read_text())
    if export_report.get("passed") is not True or export_report.get("onnx_sha256") != sha256(model):
        raise ValueError("ONNX must match a passed export parity report")
    classes_path = model.parent / "classes.json"
    classes = json.loads(classes_path.read_text())
    if (
        not isinstance(classes, list)
        or not classes
        or any(not isinstance(name, str) or not name.strip() for name in classes)
        or len(set(classes)) != len(classes)
        or export_report.get("classes_sha256") != sha256(classes_path)
        or export_report.get("names") != {str(i): name for i, name in enumerate(classes)}
    ):
        raise ValueError("Class order and classes.json hash must match the export report")
    source = export_report.get("source", {})
    if (
        export_report.get("upstream_commit") != UPSTREAM
        or source.get("upstream_commit") != UPSTREAM
        or not re.fullmatch(r"[0-9a-f]{40}", source.get("fork_commit", ""))
        or not source.get("source_root")
        or not isinstance(source.get("working_tree_changes"), list)
    ):
        raise ValueError("Export report must identify the fork source and supported upstream base")
    expected_outputs = [
        {"name": f"{kind}_s{stride}", "shape": [1, channels, 352 // stride, 640 // stride], "dtype": "float32"}
        for stride in (8, 16, 32)
        for kind, channels in (("cls", len(classes)), ("box", 64))
    ]
    if (
        export_report.get("input_name") != "images"
        or export_report.get("input_dtype") != "float32"
        or export_report.get("input") != [1, 3, 352, 640]
        or export_report.get("outputs") != expected_outputs
        or export_report.get("opset") != 11
        or export_report.get("reg_max") != 16
        or export_report.get("strides") != [8, 16, 32]
        or export_report.get("normalization") != "whole-frame LINEAR RGB/255"
    ):
        raise ValueError("Unsupported export interface or preprocessing contract")
    graph = onnx.load(model)
    onnx.checker.check_model(graph)
    expected_io = [("images", [1, 3, 352, 640])] + [(o["name"], o["shape"]) for o in expected_outputs]
    actual_io = list(graph.graph.input) + list(graph.graph.output)
    if (
        len(graph.graph.input) != 1
        or len(actual_io) != len(expected_io)
        or any(
            item.name != name
            or item.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
            or [d.dim_value for d in item.type.tensor_type.shape.dim] != shape
            for item, (name, shape) in zip(actual_io, expected_io)
        )
        or [o.version for o in graph.opset_import if o.domain in ("", "ai.onnx")] != [11]
        or {"NonMaxSuppression", "ArgMax"} & {n.op_type for n in graph.graph.node}
    ):
        raise ValueError("ONNX graph does not match the six-head float32 export interface")
    files = sorted(cal.iterdir())
    if not files or any(not p.is_file() or p.suffix != ".rgb" or p.stat().st_size != 3 * 352 * 640 * 4 for p in files):
        raise ValueError("Calibration directory must contain only CHW float32 352x640 .rgb files")
    receipt_path = cal.parent / "manifest.json"
    calibration = json.loads(receipt_path.read_text())
    if (
        calibration.get("shape") != [3, 352, 640]
        or calibration.get("dtype") != "float32"
        or calibration.get("layout") != "CHW RGB float32 raw 0..255; mapper alone applies 1/255"
    ):
        raise ValueError("Calibration preprocessing must match the export input contract")
    receipt = calibration["samples"]
    if len(receipt) != len(files) or any(r["split"] != "train" for r in receipt):
        raise ValueError("Calibration receipt must contain exactly the train-only input files")
    if len({r["id"] for r in receipt}) != len(receipt):
        raise ValueError("Repeated calibration sample ids")
    by_name = {row.get("file"): row["rgb_sha256"] for row in receipt}
    hashes = {path.name: sha256(path) for path in files}
    if len(by_name) != len(receipt) or set(by_name) != set(hashes):
        raise ValueError("Calibration receipt must name every RGB file exactly once")
    if hashes != by_name:
        raise ValueError("Calibration data changed after preparation")
    if not args.name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Model prefix must contain only letters, digits, underscores or hyphens")
    out.mkdir(parents=True, exist_ok=False)
    config = {
        "model_parameters": {
            "onnx_model": str(model),
            "march": "bayes-e",
            "layer_out_dump": False,
            "working_dir": str(out),
            "output_model_file_prefix": args.name,
        },
        "input_parameters": {
            "input_name": "images",
            "input_type_rt": "nv12",
            "input_space_and_range": "bt601_video",
            "input_type_train": "rgb",
            "input_layout_train": "NCHW",
            "norm_type": "data_scale",
            "scale_value": 1 / 255,
        },
        "calibration_parameters": {
            "cal_data_dir": str(cal),
            "cal_data_type": "float32",
            "calibration_type": "default",
            "optimization": "set_all_nodes_int8",
        },
        "compiler_parameters": {"jobs": args.jobs, "compile_mode": "latency", "debug": True, "optimize_level": "O3"},
    }
    path = out / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    (out / "classes.json").write_bytes(classes_path.read_bytes())
    identity = {
        "onnx_sha256": sha256(model),
        "export_report_sha256": sha256(export_path),
        "classes_sha256": sha256(classes_path),
        "source": source,
    }
    write_json(
        out / "inputs.json",
        {
            **identity,
            "calibration_files": len(files),
            "calibration_manifest_sha256": sha256(receipt_path),
            "calibration_sha256": hashes,
            "requested_precision": "all INT8",
            "actual_precision": "not audited; inspect compiler node report",
            "executed": False,
        },
    )
    if args.run:
        for step, command in [
            ("checker", ["hb_mapper", "checker", "--model-type", "onnx", "--model", str(model), "--march", "bayes-e"]),
            ("makertbin", ["hb_mapper", "makertbin", "--config", str(path), "--model-type", "onnx"]),
        ]:
            with (out / f"{step}.log").open("w") as log:
                subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
        binary = out / f"{args.name}.bin"
        write_json(
            out / "compiled.json",
            {
                "bin_sha256": sha256(binary),
                **identity,
                "config_sha256": sha256(path),
                "inputs_sha256": sha256(out / "inputs.json"),
                "executed": True,
                "precision_audit": audit(out, args.name),
                "runtime_validation": "pending HB_ONNXRuntime and X5 replay",
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--manifest", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--count", type=int, default=512)
    c.set_defaults(func=calibrate)
    c = sub.add_parser("compile")
    c.add_argument("--onnx", required=True)
    c.add_argument("--calibration", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--jobs", type=int, default=4)
    c.add_argument("--run", action="store_true", help="Run SDK commands; omitted means generate configuration only")
    c.set_defaults(func=compile_model)
    args = p.parse_args()
    args.func(args)
