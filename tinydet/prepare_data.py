# SPDX-License-Identifier: AGPL-3.0-only
"""Resize images without padding; normalized YOLO boxes retain their coordinates."""

import argparse
import hashlib
import json
from pathlib import Path

from .common import read_manifest, rgb_input, sha256, write_json


def prepare(args):
    import cv2
    import numpy as np
    import yaml

    names = json.loads(Path(args.classes).read_text())
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(n, str) or not n for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("classes.json must be a nonempty ordered list of unique class names")
    rows = read_manifest(args.manifest)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            (out / kind / split).mkdir(parents=True)
    result, pixels, groups = [], {}, {}
    for i, row in enumerate(rows):
        rgb = rgb_input(row["image"], normalize=False).transpose(1, 2, 0).astype(np.uint8)
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        if digest in pixels and pixels[digest] != row["split"]:
            raise ValueError(f"Train/val image leakage: {row['id']}")
        pixels[digest] = row["split"]
        if row.get("group"):
            if row["group"] in groups and groups[row["group"]] != row["split"]:
                raise ValueError(f"Capture group crosses train/val: {row['group']}")
            groups[row["group"]] = row["split"]
        text = Path(row["label"]).read_text()
        for line in text.splitlines():
            values = np.asarray([float(v) for v in line.split()])
            if values.size != 5 or not np.isfinite(values).all():
                raise ValueError(f"Expected class cx cy width height: {row['id']}")
            cls, x, y, w, h = values
            if cls != int(cls) or not 0 <= cls < len(names) or not 0 < w <= 1 or not 0 < h <= 1:
                raise ValueError(f"Invalid box/class: {row['id']}")
            if min(x - w / 2, y - h / 2) < -1e-6 or max(x + w / 2, y + h / 2) > 1 + 1e-6:
                raise ValueError(f"Box outside normalized image: {row['id']}")
        image, label = f"images/{row['split']}/{i:07}.png", f"labels/{row['split']}/{i:07}.txt"
        if not cv2.imwrite(str(out / image), rgb[:, :, ::-1]):
            raise OSError("Failed to write prepared image")
        (out / label).write_text(text)
        result.append(
            {
                "id": row["id"],
                "split": row["split"],
                "domain": row.get("domain", "default/view"),
                "group": row.get("group"),
                "image": image,
                "label": label,
                "image_sha256": sha256(out / image),
                "label_sha256": sha256(out / label),
                "source_image_sha256": sha256(row["image"]),
                "source_label_sha256": sha256(row["label"]),
                "source_image_size": list(cv2.imread(row["image"]).shape[1::-1]),
            }
        )
    if {r["split"] for r in result} != {"train", "val"}:
        raise ValueError("Both train and val are required")
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in result))
    write_json(out / "classes.json", names)
    (out / "data.yaml").write_text(
        yaml.safe_dump({"path": str(out), "train": "images/train", "val": "images/val", "names": names})
    )
    write_json(
        out / "dataset.json",
        {
            "source_manifest_sha256": sha256(args.manifest),
            "classes": names,
            "classes_sha256": sha256(out / "classes.json"),
            "input_size": [640, 352],
            "preprocessing": "whole-frame LINEAR; normalized boxes unchanged",
            "counts": {s: sum(r["split"] == s for r in result) for s in ("train", "val")},
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--classes", required=True)
    p.add_argument("--out", required=True)
    prepare(p.parse_args())
