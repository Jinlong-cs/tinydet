# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare train-only RGB calibration data for X5."""

import argparse
from pathlib import Path

from ..common import read_manifest, rgb_input, sha256, write_json


def calibrate(args):
    import numpy as np

    rows = [r for r in read_manifest(args.manifest) if r["split"] == "train"]
    if not 0 < args.count <= len(rows):
        raise ValueError("Calibration count must not exceed unique train samples")
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
    out.mkdir(parents=True, exist_ok=True)
    if any(path.name != ".gitkeep" for path in out.iterdir()):
        raise FileExistsError(f"Calibration output must be empty: {out}")
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default=str(Path(__file__).with_name("calib_data")))
    p.add_argument("--count", type=int, default=512)
    calibrate(p.parse_args())
