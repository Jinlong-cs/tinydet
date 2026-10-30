# SPDX-License-Identifier: AGPL-3.0-only

import hashlib
import json
import os
import subprocess
from pathlib import Path

UPSTREAM = "922563eab42b778445e55a2edbabb6974eed8e4b"
HEIGHT, WIDTH = 352, 640
REPO = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("TINY_WORK_DIR", "work")).resolve()
for key, suffix in {
    "YOLO_CONFIG_DIR": "cache/ultralytics",
    "TORCH_HOME": "cache/torch",
    "MPLCONFIGDIR": "cache/matplotlib",
    "XDG_CACHE_HOME": "cache",
    "TMPDIR": "tmp",
}.items():
    directory = WORK / suffix
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(directory)
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def sha256(path):
    with Path(path).open("rb") as f:
        digest = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_manifest(path):
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = set()
    for row in rows:
        if row["id"] in ids or row["split"] not in {"train", "val"}:
            raise ValueError(f"Duplicate id or invalid split: {row['id']}")
        ids.add(row["id"])
        for key in ("image", "label", "mask"):
            if key in row:
                row[key] = str((path.parent / row[key]).resolve())
                expected = row.get(key + "_sha256")
                if expected and sha256(row[key]) != expected:
                    raise ValueError(f"Prepared {key} changed: {row['id']}")
    if not rows:
        raise ValueError("Empty manifest")
    return rows


def rgb_input(path, normalize=True):
    import cv2
    import numpy as np

    image = cv2.imread(str(path))
    if image is None or image.shape not in {(352, 640, 3), (480, 640, 3), (512, 640, 3)}:
        raise ValueError(f"Expected 640x352, 640x480 or 640x512 image: {path}")
    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    result = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)
    return result / 255 if normalize else result


def check_upstream():
    import ultralytics

    if Path(ultralytics.__file__).resolve() != REPO / "ultralytics" / "__init__.py":
        raise RuntimeError(
            "TinyDet must use this fork's ultralytics source; install the checkout with pip install -e ."
        )
    if ultralytics.__version__ != "8.4.192":
        raise RuntimeError("This adapter is validated against upstream 922563e (Ultralytics 8.4.192)")


def source_provenance():

    def git(*args):
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()

    return {
        "upstream_commit": UPSTREAM,
        "fork_commit": git("rev-parse", "HEAD"),
        "working_tree_changes": git("status", "--porcelain", "--untracked-files=normal").splitlines(),
        "source_root": str(REPO),
    }
