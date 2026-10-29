# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from ultralytics.cfg import get_cfg
from ultralytics.data.converter import convert_tinyseg
from ultralytics.nn.tasks import SemanticSegmentationModel


def _manifest(tmp_path, mask, encoding):
    """Write real PNG sources with distinct train and validation images."""
    mask.save(tmp_path / "mask.png")
    rows = []
    for split, value in (("train", 32), ("val", 64)):
        cv2.imwrite(str(tmp_path / f"{split}.png"), np.full((512, 640, 3), value, np.uint8))
        rows.append({"id": split, "split": split, "image": f"{split}.png", "mask": "mask.png"})
        if encoding is not None:
            rows[-1]["mask_encoding"] = encoding
    manifest = tmp_path / "source.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest


@pytest.fixture(scope="module")
def semantic_loss():
    """Construct the real two-class semantic loss without downloading weights."""
    model = SemanticSegmentationModel("yolo26n-sem.yaml", nc=2, verbose=False)
    model.args = get_cfg()
    return model.init_criterion()


@pytest.mark.parametrize(
    "mode,encoding,foreground",
    [
        ("1", "class_ids", True),
        ("1", "binary_255", True),
        ("L", "binary_255", True),
        ("L", "class_ids", True),
        ("L", "class_ids", False),
    ],
)
def test_tinyseg_mask_encoding(tmp_path, semantic_loss, mode, encoding, foreground):
    """Keep binary foreground supervised, preserve real ignore pixels, and retain semantics on re-conversion."""
    expected = np.zeros((512, 640), np.uint8)
    if foreground:
        expected[128:384, 160:480] = 1
    if mode == "L" and encoding == "class_ids":
        expected[:32] = 255
    source = expected.astype(bool) if mode == "1" else expected * 255 if encoding == "binary_255" else expected
    mask = Image.fromarray(source)
    assert mask.mode == mode
    manifest = _manifest(tmp_path, mask, encoding)
    first = convert_tinyseg(manifest, tmp_path / "first").parent
    second = convert_tinyseg(first / "manifest.jsonl", tmp_path / "second").parent
    expected = cv2.resize(expected, (640, 352), interpolation=cv2.INTER_NEAREST)
    for output in (first, second):
        rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
        assert all(row["mask_encoding"] == "class_ids" for row in rows)
        for row in rows:
            np.testing.assert_array_equal(cv2.imread(str(output / row["mask"]), -1), expected)
    target = torch.from_numpy(cv2.imread(str(first / rows[0]["mask"]), -1)).unsqueeze(0)
    logits = torch.zeros((1, 2, 352, 640), requires_grad=True)
    loss, _ = semantic_loss(logits, {"semantic_mask": target})
    assert torch.isfinite(loss)
    loss.backward()
    gradient = logits.grad.abs().sum(dim=1)
    assert bool((target == 1).any()) == foreground
    assert (gradient[target == 0] > 0).all()
    assert (gradient[target == 1] > 0).all()
    assert (gradient[target == 255] == 0).all()


@pytest.mark.parametrize(
    "encoding,mode", [(None, "L"), ("auto", "L"), ("binary_255", "L"), ("class_ids", "P"), ("class_ids", "RGB")]
)
def test_tinyseg_rejects_ambiguous_mask_encoding(tmp_path, encoding, mode):
    """Reject missing/unknown encodings and mislabeled 8-bit 0/1 masks before publishing data.yaml."""
    mask = np.zeros((512, 640), np.uint8)
    mask[128:384, 160:480] = 1
    manifest = _manifest(tmp_path, Image.fromarray(mask).convert(mode), encoding)
    with pytest.raises(ValueError, match=r"mask_encoding|binary_255 values|grayscale"):
        convert_tinyseg(manifest, tmp_path / "output")
    assert not (tmp_path / "output/data.yaml").exists()
