# SPDX-License-Identifier: AGPL-3.0-only
"""Official YOLO11 trainer with fixed rectangular batches for both train and val."""

import argparse
import json
import shutil
from copy import copy
from pathlib import Path

from .common import WORK, check_upstream, read_manifest, sha256, source_provenance, write_json

# isort: split

import numpy as np

from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator


class FixedDataset(YOLODataset):
    def get_labels(self):
        labels = super().get_labels()
        if any(tuple(label["shape"]) != (352, 640) for label in labels):
            raise ValueError("Run prepare_data.py first; all train/val images must be 640x352")
        return labels


def fixed_dataset(args, path, data, mode, batch):
    dataset = FixedDataset(
        img_path=path,
        imgsz=640,
        batch_size=batch or 1,
        augment=mode == "train",
        hyp=args,
        rect=True,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        data=data,
        task="detect",
    )
    if not np.all(dataset.batch_shapes == [352, 640]):
        raise ValueError("Unexpected rectangular batch shape")
    return dataset


class FixedValidator(DetectionValidator):
    def preprocess(self, batch):
        batch = super().preprocess(batch)
        if tuple(batch["img"].shape[1:]) != (3, 352, 640):
            raise ValueError("Validation input must be NCHW 352x640")
        return batch

    def build_dataset(self, img_path, mode="val", batch=None):
        return fixed_dataset(self.args, img_path, self.data, mode, batch)


class FixedTrainer(DetectionTrainer):
    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if tuple(batch["img"].shape[1:]) != (3, 352, 640):
            raise ValueError("Training input must be NCHW 352x640")
        return batch

    def build_dataset(self, img_path, mode="train", batch=None):
        return fixed_dataset(self.args, img_path, self.data, mode, batch)

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return FixedValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)


def preserve_resume(trainer):
    # Upstream strips optimizer state from final last/best files. Preserve its full checkpoint before that step.
    target = trainer.save_dir / "weights" / "resume.pt"
    temporary = target.with_suffix(".part")
    shutil.copyfile(trainer.last, temporary)
    temporary.replace(target)


def validate_data(data):
    """Bind the official trainer's YAML to exactly the prepared images, labels and classes."""
    import yaml

    data = Path(data).resolve()
    root = data.parent
    config = yaml.safe_load(data.read_text())
    if Path(config["path"]).resolve() != root or any(config[s] != f"images/{s}" for s in ("train", "val")):
        raise ValueError("data.yaml must reference its own prepared images/train and images/val")
    names = json.loads((root / "classes.json").read_text())
    dataset = json.loads((root / "dataset.json").read_text())
    if (
        config["names"] != names
        or dataset["classes"] != names
        or dataset["classes_sha256"] != sha256(root / "classes.json")
    ):
        raise ValueError("Prepared classes.json, dataset.json and data.yaml must agree")
    rows = read_manifest(root / "manifest.jsonl")
    for split in ("train", "val"):
        selected = [r for r in rows if r["split"] == split]
        if not selected:
            raise ValueError(f"Missing prepared {split} samples")
        expected_images = {Path(r["image"]) for r in selected}
        expected_labels = {Path(r["label"]) for r in selected}
        actual_images = {p.resolve() for p in (root / "images" / split).rglob("*") if p.is_file()}
        actual_labels = {p.resolve() for p in (root / "labels" / split).rglob("*") if p.is_file()}
        if expected_images != actual_images or expected_labels != actual_labels:
            raise ValueError(f"Prepared {split} files must match the frozen manifest exactly")
        for row in selected:
            image, label = Path(row["image"]), Path(row["label"])
            if label != root / "labels" / split / (image.stem + ".txt"):
                raise ValueError("Prepared image and label names must match")
    return {
        name + "_sha256": sha256(root / name)
        for name in ("data.yaml", "manifest.jsonl", "classes.json", "dataset.json")
    }


def check_training_run(output, identity, resume, model_path):
    """Reject changed data or a different run before official resume restores its arguments."""
    output, model_path = Path(output).resolve(), Path(model_path).resolve()
    if not resume:
        if output.exists():
            raise FileExistsError("Choose a new output directory or explicitly resume")
        return
    if model_path != output / "weights" / "resume.pt":
        raise ValueError("Resume must use this run's weights/resume.pt and the original output directory")
    prior = json.loads((output / "provenance.json").read_text())
    if prior["data_identity"] != identity:
        raise ValueError("Resume data identity differs from the original run")


def main(args):
    check_upstream()
    if "," in args.device:
        raise ValueError("TinyDet currently supports a single GPU or CPU/MPS; multi-GPU resume is not supported")
    data = Path(args.data).resolve()
    identity = validate_data(data)
    if not Path(args.model).is_file():
        raise ValueError("Provide an already downloaded official YOLO11n/s .pt or a compatible training checkpoint")
    output = Path(args.out).resolve()
    check_training_run(output, identity, args.resume, args.model)
    initial_sha256 = sha256(args.model)
    model = YOLO(args.model)
    if model.task != "detect":
        raise ValueError("Expected a detection checkpoint")
    output.mkdir(parents=True, exist_ok=bool(args.resume))
    provenance = (
        json.loads((output / "provenance.json").read_text())
        if args.resume
        else {
            "initial_checkpoint_sha256": initial_sha256,
            "data_identity": identity,
            "input": [3, 352, 640],
            "resumes": [],
        }
    )
    if args.resume:
        provenance["resumes"].append({"checkpoint_sha256": initial_sha256, "source": source_provenance()})
    else:
        provenance["source"] = source_provenance()
    provenance["status"] = "running"
    provenance["recipe"] = "YOLO11n initial baseline; not a validated product accuracy recipe"
    write_json(output / "provenance.json", provenance)
    model.add_callback("on_model_save", preserve_resume)
    model.train(
        trainer=FixedTrainer,
        data=str(data),
        project=str(output.parent),
        name=output.name,
        exist_ok=True,
        resume=args.model if args.resume else False,
        device=args.device,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        workers=args.workers,
        imgsz=640,
        rect=True,
        multi_scale=0.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        close_mosaic=0,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        seed=42,
        deterministic=True,
        amp=args.device != "cpu",
        save_period=5,
        plots=False,
    )
    provenance["status"] = "completed"
    write_json(output / "provenance.json", provenance)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="data.yaml produced by prepare_data.py")
    p.add_argument(
        "--model", default=str(WORK / "weights" / "yolo11n.pt"), help="Local trusted checkpoint; never auto-downloaded"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true", help="Use --model <run>/weights/resume.pt with the same run/data")
    main(p.parse_args())
