# SPDX-License-Identifier: AGPL-3.0-only
"""CPU image/video inference with original-coordinate detections and visualization."""

import argparse
import json
from pathlib import Path

from .common import HEIGHT, WIDTH, write_json
from .inference import Detector, add_arguments, latency_summary

# isort: split

import cv2
import numpy as np

from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils.plotting import Annotator, colors


def predict(args):
    source = Path(args.source).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    video = source.is_file() and source.suffix.lower()[1:] in VID_FORMATS
    images = (
        sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower()[1:] in IMG_FORMATS)
        if source.is_dir()
        else [source]
    )
    if not video and (not images or any(p.suffix.lower()[1:] not in IMG_FORMATS for p in images)):
        raise ValueError("Source must be an image, an image directory or a video")
    detector = Detector(args)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    capture, writer = None, None
    timings = []
    try:
        if video:
            capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                raise OSError(f"Cannot open video: {source}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("Video must have a positive frame rate")
        with (out / "predictions.jsonl").open("w") as stream:
            index = 0
            while video or index < len(images):
                if video:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    path = source
                else:
                    path = images[index]
                    frame = cv2.imread(str(path))
                if index == 0:
                    detector.warmup(frame, args.warmup)
                boxes, elapsed = detector(frame)
                timings.append(elapsed)
                height, width = frame.shape[:2]
                boxes = boxes.clone()
                boxes[:, [0, 2]] *= width / WIDTH
                boxes[:, [1, 3]] *= height / HEIGHT
                annotator = Annotator(frame, line_width=2, pil=False, example="abc")
                for x1, y1, x2, y2, score, cls in boxes.tolist():
                    name = detector.names[int(cls)]
                    label = f"{int(cls)} {name if name.isascii() else ''} {score:.2f}"
                    annotator.box_label([x1, y1, x2, y2], label, color=colors(int(cls), bgr=True))
                rendered = annotator.result()
                if video:
                    if writer is None:
                        video_size = (width, height)
                        writer = cv2.VideoWriter(
                            str(out / "visualization.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, video_size
                        )
                        if not writer.isOpened():
                            raise OSError("Cannot create MP4 writer")
                    if (width, height) != video_size:
                        raise ValueError("Video frame size changed")
                    writer.write(rendered)
                elif not cv2.imwrite(str(out / f"{index:06d}_{path.stem}.jpg"), rendered):
                    raise OSError("Cannot write visualization image")
                stream.write(
                    json.dumps(
                        {
                            "source": str(path),
                            "frame": index,
                            "size": [width, height],
                            "detections": boxes.tolist(),
                            "latency_ms": elapsed,
                        }
                    )
                    + "\n"
                )
                index += 1
        if not timings:
            raise ValueError("Source contains no readable frames")
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
    write_json(
        out / "report.json",
        {
            "model": detector.metadata,
            "source": str(source),
            "detections_format": "xyxy pixels in original image, score, class_id",
            "latency_ms": latency_summary(timings),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, conf=0.25)
    parser.add_argument("--source", required=True, help="Image, flat image directory or video")
    predict(parser.parse_args())
