# TinyPerception

TinyPerception is the shared [Ultralytics](https://github.com/ultralytics/ultralytics) fork for
TinyDet object detection and TinySeg semantic segmentation. Task entry points, configurations,
class mappings and export workflows remain separate; both reuse upstream model and training implementations.
Both tasks use the same upstream base; their dataset labels and preprocessing remain task-specific.

## TinyDet (YOLO11n)

Detection adapters live in `tinydet/`. See [TinySeg](#tinyseg-yolo26n-sem) for semantic data preparation.

### Setup

Use Python 3.11+ and install this checkout:

```bash
git clone -b main https://github.com/Jinlong-cs/tinyperception.git
cd tinyperception
python3.11 -m venv venv
source venv/bin/activate
export TINY_WORK_DIR="$PWD/work"
export PIP_CACHE_DIR="$TINY_WORK_DIR/cache/pip"
export TMPDIR="$TINY_WORK_DIR/tmp"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TINY_WORK_DIR/weights"
python -m pip install -e ".[tinydet]"
```

### Data preparation

Provide an ordered `classes.json`, such as `["person", "chair"]`, and YOLO labels in
`class_id cx cy width height` format with normalized coordinates. Empty label files must be
reviewed negatives; missing files raise an error. JSONL paths are relative to the manifest:

```jsonl
{"id":"001","split":"train","domain":"toc/front","group":"session_a","image":"images/001.png","label":"labels/001.txt"}
{"id":"002","split":"val","domain":"toc/front","group":"session_b","image":"images/002.png","label":"labels/002.txt"}
```

Inputs are 640×352, 640×480 or 640×512, resized over the full frame to 640×352 with LINEAR interpolation.
Normalized boxes retain their coordinates. Split by capture sequence; groups and duplicate pixels
cannot cross train/val. Partial labels, ignore regions and a separate test split are not supported.

```bash
python -m tinydet.prepare_data --manifest /path/to/source.jsonl \
  --classes /path/to/classes.json --out work/data
```

The output directory must not exist. It contains matching images/labels, `data.yaml`,
`classes.json`, `dataset.json` and a manifest with source/output hashes. Training validates the
prepared file set and class order; interrupted runs require the original data and output directory.

### Training

Place trusted official [YOLO11n weights](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)
in `work/weights/yolo11n.pt` (SHA256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`).
Weights are not downloaded automatically; `--model` accepts another local checkpoint.

```bash
python -m tinydet.train --data work/data/data.yaml --out work/train --device 0
# Resume an interrupted run with the original data, output directory and device environment.
python -m tinydet.train --data work/data/data.yaml --model work/train/weights/resume.pt \
  --out work/train --device 0 --resume
```

Training and validation use RGB NCHW 352×640 inputs scaled to 0–1. Only one device is supported;
use `--device cpu` for CPU. The full `resume.pt` retains optimizer/EMA/scaler state.
See each module's `--help` for additional options.

### ONNX export and X5 compilation

```bash
python -m tinydet.export --checkpoint work/train/weights/best.pt \
  --check-manifest work/data/manifest.jsonl --out work/export
python -m tinydet.quantization.prepare_calibration --manifest work/data/manifest.jsonl \
  --count 512
```

The opset11 graph accepts FLOAT32 `[1,3,352,640]` and returns
`cls_s8, box_s8, cls_s16, box_s16, cls_s32, box_s32`. Grids are 44×80, 22×40 and 11×20;
class channels follow the checkpoint, and box heads have 64 channels (DFL reg_max16).
Decoding/NMS runs outside the graph; class changes require matching deployment metadata.

By default, export checks raw heads, decoded boxes and scores on up to 64 validation images plus random input.
Calibration uses train samples only; `--count` cannot exceed the train set size. Files contain
RGB 0–255 float32 CHW values, and the mapper applies 1/255 once. Export records source, class and model
hashes; calibration records source-manifest, sample and RGB-file hashes.

Calibration defaults to `tinydet/quantization/calib_data/`, with `.rgb` files in `rgb/` and a
`manifest.json` beside it. The script creates the directory if absent and rejects nonempty output.
The directory is ignored by Git. Use `--out` and the compiler's `--calibration` to override these script-relative defaults.

Both workflows use [tinydet/quantization/config.yaml](tinydet/quantization/config.yaml) and an x86 Linux OpenExplorer v1.2.8 environment.

```bash
IMAGE=registry.d-robotics.cc/deliver/hub.hobot.cc/aitools/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
```

#### Automatic compilation

```bash
docker run --rm --cpus=4 --memory=16g \
  -v "$PWD:/workspace" -w /workspace \
  -e TINY_WORK_DIR=/workspace/work -e PYTHONDONTWRITEBYTECODE=1 \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  "$IMAGE" python3 -m tinydet.quantization.x5_compile \
  --onnx work/export/model.onnx \
  --out work/ptq --name yolo11n_640x352_int8_nv12 --run
```

Omit `--run` to generate configuration only; use a new output directory. The configuration requests
bayes-e, NV12 BT601 video range, a single 1/255 scale, INT8 and latency/O3. `compiled.json` is written
only after SDK execution and node/input audits pass. HB/X5 validation remains a separate step.
Local CPU, ONNX and configuration checks do not establish product accuracy or board acceptance.

#### Manual compilation

Edit `tinydet/quantization/config.yaml` with paths visible inside the SDK container. Its defaults match the export
and calibration directories above and write to `work/ptq-manual`. The Python workflow loads the same
defaults, then overrides model/calibration/output paths, model name and jobs from its CLI arguments.

From the repository root, open the same SDK image:

```bash
docker run --rm -it --cpus=4 --memory=16g \
  -v "$PWD:/workspace" -w /workspace \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 "$IMAGE" bash
```

Inside the container:

```bash
mkdir -p /workspace/work/ptq-manual
cd /workspace/work/ptq-manual
hb_mapper checker --model-type onnx --model /workspace/work/export/model.onnx --march bayes-e
hb_mapper makertbin --config /workspace/tinydet/quantization/config.yaml --model-type onnx
hb_model_info /workspace/work/ptq-manual/yolo11n_640x352_int8_nv12.bin
```

Use a fresh output directory and update the commands if you change the YAML paths or model name.
Keep calibration as RGB float32 CHW values in 0–255; the YAML applies 1/255 once. This direct SDK path
does not perform the Python wrapper's provenance checks or write `compiled.json`; retain the SDK logs
and verify actual precision, placement and NV12 BT601 video range before board validation.

The upstream base is `922563eab42b778445e55a2edbabb6974eed8e4b` (8.4.192). Adapters were migrated from
[tinydet-legacy](https://github.com/Jinlong-cs/tinydet-legacy) at `7e9a16edd4ae25c8301a6fba82e57b4ae518a4e1`.
Raw-head/Attention export code derives from [D-Robotics](https://github.com/D-Robotics/rdk_model_zoo/blob/rdk_x5/samples/vision/ultralytics_yolo/conversion/export_monkey_patch.py);
attribution and its [Apache-2.0 license](tinydet/docs/LICENSE-D-Robotics) are retained.

## TinySeg (YOLO26n-sem)

TinySeg provides binary drivable-area data preparation for the ToC robot.

### Data preparation

The [dataset configuration](ultralytics/cfg/datasets/tinyseg.yaml) defines `0=non_drivable`, `1=drivable`,
and `255=ignore`. Inputs are 640×352 or 640×512; images use full-frame LINEAR resize to 640×352 and native
PNG masks use NEAREST. Legacy normalized polygons use `0=drivable`, `1=exclusion` and are rasterized at
the target size, with exclusions taking priority.

Provide a JSONL manifest with unique IDs, nonempty train/val splits, and exactly one `label` or `mask`
per image. Paths are relative to the manifest. Split by capture sequence and supply `group` to prevent
sequence leakage; `domain` retains the product/view. Empty polygon files must be reviewed negatives.
Grayscale PNG rows require `mask_encoding`: `binary_255` means 0=background/255=foreground; `class_ids` means
0=background/1=foreground/255=ignore. Native 1-bit PNGs use the upstream decoder to preserve 0/1.
Missing or unknown encodings raise an error; no pixel-value guessing is performed.

```jsonl
{"id":"front-01","split":"train","group":"capture-01","image":"raw/front-01.png","label":"labels/front-01.txt"}
{"id":"back-02","split":"val","group":"capture-02","image":"raw/back-02.png","mask":"masks/back-02.png","mask_encoding":"class_ids"}
```

```python
from ultralytics.data.converter import convert_tinyseg

data_yaml = convert_tinyseg("source.jsonl", "work/tinyseg")
```

The output directory must not exist. It contains matching `images/{train,val}` and `masks/{train,val}`
PNGs, a provenance `manifest.jsonl`, `dataset.json`, and `data.yaml`. Optional `image_sha256` and
`label_sha256`/`mask_sha256` fields are verified; source/output hashes are recorded. Invalid annotations
and cross-split pixel/group leakage raise errors. Failed conversions can leave partial output without
`data.yaml`; inspect and remove it or retry in a new directory.
Prepared manifests always declare `mask_encoding: class_ids` for reuse.

This prepares dataset files only; native semantic training transforms can still change tensor geometry.
Fixed 640×352 semantic training, the historical recipe, opset11 export, and X5 INT8 compilation will follow
in separate reviewed PRs.

The upstream README and [AGPL-3.0 license](LICENSE) are preserved below.

---

<div align="center">
  <p>
    <a href="https://www.ultralytics.com/yolo/yolo27?utm_source=github&utm_medium=social&utm_campaign=yolo27-launch-2026&utm_content=banner" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="Ultralytics YOLO banner"></a>
  </p>

[中文](https://docs.ultralytics.com/zh) | [한국어](https://docs.ultralytics.com/ko) | [日本語](https://docs.ultralytics.com/ja) | [Русский](https://docs.ultralytics.com/ru) | [Deutsch](https://docs.ultralytics.com/de) | [Français](https://docs.ultralytics.com/fr) | [Español](https://docs.ultralytics.com/es) | [Português](https://docs.ultralytics.com/pt) | [Türkçe](https://docs.ultralytics.com/tr) | [Tiếng Việt](https://docs.ultralytics.com/vi) | [العربية](https://docs.ultralytics.com/ar) <br>

<div>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://clickpy.clickhouse.com/dashboard/ultralytics"><img src="https://static.pepy.tech/badge/ultralytics" alt="Ultralytics Downloads"></a>
    <a href="https://discord.com/invite/ultralytics"><img alt="Ultralytics Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>
    <a href="https://community.ultralytics.com"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a>
    <a href="https://www.reddit.com/r/ultralytics/"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run Ultralytics on Gradient"></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Ultralytics In Colab"></a>
    <a href="https://www.kaggle.com/models/ultralytics/yolo26"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open Ultralytics In Kaggle"></a>
    <a href="https://mybinder.org/v2/gh/ultralytics/ultralytics/HEAD?labpath=examples%2Ftutorial.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Open Ultralytics In Binder"></a>
</div>
</div>
<br>

<div align="center">
  <a href="https://trendshift.io/repositories/1556?utm_source=repository-badge&amp;utm_medium=badge&amp;utm_campaign=badge-repository-1556" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/repositories/1556" alt="ultralytics%2Fultralytics | Trendshift" width="250" height="55"/></a>
</div>
<br>

[Ultralytics](https://www.ultralytics.com) creates cutting-edge, state-of-the-art (SOTA) [YOLO models](https://www.ultralytics.com/yolo) built on years of foundational research in computer vision and AI. Constantly updated for performance and flexibility, our models are **fast**, **accurate**, and **easy to use**. They excel at [object detection](https://docs.ultralytics.com/tasks/detect), [instance segmentation](https://docs.ultralytics.com/tasks/segment), [semantic segmentation](https://docs.ultralytics.com/tasks/semantic), [depth estimation](https://docs.ultralytics.com/tasks/depth), [image classification](https://docs.ultralytics.com/tasks/classify), [pose estimation](https://docs.ultralytics.com/tasks/pose), and [oriented object detection](https://docs.ultralytics.com/tasks/obb) tasks, and can [track](https://docs.ultralytics.com/modes/track) detected objects across video frames.

Find detailed documentation in the [Ultralytics Docs](https://docs.ultralytics.com). Get support via [GitHub Issues](https://github.com/ultralytics/ultralytics/issues/new/choose). Join discussions on [Discord](https://discord.com/invite/ultralytics), [Reddit](https://www.reddit.com/r/ultralytics/), and the [Ultralytics Community Forums](https://community.ultralytics.com)!

Request an Enterprise License for commercial use at [Ultralytics Licensing](https://www.ultralytics.com/license).

<a href="https://platform.ultralytics.com/ultralytics/yolo26" target="_blank">
  <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/refs/heads/main/yolo/performance-comparison.png" alt="YOLO26 performance plots">
</a>

<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="2%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="2%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="2%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="2%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="2%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="2%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="2%" alt="Ultralytics Discord"></a>
</div>

## 📄 Documentation

See below for quickstart installation and usage examples. For comprehensive guidance on training, validation, prediction, and deployment, refer to our full [Ultralytics Docs](https://docs.ultralytics.com).

<details open>
<summary>Install</summary>

Install the `ultralytics` package, including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml), in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/) [![Ultralytics Downloads](https://static.pepy.tech/badge/ultralytics)](https://clickpy.clickhouse.com/dashboard/ultralytics) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics/)

```bash
pip install ultralytics
```

For alternative installation methods, including [Conda](https://anaconda.org/conda-forge/ultralytics), [Docker](https://hub.docker.com/r/ultralytics/ultralytics), and building from source via Git, please consult the [Quickstart Guide](https://docs.ultralytics.com/quickstart).

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/ultralytics?logo=condaforge)](https://anaconda.org/conda-forge/ultralytics) [![Docker Image Version](https://img.shields.io/docker/v/ultralytics/ultralytics?sort=semver&logo=docker)](https://hub.docker.com/r/ultralytics/ultralytics) [![Ultralytics Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker)](https://hub.docker.com/r/ultralytics/ultralytics)

</details>

<details open>
<summary>Usage</summary>

### CLI

You can use Ultralytics YOLO directly from the Command Line Interface (CLI) with the `yolo` command:

```bash
# Predict using a pretrained YOLO model (e.g., YOLO26n) on an image
yolo predict model=yolo26n.pt source='https://ultralytics.com/images/bus.jpg'
```

The `yolo` command supports various tasks and modes, accepting additional arguments like `imgsz=640`. Explore the YOLO [CLI Docs](https://docs.ultralytics.com/usage/cli) for more examples.

### Python

Ultralytics YOLO can also be integrated directly into your Python projects. It accepts the same [configuration arguments](https://docs.ultralytics.com/usage/cfg) as the CLI:

```python
from ultralytics import YOLO

# Load a pretrained YOLO26n model
model = YOLO("yolo26n.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="coco8.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

# Evaluate the model's performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")  # Predict on an image
results[0].show()  # Display results

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model
```

Discover more examples in the YOLO [Python Docs](https://docs.ultralytics.com/usage/python).

</details>

## ✨ Models

> **[YOLO27 — Coming Soon](https://docs.ultralytics.com/models/yolo27/):** Preview the upcoming family, currently undergoing final R&D. Models are not yet available, and no launch date is set.

Ultralytics supports a wide range of YOLO models, from early versions like [YOLOv3](https://docs.ultralytics.com/models/yolov3) to the latest [YOLO26](https://docs.ultralytics.com/models/yolo26). The tables below showcase YOLO26 models pretrained on [COCO](https://docs.ultralytics.com/datasets/detect/coco) for [Detection](https://docs.ultralytics.com/tasks/detect), [Segmentation](https://docs.ultralytics.com/tasks/segment), and [Pose Estimation](https://docs.ultralytics.com/tasks/pose). [Semantic Segmentation](https://docs.ultralytics.com/tasks/semantic) models are pretrained on [Cityscapes](https://docs.ultralytics.com/datasets/semantic/cityscapes), [Depth Estimation](https://docs.ultralytics.com/tasks/depth) models are pretrained on a broad multi-dataset mix and evaluated on [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html), [Classification](https://docs.ultralytics.com/tasks/classify) models are pretrained on [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet), and [OBB](https://docs.ultralytics.com/tasks/obb) models are pretrained on [DOTAv1](https://docs.ultralytics.com/datasets/obb/dota-v2#dota-v10). [Tracking](https://docs.ultralytics.com/modes/track) mode is compatible with Detection, Segmentation, Pose, and OBB models. The listed [pretrained models](https://docs.ultralytics.com/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

<a href="https://docs.ultralytics.com/tasks" target="_blank">
    <img width="100%" src="https://cdn.ul.run/i/c99d914c3958d0755b5a3d7204b6f24a.avif" alt="Ultralytics YOLO supported tasks">
</a>
<br>
<br>

<details open><summary>Detection (COCO)</summary>

Explore the [Detection Docs](https://docs.ultralytics.com/tasks/detect) for usage examples. These models are trained on the [COCO dataset](https://cocodataset.org/), featuring 80 object classes.

| Model                                                                  | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | mAP<sup>val<br>50-95(e2e)</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------------------------------------------------------------------- | --------------------------- | -------------------------- | ------------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n) | 640                         | 40.9                       | 40.1                            | 38.9 ± 0.7                           | 1.7 ± 0.0                                 | 2.4                      | 5.5                     |
| [YOLO26s](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s) | 640                         | 48.6                       | 47.8                            | 87.2 ± 0.9                           | 2.5 ± 0.0                                 | 9.5                      | 20.9                    |
| [YOLO26m](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m) | 640                         | 53.1                       | 52.5                            | 220.0 ± 1.4                          | 4.7 ± 0.1                                 | 20.4                     | 68.4                    |
| [YOLO26l](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l) | 640                         | 55.0                       | 54.4                            | 286.2 ± 2.0                          | 6.2 ± 0.2                                 | 24.8                     | 86.8                    |
| [YOLO26x](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x) | 640                         | 57.5                       | 56.9                            | 525.8 ± 4.0                          | 11.8 ± 0.2                                | 55.7                     | 194.4                   |

- **mAP<sup>val</sup>** values refer to single-model single-scale performance on the [COCO val2017](https://cocodataset.org/) dataset. See [YOLO Performance Metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics) for details. <br>Reproduce with `yolo val detect data=coco.yaml device=0`. Add `nms=False` to reproduce the e2e column.
- **Speed** metrics are averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. CPU speeds measured with [ONNX](https://onnx.ai/) export. GPU speeds measured with [TensorRT](https://developer.nvidia.com/tensorrt) export. <br>Reproduce with `yolo val detect data=coco.yaml batch=1 device=0|cpu nms=False`

</details>

<details><summary>Segmentation (COCO)</summary>

Refer to the [Segmentation Docs](https://docs.ultralytics.com/tasks/segment) for usage examples. These models are trained on [COCO-Seg](https://docs.ultralytics.com/datasets/segment/coco), including 80 classes.

| Model                                                                          | size<br><sup>(pixels)</sup> | mAP<sup>box<br>50-95(e2e)</sup> | mAP<sup>mask<br>50-95(e2e)</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ------------------------------------------------------------------------------ | --------------------------- | ------------------------------- | -------------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| [YOLO26n-seg](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-seg) | 640                         | 39.6                            | 33.9                             | 53.3 ± 0.5                           | 2.1 ± 0.0                                 | 2.7                      | 9.3                     |
| [YOLO26s-seg](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s-seg) | 640                         | 47.3                            | 40.0                             | 118.4 ± 0.9                          | 3.3 ± 0.0                                 | 10.4                     | 34.5                    |
| [YOLO26m-seg](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m-seg) | 640                         | 52.5                            | 44.1                             | 328.2 ± 2.4                          | 6.7 ± 0.1                                 | 23.6                     | 121.7                   |
| [YOLO26l-seg](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l-seg) | 640                         | 54.4                            | 45.5                             | 387.0 ± 3.7                          | 8.0 ± 0.1                                 | 28.0                     | 140.1                   |
| [YOLO26x-seg](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x-seg) | 640                         | 56.5                            | 47.0                             | 787.0 ± 6.8                          | 16.4 ± 0.1                                | 62.8                     | 314.0                   |

- **mAP<sup>val</sup>** values are for single-model single-scale on the [COCO val2017](https://cocodataset.org/) dataset. See [YOLO Performance Metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics) for details. <br>Reproduce with `yolo val segment data=coco.yaml device=0 nms=False`
- **Speed** metrics are averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. CPU speeds measured with [ONNX](https://onnx.ai/) export. GPU speeds measured with [TensorRT](https://developer.nvidia.com/tensorrt) export. <br>Reproduce with `yolo val segment data=coco.yaml batch=1 device=0|cpu nms=False`

</details>

<details><summary>Semantic Segmentation (Cityscapes)</summary>

See the [Semantic Segmentation Docs](https://docs.ultralytics.com/tasks/semantic) for usage examples. These models are trained on [Cityscapes](https://docs.ultralytics.com/datasets/semantic/cityscapes), including 19 classes.

| Model                                                                          | size<br><sup>(pixels)</sup> | mIoU<sup>val</sup> | Speed<br><sup>RTX3090 PyTorch<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ------------------------------------------------------------------------------ | --------------------------- | ------------------ | ------------------------------------------- | ------------------------ | ----------------------- |
| [YOLO26n-sem](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-sem) | 1024 &times; 2048           | 78.3               | 4.4 ± 0.0                                   | 1.6                      | 23.8                    |
| [YOLO26s-sem](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s-sem) | 1024 &times; 2048           | 80.8               | 8.4 ± 0.0                                   | 6.5                      | 91.0                    |
| [YOLO26m-sem](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m-sem) | 1024 &times; 2048           | 82.0               | 19.9 ± 0.1                                  | 14.3                     | 305.5                   |
| [YOLO26l-sem](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l-sem) | 1024 &times; 2048           | 82.9               | 26.5 ± 0.1                                  | 17.8                     | 388.2                   |
| [YOLO26x-sem](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x-sem) | 1024 &times; 2048           | 83.6               | 48.9 ± 0.2                                  | 40.1                     | 866.9                   |

- **mIoU<sup>val</sup>** values are for single-model single-scale on the [Cityscapes](https://www.cityscapes-dataset.com/) validation set. <br>Reproduce with `yolo semantic val data=cityscapes.yaml device=0 imgsz=2048`
- **Speed** metrics are averaged over Cityscapes validation images using an RTX3090 instance. <br>Reproduce with `yolo semantic val data=cityscapes.yaml batch=1 device=0|cpu imgsz=2048`

</details>

<details><summary>Depth Estimation (NYU Depth V2)</summary>

See the [Depth Estimation Docs](https://docs.ultralytics.com/tasks/depth) for usage examples. These models are pretrained on a broad multi-dataset mix and evaluated on the [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) Eigen test split, predicting per-pixel depth in meters.

| Model                                                                              | size<br><sup>(pixels)</sup> | delta1<sup>NYU</sup> | abs_rel<sup>NYU</sup> | rmse<sup>NYU</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ---------------------------------------------------------------------------------- | --------------------------- | -------------------- | --------------------- | ------------------ | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| [YOLO26n-depth](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-depth) | 768                         | 0.882                | 0.109                 | 0.414              | 272.0 ± 27.2                         | 2.7 ± 0.1                                 | 6.3                      | 46.9                    |
| [YOLO26s-depth](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s-depth) | 768                         | 0.896                | 0.104                 | 0.399              | 393.7 ± 13.1                         | 3.8 ± 0.0                                 | 13.2                     | 68.0                    |
| [YOLO26m-depth](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m-depth) | 768                         | 0.921                | 0.089                 | 0.364              | 621.5 ± 49.7                         | 6.0 ± 0.1                                 | 23.3                     | 130.4                   |
| [YOLO26l-depth](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l-depth) | 768                         | 0.930                | 0.083                 | 0.351              | 821.9 ± 50.7                         | 7.7 ± 0.1                                 | 27.7                     | 157.0                   |
| [YOLO26x-depth](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x-depth) | 768                         | 0.933                | 0.080                 | 0.344              | 1240.9 ± 73.3                        | 13.6 ± 0.2                                | 57.0                     | 301.7                   |

- **delta1<sup>NYU</sup>** is the percentage of pixels where the predicted depth is within a factor of 1.25 of the ground truth, on the NYU Depth V2 Eigen test split (654 images) with multi-scale + horizontal-flip TTA and log-least-squares alignment.
- Single-scale accuracy without TTA is reproducible with `yolo depth val model=yolo26n-depth.pt data=nyu-depth.yaml imgsz=768 device=0` (substitute `model=` for each size), which uses median (scale-only) alignment and scores lower: delta1 0.783 (n), 0.793 (s), 0.840 (m), 0.853 (l), 0.860 (x).
- **abs_rel** is the mean absolute relative error between predicted and ground-truth depth values.
- **rmse** is the root mean squared error in meters.
- **Speed** is inference-only latency (pre/post-processing excluded) at `imgsz=768`, `batch=1`, reported as mean ± std over timed runs after warmup. **CPU ONNX** is ONNX Runtime fp32 on a 32-core Intel Xeon (Skylake); **T4 TensorRT10** is TensorRT fp16 on a Tesla T4.
- **params** and **FLOPs** are measured at 768×768, the training resolution of the released weights.

</details>

<details><summary>Classification (ImageNet)</summary>

Consult the [Classification Docs](https://docs.ultralytics.com/tasks/classify) for usage examples. These models are trained on [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet), covering 1000 classes.

| Model                                                                          | size<br><sup>(pixels)</sup> | acc<br><sup>top1</sup> | acc<br><sup>top5</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B) at 224</sup> |
| ------------------------------------------------------------------------------ | --------------------------- | ---------------------- | ---------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ------------------------------ |
| [YOLO26n-cls](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-cls) | 224                         | 71.4                   | 90.1                   | 5.0 ± 0.3                            | 1.1 ± 0.0                                 | 2.8                      | 0.4                            |
| [YOLO26s-cls](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s-cls) | 224                         | 76.0                   | 92.9                   | 7.9 ± 0.2                            | 1.3 ± 0.0                                 | 6.7                      | 1.5                            |
| [YOLO26m-cls](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m-cls) | 224                         | 78.1                   | 94.2                   | 17.2 ± 0.4                           | 2.0 ± 0.0                                 | 11.6                     | 4.8                            |
| [YOLO26l-cls](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l-cls) | 224                         | 79.0                   | 94.6                   | 23.2 ± 0.3                           | 2.8 ± 0.0                                 | 14.1                     | 6.0                            |
| [YOLO26x-cls](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x-cls) | 224                         | 79.9                   | 95.0                   | 41.4 ± 0.9                           | 3.8 ± 0.0                                 | 29.6                     | 13.5                           |

- **acc** values represent model accuracy on the [ImageNet](https://www.image-net.org/) dataset validation set. <br>Reproduce with `yolo val classify data=path/to/ImageNet device=0`
- **Speed** metrics are averaged over ImageNet val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. CPU speeds measured with [ONNX](https://onnx.ai/) export. GPU speeds measured with [TensorRT](https://developer.nvidia.com/tensorrt) export. <br>Reproduce with `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`

</details>

<details><summary>Pose (COCO)</summary>

See the [Pose Estimation Docs](https://docs.ultralytics.com/tasks/pose) for usage examples. These models are trained on [COCO-Pose](https://docs.ultralytics.com/datasets/pose/coco), focusing on the 'person' class.

| Model                                                                            | size<br><sup>(pixels)</sup> | mAP<sup>pose<br>50-95(e2e)</sup> | mAP<sup>pose<br>50(e2e)</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| -------------------------------------------------------------------------------- | --------------------------- | -------------------------------- | ----------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| [YOLO26n-pose](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-pose) | 640                         | 57.2                             | 83.3                          | 40.3 ± 0.5                           | 1.8 ± 0.0                                 | 2.9                      | 7.6                     |
| [YOLO26s-pose](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s-pose) | 640                         | 63.0                             | 86.6                          | 85.3 ± 0.9                           | 2.7 ± 0.0                                 | 10.4                     | 24.1                    |
| [YOLO26m-pose](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m-pose) | 640                         | 68.8                             | 89.6                          | 218.0 ± 1.5                          | 5.0 ± 0.1                                 | 21.5                     | 73.3                    |
| [YOLO26l-pose](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l-pose) | 640                         | 70.4                             | 90.5                          | 275.4 ± 2.4                          | 6.5 ± 0.1                                 | 25.9                     | 91.7                    |
| [YOLO26x-pose](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x-pose) | 640                         | 71.6                             | 91.6                          | 565.4 ± 3.0                          | 12.2 ± 0.2                                | 57.6                     | 202.3                   |

- **mAP<sup>val</sup>** values are for single-model single-scale on the [COCO Keypoints val2017](https://docs.ultralytics.com/datasets/pose/coco) dataset. See [YOLO Performance Metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics) for details. <br>Reproduce with `yolo val pose data=coco-pose.yaml device=0 nms=False`
- **Speed** metrics are averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. CPU speeds measured with [ONNX](https://onnx.ai/) export. GPU speeds measured with [TensorRT](https://developer.nvidia.com/tensorrt) export. <br>Reproduce with `yolo val pose data=coco-pose.yaml batch=1 device=0|cpu nms=False`

</details>

<details><summary>Oriented Bounding Boxes (DOTAv1)</summary>

Check the [OBB Docs](https://docs.ultralytics.com/tasks/obb) for usage examples. These models are trained on [DOTAv1](https://docs.ultralytics.com/datasets/obb/dota-v2#dota-v10), including 15 classes.

| Model                                                                          | size<br><sup>(pixels)</sup> | mAP<sup>test<br>50-95(e2e)</sup> | mAP<sup>test<br>50(e2e)</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ------------------------------------------------------------------------------ | --------------------------- | -------------------------------- | ----------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| [YOLO26n-obb](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-obb) | 1024                        | 52.4                             | 78.9                          | 97.7 ± 0.9                           | 2.8 ± 0.0                                 | 2.4                      | 14.8                    |
| [YOLO26s-obb](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s-obb) | 1024                        | 54.8                             | 80.9                          | 218.0 ± 1.4                          | 4.9 ± 0.1                                 | 9.8                      | 56.7                    |
| [YOLO26m-obb](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m-obb) | 1024                        | 55.3                             | 81.0                          | 579.2 ± 3.8                          | 10.2 ± 0.3                                | 21.2                     | 184.9                   |
| [YOLO26l-obb](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l-obb) | 1024                        | 56.2                             | 81.6                          | 735.6 ± 3.1                          | 13.0 ± 0.2                                | 25.6                     | 232.4                   |
| [YOLO26x-obb](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x-obb) | 1024                        | 56.7                             | 81.7                          | 1485.7 ± 11.5                        | 30.5 ± 0.9                                | 57.6                     | 520.1                   |

- **mAP<sup>test</sup>** values are for single-model multiscale performance on the [DOTAv1 test set](https://captain-whu.github.io/DOTA/dataset.html). <br>Reproduce by `yolo val obb data=DOTAv1.yaml device=0 split=test nms=False save_json=True` and submit merged results to the [DOTA evaluation server](https://captain-whu.github.io/DOTA/evaluation.html).
- **Speed** metrics are averaged over [DOTAv1 val images](https://docs.ultralytics.com/datasets/obb/dota-v2#dota-v10) using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. CPU speeds measured with [ONNX](https://onnx.ai/) export. GPU speeds measured with [TensorRT](https://developer.nvidia.com/tensorrt) export. <br>Reproduce by `yolo val obb data=DOTAv1.yaml batch=1 device=0|cpu nms=False`

</details>

## 🧩 Integrations

Our key integrations with leading AI platforms extend the functionality of Ultralytics' offerings, enhancing tasks like dataset labeling, training, visualization, and model management. Discover how Ultralytics, in collaboration with partners like [Weights & Biases](https://docs.ultralytics.com/integrations/weights-biases), [Comet ML](https://docs.ultralytics.com/integrations/comet), [Roboflow](https://docs.ultralytics.com/integrations/roboflow), and [Intel OpenVINO](https://docs.ultralytics.com/integrations/openvino), can optimize your AI workflow. Explore more at [Ultralytics Integrations](https://docs.ultralytics.com/integrations).

<a href="https://platform.ultralytics.com" target="_blank">
    <img width="100%" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics active learning integrations">
</a>

## 🤝 Contribute

We thrive on community collaboration! Ultralytics YOLO wouldn't be the SOTA framework it is without contributions from developers like you. Please see our [Contributing Guide](https://docs.ultralytics.com/help/contributing) to get started. We also welcome your feedback—share your experience by completing our [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). A huge **Thank You** 🙏 to everyone who contributes!

<!-- SVG image from https://opencollective.com/ultralytics/contributors.svg?width=1280 -->

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

We look forward to your contributions to help make the Ultralytics ecosystem even better!

## 📜 License

Ultralytics offers two licensing options to suit different needs:

- **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/license/agpl-3.0) open-source license is perfect for students, researchers, and enthusiasts. It encourages open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for full details.
- **Ultralytics Enterprise License**: For development and production use, this license enables seamless integration of Ultralytics software and AI models into business products and services, including internal tools, automated workflows, and production deployments, bypassing the open-source requirements of AGPL-3.0. To get started, please contact us via [Ultralytics Licensing](https://www.ultralytics.com/license).

## 📞 Contact

For bug reports and feature requests related to Ultralytics software, please visit [GitHub Issues](https://github.com/ultralytics/ultralytics/issues). For questions, discussions, and community support, join our active communities on [Discord](https://discord.com/invite/ultralytics), [Reddit](https://www.reddit.com/r/ultralytics/), and the [Ultralytics Community Forums](https://community.ultralytics.com). We're here to help with all things Ultralytics!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
