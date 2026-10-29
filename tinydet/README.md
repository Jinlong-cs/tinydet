# TinyDet

YOLO11n 数据准备、训练、六头 ONNX 导出和 RDK X5 INT8 编译。项目分支为 `tinydet`，适配代码在本目录。

## 安装

从本 fork 根目录安装，训练/导出使用 Python 3.11+。

```bash
git clone -b tinydet https://github.com/Jinlong-cs/tinydet.git
cd tinydet
python3.11 -m venv venv
source venv/bin/activate
export TINY_WORK_DIR="$PWD/work"
export PIP_CACHE_DIR="$TINY_WORK_DIR/cache/pip"
export TMPDIR="$TINY_WORK_DIR/tmp"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TINY_WORK_DIR/weights"
python -m pip install -e ".[tinydet]"
```

## 数据和训练

`classes.json` 为有序类别列表，例如 `["person", "chair"]`。标签为 YOLO 格式 `class_id cx cy width height`，坐标归一化到 0–1；审核过的负样本可使用空文件，缺失标签报错。JSONL 中路径相对清单目录：

```json
{"id":"001","split":"train","domain":"toc/front","group":"session_a","image":"images/001.png","label":"labels/001.txt"}
{"id":"002","split":"val","domain":"toc/front","group":"session_b","image":"images/002.png","label":"labels/002.txt"}
```

支持 640×352/480/512，整帧 LINEAR 到 640×352，无裁剪或 letterbox。按采集组分集；group 和重复像素不得跨 train/val。入口要求完整标签，不支持 partial/ignore 或独立 test 集。

将可信的官方 [YOLO11n 权重](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)放到 `work/weights/yolo11n.pt`：
SHA256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`。入口不自动下载权重，`--model` 可指定其他本地 checkpoint。

```bash
python -m tinydet.prepare_data --manifest /path/to/source.jsonl \
  --classes /path/to/classes.json --out work/data
python -m tinydet.train --data work/data/data.yaml --out work/train --device 0
# 中断后恢复：保持原数据、运行目录和设备环境
python -m tinydet.train --data work/data/data.yaml --model work/train/weights/resume.pt \
  --out work/train --device 0 --resume
```

当前仅支持单设备；CPU 使用 `--device cpu`，拒绝多 GPU 列表。训练/验证固定 RGB NCHW 352×640、0–1，复用官方 Trainer/Validator。恢复文件保留 optimizer/EMA/scaler，数据身份须与原运行一致；更多参数见各入口 `--help`。

## ONNX 和 X5

```bash
python -m tinydet.export --checkpoint work/train/weights/best.pt \
  --check-manifest work/data/manifest.jsonl --out work/export
python -m tinydet.x5 calibrate --manifest work/data/manifest.jsonl \
  --count 512 --out work/calibration
```

ONNX 为 opset11、FLOAT32 输入 `[1,3,352,640]`，输出顺序为
`cls_s8, box_s8, cls_s16, box_s16, cls_s32, box_s32`；网格 44×80、22×40、11×20，cls 通道为实际 nc，box 为 64（DFL reg_max16）。图内无 NMS/ArgMax，部署端负责解码；更换类别须同步通道与标签表。

导出默认检查最多 64 张 val 图及随机输入，记录原始头/框/分数数值对齐、源码、模型和类别 hash。校准只取 train，`--count` 不得超过 train 样本数；输出 RGB 0–255 float32 CHW，由 mapper 做一次 1/255。

在 x86 Linux OpenExplorer v1.2.8 环境执行编译：

```bash
IMAGE=registry.d-robotics.cc/deliver/hub.hobot.cc/aitools/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
docker run --rm --cpus=4 --memory=16g \
  -v "$PWD:/workspace" -w /workspace \
  -e TINY_WORK_DIR=/workspace/work -e PYTHONDONTWRITEBYTECODE=1 \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  "$IMAGE" python3 -m tinydet.x5 compile \
  --onnx work/export/model.onnx --calibration work/calibration/rgb \
  --out work/ptq --name yolo11n_640x352_int8_nv12 --run
```

省略 `--run` 只生成配置；输出目录须为新目录。编译固定 bayes-e、NV12 BT601 video、单次 1/255、INT8 请求及 latency/O3；实际节点与 BIN 输入审计通过后才写 `compiled.json`。编译后仍需 HB/X5 验证。当前本地 CPU/ONNX/配置检查不代表产品精度或板端验收。

## 来源

官方基线为 `ultralytics/ultralytics@3d75949c41f78a2c6763eb347ac7a94793098e89`（8.4.175），`ultralytics/` 核心未修改。
适配器迁自 [tinydet-legacy](https://github.com/Jinlong-cs/tinydet-legacy) 的 `7e9a16edd4ae25c8301a6fba82e57b4ae518a4e1`。
raw-head/Attention 导出改写源于 [D-Robotics](https://github.com/D-Robotics/rdk_model_zoo/blob/rdk_x5/samples/vision/ultralytics_yolo/conversion/export_monkey_patch.py)，保留代码归属和 [Apache-2.0 许可证](LICENSE-D-Robotics)；仓库沿用上游 [AGPL-3.0](../LICENSE)。
