# TinyDet

基于官方 Ultralytics 源码的 YOLO11n 项目适配：准备数据、训练、六头 ONNX 导出、RDK X5 INT8 PTQ 与编译。
当前训练计划以官方 YOLO11n 预训练权重起步；YOLO11s 兼容相同接口，其既有工程结果保留为历史基线。
本仓库不包含自有产品数据、已训练产品权重或已验收的激光融合实现。

## 仓库与职责

- `main`：官方上游分支；`tinydet`：固定官方基线上的项目分支，后续改动通过 feature branch 和 PR 进入。
- `ultralytics/`：官方模型、检测损失、Trainer/Validator，不修改训练 forward 或复制另一套框架。
- `tinydet/prepare_data.py`：有序类别、分集校验、固定几何和数据身份。
- `tinydet/train.py`：官方训练器的矩形输入 hooks、完整断点和数据绑定。
- `tinydet/export.py`：导出副本的 BPU 适配、原始六头和数值验证。
- `tinydet/x5.py`：train-only 校准、Horizon 配置/命令、实际编译产物审计。

官方基线固定为 `3d75949c41f78a2c6763eb347ac7a94793098e89`（8.4.175），详见 [UPSTREAM.md](UPSTREAM.md)。
旧独立适配仓库已保留为 [tinydet-legacy](https://github.com/Jinlong-cs/tinydet-legacy)。

## 环境

训练/导出使用 Python 3.11+；Horizon SDK 使用其独立 x86 Linux 镜像，不混装训练依赖。
从本 fork 根目录安装，不能再安装旧适配器中指向官方 ZIP 的依赖，否则可能绕过项目源码。

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

适配器会检查加载的是本 checkout 的 `ultralytics`，并记录实际 Git commit、脏工作树状态及官方基线。
训练/导出使用本地可信 `.pt`；不会自动下载权重。模型、数据、环境、日志、ONNX、BIN均不提交Git。

## 数据准备

`classes.json` 是唯一有序类别表，例如 `["person", "chair"]`；这个示例不是产品类别批准。
每行标签为 `class_id cx cy width height`，坐标归一化到0–1。完整审核过的负样本使用空标签文件，缺失文件会报错。

原始JSONL路径相对清单所在目录：

```json
{"id":"frame_001","split":"train","domain":"toc/front","group":"session_a","image":"images/001.png","label":"labels/001.txt"}
{"id":"frame_002","split":"val","domain":"toc/front","group":"session_b","image":"images/002.png","label":"labels/002.txt"}
```

先按采集组划分train/val，再抽帧。提供的group不能跨集，准备后的重复像素不能跨集。
现有入口只支持train/val，不提供partial/ignore标签或独立test导入。

```bash
python -m tinydet.prepare_data --manifest /path/to/source.jsonl \
  --classes /path/to/classes.json --out work/data
```

支持原生640×352、640×480、640×512，整帧LINEAR到640×352，不裁剪或letterbox；归一化bbox不变。
生成PNG、YOLO标签、`data.yaml`、`classes.json`、`dataset.json`和带hash的manifest，保存源图尺寸与源标签hash。
训练会核对YAML实际指向这批文件和类别；修改/搬动数据后应生成新的准备版本，不复用旧运行断点。

## 官方预训练与训练

官方 [YOLO11n权重](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)
保存为 `work/weights/yolo11n.pt`，SHA256应为
`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`。
将权重作为迁移学习起点，自定义类别由准备数据的有序类别表提供。

```bash
python -m tinydet.train --data work/data/data.yaml --out work/train --device 0
```

默认模型是 `$TINY_WORK_DIR/weights/yolo11n.pt`，可用 `--model` 指定已下载的可信权重。
初始配方：AdamW、lr0=0.001、cosine、batch32、seed42、200epochs、patience40、GPU AMP；
关闭mosaic/mixup/cutmix/多尺度，保留普通颜色/翻转/仿射增强。它是待产品数据评估的初始配方。
train/val都断言RGB NCHW `[N,3,352,640]`、0–1，关闭官方验证额外补边。

```bash
python -m tinydet.train --data work/data/data.yaml --model work/train/weights/resume.pt \
  --out work/train --device 0 --resume
```

完整 `resume.pt` 保留optimizer/EMA/scaler，避免官方收尾清理best/last状态。
恢复必须使用同一运行目录和数据身份；训练前写入provenance，因此中断运行也能核验。
保持原设备环境；不承诺跨GPU/AMP环境或中断恢复的逐位一致。

## ONNX导出

```bash
python -m tinydet.export --checkpoint work/train/weights/best.pt \
  --check-manifest work/data/manifest.jsonl --out work/export
```

opset11、输入FLOAT32 `[1,3,352,640]`；输出顺序：
`cls_s8, box_s8, cls_s16, box_s16, cls_s32, box_s32`。
网格44×80、22×40、11×20；cls通道为实际nc，box为64通道（DFL reg_max16）。
Attention及检测头改写只作用于导出副本；图内无NMS/ArgMax。

默认64张val图加随机输入，比较原始头、DFL框和sigmoid分数；`export.json`记录实际检查数、源码、模型和类别hash。
没有bbox GT时，可以用固定图像清单和官方COCO权重验证导出链路，不能据此报告产品mAP。
检测后处理由部署端负责；更换nc或类别顺序必须同步部署通道、解码和标签表。

## X5校准、量化与编译

```bash
python -m tinydet.x5 calibrate --manifest work/data/manifest.jsonl \
  --count 512 --out work/calibration
```

仅从train按domain轮转抽样，不重复。输出原始RGB 0–255 float32 CHW；mapper做一次1/255。
每个校准文件都有名称、样本身份和hash，顺序不作为文件身份。

在x86 Linux的OpenExplorer v1.2.8 CPU环境运行（不在Mac上假装执行SDK）：

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

省略`--run`只生成配置与输入记录，必须使用新的输出目录。固定bayes-e、NV12、
`input_space_and_range=bt601_video`、scale=1/255、`set_all_nodes_int8`、latency/O3。
会核对实际ONNX、六头、类别表及导出报告身份；SDK命令成功后审计实际节点放置/精度、BIN输入色彩，才写`compiled.json`。
FLOAT32公开输出不表示内部float回退。镜像标签不是固定身份，实际运行另存image ID和SDK版本。

编译后仍需独立验证HB量化图、X5原始输出、框、视频及ROS并发。历史BIN结果不替代新产物验收。
本次仓库迁移的本地验证不构成新GPU正式训练、Horizon编译或板端通过。

## 后续维护

从`tinydet`创建独立feature worktree，经PR合入项目分支。官方`main`保持上游用途，升级官方基线另做兼容性PR，
不要只修改版本号绕过导出结构检查。发布绑定数据/split/类别、源码、预处理、checkpoint/ONNX/BIN hash和分层验证。
LiDAR/双目融合属于后续独立任务，当前仓库保持2D Detection训练和导出职责。
