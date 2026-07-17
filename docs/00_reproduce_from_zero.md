# 从 0 到 VAD-Tiny 小样本推理：完整复现流程

这份文档假设你拿到一台全新的 Windows 11 电脑，只知道它有 NVIDIA GPU。目标不是训练 VAD，而是用官方权重在 nuScenes mini 的连续小样本上得到真实预测，并且能证明每一环确实成功。

## 0. 先定义“复现成功”

只有下面证据全部成立，才算完成：

1. 任选 Windows 已有环境或 WSL 标准环境；所选环境能看到 GPU，且真实 CUDA tensor 运算通过；
2. 4 个核心下载文件的字节数精确匹配；
3. nuScenes devkit 能加载 `v1.0-mini` 的 10 个 scene / 404 个 sample；
4. VAD converter 生成带 `metadata.version == v1.0-mini` 的 train/val pickle；
5. 真正启动 MMCV deformable-attention CUDA kernel；WSL 完整环境还需通过 mmdet3d 点云算子 smoke test；
6. `tools/test.py` 读取官方 `VAD_tiny.pth`，对连续帧完成前向；
7. 输出 pickle 可被再次加载，样本数与子集标注一致；
8. 至少一张结果图能对应回 nuScenes sample token。

“脚本没有立刻报错”不等于以上任一项。

## 1. 本项目为何选择 WSL2

官方栈来自 2023 年：Python 3.8、MMCV 1.x、MMDetection3D 0.x，并依赖多个 CUDA/C++ 扩展。Windows 原生需要额外配置 MSVC、匹配版 CUDA Toolkit，并处理大量只在 Linux CI 测过的编译路径。WSL2 直接复用 Windows NVIDIA 驱动，同时提供上游预期的 Linux ABI。

本机实测兼容路线采用：

| 组件 | 版本/策略 | 原因 |
|---|---|---|
| Ubuntu | 20.04 on WSL2 | GCC/旧研究栈兼容性较好 |
| Python | 3.8 | VAD 与 mmdet3d 0.17.1 的共同稳定区间 |
| PyTorch | 1.13.1+cu117 | 比官方 1.9.1+cu111 更适合 Ada/RTX 40 系 |
| MMCV | 1.7.2 | 有 torch 1.13/cu117 发布轮子 |
| mmdet | 2.28.2 | 仍保持 2.x API |
| mmseg | 0.30.0 | 仍保持 0.x API |
| mmdet3d | 0.17.1 | VAD 官方指定的旧 API |
| CUDA Toolkit | 11.7 nvcc | 编译 mmdet3d 扩展；不在 WSL 安装显卡驱动 |
| 架构标志 | `8.6+PTX` | CUDA 11.7 不认识 sm_89；PTX 交给新驱动 JIT |

`patch_mmdet3d_compat.py` 只把 mmdet3d 的 MMCV 版本上限从 1.4.0 放宽到 1.7.2，不修改模型或算子。是否真兼容由编译算子与完整前向验证，而不是靠删除 assert 自我安慰。

## 2. 硬件、空间和目录

最低建议：

- NVIDIA GPU，建议 12 GB 以上显存；
- 32 GB 内存；
- 40 GB 可用磁盘，50 GB 更稳妥；
- Windows 11，BIOS 已开启 CPU virtualization；
- 网络可访问 AWS/S3、Google Drive、PyTorch、OpenMMLab、Anaconda 和 NVIDIA 软件源。

本教程默认仓库位于：

```text
C:\E2E\VAD
```

WSL 中对应：

```text
/mnt/c/E2E/VAD
```

不要把 checkpoint、nuScenes 或 Conda 环境提交到 Git。

## 3. 获取仓库

已有本仓库时直接进入目录。自行从 GitHub 复现时：

```powershell
git clone https://github.com/jcchen666888-cyber/bevformer-vad-study.git C:\E2E\VAD
Set-Location C:\E2E\VAD
```

检查根目录：

```powershell
Get-ChildItem README.md, scripts, demo, docs
```

## 4. 下载源码、数据与权重

在普通 PowerShell 中执行：

```powershell
Set-Location C:\E2E\VAD
powershell -ExecutionPolicy Bypass -File .\scripts\download_assets.ps1
```

脚本会完成：

1. 从 Motional 的 S3 公共对象下载 nuScenes mini；
2. 下载 CAN bus expansion；
3. 下载 map expansion v1.3；
4. 处理 Google Drive 的“大文件无法扫描病毒”确认页并下载 VAD-Tiny；
5. 下载固定提交的 VAD 源码归档；
6. 下载 MMDetection3D v0.17.1；
7. 解压到正确目录；
8. 按精确字节数验收。

再次独立校验：

```powershell
python .\scripts\verify_assets.py --root . --hash
```

预期每行 `"ok": true`。`--hash` 会多读约 5.8 GB 文件，耗时取决于磁盘速度。

最终数据结构必须是：

```text
data
├── can_bus
│   ├── scene-0001_pose.json
│   └── ...
└── nuscenes
    ├── maps
    │   ├── expansion
    │   ├── basemap
    │   └── prediction
    ├── samples
    ├── sweeps
    └── v1.0-mini
        ├── sample.json
        ├── sample_data.json
        └── ...
```

### 为什么不能从两个 URL 混合断点续传

`www.nuscenes.org/data/v1.0-mini.tgz` 与 S3 公共对象在本次实测中返回了不同的对象长度。若先从一个端点下载一部分，再用 `curl -C -` 从另一个端点接续，最终文件长度可能接近预期，却在 `tar` 中段报 `Truncated tar archive`。脚本固定单一 S3 URL，并在解压前后做检查。

### 本机已有环境的 Windows 实测捷径

这台目标主机已有隔离的 `vad-study`（Python 3.8、torch 1.13.1+cu117、mmcv-full 1.7.2、mmdet 2.28.2）。MMCV deformable attention 已实际在 RTX 4090 上执行。可以直接运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_vad_mini_windows.ps1
```

该脚本使用 `patch_camera_only_windows.py` 避免 MMDetection3D 0.17.1 顶层导入与 VAD 相机模型无关的 ball-query/voxel/IoU 扩展；真正参与 VAD 的 MMCV CUDA deformable attention 仍是编译算子。任何点云 IoU 调用都会明确失败，而不会返回伪结果。此路线已在本机得到 12 帧真实预测，但它依赖本机已有 Conda 环境；从全新电脑开始，请继续执行下面可完整重建的 WSL2 路线。

## 5. 安装 WSL2

在“以管理员身份运行”的 PowerShell 中：

```powershell
Set-Location C:\E2E\VAD
powershell -ExecutionPolicy Bypass -File .\scripts\install_wsl.ps1
```

若系统要求重启，重启一次。随后在普通 PowerShell 启动 Ubuntu：

```powershell
wsl.exe --distribution Ubuntu-20.04
```

首次启动时 Ubuntu 会让你创建 Linux 用户名和密码。这个密码只用于 `sudo`，输入时终端不会显示星号，属于正常现象。

检查 WSL 版本：

```powershell
wsl.exe --list --verbose
```

`Ubuntu-20.04` 的 VERSION 应为 `2`。不是 2 时执行：

```powershell
wsl.exe --set-version Ubuntu-20.04 2
```

## 6. 先验证 WSL GPU 透传

进入 Ubuntu 后：

```bash
nvidia-smi
```

应看到 GPU 名称和 Windows 驱动版本。WSL 内不要安装 `nvidia-driver-*`；GPU 驱动由 Windows 提供。我们只安装编译扩展所需的 CUDA Toolkit/nvcc。

如果 `nvidia-smi` 不存在或看不到 GPU，先停止，参见 [诊断树](03_troubleshooting.md)。

## 7. 一键建立隔离环境

在 Ubuntu 中：

```bash
cd /mnt/c/E2E/VAD
bash scripts/setup_env.sh
```

脚本依次执行：

1. 安装 Linux 编译依赖；
2. 安装用户目录下的 Miniconda；
3. 创建 `vad-study`；
4. 安装 PyTorch 1.13.1+cu117；
5. 安装 MMCV/mmdet/mmseg；
6. 安装 CUDA 11.7 的 nvcc 与 headers，不安装 Linux GPU driver；
7. 对 mmdet3d 0.17.1 施加最小版本守卫补丁；
8. 从源码编译 mmdet3d CUDA ops；
9. 真正执行一次 GPU 矩阵乘法并导入编译算子。

编译会明显占用 CPU，通常是环境阶段最慢的一步。完成后预期打印类似：

```json
{
  "torch": "1.13.1+cu117",
  "torch_cuda": "11.7",
  "gpu": "NVIDIA GeForce RTX ...",
  "mmcv": "1.7.2",
  "mmdet3d": "0.17.1",
  "cuda_kernel_smoke_test": "passed",
  "compiled_ops_import": "passed"
}
```

新终端里激活环境：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vad-study
export CUDA_HOME=/usr/local/cuda-11.7
export TORCH_CUDA_ARCH_LIST='8.6+PTX'
```

## 8. 理解标注生成，而不是盲跑命令

nuScenes 原始 JSON 只描述关系表。VAD 还需要每个 sample 的：

- 6 路相机图像路径与 `lidar2img`；
- 当前及历史 ego pose；
- CAN bus 状态；
- Agent 3D box、历史/未来轨迹和有效 mask；
- 地图 location；
- temporal queue 所需的 `prev/next`、scene token、frame index。

官方 converter 已支持 `v1.0-mini`。它本身只需要类别映射和 NumPy 投影，却会因顶层导入提前加载尚未编译的 MMDet3D CUDA ops。本仓库先生成一个受保护的 standalone 副本：模型代码没有改变，数据转换也不必等待 CUDA 编译。

```bash
cd /mnt/c/E2E/VAD/_deps/VAD
ln -sfn /mnt/c/E2E/VAD/data data
python /mnt/c/E2E/VAD/scripts/make_standalone_converter.py \
  --vad-root /mnt/c/E2E/VAD/_deps/VAD
python tools/data_converter/vad_nuscenes_converter_standalone.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag vad_nuscenes \
  --version v1.0-mini \
  --canbus ./data
```

应生成：

```text
data/nuscenes/vad_nuscenes_infos_temporal_train.pkl
data/nuscenes/vad_nuscenes_infos_temporal_val.pkl
```

注意这里必须传 `v1.0-mini`，不是默认文档中的 `v1.0`。

## 9. 为什么要截取“同一 scene 的连续帧”

BEVFormer/VAD 的 previous BEV 是时序状态。随机抽 12 张图会破坏：

- `prev_bev` 的时间顺序；
- ego-motion 对齐；
- scene 切换时的状态重置；
- 速度和未来轨迹含义。

因此使用：

```bash
python /mnt/c/E2E/VAD/scripts/make_mini_subset.py \
  --input data/nuscenes/vad_nuscenes_infos_temporal_val.pkl \
  --output data/nuscenes/vad_nuscenes_infos_temporal_val_subset.pkl \
  --frames 12
```

脚本只保留验证集第一个 scene 的前 12 帧，并逐对检查 `next/prev`、时间戳严格递增、scene token 一致，以及 metadata 仍是 `v1.0-mini`。

## 10. 生成正确的 mini 推理配置

官方 README 指出：发布权重训练时使用的是 BGR mean subtraction：

```python
mean = [103.530, 116.280, 123.675]
std = [1.0, 1.0, 1.0]
to_rgb = False
```

仓库当前默认配置已换成 RGB ImageNet normalization；直接使用会得到错误指标和可视化。不要只改顶层 `img_norm_cfg` 后用 `_base_` 继承，因为 base config 中的 pipeline 已经把旧字典展开复制。

`make_mini_config.py` 从官方完整配置复制，再做受保护的精确替换，并追加 mini subset 路径：

```bash
python /mnt/c/E2E/VAD/scripts/make_mini_config.py \
  --vad-root /mnt/c/E2E/VAD/_deps/VAD \
  --project-root /mnt/c/E2E/VAD
```

若上游文本发生变化，脚本会失败而不是静默生成错误配置。

## 11. 运行官方 VAD-Tiny 小样本推理

一条命令完成配置、标注、子集和推理：

```bash
cd /mnt/c/E2E/VAD
bash scripts/run_vad_mini.sh 12
```

等价的核心前向命令是：

```bash
cd _deps/VAD
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  projects/configs/VAD/VAD_tiny_mini.py \
  ckpts/VAD_tiny.pth \
  --launcher none \
  --out work_dirs/vad_tiny_mini/predictions.pkl
```

为什么不传 `--eval bbox`：12 帧只是理解前向和可视化的子集，并不覆盖整个 `mini_val`，拿它计算官方 detection/planning aggregate metric 没有统计意义。要比较 mini 指标，应改用完整 mini val annotation，再跑全部验证帧。

## 12. 检查输出，而不是只看进度条

在 VAD 环境执行：

```bash
python - <<'PY'
import mmcv

path = 'work_dirs/vad_tiny_mini/predictions.pkl'
result = mmcv.load(path)
print(type(result), len(result))
first = result[0]
print(first.keys())
print(first['pts_bbox'].keys() if 'pts_bbox' in first else first.keys())
PY
```

必须满足：

- 长度为 12；
- 每帧包含 3D box/class/score；
- 包含 agent future trajectories；
- 包含 map vectors；
- 包含 ego future plan 与 command。

输出结构会随 test helper 的包装方式略有不同，所以先打印 key，再写后处理；不要假定所有版本都固定有 `pts_bbox`。

## 13. 可视化

VAD 上游可视化脚本默认把 nuScenes 版本写死为 `v1.0-trainval`，不能直接用于 mini。复现脚本会使用 mini 兼容入口，并显式传：

```text
version = v1.0-mini
dataroot = /mnt/c/E2E/VAD/data/nuscenes
```

最终图与视频放在：

```text
outputs/vad_tiny_mini/
```

验收图中至少应看见：6 路相机、BEV agent box/未来轨迹、地图向量，以及 ego 规划轨迹。若图能生成但颜色/位置极端异常，优先检查第 10 节的 normalization。

## 14. 运行透明教学闭环

教学 demo 不依赖 VAD 环境，可先确认自己的数学直觉：

```bash
python demo/minimal_bevformer_vad.py --self-test
python demo/minimal_bevformer_vad.py --save-gif --no-show --frames 40
```

对应关系：

| demo 函数 | 论文模块 |
|---|---|
| `camera_visibility` | 3D reference point 投影与 hit view |
| `multiview_to_bev` | Spatial Cross-Attention 的稀疏采样/归一化 |
| `warp_previous_bev` | ego-motion 对齐的 history BEV |
| `find_peaks` | query 解码的透明替身 |
| `match_velocities` | 多帧 agent motion 的透明替身 |
| `plan_ego` | 轨迹候选与 vectorized safety cost |

## 15. 完整 mini 与论文数字

若想跑完整 mini val，把配置的 `ann_file` 改回：

```text
vad_nuscenes_infos_temporal_val.pkl
```

然后可添加 `--eval bbox`。但要明确：

- 论文表 1 使用完整 nuScenes val，不是 mini；
- mini 只有 10 个 scene，不适合复刻论文 aggregate 指标；
- 本仓库目标是验证推理闭环，不把 mini 数字包装成论文复现精度。

## 16. 可重复重跑清单

从头重跑前逐项确认：

```text
[ ] git commit 固定
[ ] 4 个核心文件字节数匹配
[ ] WSL distro 和版本记录
[ ] pip/conda freeze 保存
[ ] nvidia-smi 保存
[ ] config 中 checkpoint normalization 正确
[ ] subset 为同一 scene 连续帧
[ ] CUDA_VISIBLE_DEVICES=0 且 launcher=none
[ ] prediction pickle 数量和 key 验收
[ ] 可视化 sample token 能回查原数据
```

遇到失败不要随机升级依赖，先按 [易错点与诊断树](03_troubleshooting.md) 从最短证据链定位。
