# 易错点与故障排查手册

这份清单按“症状 → 根因 → 如何确认 → 修复”编写。先找症状，再做证据检查；不要看到 CUDA 报错就重装整个环境。

## 1. 下载与文件完整性

### 1.1 `tar: Unexpected EOF` / `gzip: invalid compressed data`

**常见根因**：断点续传时换了下载端点。两个看似同名的 nuScenes URL 可能不是同一对象长度；把 A 的前半段与 B 的后半段拼接，文件总大小甚至可能“看起来合理”，内容仍是坏的。

**确认**：

```powershell
python scripts/verify_assets.py --root .
tar -tzf artifacts/downloads/v1.0-mini.tgz > $null
```

**修复**：删除损坏文件后从一个固定 URL 重下；本仓库脚本固定使用 Motional S3，并校验精确字节数 `4,168,148,189`。

### 1.2 Google Drive 下载得到几 KB HTML

**根因**：大文件需要 confirmation token，保存的是警告页，不是 `.pth`。

**确认**：VAD-Tiny 必须恰好为 `484,968,871` bytes；文本开头含 HTML 就一定错误。

**修复**：运行 `scripts/download_gdrive_resumable.ps1`。脚本先解析 confirmation UUID，再检查 `206 Partial Content` 和 `Content-Range` 后才追加，避免断流后重复下载。

### 1.3 文件大小正确是否就绝对可信

不是。精确长度能高概率发现截断和 HTML，但不能抵抗同长度篡改。首次复现建议：

```powershell
python scripts/verify_assets.py --root . --hash
```

保存 SHA-256 到实验记录，并用 `torch.load(..., map_location='cpu')` 验证 checkpoint 结构。只加载可信来源的 pickle/PyTorch 权重；反序列化不受信任文件可能执行恶意代码。

## 2. 数据目录与 mini 标注

### 2.1 `Database version not found: v1.0-trainval`

**根因**：配置或可视化脚本仍硬编码 `v1.0-trainval`，但本地只有 `v1.0-mini`。

**确认**：`data/nuscenes/v1.0-mini/scene.json` 存在，而 `v1.0-trainval` 不存在。

**修复**：数据转换命令显式传 `--version v1.0-mini`，生成的配置使用 mini 标注文件。

### 2.2 有图像却报地图/CAN bus 缺失

VAD 不仅用六相机图像。最小闭环还需要：

```text
data/nuscenes/maps/expansion/
data/can_bus/
```

地图扩展用于向量地图监督/评估，CAN bus 用于自车状态与规划信息。不要把“nuScenes SDK 能加载”误认为“VAD 转换器所需数据全齐”。

### 2.3 随机选 12 帧后结果异常

**根因**：BEVFormer/VAD 是时序模型。随机帧跨 scene，`prev_bev` 没有连续物理意义。

**修复**：本仓库 `make_mini_subset.py` 从同一场景首帧开始沿时间顺序保留连续帧。小样本不是随机图片子集，而是一个短的连续序列。

### 2.4 第一帧或 scene 切换后出现历史鬼影

**根因**：没有清空 `prev_bev`，或 scene token 判定错误。

**确认**：在 `VAD.py` 的 `forward_test` 附近记录当前 scene token 与 `prev_bev is None`。

**修复**：新场景第一帧必须清空历史；同场景内按时间单线程顺序推理。

## 3. 环境与二进制兼容

### 3.1 `ModuleNotFoundError: mmcv._ext`

**根因**：安装了纯 Python 的 `mmcv`，而非带 CUDA/C++ 运算的 `mmcv-full`，或 wheel 与 PyTorch/CUDA 不匹配。

**确认**：

```bash
python -c "import mmcv; print(mmcv.__version__); import mmcv._ext"
```

**修复**：按 `setup_env.sh` 的 cu117/torch1.13 索引安装 `mmcv-full==1.7.2`，不要在之后再执行会把它替换掉的 `pip install mmcv`。

### 3.2 `undefined symbol` / DLL 或 `.so` 加载失败

**根因**：PyTorch、CUDA runtime、mmcv-full 或 mmdet3d 扩展的 ABI 不一致。

**诊断顺序**：

1. `python -c "import torch; print(torch.__version__, torch.version.cuda)"`
2. `nvcc --version`
3. `python -c "import mmcv; print(mmcv.__version__)"`
4. `python scripts/smoke_test_env.py`

不要混用一个环境中编译的 `.so` 与另一个环境的 torch。

### 3.3 `no kernel image is available for execution`

**根因**：旧 CUDA 扩展没有为 RTX 4090 的 Ada 架构 `sm_89` 编译，或旧工具链不认识该架构。

本仓库采用 CUDA 11.7 + PyTorch 1.13.1，并设置：

```bash
export TORCH_CUDA_ARCH_LIST="8.6+PTX"
```

11.7 生成 Ampere cubin + PTX，驱动可为 Ada JIT。若你改用支持 `sm_89` 的新工具链，要同步验证旧版 MMCV/MMDet3D 的源码兼容性，不能只升级 CUDA 一个包。

### 3.4 为什么不直接原生 Windows 安装

官方栈依赖旧版 MMCV/MMDetection3D 自定义 CUDA 运算。当前机器的 Windows 环境没有 `nvcc`/MSVC build tools，旧项目也主要按 Linux 编译链测试。WSL2 能复用 Windows NVIDIA 驱动并提供更接近官方的 Linux 用户态，故本教程把它作为主路径。

### 3.5 `mmdet3d` 拒绝 MMCV 版本

mmdet3d 0.17.1 的版本上限早于 VAD 后来实际使用的 MMCV 1.7.x。仓库补丁只对精确的版本断言做守卫式修改；若上游文件内容不匹配会直接停止，而不是盲改。

## 4. 配置与权重不匹配

### 4.1 推理可运行但结果明显很差

优先检查图像归一化。官方当前仓库某些配置曾改成 ImageNet RGB 形式，而发布 VAD-Tiny 权重对应的原设置是：

```python
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    to_rgb=False)
```

错误归一化不会总是报 shape mismatch，却会让输入分布完全改变。本仓库 `make_mini_config.py` 使用精确文本守卫生成配置；若上游内容变了，它会失败并要求人工审查。

### 4.2 `Missing key(s)` / `Unexpected key(s)`

**根因**：VAD-Tiny 权重配了 Base 配置、stage 配错、query 数/encoder 结构被改，或 checkpoint 下载损坏。

**确认**：先打印 checkpoint 顶层键和 `state_dict` 数量，再查看加载日志中缺失键比例。少数非参数缓冲差异与大面积 backbone/head 缺失不是一回事。

### 4.3 `size mismatch` 不应使用 `strict=False` 掩盖

`strict=False` 只会跳过部分键，可能让关键 head 随机初始化而脚本仍“跑完”。复现目标是使用发布权重，任何关键结构 size mismatch 都应修配置，而非压掉警告。

### 4.4 明明写了 VAD dataset，运行时却变成基础 nuScenes dataset

**根因**：在完整 VAD config 末尾重新写 `data = dict(test=...)`。Python 文件内第二次赋值先覆盖第一次完整配置，MMCV 随后把这份残缺 dict 与 `_base_` 合并，于是 `type` 从 base 回流成错误数据集类。

**确认**：

```python
from mmcv import Config
c = Config.fromfile('projects/configs/VAD/VAD_tiny_mini.py')
print(c.data.test.type)
```

必须是 `VADCustomNuScenesDataset`。本仓库模板用 `data['test']['ann_file'] = ...` 原位修改，不重新赋值整个 `data`。

## 5. 推理、评估与可视化

### 5.1 单卡小样本为何比完整 mini 更合适

本任务验证的是安装、数据转换、时序状态、checkpoint 和输出结构，不追求论文指标。连续 12 帧足以穿过这些路径，显著减少首次排错时间。之后可切换完整 404 帧 mini 做稳定性检查。

### 5.2 多 GPU 可能让小样本时序状态失真

标准 distributed sampler 会把样本分片；若每个进程收到不连续帧，递归 `prev_bev` 不再对应前一时刻。第一次闭环请固定单 GPU、`workers_per_gpu=0`、batch size 1。

### 5.3 输出是增量而不是绝对点

若轨迹像一团靠近原点的小线段，先检查是否忘了 `cumsum(dim=-2)`。反过来，若已累积的轨迹再次 `cumsum`，数值会快速发散。

### 5.4 可视化与模型推理应分开验证

先确认 `predictions.pkl` 可加载、样本数正确、每帧有 planning 输出，再运行可视化。可视化脚本的坐标轴、类别阈值或 nuScenes version 写错，不代表模型没输出。

### 5.5 可视化报轨迹数组维度不一致

上游脚本把所有模态写成 `mode_idx=[0,1,...]`，而 box helper 又执行 `fut_coords[[mode_idx]]`，会多引入一维。传 `mode_idx=None` 本来就表示渲染全部模态。本仓库生成 mini visualizer 时做受保护替换，并保留 12 张合成帧用于逐帧检查。

## 6. 最短诊断清单

按顺序执行：

```text
[ ] verify_assets.py 全部 ok
[ ] NuScenes(v1.0-mini) = 10 scenes / 404 samples
[ ] nvidia-smi 在 WSL 可见 GPU
[ ] smoke_test_env.py 的 torch CUDA、MMCV op、MMDet3D op 全过
[ ] 生成 mini info 后，连续子集样本数为 12
[ ] 配置归一化与发布权重一致
[ ] checkpoint 关键参数完整加载
[ ] 单 GPU 顺序推理，scene 首帧清 prev_bev
[ ] 输出先结构检查，再可视化
```

如果你在 issue 中求助，请同时贴出：精确命令、完整首个异常栈、`nvidia-smi`、torch/CUDA/MMCV/MMDet/MMDet3D 版本、配置路径、输入 info 样本数。只贴最后一句错误通常无法定位。
