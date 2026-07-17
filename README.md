# BEVFormer → VAD Study

这是一个面向“真正从零复现”的端到端自动驾驶学习仓库。它把论文公式、可运行的透明教学 demo，以及官方 VAD-Tiny 在 nuScenes mini 上的小样本推理放在同一条证据链里：

```text
6 路相机 → 图像特征 → 时空 BEV → Agent / Map 向量 → 运动预测 → Ego 规划
```

![minimal BEVFormer to VAD closed loop](demo/outputs/minimal_bevformer_vad.png)

> 教学 demo 用 NumPy 明确展示数据流和公式，不冒充官方神经网络。真正的复现结果来自官方 VAD 代码与发布的 VAD-Tiny 权重。

## 你能在这里完成什么

- 在 Windows 11 + NVIDIA GPU 上建立隔离的 WSL2/CUDA 环境；
- 下载、校验并解压最小所需资源，而不是约 300 GB 的完整 nuScenes trainval；
- 生成 VAD 专用的 nuScenes mini 时序标注；
- 从 mini 验证集截取同一 scene 的连续 12 帧；
- 用官方 VAD-Tiny 权重做单卡推理并保存预测；
- 运行一个透明的 BEVFormer→VAD 最小闭环和 5 组数学自测；
- 从公式、张量形状、代码位置和常见错误四个角度理解结果。

## 两条闭环，不能混淆

| 路线 | 目的 | 输入 | 输出 | 能证明什么 |
|---|---|---|---|---|
| 官方复现 | 验证真实 VAD 模型 | nuScenes mini、官方 checkpoint | `predictions.pkl`、可视化 | 环境、数据、权重、模型前向均跑通 |
| 教学 demo | 拆开黑盒逐步自测 | 合成 6 视角场景 | PNG、GIF、代价分解 | 投影、时序对齐、向量预测和规划逻辑可解释 |

## 最快开始

当前这台 Windows 主机已经有 `vad-study` Conda 环境，实测的一条命令闭环是：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_vad_mini_windows.ps1
```

它会重新校验资产、转换 mini、核验 12 帧时序、运行官方 VAD-Tiny、格式化结果并渲染。全新电脑建议走下面的 WSL2 标准路线。

在管理员 PowerShell 中安装 WSL2（只需首次执行）：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_wsl.ps1
```

在普通 PowerShell 中下载与解压最小资源：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_assets.ps1
python .\scripts\verify_assets.py --hash
```

进入 Ubuntu/WSL，配置 GPU 环境并运行 12 帧推理：

```bash
cd /mnt/c/E2E/VAD
bash scripts/setup_env.sh
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vad-study
bash scripts/run_vad_mini.sh 12
```

先运行不依赖 CUDA 的教学闭环：

```powershell
conda create -n vad-demo python=3.10 -y
conda run -n vad-demo pip install -r requirements-demo.txt
conda run -n vad-demo python demo\minimal_bevformer_vad.py --self-test
conda run -n vad-demo python demo\minimal_bevformer_vad.py --save-gif --no-show
```

完整解释、每一步预期输出和失败恢复见 [从零复现全流程](docs/00_reproduce_from_zero.md)。

## 当前最小资源合同

| 文件 | 精确字节数 | 用途 |
|---|---:|---|
| `v1.0-mini.tgz` | 4,168,148,189 | 10 个 nuScenes 场景 |
| `can_bus.zip` | 780,974,697 | Ego 位姿、速度、控制等时序信息 |
| `nuScenes-map-expansion-v1.3.zip` | 398,535,531 | 可向量化的地图边界、分隔线等 |
| `VAD_tiny.pth` | 484,968,871 | 官方 VAD-Tiny stage-2 权重 |

全部下载、环境、解压和输出建议预留 40 GB；脚本按 50 GB 上限设计。大文件均被 `.gitignore` 排除。

## 已得到的教学结果

- 数学与闭环自测：`PASS: 5/5`；
- 解码出 3 个 agent query；
- 示例规划在 15 条候选中选择 `terminal_offset=-3.0 m, speed=5.0 m/s`；
- 被选轨迹总代价 `29.4573`，其中碰撞项 `0.7770`、车道项 `0.3184`、舒适项 `0.0079`；
- 输出：[PNG](demo/outputs/minimal_bevformer_vad.png) / [GIF](demo/outputs/minimal_bevformer_vad.gif)。

官方小样本推理的机器实测、日志摘要和预测图记录在 [实验报告](docs/05_experiment_report.md)。

![官方 VAD-Tiny 在 nuScenes mini 首帧的预测](outputs/vad_tiny_mini/frame_000.jpg)

## 学习顺序

1. [从零复现全流程](docs/00_reproduce_from_zero.md)：环境、下载、标注、推理、验证和重跑。
2. [BEVFormer 公式推导](docs/01_bevformer_math.md)：3D 参考点投影、空间交叉注意力、时序自注意力。
3. [VAD 公式推导](docs/02_vad_math.md)：向量化场景、交互式规划、三类向量约束和总损失。
4. [易错点与诊断树](docs/03_troubleshooting.md)：从症状直接定位数据、版本、CUDA 或配置错误。
5. [详细自测](docs/04_self_test.md)：从概念题、手算题、代码题到复现验收。
6. [实验报告](docs/05_experiment_report.md)：真实硬件、版本、结果和与论文数字的边界。

## 目录

```text
.
├── README.md
├── configs
│   └── VAD_tiny_mini.template.py
├── demo
│   ├── minimal_bevformer_vad.py
│   └── outputs
│       ├── minimal_bevformer_vad.png
│       └── minimal_bevformer_vad.gif
├── docs
│   ├── 00_reproduce_from_zero.md
│   ├── 01_bevformer_math.md
│   ├── 02_vad_math.md
│   ├── 03_troubleshooting.md
│   ├── 04_self_test.md
│   └── 05_experiment_report.md
├── scripts
│   ├── download_assets.ps1
│   ├── download_gdrive_resumable.ps1
│   ├── install_wsl.ps1
│   ├── make_mini_config.py
│   ├── make_mini_subset.py
│   ├── make_mini_visualizer.py
│   ├── make_standalone_converter.py
│   ├── patch_mmdet3d_compat.py
│   ├── run_vad_mini.sh
│   ├── setup_env.sh
│   ├── smoke_test_env.py
│   └── verify_assets.py
└── requirements-demo.txt
```

运行时还会生成 `_deps/`、`artifacts/`、`data/` 和 `work_dirs/`。它们是可重建的大文件，不进入 Git。

## 上游版本与来源

- VAD 固定到提交 [`1688c4b`](https://github.com/hustvl/VAD/commit/1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8)；
- MMDetection3D 固定到 `v0.17.1`；
- VAD 论文：[Vectorized Scene Representation for Efficient Autonomous Driving](https://arxiv.org/abs/2303.12077)；
- BEVFormer 论文：[Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers](https://arxiv.org/abs/2203.17270)；
- nuScenes：[官网与下载说明](https://www.nuscenes.org/nuscenes)。

本仓库自己的文档和 demo 使用 MIT License。VAD 与各上游依赖仍分别遵守其原始许可证；下载的 nuScenes 数据和模型权重不会被重新分发到本仓库。
