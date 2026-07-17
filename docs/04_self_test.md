# 详细自测：从公式到可复现性

建议先独立作答，再展开答案。自测分四层：概念、手算、代码、真实复现。只有最后一层通过，才算闭环完成。

## A. 概念题

1. BEV query 与相机像素的本质区别是什么？
2. 为什么 SCA 先做几何投影，还要学习采样偏移？
3. 为什么一个 BEV 网格要沿高度取多个 3D 参考点？
4. `prev_bev` 为什么必须先做 ego-motion 对齐？
5. 小样本推理为什么必须取同一 scene 的连续帧？
6. agent query 的固定编号是否对应固定物理车辆？
7. `ego_fut_cmd` 来自哪里？它解决什么歧义？
8. VAD 的 collision/boundary/direction 项在推理时是否还运行成一个显式优化器？
9. 为什么 checkpoint `strict=False` 不是配置不匹配的通用修复？
10. 为什么“脚本退出码为 0”还不足以说明复现成功？

<details><summary>答案</summary>

1. query 表示 BEV 米制位置的可学习查询；像素是相机成像平面的观测。
2. 几何给出强先验位置，偏移用来吸收深度离散、标定误差、目标尺度及局部形变。
3. 单一地面高度无法覆盖有高度的目标；多个参考高度构成一根 3D 柱。
4. 两帧自车坐标系不同；不对齐会把静态物体融合到两个位置。
5. TSA 是递归状态模型，随机/跨 scene 帧的历史没有物理意义。
6. 否。集合预测通过 Hungarian matching 动态分配 query 与实例。
7. 来自上游导航/数据标签的高层左转、直行、右转命令，用于选择条件轨迹分支。
8. 否。它们是训练损失，参数被塑形后规划头直接前向输出。
9. 它可能静默跳过关键层并留下随机初始化参数，应修复结构/配置。
10. 还需验证资产、GPU 运算、样本时序、checkpoint 加载、输出结构与可视结果。

</details>

## B. 手算题

### B1. BEV 米制坐标

$H=W=200$，分辨率 $s=0.5$ m/grid。网格 $(x,y)=(120,80)$ 对应哪个米制位置？

<details><summary>答案</summary>

$$x'=(120-100)\times0.5=10\text{ m},$$

$$y'=(80-100)\times0.5=-10\text{ m}.$$

</details>

### B2. 投影有效性

相机内参 $f_x=f_y=800,c_x=640,c_y=360$。相机坐标点 $(X,Y,Z)=(3,-1,20)$，图像 $1280\times720$。求像素并判断是否有效。

<details><summary>答案</summary>

$$u=800(3/20)+640=760,$$

$$v=800(-1/20)+360=320.$$

$Z>0$ 且像素在图内，因此有效。

</details>

### B3. 双线性插值

四邻域左上、右上、左下、右下特征分别为 $0,2,4,6$，采样点小数偏移 $\alpha=0.25,\beta=0.5$。结果是多少？

<details><summary>答案</summary>

权重为 $0.375,0.125,0.375,0.125$，结果

$$0\times0.375+2\times0.125+4\times0.375+6\times0.125=2.5.$$

</details>

### B4. 历史 BEV 对齐

忽略旋转。自车从世界坐标 $(0,0)$ 前进到 $(2,0)$。上一帧在自车坐标 $(10,1)$ 的静态点，在当前自车坐标是什么？

<details><summary>答案</summary>

世界坐标为 $(10,1)$，当前自车原点为 $(2,0)$，所以当前自车坐标为 $(8,1)$。

</details>

### B5. 增量轨迹

规划 head 输出 $[(1,0),(1,0.5),(0.5,0.5)]$，累积轨迹是什么？

<details><summary>答案</summary>

$[(1,0),(2,0.5),(2.5,1.0)]$。

</details>

### B6. 碰撞 hinge

阈值 $\delta=2$ m，三个时刻最近 agent 距离为 $3,1.5,0.5$ m。忽略其他 agent，平均碰撞损失是多少？

<details><summary>答案</summary>

逐时刻为 $0,0.5,1.5$，平均 $(0+0.5+1.5)/3=2/3$。

</details>

### B7. 方向误差

自车切向为 $(1,1)$，最近车道方向为 $(1,0)$，角度误差是多少？

<details><summary>答案</summary>

余弦为 $1/\sqrt2$，误差 $\pi/4=45^\circ$。

</details>

## C. 代码级自测

### C1. 最小闭环的五个断言

```powershell
conda run -n vad-study python demo/minimal_bevformer_vad.py --self-test
```

成功判据：输出 `PASS: 5/5 mathematical and closed-loop self-tests`。五项分别覆盖：相机融合峰值、ego-motion warp、局部极大值查询、速度估计、规划候选选择。

### C2. Python 与 PowerShell 语法

```powershell
conda run -n vad-study python -m py_compile `
  demo/minimal_bevformer_vad.py `
  scripts/make_mini_config.py `
  scripts/make_mini_subset.py `
  scripts/smoke_test_env.py `
  scripts/verify_assets.py
```

对 PowerShell 使用 AST parser，而不是等下载数 GB 后才发现语法错：

```powershell
$files = Get-ChildItem scripts -Filter *.ps1
foreach ($f in $files) {
  $tokens=$null; $errors=$null
  [System.Management.Automation.Language.Parser]::ParseFile(
    $f.FullName,[ref]$tokens,[ref]$errors) > $null
  if ($errors) { $errors; throw "parse failed: $($f.Name)" }
}
```

### C3. 二进制算子烟雾测试

在 WSL 环境内：

```bash
conda activate vad-study
python scripts/smoke_test_env.py
```

必须同时覆盖：真实 CUDA tensor 运算、MMCV deformable attention、MMDet3D `points_in_boxes`。只执行 `import torch` 没有触发最容易出错的自定义算子。

### C4. 数据语义测试

```powershell
conda run -n vad-study python -c "from nuscenes import NuScenes; n=NuScenes(version='v1.0-mini',dataroot=r'data/nuscenes'); print(len(n.scene),len(n.sample),len(n.sample_data))"
```

应为 `10 404 31206`。再检查 12 帧子集：所有相邻记录的 scene token 相同，且前一条 `next` 等于后一条 token。

## D. 真实复现验收

逐项勾选并把输出保存到实验报告：

- [ ] 四个大文件精确长度全通过；
- [ ] nuScenes mini SDK 加载为 10/404/31206；
- [ ] WSL `nvidia-smi` 能看到 RTX 4090；
- [ ] 环境 smoke test 的三类 GPU/CUDA 运算通过；
- [ ] 数据转换器生成 mini info，无 trainval 路径泄漏；
- [ ] 子集为同一 scene 的 12 个连续 sample；
- [ ] VAD-Tiny checkpoint 在 CPU 可加载，关键参数无结构 mismatch；
- [ ] 单 GPU 推理输出 12 条结果；
- [ ] 每条结果含 agent/map/planning 预期字段；
- [ ] 至少一帧渲染图能看到相机/地图/agent/自车规划对应关系；
- [ ] 从全新终端按照文档重跑仍成功。

## E. 进阶反思题

1. SCA 对可见相机简单平均，在不同相机曝光差异明显时可能有什么限制？怎样改？
2. 使用 minFDE 选运动模态是否会忽视中途碰撞？可设计什么匹配代价？
3. VAD 的规划损失依赖自身预测地图与 agent，感知偏差如何传到规划？
4. open-loop L2 更小是否必然意味着 closed-loop 更安全？为什么？
5. 若将 12 帧扩到完整 mini，哪些指标是“系统稳定性指标”，哪些仍不能与论文表格比较？

这些题没有唯一一句话答案。理想回答应明确假设、给出失败例、提出可测的改进，并说明需要哪些数据验证。
