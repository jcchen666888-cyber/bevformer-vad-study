# BEVFormer：从像素到时序 BEV 的公式推导

> 目标：读完后，你不仅能复述 BEVFormer，还能手算一个参考点如何投影到相机、解释为什么要做时序对齐，并能在 VAD 源码里定位每一步。本文公式以 [BEVFormer 论文](https://arxiv.org/abs/2203.17270) 为准。

## 1. 先固定坐标与张量

六路环视相机在时刻 $t$ 经过共享骨干网络和 FPN，得到多尺度特征

$$
F_t^i=\{F_{t,l}^i\}_{l=1}^{L},\qquad i\in\{1,\ldots,N_{cam}\}.
$$

BEV 平面被离散成 $H\times W$ 个网格，每格一个可学习查询：

$$
Q\in\mathbb{R}^{H\times W\times C},\qquad Q_p\in\mathbb{R}^{C}.
$$

注意 $Q_p$ 不是“某个像素”。它代表自车坐标系地面上的一个位置，图像特征只是它要查询的证据。

设 BEV 分辨率为 $s$ 米/格，以中心为原点，则整数网格 $(x,y)$ 对应的米制位置为

$$
x'=(x-W/2)s,\qquad y'=(y-H/2)s. \tag{1}
$$

若横纵范围不相同，应分别使用 $s_x,s_y$。代码中的 `pc_range` 决定实际边界，不能只靠图片宽高猜。

## 2. 为什么不用普通全局注意力

普通 cross-attention 让每个 BEV query 访问每个相机、每个尺度的所有像素，复杂度近似

$$
O(HW\cdot N_{cam}\cdot\sum_l H_lW_l).
$$

BEVFormer 使用 deformable attention：每个 query 只在参考点附近采样少量位置。对查询 $q$、参考点 $p$ 和特征 $x$，论文写作

$$
\operatorname{DeformAttn}(q,p,x)
=\sum_{h=1}^{N_h}W_h
\left[\sum_{k=1}^{N_k}A_{hk}\,W'_h x(p+\Delta p_{hk})\right], \tag{2}
$$

其中

- $N_h$：注意力头数；
- $N_k$：每个头的采样点数；
- $\Delta p_{hk}$：由 query 预测的二维偏移；
- $A_{hk}$：归一化权重，$\sum_kA_{hk}=1$；
- $W'_h,W_h$：每个头的值投影与输出投影。

采样位置通常不是整数像素。令 $u=u_0+\alpha,v=v_0+\beta$，$\alpha,\beta\in[0,1)$，双线性插值为

$$
\begin{aligned}
x(u,v)=&(1-\alpha)(1-\beta)x_{u_0,v_0}
+\alpha(1-\beta)x_{u_0+1,v_0}\\
&+(1-\alpha)\beta x_{u_0,v_0+1}
+\alpha\beta x_{u_0+1,v_0+1}. \tag{3}
\end{aligned}
$$

四个权重之和为 1，且对 $u,v$ 分段可导，因此采样偏移可端到端学习。稀疏采样把主要复杂度降到约

$$
O(HW\cdot N_{cam}\cdot N_hN_kL),
$$

其中 $N_k,L$ 都很小。

## 3. 空间交叉注意力 SCA

### 3.1 从 BEV 柱到图像点

地面网格只给出 $(x',y')$。同一地面位置可能对应汽车、行人、路牌等不同高度，所以在竖直方向取 $N_{ref}$ 个参考高度 $z'_j$，形成三维柱：

$$
r_{p,j}=[x',y',z'_j,1]^\top.
$$

已知从自车/LiDAR 坐标到第 $i$ 个相机像素齐次坐标的变换 $T_i$，投影为

$$
z_{ij}
\begin{bmatrix}u_{ij}\\v_{ij}\\1\end{bmatrix}
=T_i
\begin{bmatrix}x'\\y'\\z'_j\\1\end{bmatrix}. \tag{4}
$$

这里 $z_{ij}$ 是深度尺度。只有 $z_{ij}>0$ 且 $(u_{ij},v_{ij})$ 落在图像内，该投影才有效。可见相机集合记为 $\mathcal V_{hit}(p)$。

### 3.2 多相机融合

空间交叉注意力为

$$
\operatorname{SCA}(Q_p,F_t)
=\frac{1}{|\mathcal V_{hit}(p)|}
\sum_{i\in\mathcal V_{hit}(p)}
\sum_{j=1}^{N_{ref}}
\operatorname{DeformAttn}
\left(Q_p,\mathcal P(p,i,j),F_t^i\right). \tag{5}
$$

先用几何标定找到“应该看哪里”，再由可学习偏移在邻域内纠正标定误差、深度不确定性和尺度变化。除以可见相机数能减小相机重叠区域与单相机区域之间的幅值偏差。

### 3.3 一个可手算的投影例子

若相机坐标中的点为 $(X_c,Y_c,Z_c)=(2,1,10)$ 米，内参

$$
K=\begin{bmatrix}1000&0&800\\0&1000&450\\0&0&1\end{bmatrix},
$$

则

$$
u=1000\frac{2}{10}+800=1000,\qquad
v=1000\frac{1}{10}+450=550.
$$

若图像为 $1600\times900$，该点有效；若 $Z_c<0$，即使 $u,v$ 数值落在图内也必须丢弃。

## 4. 时序自注意力 TSA

### 4.1 为什么必须先对齐历史 BEV

上一帧 BEV $B_{t-1}$ 是以上一帧自车为原点。车辆移动后，同一静态路面在两张 BEV 上的索引不同，直接相加会产生双影。

设上一帧自车到世界的位姿为 $T^w_{e,t-1}$，当前帧为 $T^w_{e,t}$。把上一帧点变到当前自车系：

$$
p_{e,t}=(T^w_{e,t})^{-1}T^w_{e,t-1}p_{e,t-1}. \tag{6}
$$

平面近似下，这就是一个 $SE(2)$ 变换：

$$
\begin{bmatrix}x_t\\y_t\\1\end{bmatrix}
=
\begin{bmatrix}
\cos\Delta\theta&-\sin\Delta\theta&\Delta x\\
\sin\Delta\theta& \cos\Delta\theta&\Delta y\\
0&0&1
\end{bmatrix}
\begin{bmatrix}x_{t-1}\\y_{t-1}\\1\end{bmatrix}. \tag{7}
$$

对齐后的历史特征记作 $B'_{t-1}$。

### 4.2 TSA 公式

论文将当前 query 和对齐历史都作为 deformable attention 的值源：

$$
\operatorname{TSA}
\left(Q_p,\{Q,B'_{t-1}\}\right)
=\sum_{V\in\{Q,B'_{t-1}\}}
\operatorname{DeformAttn}(Q_p,p,V). \tag{8}
$$

采样偏移和权重由 $Q_p$ 与该位置的历史特征拼接后预测。第一帧没有历史时使用 $\{Q,Q\}$，从而保持网络形状不变。

推理必须按时间顺序递归：

$$
B_t=f_{enc}(Q,F_t,B'_{t-1}),\qquad B_{t-1}\leftarrow B_t. \tag{9}
$$

因此“随机抽 12 张图”不是真正的小样本时序推理；正确做法是从同一 scene 沿 `sample.next` 取连续帧。

## 5. 一层编码器的计算顺序

典型 BEVFormer encoder layer 可概括为

$$
\tilde Q=\operatorname{LN}(Q+\operatorname{TSA}(Q,B'_{t-1})),
$$

$$
\hat Q=\operatorname{LN}(\tilde Q+\operatorname{SCA}(\tilde Q,F_t)),
$$

$$
Q^{out}=\operatorname{LN}(\hat Q+\operatorname{FFN}(\hat Q)). \tag{10}
$$

多层堆叠后输出 $B_t$。VAD 并未抛弃这部分，而是把它作为统一向量化检测、地图、运动和规划的感知底座。

## 6. 与本仓库闭环教学的对应

`demo/minimal_bevformer_vad.py` 故意把神经网络替换成透明的数值部件：

| 正式模型 | 教学闭环中的可观察替身 |
|---|---|
| SCA | 六相机可见性掩码 + 局部高斯证据融合 |
| TSA | 按自车位移 warp 上一帧热图并加权融合 |
| object queries | 热图局部极大值/NMS |
| motion head | 两帧匹配后恒速外推 |
| planning head | 五次多项式候选轨迹 + 显式代价 |

它不是精度替代品，而是一条可以逐项断言的最小闭环。先用它建立因果直觉，再运行官方权重，就不会把复杂框架调用误当成“理解了模型”。

## 7. 源码定位

VAD 内嵌的 BEVFormer 主要位于：

- `projects/mmdet3d_plugin/VAD/modules/transformer.py`：历史 BEV 旋转/平移和 encoder 调用；
- `projects/mmdet3d_plugin/VAD/modules/encoder.py`：参考点、`point_sampling`、逐层编码；
- `projects/mmdet3d_plugin/VAD/modules/spatial_cross_attention.py`：SCA；
- `projects/mmdet3d_plugin/VAD/modules/temporal_self_attention.py`：TSA；
- `projects/mmdet3d_plugin/ops/`：多尺度可变形注意力 CUDA 运算。

建议调试时打印以下形状，而不是只看最终 loss：`mlvl_feats`、`bev_queries`、`reference_points_cam`、`bev_mask`、`prev_bev`、`bev_embed`。

## 8. 最容易混淆的四点

1. **BEV query 是位置查询，不是相机像素。** 投影只负责告诉查询去哪些像素取证。
2. **SCA 不是盲目融合六相机。** 每个 query 只访问几何上可见的相机。
3. **TSA 不是简单叠帧。** 历史 BEV 必须先做 ego-motion 对齐。
4. **时序模型不能用随机 sampler 验证状态逻辑。** scene 变化时必须清空 `prev_bev`。

下一篇：[VAD：从向量查询到规划轨迹](02_vad_math.md)。
