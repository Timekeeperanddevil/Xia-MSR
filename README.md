# Xia-MSR — Glow-NILM: 基于归一化流的电器功率序列建模

> 利用 **Glow**（Generative Flow）归一化流模型对单电器功率时间序列的分布进行精确建模，
> 支持高质量生成采样与非侵入式负荷监测（NILM）研究。

---

## 目录

- [项目简介](#项目简介)
- [模型架构](#模型架构)
- [环境安装](#环境安装)
- [快速开始](#快速开始)
  - [训练](#训练)
  - [评估](#评估)
  - [采样](#采样)
- [支持电器](#支持电器)
- [目录结构](#目录结构)
- [评估指标](#评估指标)

---

## 项目简介

Xia-MSR 将 Glow 归一化流应用于 NILM 场景，以功率时间序列的**精确对数似然**为训练目标，
学习电器运行时的功率分布。训练完成后可：

- 从学习到的分布中**无条件采样**，生成统计特性与真实信号高度吻合的合成数据；
- 通过**潜变量正态性**（Shapiro-Wilk、QQ 图）验证模型编码质量；
- 使用 Wasserstein 距离、KL 散度、自相关 MAE 等指标**量化生成质量**。

---

## 模型架构

```
输入 (B, 1, L)
  │
  ▼  Squeeze1d × num_scales          将时间步折叠到通道维度
  │
  ▼  K × GlowStep（每个尺度）
  │     ├── ActNorm1d                 数据驱动的仿射归一化
  │     ├── Invertible1x1Conv         LU 分解的可逆通道混合
  │     └── AffineCouplingLayer       基于残差卷积网络的仿射耦合
  │
  ▼  潜变量 z ~ N(0, I)
```

训练目标（负对数似然）：

```
loss = 0.5 × mean(z²) + 0.5 × log(2π) − log_det / (B × C × L)
```

---

## 环境安装

```bash
# 建议 Python ≥ 3.9，PyTorch ≥ 2.0
pip install -r glow_nilm/requirements.txt
```

---

## 快速开始

所有脚本均在 `glow_nilm/` 目录下执行。

### 训练

```bash
cd glow_nilm

# 使用合成数据训练 kettle 模型（快速验证）
python train.py --appliance kettle --epochs 50 --use_synthetic

# 使用真实 UK-DALE CSV 数据训练
python train.py --appliance fridge \
    --data_path /path/to/fridge.csv \
    --epochs 100 --batch_size 64 --lr 1e-3
```

训练完成后，最优权重保存至 `checkpoints/<appliance>/best_model.pt`。

### 评估

```bash
# 评估单个电器
python evaluate.py --appliance kettle

# 批量评估所有已训练电器并输出汇总表格
python evaluate.py --all
```

评估结果（QQ 图、分布对比图）保存至 `results/<appliance>/`。

### 采样

```bash
# 从训练好的 kettle 模型采样 200 条序列
python sample.py --appliance kettle --n_samples 200 --temperature 0.9
```

生成样本保存至 `results/<appliance>/samples.npy`，附对比可视化图像。

---

## 支持电器

| 电器                | 合成数据特征                       | 额定功率参考 |
|---------------------|------------------------------------|-------------|
| `kettle`            | 稀疏高功率脉冲（~120 s）           | ~2500 W     |
| `fridge`            | 周期性开关循环（周期 ~600 s）      | ~150 W      |
| `washing_machine`   | 三阶段：预热 → 洗涤 → 甩干        | ~400–2000 W |
| `microwave`         | 短时高功率脉冲（30–180 s）         | ~1200 W     |
| `dishwasher`        | 长时中功率波形（周期 ~7200 s）     | ~1000–2200 W|

---

## 目录结构

```
glow_nilm/
├── train.py            # 模型训练入口
├── evaluate.py         # 模型评估入口
├── sample.py           # 采样与可视化入口
├── requirements.txt    # 依赖列表
├── data/
│   └── dataset.py      # 数据加载、合成数据生成、滑动窗口
├── models/
│   ├── glow.py         # Glow 主模型（GlowStep / Glow）
│   ├── actnorm.py      # ActNorm1d 激活归一化层
│   ├── coupling.py     # AffineCouplingLayer 仿射耦合层
│   └── invertible_conv.py  # Invertible1x1Conv 可逆卷积
└── utils/
    └── metrics.py      # MAE / RMSE / SAE / MRE / Wasserstein / KL 等指标
```

---

## 评估指标

| 指标             | 说明                                               |
|------------------|----------------------------------------------------|
| **MAE**          | 均绝对误差                                         |
| **RMSE**         | 均方根误差                                         |
| **SAE**          | 信号总量误差（Signal Aggregate Error）             |
| **MRE**          | 平均相对误差                                       |
| **Wasserstein**  | 生成分布与真实分布之间的 Wasserstein-1 距离         |
| **KL 散度**      | 直方图近似 KL(真实 ∥ 生成)                         |
| **ACF-MAE**      | 平均自相关曲线均绝对误差                           |
| **Shapiro-W**    | 潜变量 z 的正态性检验统计量                         |
