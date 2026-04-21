"""
数据集加载与预处理模块
======================
支持 UK-DALE、REDD 等公开 NILM 数据集，以及自定义 CSV 格式。
提供滑动窗口切分、MinMax 归一化和合成数据生成功能。
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Tuple


# 各电器的功率开启阈值（瓦），用于弱标签状态判断
APPLIANCE_THRESHOLDS = {
    "kettle":        2000,
    "microwave":      200,
    "fridge":          50,
    "washing_machine": 20,
    "dishwasher":      10,
}

# UK-DALE 各电器在不同 house 中的子计量通道编号（仅供参考）
UKDALE_APPLIANCE_CHANNELS = {
    "kettle":        {1: 10, 3: 2, 5: 0},
    "microwave":     {1: 13, 5: 23},
    "fridge":        {1: 12, 2: 14},
    "washing_machine": {1: 5, 5: 24},
    "dishwasher":    {1: 6},
}


def sliding_window(sequence: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """将一维时间序列切分为滑动窗口片段。

    Args:
        sequence:    一维 numpy 数组，shape (T,)
        window_size: 窗口长度（时间步数）
        stride:      滑动步长

    Returns:
        shape (N, window_size) 的二维数组
    """
    assert sequence.ndim == 1, "输入序列必须为一维数组"
    n_windows = (len(sequence) - window_size) // stride + 1
    indices = np.arange(window_size)[None, :] + stride * np.arange(n_windows)[:, None]
    return sequence[indices]


def load_ukdale(
    data_path: str,
    appliance: str,
    house: int = 1,
    sample_rate: str = "6s",
    train_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """从 UK-DALE 格式的 CSV 文件加载单电器功率序列并进行归一化处理。

    CSV 格式要求：索引列为时间戳（timestamp），数据列为功率值（power，单位：瓦）。
    若无真实数据，可使用 :func:`generate_synthetic_data` 替代。

    Args:
        data_path:   CSV 文件路径
        appliance:   电器名称（仅用于阈值参考）
        house:       房间编号
        sample_rate: 重采样频率
        train_ratio: 训练集比例

    Returns:
        (train_series, test_series)，均为 MinMax 归一化后的 float32 数组
    """
    df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")
    df = df.resample(sample_rate).mean().fillna(method="ffill").fillna(0.0)
    series = df["power"].values.astype(np.float32)
    series = np.clip(series, 0, None)

    split = int(len(series) * train_ratio)
    train_raw, test_raw = series[:split], series[split:]

    scaler = MinMaxScaler()
    train_norm = scaler.fit_transform(train_raw.reshape(-1, 1)).flatten()
    test_norm  = scaler.transform(test_raw.reshape(-1, 1)).flatten()
    return train_norm, test_norm


def generate_synthetic_data(
    appliance: str = "kettle",
    n_samples: int = 50000,
    seed: int = 42,
) -> np.ndarray:
    """生成合成电器功率序列，供无真实数据时调试与快速验证使用。

    各电器模型假设：

    - **fridge**:           低频持续噪声 + 周期性开关脉冲（周期 ~600 s）
    - **kettle**:           稀疏高功率脉冲（持续约 60–180 s）
    - **washing_machine**:  三阶段用电（预热→洗涤→甩干）
    - **microwave**:        短时高功率脉冲（持续约 30–180 s）
    - **dishwasher**:       长时中功率波形（周期 ~7200 s）
    - 其他：                高斯基础噪声 + 随机功率尖峰
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples, dtype=np.float32)
    power = np.zeros(n_samples, dtype=np.float32)

    if appliance == "fridge":
        # 周期约 600 s 的开关循环
        cycle = 600
        on_mask = (t % cycle) < (cycle * 0.4)
        power[on_mask] = rng.normal(loc=150, scale=20, size=on_mask.sum())
        power[~on_mask] = rng.normal(loc=2, scale=1, size=(~on_mask).sum())

    elif appliance == "kettle":
        # 稀疏的约 60–180 s 烧水脉冲
        pulse_starts = rng.choice(n_samples - 200, size=n_samples // 2000, replace=False)
        for s in pulse_starts:
            duration = rng.randint(60, 180)
            end = min(s + duration, n_samples)
            power[s:end] = rng.normal(loc=2500, scale=100, size=end - s)

    elif appliance == "washing_machine":
        # 三阶段：预热（高功率）→ 洗涤（中功率）→ 甩干（间歇脉冲）
        cycle_len = 3600
        n_cycles = n_samples // cycle_len
        for c in range(n_cycles):
            base = c * cycle_len
            p1_end = base + 600
            p2_end = base + 2400
            p3_end = base + cycle_len
            power[base:p1_end] = rng.normal(2000, 200, p1_end - base)
            power[p1_end:p2_end] = rng.normal(400, 50, p2_end - p1_end)
            for i in range(p2_end, p3_end, 60):
                power[i:i + 30] = rng.normal(800, 100, min(30, p3_end - i))

    elif appliance == "microwave":
        pulse_starts = rng.choice(n_samples - 200, size=n_samples // 5000, replace=False)
        for s in pulse_starts:
            duration = rng.randint(30, 180)
            end = min(s + duration, n_samples)
            power[s:end] = rng.normal(1200, 80, end - s)

    elif appliance == "dishwasher":
        cycle_len = 7200
        n_cycles = n_samples // cycle_len
        for c in range(n_cycles):
            base = c * cycle_len
            end = min(base + cycle_len, n_samples)
            power[base:end] = rng.normal(1000, 150, end - base)
            power[base:base + 600] = rng.normal(2200, 100, 600)

    else:
        power = rng.normal(loc=50, scale=20, size=n_samples).astype(np.float32)
        spikes = rng.choice(n_samples, size=n_samples // 200)
        for s in spikes:
            power[s] += rng.uniform(200, 1000)

    power = np.clip(power, 0, None)
    scaler = MinMaxScaler()
    power = scaler.fit_transform(power.reshape(-1, 1)).flatten().astype(np.float32)
    return power


class ApplianceDataset(Dataset):
    """单电器功率时间序列数据集。

    将连续时间序列切分为固定长度窗口，每个样本形状为 ``(window_size,)``。
    训练时送入 Glow 模型，学习该电器的功率分布。

    Args:
        series:      一维功率序列（已归一化至 [0, 1]）
        window_size: 每个样本的时间步数
        stride:      滑动步长
        augment:     是否对训练样本执行随机时序翻转增强
    """

    def __init__(
        self,
        series: np.ndarray,
        window_size: int = 128,
        stride: int = 64,
        augment: bool = False,
    ):
        self.augment = augment
        windows = sliding_window(series, window_size, stride)
        self.data = torch.from_numpy(windows).float()  # (N, window_size)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.data[idx]  # (window_size,)
        if self.augment and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[0])
        return x
