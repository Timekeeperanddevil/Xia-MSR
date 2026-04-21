"""
评估指标工具函数
================
常用 NILM 误差指标及分布相似度度量：

- **MAE**  均绝对误差（Mean Absolute Error）
- **RMSE** 均方根误差（Root Mean Square Error）
- **SAE**  信号总量误差（Signal Aggregate Error）
- **MRE**  平均相对误差（Mean Relative Error）
- **Wasserstein** Wasserstein-1 距离
- **KL 散度**  直方图近似 KL 散度
"""

import numpy as np


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均绝对误差（Mean Absolute Error）。"""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差（Root Mean Square Error）。"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_sae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """信号总量误差（Signal Aggregate Error）。

    .. code-block:: text

        SAE = |sum(y_pred) − sum(y_true)| / sum(y_true)
    """
    total_true = np.sum(y_true)
    if total_true == 0:
        return 0.0
    return float(np.abs(np.sum(y_pred) - total_true) / total_true)


def compute_mre(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """平均相对误差（Mean Relative Error）。"""
    return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)))


def compute_statistics(samples: np.ndarray, real: np.ndarray) -> dict:
    """比较生成样本与真实数据的统计特性，返回多维度分布相似度指标。

    Args:
        samples: (N, L)，生成样本窗口
        real:    (M, L)，真实数据窗口

    Returns:
        包含均值、标准差、峰度、Wasserstein 距离、KL 散度等的字典
    """
    from scipy.stats import kurtosis, wasserstein_distance

    stats = {}
    s_flat = samples.flatten()
    r_flat = real.flatten()

    stats["mean_real"]    = float(np.mean(r_flat))
    stats["mean_gen"]     = float(np.mean(s_flat))
    stats["std_real"]     = float(np.std(r_flat))
    stats["std_gen"]      = float(np.std(s_flat))
    stats["kurt_real"]    = float(kurtosis(r_flat))
    stats["kurt_gen"]     = float(kurtosis(s_flat))
    stats["wasserstein"]  = float(wasserstein_distance(r_flat, s_flat))

    # 直方图近似 KL 散度（离散化近似）
    bins = 50
    p, edges = np.histogram(r_flat, bins=bins, density=True)
    q, _     = np.histogram(s_flat, bins=edges, density=True)
    eps = 1e-10
    p, q = p + eps, q + eps
    p /= p.sum(); q /= q.sum()
    stats["kl_div"] = float(np.sum(p * np.log(p / q)))

    return stats
