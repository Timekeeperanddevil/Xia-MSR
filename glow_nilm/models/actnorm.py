"""
ActNorm（激活归一化层）
=======================
对每个通道独立进行仿射变换，参数在第一批数据时通过数据驱动的方式初始化（
使得输出近似均值为 0、方差为 1），之后作为可训练参数更新。

前向（归一化方向）:
    y = (x - bias) / exp(logs)
    log|det J| = -sum(logs) * pixels

逆向（生成方向）:
    x = y * exp(logs) + bias
"""

import torch
import torch.nn as nn


class ActNorm1d(nn.Module):
    """1D 序列的 ActNorm。

    Args:
        num_channels: 特征/通道数
        eps:          数值稳定项
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps

        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.logs = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.initialized = False

    def _initialize(self, x: torch.Tensor):
        """用第一个 mini-batch 做数据驱动初始化。x: (B, C, L)"""
        with torch.no_grad():
            # 在 batch 和 length 维度上统计
            mean = x.mean(dim=[0, 2], keepdim=True)          # (1, C, 1)
            std  = x.std(dim=[0, 2], keepdim=True) + self.eps
            self.bias.data.copy_(-mean)
            self.logs.data.copy_(-torch.log(std))
        self.initialized = True

    def forward(
        self, x: torch.Tensor, reverse: bool = False
    ):
        """
        Args:
            x:       (B, C, L)
            reverse: 若 True 则执行逆变换（生成方向）

        Returns:
            (y, log_det)  log_det 为 scalar
        """
        if not self.initialized:
            self._initialize(x)

        B, C, L = x.shape
        log_det = self.logs.sum() * L  # scalar

        if not reverse:
            y = (x + self.bias) * torch.exp(self.logs)
            return y, -log_det          # 归一化方向：对数行列式为负
        else:
            y = x * torch.exp(-self.logs) - self.bias
            return y, log_det
