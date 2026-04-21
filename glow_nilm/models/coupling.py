"""
仿射耦合层（Affine Coupling Layer）
=====================================
将通道对半切分后执行仿射变换：

.. code-block:: text

    x1, x2 = split(x)
    s, t   = NN(x1)
    y1     = x1
    y2     = x2 × exp(s) + t
    log|det J| = sum(s)

逆变换：

.. code-block:: text

    y1, y2 = split(y)
    s, t   = NN(y1)
    x1     = y1
    x2     = (y2 − t) × exp(−s)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1d(nn.Module):
    """带残差连接的 1D 卷积块（用于耦合网络的特征提取）。"""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.GroupNorm(min(8, channels), channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class CouplingNetwork(nn.Module):
    """耦合层中用于预测缩放因子 s 和平移量 t 的网络（1D 卷积 + 残差块）。

    Args:
        in_channels:     输入通道数（= num_channels // 2）
        out_channels:    输出通道数（= num_channels，s 和 t 各占一半）
        hidden_channels: 隐藏层通道数
        n_blocks:        残差块数量
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_blocks: int = 2,
    ):
        super().__init__()
        self.input_conv = nn.Conv1d(in_channels, hidden_channels, 3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResBlock1d(hidden_channels) for _ in range(n_blocks)]
        )
        self.output_conv = nn.Conv1d(hidden_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)  # 初始输出为零，使初始变换接近恒等映射

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.input_conv(x))
        h = self.res_blocks(h)
        return self.output_conv(h)


class AffineCouplingLayer(nn.Module):
    """仿射耦合层。

    Args:
        num_channels:    总通道数（必须为偶数）
        hidden_channels: 耦合网络隐藏层通道数
        n_blocks:        耦合网络残差块数量
        scale_bound:     对缩放因子 s 的范围约束，防止数值爆炸
    """

    def __init__(
        self,
        num_channels: int,
        hidden_channels: int = 64,
        n_blocks: int = 2,
        scale_bound: float = 2.0,
    ):
        super().__init__()
        assert num_channels % 2 == 0, "通道数必须为偶数"
        self.c_half = num_channels // 2
        self.scale_bound = scale_bound

        self.net = CouplingNetwork(
            in_channels=self.c_half,
            out_channels=self.c_half * 2,  # s 和 t 各 c_half 通道
            hidden_channels=hidden_channels,
            n_blocks=n_blocks,
        )

    def forward(
        self, x: torch.Tensor, reverse: bool = False
    ):
        """
        Args:
            x:       (B, C, L)
            reverse: True 为逆变换

        Returns:
            (y, log_det)
        """
        x1, x2 = x[:, : self.c_half], x[:, self.c_half :]
        st = self.net(x1)  # (B, c_half*2, L)
        s, t = st[:, : self.c_half], st[:, self.c_half :]
        s = torch.tanh(s) * self.scale_bound

        if not reverse:
            y2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=[1, 2]).mean()  # 对 batch 取均值
            y = torch.cat([x1, y2], dim=1)
            return y, log_det
        else:
            x2_rec = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=[1, 2]).mean()
            x_rec = torch.cat([x1, x2_rec], dim=1)
            return x_rec, log_det
