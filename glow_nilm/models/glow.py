"""
Glow 主模型
===========
每个 Glow 步骤（GlowStep）由三个子模块依次组成：

1. **ActNorm1d**           — 数据驱动的仿射激活归一化
2. **Invertible1x1Conv**   — LU 分解参数化的可逆 1×1 通道混合卷积
3. **AffineCouplingLayer** — 基于残差卷积网络的仿射耦合变换

多个 GlowStep 堆叠为 GlowBlock；支持多尺度结构（每个 block 后通过
Squeeze1d 将时间步折叠到通道维度），单尺度配置同样适用于功率时间序列。

训练目标（逐元素负对数似然）：

.. code-block:: text

    loss = 0.5 × mean(z²) + 0.5 × log(2π) − log_det / (B × C × L)

其中 z 是将输入 x 编码到高斯空间后的潜变量，log_det 为整条流的
对数雅可比行列式之和。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

from .actnorm import ActNorm1d
from .invertible_conv import Invertible1x1Conv
from .coupling import AffineCouplingLayer


class GlowStep(nn.Module):
    """单个 Glow 步骤：ActNorm → Invertible1×1Conv → AffineCoupling。

    Args:
        num_channels:      当前通道数
        hidden_channels:   耦合网络隐藏层通道数
        n_coupling_blocks: 耦合网络残差块数量
    """

    def __init__(
        self,
        num_channels: int,
        hidden_channels: int = 64,
        n_coupling_blocks: int = 2,
    ):
        super().__init__()
        self.actnorm = ActNorm1d(num_channels)
        self.inv_conv = Invertible1x1Conv(num_channels)
        self.coupling = AffineCouplingLayer(
            num_channels,
            hidden_channels=hidden_channels,
            n_blocks=n_coupling_blocks,
        )

    def forward(
        self, x: torch.Tensor, reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not reverse:
            x, ld1 = self.actnorm(x, reverse=False)
            x, ld2 = self.inv_conv(x, reverse=False)
            x, ld3 = self.coupling(x, reverse=False)
            return x, ld1 + ld2 + ld3
        else:
            x, ld3 = self.coupling(x, reverse=True)
            x, ld2 = self.inv_conv(x, reverse=True)
            x, ld1 = self.actnorm(x, reverse=True)
            return x, ld1 + ld2 + ld3


class Squeeze1d(nn.Module):
    """将时间步维度折叠到通道维度（L → L//factor，C → C×factor）。"""

    def __init__(self, factor: int = 2):
        self.factor = factor
        super().__init__()

    def forward(self, x: torch.Tensor, reverse: bool = False):
        B, C, L = x.shape
        f = self.factor
        if not reverse:
        assert L % f == 0, f"序列长度 {L} 必须能被 squeeze_factor {f} 整除"
            x = x.view(B, C, L // f, f).permute(0, 1, 3, 2)  # (B, C, f, L//f)
            x = x.contiguous().view(B, C * f, L // f)
        else:
            assert C % f == 0
            x = x.view(B, C // f, f, L).permute(0, 1, 3, 2)  # (B, C//f, L, f)
            x = x.contiguous().view(B, C // f, L * f)
        return x


class Glow(nn.Module):
    """基于 Glow 归一化流的电器功率时间序列分布建模网络。

    网络结构（单尺度示例）：

    .. code-block:: text

        输入 (B, 1, L)
          ↓  Squeeze1d → (B, squeeze_factor, L // squeeze_factor)
          ↓  K × GlowStep
          ↓  [可选多尺度] 继续 Squeeze1d + K × GlowStep
          ↓  潜变量 z，形状与 squeeze 后的输入相同

    Args:
        in_channels:       输入通道数（功率单序列为 1）
        num_steps:         每个尺度的 Glow 步骤数（K）
        num_scales:        多尺度层数（每层 squeeze 2 倍）
        hidden_channels:   耦合网络隐藏层维度
        n_coupling_blocks: 耦合网络残差块数量
        squeeze_factor:    squeeze 倍数
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_steps: int = 8,
        num_scales: int = 1,
        hidden_channels: int = 64,
        n_coupling_blocks: int = 2,
        squeeze_factor: int = 2,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.squeeze_factor = squeeze_factor

        self.squeezes: nn.ModuleList = nn.ModuleList()
        self.flow_blocks: nn.ModuleList = nn.ModuleList()

        c = in_channels
        for scale in range(num_scales):
            self.squeezes.append(Squeeze1d(squeeze_factor))
            c_sq = c * squeeze_factor
            block = nn.ModuleList([
                GlowStep(c_sq, hidden_channels, n_coupling_blocks)
                for _ in range(num_steps)
            ])
            self.flow_blocks.append(block)
            c = c_sq

        self.out_channels = c  # 最终潜变量通道数

    # ------------------------------------------------------------------ #
    #  前向（编码方向 x → z）
    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """将功率序列 x 编码为潜变量 z，并返回对数行列式之和。

        Args:
            x: (B, 1, L)，归一化功率序列

        Returns:
            z:          (B, C_out, L_out) 潜变量
            log_det:    scalar，对数行列式之和
        """
        log_det_total = torch.zeros(1, device=x.device)
        h = x
        for scale in range(self.num_scales):
            h = self.squeezes[scale](h, reverse=False)
            for step in self.flow_blocks[scale]:
                h, ld = step(h, reverse=False)
                log_det_total = log_det_total + ld
        return h, log_det_total

    # ------------------------------------------------------------------ #
    #  逆向（生成方向 z → x）
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """从潜变量 z 生成功率序列样本。

        Args:
            z: (B, C_out, L_out)，从标准正态分布采样

        Returns:
            x: (B, 1, L)
        """
        h = z
        for scale in reversed(range(self.num_scales)):
            for step in reversed(list(self.flow_blocks[scale])):
                h, _ = step(h, reverse=True)
            h = self.squeezes[scale](h, reverse=True)
        return h

    # ------------------------------------------------------------------ #
    #  负对数似然损失
    # ------------------------------------------------------------------ #
    def nll_loss(self, x: torch.Tensor) -> torch.Tensor:
        """计算逐元素负对数似然损失（越小表示拟合越好）。

        .. code-block:: text

            loss = 0.5 × mean(z²) + 0.5 × log(2π) − log_det / (B × C × L)

        Args:
            x: (B, 1, L)，归一化功率序列

        Returns:
            loss: 标量
        """
        z, log_det = self.encode(x)
        B, C, L = z.shape
        # 高斯对数似然：-0.5 * z² - 0.5 * log(2π)
        nll = 0.5 * (z ** 2 + np.log(2 * np.pi))  # (B, C, L)
        nll = nll.mean()  # 对所有维度平均
        # 减去每元素的对数行列式贡献
        nll = nll - log_det / (B * C * L)
        return nll

    # ------------------------------------------------------------------ #
    #  采样
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        seq_len: int,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> torch.Tensor:
        """从模型中采样功率时间序列。

        Args:
            n_samples:   生成样本数量
            seq_len:     原始序列长度（squeeze 前的 L）
            temperature: 采样温度（< 1 更保守，> 1 多样性更高）
            device:      计算设备字符串

        Returns:
            samples: (n_samples, 1, seq_len)，值域 [0, 1]（经 sigmoid 裁剪）
        """
        factor = self.squeeze_factor ** self.num_scales
        assert seq_len % factor == 0, (
            f"seq_len ({seq_len}) 必须能被 {factor}（squeeze 总倍数）整除"
        )
        L_out = seq_len // factor
        C_out = self.out_channels

        z = torch.randn(n_samples, C_out, L_out, device=device) * temperature
        x = self.decode(z)
        # 将输出裁剪到 [0, 1]
        x = torch.sigmoid(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """默认前向为编码方向（训练使用 nll_loss）。"""
        return self.encode(x)
