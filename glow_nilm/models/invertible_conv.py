"""
可逆 1×1 卷积（Invertible 1×1 Convolution）
===========================================
对 1D 序列在通道维度执行可逆线性变换，实现可学习的通道混合与排列，
等价于 Glow 论文中对图像空间维度的通道混合操作。

前向::

    y = W @ x,      log|det J| = L × log|det W|

逆向::

    x = W⁻¹ @ y

为保证数值稳定性，采用 LU 分解参数化权重矩阵：W = P L U，
其中 P 为固定置换矩阵，L 为对角元固定为 1 的下三角矩阵，U 为上三角矩阵。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Invertible1x1Conv(nn.Module):
    """可逆 1×1 卷积（LU 分解参数化）。

    Args:
        num_channels: 通道数
        lu_decompose: 若为 True，使用 LU 分解参数化（节省存储并加速矩阵求逆）
    """

    def __init__(self, num_channels: int, lu_decompose: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.lu_decompose = lu_decompose

        # 以随机正交矩阵初始化权重
        W_init = np.linalg.qr(np.random.randn(num_channels, num_channels))[0]
        W_init = W_init.astype(np.float32)

        if lu_decompose:
            # P、L、U 三角分解
            P, L, U = self._plu(W_init)
            self.register_buffer("P", torch.from_numpy(P))
            self.L_raw   = nn.Parameter(torch.from_numpy(L))
            self.U_raw   = nn.Parameter(torch.from_numpy(U))
            # 固定上三角对角元的符号，保证行列式符号不变
            s = np.diag(U)
            sign_s = np.sign(s)
            sign_s[sign_s == 0] = 1.0
            log_abs_s = np.log(np.abs(s) + 1e-8)
            self.register_buffer("sign_s", torch.from_numpy(sign_s.astype(np.float32)))
            self.log_s = nn.Parameter(torch.from_numpy(log_abs_s.astype(np.float32)))
        else:
            self.W = nn.Parameter(torch.from_numpy(W_init))

    @staticmethod
    def _plu(W: np.ndarray):
        from scipy.linalg import lu
        P, L, U = lu(W)
        return P.astype(np.float32), L.astype(np.float32), U.astype(np.float32)

    def _get_W(self, inverse: bool = False):
        """构造权重矩阵；若 inverse=True 则返回其逆矩阵。"""
        if self.lu_decompose:
            C = self.num_channels
            # 下三角矩阵：强制对角元为 1
            L = torch.tril(self.L_raw, diagonal=-1) + torch.eye(C, device=self.L_raw.device)
            # 上三角矩阵：以 log|s| + sign_s 重建对角
            U_diag = self.sign_s * torch.exp(self.log_s)
            U = torch.triu(self.U_raw, diagonal=1) + torch.diag(U_diag)
            W = self.P @ L @ U
            log_det = self.log_s.sum()
        else:
            W = self.W
            log_det = torch.log(torch.abs(torch.det(W)))

        if inverse:
            W = torch.inverse(W)
        return W, log_det

    def forward(
        self, x: torch.Tensor, reverse: bool = False
    ):
        """
        Args:
            x:       (B, C, L)
            reverse: True 为逆变换（生成方向）

        Returns:
            (y, log_det)  log_det 为 scalar（L 倍乘以对数行列式）
        """
        B, C, L = x.shape
        W, log_det_W = self._get_W(inverse=reverse)
        log_det = log_det_W * L

        # (B, C, L) -> 对通道做矩阵乘
        y = torch.einsum("oi, bil -> bol", W, x)
        if reverse:
            return y, -log_det
        return y, log_det
