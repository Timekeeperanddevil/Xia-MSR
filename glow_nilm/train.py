"""
训练脚本
========
用法（示例）：

    # 使用合成数据训练 kettle 模型
    python train.py --appliance kettle --epochs 50 --use_synthetic

    # 使用真实 CSV 数据训练
    python train.py --appliance fridge --data_path /path/to/fridge.csv --epochs 100

训练完成后，模型权重保存在 checkpoints/<appliance>/best_model.pt。
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import ApplianceDataset, generate_synthetic_data, load_ukdale
from models.glow import Glow


def parse_args():
    parser = argparse.ArgumentParser(description="Glow NILM 训练脚本")
    parser.add_argument("--appliance", type=str, default="kettle",
                        help="电器名称（kettle/fridge/microwave/washing_machine/dishwasher）")
    parser.add_argument("--data_path", type=str, default=None,
                        help="真实数据 CSV 路径（含 timestamp, power 列）")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="使用合成数据替代真实数据（调试用）")
    parser.add_argument("--window_size", type=int, default=128,
                        help="滑动窗口长度")
    parser.add_argument("--stride", type=int, default=32,
                        help="滑动步长")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_steps", type=int, default=8,
                        help="每个尺度的 Glow 步骤数")
    parser.add_argument("--num_scales", type=int, default=1,
                        help="多尺度层数")
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--n_coupling_blocks", type=int, default=2)
    parser.add_argument("--squeeze_factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto",
                        help="cpu / cuda / auto")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def build_dataloaders(args):
    """构建训练/验证 DataLoader。"""
    if args.use_synthetic or args.data_path is None:
        print(f"[数据] 使用合成数据（电器：{args.appliance}）")
        series = generate_synthetic_data(appliance=args.appliance, seed=args.seed)
        split = int(len(series) * 0.8)
        train_series, val_series = series[:split], series[split:]
    else:
        print(f"[数据] 从 {args.data_path} 加载真实数据")
        train_series, val_series = load_ukdale(
            args.data_path, appliance=args.appliance
        )

    train_ds = ApplianceDataset(train_series, args.window_size, args.stride, augment=True)
    val_ds   = ApplianceDataset(val_series,   args.window_size, args.stride, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    print(f"[数据] 训练样本数: {len(train_ds)}, 验证样本数: {len(val_ds)}")
    return train_loader, val_loader


def build_model(args) -> Glow:
    model = Glow(
        in_channels=1,
        num_steps=args.num_steps,
        num_scales=args.num_scales,
        hidden_channels=args.hidden_channels,
        n_coupling_blocks=args.n_coupling_blocks,
        squeeze_factor=args.squeeze_factor,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 可训练参数量: {n_params:,}")
    return model


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    for batch in loader:
        x = batch.to(device)           # (B, window_size)
        x = x.unsqueeze(1)             # (B, 1, window_size)
        # 加少量抖动防止过拟合精确值
        x = x + torch.randn_like(x) * 1e-3
        optimizer.zero_grad()
        loss = model.nll_loss(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        x = batch.to(device).unsqueeze(1)
        loss = model.nll_loss(x)
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"[设备] 使用: {device}")

    train_loader, val_loader = build_dataloaders(args)
    model = build_model(args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_path = Path(args.save_dir) / args.appliance
    save_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss   = validate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:>3}/{args.epochs}] "
            f"Train NLL: {train_loss:.4f} | Val NLL: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }
            torch.save(ckpt, save_path / "best_model.pt")
            print(f"  ✓ 保存最优模型 (val_loss={val_loss:.4f})")

    print(f"\n训练完成！最优验证损失: {best_val_loss:.4f}")
    print(f"模型已保存至: {save_path / 'best_model.pt'}")


if __name__ == "__main__":
    main()
