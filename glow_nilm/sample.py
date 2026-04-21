"""
采样脚本
========
从已训练的 Glow-NILM 模型中采样，生成与真实电器功率信号统计特性高度相近的
合成序列，并输出定量对比指标与可视化图像。

用法
----
::

    python sample.py --appliance kettle --n_samples 200 --temperature 0.9

生成结果保存至 ``results/<appliance>/samples.npy``，附样本对比图与分布对比图。
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from models.glow import Glow
from data.dataset import generate_synthetic_data, sliding_window
from utils.metrics import compute_statistics


def parse_args():
    parser = argparse.ArgumentParser(description="Glow-NILM 采样脚本")
    parser.add_argument("--appliance", type=str, default="kettle")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="生成样本数量")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="采样温度（推荐范围 0.5–1.0；值越小生成越保守）")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_model(ckpt_path: Path, device: torch.device) -> tuple:
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt["args"]
    model = Glow(
        in_channels=1,
        num_steps=saved_args["num_steps"],
        num_scales=saved_args["num_scales"],
        hidden_channels=saved_args["hidden_channels"],
        n_coupling_blocks=saved_args["n_coupling_blocks"],
        squeeze_factor=saved_args["squeeze_factor"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    print(f"[模型]  已加载 {ckpt_path}（epoch {ckpt['epoch']}，val_loss={ckpt['val_loss']:.4f}）")
    return model, saved_args


def plot_samples(samples: np.ndarray, real: np.ndarray, appliance: str, save_dir: Path):
    """绘制生成样本与真实信号的逐窗口对比图。"""
    n_show = min(6, len(samples), len(real))
    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 6))
    fig.suptitle(f"Glow 生成样本 vs 真实信号 — {appliance}", fontsize=14)

    for i in range(n_show):
        axes[0, i].plot(real[i], color="steelblue", linewidth=0.8)
        axes[0, i].set_title("真实信号", fontsize=9)
        axes[0, i].set_ylim(-0.05, 1.05)

        axes[1, i].plot(samples[i], color="tomato", linewidth=0.8)
        axes[1, i].set_title("生成样本", fontsize=9)
        axes[1, i].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    out_path = save_dir / "sample_comparison.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[可视化]  样本对比图已保存至：{out_path}")


def plot_distribution(samples: np.ndarray, real: np.ndarray, appliance: str, save_dir: Path):
    """绘制归一化功率分布直方图对比。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(real.flatten(),    bins=60, density=True, alpha=0.6, label="真实数据", color="steelblue")
    ax.hist(samples.flatten(), bins=60, density=True, alpha=0.6, label="Glow 生成", color="tomato")
    ax.set_xlabel("归一化功率值")
    ax.set_ylabel("概率密度")
    ax.set_title(f"{appliance} — 功率分布对比")
    ax.legend()
    plt.tight_layout()
    out_path = save_dir / "distribution_comparison.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[可视化]  分布对比图已保存至：{out_path}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device_str = args.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    ckpt_path = Path(args.ckpt_dir) / args.appliance / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"未找到模型权重：{ckpt_path}。请先执行 train.py 完成训练。"
        )

    model, saved_args = load_model(ckpt_path, device)

    window_size = saved_args["window_size"]
    temperature = args.temperature

    # 采样
    print(f"[采样]   正在生成 {args.n_samples} 个样本（温度={temperature}）...")
    with torch.no_grad():
        samples_tensor = model.sample(
            n_samples=args.n_samples,
            seq_len=window_size,
            temperature=temperature,
            device=str(device),
        )
    samples = samples_tensor.squeeze(1).cpu().numpy()  # (N, L)
    print(f"[采样]   完成，样本张量形状：{samples.shape}")

    # 加载参考真实数据（用于对比）
    real_series = generate_synthetic_data(appliance=args.appliance)
    real_windows = sliding_window(real_series, window_size, stride=window_size)
    np.random.shuffle(real_windows)
    real_windows = real_windows[: args.n_samples]

    # 统计指标
    stats = compute_statistics(samples, real_windows)
    print("\n[统计对比]")
    print(f"  真实均值：{stats['mean_real']:.4f}    生成均值：{stats['mean_gen']:.4f}")
    print(f"  真实标准差：{stats['std_real']:.4f}   生成标准差：{stats['std_gen']:.4f}")
    print(f"  真实峰度：{stats['kurt_real']:.4f}    生成峰度：{stats['kurt_gen']:.4f}")
    print(f"  Wasserstein 距离：{stats['wasserstein']:.6f}")
    print(f"  KL 散度 (P‖Q)：{stats['kl_div']:.6f}")

    # 保存结果
    save_dir = Path(args.save_dir) / args.appliance
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "samples.npy", samples)
    print(f"\n[保存]   样本已保存至：{save_dir / 'samples.npy'}")

    plot_samples(samples, real_windows, args.appliance, save_dir)
    plot_distribution(samples, real_windows, args.appliance, save_dir)

    print("\n采样完成。")


if __name__ == "__main__":
    main()
