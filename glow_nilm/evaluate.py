"""
评估脚本
========
对已训练的 Glow 模型进行全面评估：
    1. 编码质量：验证潜变量 z 是否近似服从标准正态分布（Shapiro-Wilk / QQ 图）
    2. 生成质量：Wasserstein 距离、KL 散度、均值/方差比较
    3. 多电器比较：在所有已训练的电器模型上输出评估表格

用法：

    python evaluate.py --appliance kettle
    python evaluate.py --all        # 评估 checkpoints/ 下所有电器
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import shapiro, probplot

from models.glow import Glow
from data.dataset import generate_synthetic_data, sliding_window, ApplianceDataset
from utils.metrics import compute_statistics, compute_mae, compute_rmse


def parse_args():
    parser = argparse.ArgumentParser(description="Glow NILM 评估脚本")
    parser.add_argument("--appliance", type=str, default="kettle")
    parser.add_argument("--all", action="store_true", help="评估所有已训练电器")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_dir", type=str, default="results")
    return parser.parse_args()


def load_model(ckpt_path: Path, device: torch.device):
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
    model.eval().to(device)
    return model, saved_args


def evaluate_latent_normality(model, real_series, window_size, device, save_dir):
    """检验潜变量 z 是否近似服从标准正态分布。"""
    windows = sliding_window(real_series, window_size, stride=window_size)
    idx = np.random.choice(len(windows), size=min(200, len(windows)), replace=False)
    x = torch.from_numpy(windows[idx]).float().unsqueeze(1).to(device)
    with torch.no_grad():
        z, _ = model.encode(x)
    z_np = z.cpu().numpy().flatten()

    # Shapiro-Wilk 检验（取子样本，因 n<5000）
    sub = np.random.choice(z_np, size=min(5000, len(z_np)), replace=False)
    stat, p_val = shapiro(sub[:500])  # shapiro 最大支持 5000，但建议 ≤500

    print(f"\n[潜变量正态性检验] Shapiro-Wilk W={stat:.4f}, p={p_val:.4f}")
    print(f"  均值: {z_np.mean():.4f}, 标准差: {z_np.std():.4f}")

    # QQ 图
    fig, ax = plt.subplots(figsize=(5, 5))
    probplot(sub, dist="norm", plot=ax)
    ax.set_title("潜变量 z — QQ 图（vs 标准正态）")
    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "latent_qq_plot.png", dpi=120)
    plt.close()
    print(f"  QQ 图已保存至 {save_dir / 'latent_qq_plot.png'}")
    return {"shapiro_W": float(stat), "shapiro_p": float(p_val)}


def evaluate_generation(model, real_series, window_size, n_samples, temperature, device, save_dir):
    """评估生成质量（统计指标 + 分布可视化）。"""
    # 真实窗口
    real_windows = sliding_window(real_series, window_size, stride=window_size)
    np.random.shuffle(real_windows)
    real_windows = real_windows[:n_samples]

    # 生成样本
    with torch.no_grad():
        samples = model.sample(n_samples, window_size, temperature=temperature, device=str(device))
    samples_np = samples.squeeze(1).cpu().numpy()

    stats = compute_statistics(samples_np, real_windows)
    print(f"\n[生成质量评估]")
    print(f"  Wasserstein 距离: {stats['wasserstein']:.6f}")
    print(f"  KL 散度:          {stats['kl_div']:.6f}")
    print(f"  均值差异:         |{stats['mean_real']:.4f} - {stats['mean_gen']:.4f}| = "
          f"{abs(stats['mean_real'] - stats['mean_gen']):.4f}")
    print(f"  标准差差异:       |{stats['std_real']:.4f} - {stats['std_gen']:.4f}| = "
          f"{abs(stats['std_real'] - stats['std_gen']):.4f}")

    # 自相关对比
    def mean_autocorr(windows, max_lag=32):
        acf = []
        for w in windows:
            w = w - w.mean()
            if w.std() < 1e-6:
                continue
            c = np.correlate(w, w, mode="full")
            c = c[len(c) // 2: len(c) // 2 + max_lag]
            c /= c[0] + 1e-10
            acf.append(c)
        return np.mean(acf, axis=0)

    acf_real = mean_autocorr(real_windows)
    acf_gen  = mean_autocorr(samples_np)
    acf_mae  = compute_mae(acf_real, acf_gen)

    print(f"  自相关曲线 MAE:   {acf_mae:.6f}")

    # 分布 + 自相关可视化
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(real_windows.flatten(), bins=60, density=True, alpha=0.6, label="真实数据", color="steelblue")
    ax1.hist(samples_np.flatten(), bins=60, density=True, alpha=0.6, label="Glow 生成", color="tomato")
    ax1.set_xlabel("归一化功率"); ax1.set_ylabel("密度"); ax1.legend(); ax1.set_title("功率分布对比")

    ax2.plot(acf_real, label="真实数据", color="steelblue")
    ax2.plot(acf_gen,  label="Glow 生成", color="tomato", linestyle="--")
    ax2.set_xlabel("延迟（lag）"); ax2.set_ylabel("自相关"); ax2.legend(); ax2.set_title("平均自相关对比")

    plt.tight_layout()
    fig.savefig(save_dir / "eval_comparison.png", dpi=120)
    plt.close()
    print(f"  评估图已保存至 {save_dir / 'eval_comparison.png'}")

    stats["acf_mae"] = acf_mae
    return stats


def evaluate_appliance(appliance, args, device):
    ckpt_path = Path(args.ckpt_dir) / appliance / "best_model.pt"
    if not ckpt_path.exists():
        print(f"[跳过] {appliance}：未找到模型权重 {ckpt_path}")
        return None

    save_dir = Path(args.save_dir) / appliance
    model, saved_args = load_model(ckpt_path, device)
    window_size = saved_args.get("window_size", args.window_size)

    real_series = generate_synthetic_data(appliance=appliance)

    print(f"\n{'='*50}")
    print(f"电器: {appliance}")
    print('='*50)

    latent_stats = evaluate_latent_normality(model, real_series, window_size, device, save_dir)
    gen_stats    = evaluate_generation(
        model, real_series, window_size, args.n_samples, args.temperature, device, save_dir
    )
    return {**latent_stats, **gen_stats}


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.all:
        appliances = [d.name for d in Path(args.ckpt_dir).iterdir() if d.is_dir()]
        if not appliances:
            print(f"[错误] {args.ckpt_dir} 下未找到任何已训练的模型。")
            return
        print(f"[评估] 发现 {len(appliances)} 个电器：{appliances}")
        results = {}
        for ap in appliances:
            r = evaluate_appliance(ap, args, device)
            if r:
                results[ap] = r

        # 打印汇总表格
        print(f"\n{'='*70}")
        print(f"{'电器':<20} {'Wasserstein':>14} {'KL散度':>12} {'ACF-MAE':>12} {'Shapiro-W':>12}")
        print('-'*70)
        for ap, r in results.items():
            print(f"{ap:<20} {r['wasserstein']:>14.6f} {r['kl_div']:>12.6f} "
                  f"{r['acf_mae']:>12.6f} {r['shapiro_W']:>12.4f}")
    else:
        evaluate_appliance(args.appliance, args, device)

    print("\n评估完成！")


if __name__ == "__main__":
    main()
