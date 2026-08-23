"""Flow Matching を MNIST (28x28 グレースケール) に適用するサンプル。

2D トイデータ版 (`flow_matching_2d.py`) と理論・損失は完全に同じ:

    x_t = (1-t) x0 + t x1,   x0 ~ N(0,I),  x1 ~ p_data,  t ~ U(0,1)
    loss = E || v_theta(x_t, t, y) - (x1 - x0) ||^2

違うのは v_theta が MLP ではなく小さな U-Net (CNN) になり、入力がベクトルでは
なく 1x28x28 の画像になる点だけ。クラスラベル y (0-9) を条件付け、CFG
(classifier-free guidance) にも対応。

実行:
    python flow_matching_mnist.py --steps 6000
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from torchvision.datasets import MNIST  # noqa: E402
from torchvision.utils import make_grid  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
OUT_DIR = ROOT / "out" / "mnist"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
N_CLASSES = 10
NULL_LABEL = N_CLASSES


# ------------------------------------------------------------------ データ
def load_mnist_tensor(device) -> tuple[torch.Tensor, torch.Tensor]:
    """全画像をメモリに載せる (60000 x 1 x 28 x 28, [-1,1] 正規化)。MNIST は小さいので可能。"""
    ds = MNIST(root=DATA_DIR, train=True, download=True)
    x = ds.data.float().unsqueeze(1) / 127.5 - 1.0  # [0,255] -> [-1,1]
    y = ds.targets.long()
    return x.to(device), y.to(device)


# ------------------------------------------------------------------ モデル
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(torch.linspace(0.0, math.log(1000.0), half))
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t[:, None] * self.freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    """time (+class) embedding を加算する GroupNorm 残差ブロック。"""

    def __init__(self, ch_in: int, ch_out: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch_in)
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, ch_out)
        self.norm2 = nn.GroupNorm(8, ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, padding=1)
        self.skip = nn.Conv2d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNetVelocity(nn.Module):
    """28x28 -> 14x14 -> 7x7 -> 14x14 -> 28x28 の小さな U-Net。"""

    def __init__(self, base_ch: int = 64, emb_dim: int = 128):
        super().__init__()
        self.time_embed = TimeEmbedding(emb_dim)
        self.label_embed = nn.Embedding(N_CLASSES + 1, emb_dim)  # +1 = null (CFG 用)

        self.in_conv = nn.Conv2d(1, base_ch, 3, padding=1)

        self.down1 = ResBlock(base_ch, base_ch, emb_dim)
        self.pool1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)      # 28 -> 14
        self.down2 = ResBlock(base_ch, base_ch * 2, emb_dim)
        self.pool2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)  # 14 -> 7

        self.mid = ResBlock(base_ch * 2, base_ch * 2, emb_dim)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)  # 7 -> 14
        self.dec2 = ResBlock(base_ch * 4, base_ch, emb_dim)   # skip 連結で ch 2 倍
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1)          # 14 -> 28
        self.dec1 = ResBlock(base_ch * 2, base_ch, emb_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        emb = self.time_embed(t) + self.label_embed(y)

        h0 = self.in_conv(x)
        h1 = self.down1(h0, emb)
        h2 = self.down2(self.pool1(h1), emb)
        m = self.mid(self.pool2(h2), emb)

        u2 = self.up2(m)
        d2 = self.dec2(torch.cat([u2, h2], dim=1), emb)
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, h1], dim=1), emb)

        return self.out_conv(F.silu(self.out_norm(d1)))


# ------------------------------------------------------------------- 学習
def cfm_loss(model, x1: torch.Tensor, y: torch.Tensor, p_uncond: float) -> torch.Tensor:
    b = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = torch.rand(b, device=x1.device)
    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
    target = x1 - x0

    drop = torch.rand(b, device=x1.device) < p_uncond
    y = torch.where(drop, torch.full_like(y, NULL_LABEL), y)

    return (model(xt, t, y) - target).pow(2).mean()


def train(model, images, labels, steps: int, batch_size: int, lr: float, p_uncond: float, device) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    n = images.shape[0]
    history = []

    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch_size,), device=device)
        loss = cfm_loss(model, images[idx], labels[idx], p_uncond)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        history.append(loss.item())
        if step % max(1, steps // 20) == 0 or step == 1:
            recent = sum(history[-100:]) / len(history[-100:])
            print(f"step {step:>6d}/{steps}  loss {loss.item():.4f}  (avg100 {recent:.4f})")
    return history


# --------------------------------------------------------------- サンプリング
@torch.no_grad()
def sample_ode(model, labels: torch.Tensor, steps: int, guidance: float, device, return_traj: bool = False):
    """midpoint 積分 + CFG。labels は (N,) の int テンソル。"""
    model.eval()
    n = labels.shape[0]
    x = torch.randn(n, 1, 28, 28, device=device)
    y = labels.to(device)
    y_null = torch.full_like(y, NULL_LABEL)
    dt = 1.0 / steps
    traj = [x.clone()] if return_traj else None

    def velocity(x_in, t_in):
        v_cond = model(x_in, t_in, y)
        if guidance == 0.0:
            return v_cond
        v_uncond = model(x_in, t_in, y_null)
        return v_uncond + (1.0 + guidance) * (v_cond - v_uncond)

    for i in range(steps):
        t = torch.full((n,), i * dt, device=device)
        k1 = velocity(x, t)
        k2 = velocity(x + 0.5 * dt * k1, t + 0.5 * dt)
        x = x + dt * k2
        if return_traj:
            traj.append(x.clone())

    return (x, torch.stack(traj)) if return_traj else x


# ------------------------------------------------------------------- 可視化
def to_img(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1) / 2).cpu()


def make_figures(model, device, out_dir: Path) -> None:
    # (1) 0-9 を各行 8 枚ずつ生成
    labels = torch.arange(N_CLASSES).repeat_interleave(8)
    imgs = sample_ode(model, labels, steps=50, guidance=1.0, device=device)
    grid = make_grid(to_img(imgs), nrow=8, padding=2)
    plt.figure(figsize=(6, 7.5))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    plt.axis("off")
    plt.title("generated digits (rows = class 0-9, CFG w=1)")
    plt.tight_layout()
    plt.savefig(out_dir / "mnist_grid.png", dpi=140)
    plt.close()
    print(f"saved: {out_dir / 'mnist_grid.png'}")

    # (2) 1 個の数字が t とともにどう現れるかを追う
    label = torch.full((6,), 7)
    _, traj = sample_ode(model, label, steps=50, guidance=1.0, device=device, return_traj=True)
    show_t = [0, 5, 10, 20, 30, 40, 50]
    fig, axes = plt.subplots(6, len(show_t), figsize=(1.6 * len(show_t), 1.6 * 6))
    for row in range(6):
        for col, ti in enumerate(show_t):
            ax = axes[row, col]
            ax.imshow(to_img(traj[ti, row, 0]).numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"t={ti/50:.2f}", fontsize=9)
    fig.suptitle("denoising trajectory for class 7 (6 independent samples)")
    fig.tight_layout()
    fig.savefig(out_dir / "mnist_trajectory.png", dpi=140)
    plt.close(fig)
    print(f"saved: {out_dir / 'mnist_trajectory.png'}")

    # (3) ステップ数を減らしたときの崩れ方
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
    labels8 = torch.arange(8)
    for ax, n_steps in zip(axes, [1, 2, 4, 10, 50]):
        imgs = sample_ode(model, labels8, steps=n_steps, guidance=1.0, device=device)
        grid = make_grid(to_img(imgs), nrow=8, padding=2)
        ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
        ax.axis("off")
        ax.set_title(f"{n_steps} step(s)")
    fig.suptitle("fewer ODE steps -> lower quality (digits 0-7, one sample each)")
    fig.tight_layout()
    fig.savefig(out_dir / "mnist_steps.png", dpi=140)
    plt.close(fig)
    print(f"saved: {out_dir / 'mnist_steps.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description="MNIST Flow Matching sample")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--p-uncond", type=float, default=0.1)
    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", type=str, default="velocity_mnist.pt")
    p.add_argument("--retrain", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    images, labels = load_mnist_tensor(device)
    print(f"MNIST: {images.shape[0]} images, {images.shape[2]}x{images.shape[3]}")

    model = UNetVelocity(base_ch=args.base_ch).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M")

    ckpt = CKPT_DIR / args.ckpt
    if ckpt.exists() and not args.retrain:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"loaded checkpoint: {ckpt}")
    else:
        train(model, images, labels, args.steps, args.batch_size, args.lr, args.p_uncond, device)
        torch.save(model.state_dict(), ckpt)
        print(f"saved checkpoint: {ckpt}")

    make_figures(model, device, OUT_DIR)


if __name__ == "__main__":
    main()
