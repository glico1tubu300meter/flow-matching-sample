"""条件付き Flow Matching + Classifier-Free Guidance (CFG) のサンプル。

データは 8 個のガウス分布を円周上に並べたもので、それぞれのモードに
クラスラベル y ∈ {0..7} が付いている。速度場を v_theta(x, t, y) として学習し、
生成時に

    v_cfg = v(x, t, null) + w * ( v(x, t, y) - v(x, t, null) )

とすることでクラス条件を強調する (拡散モデルの CFG と全く同じ形)。
学習中は確率 p_uncond でラベルを null トークンに置き換える。

実行:
    python flow_matching_conditional.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N_CLASSES = 8
NULL_LABEL = N_CLASSES  # CFG 用の無条件トークン


def sample_8gaussians(n: int, std: float = 0.12, radius: float = 2.0):
    """(x, y) を返す。y はどのモードから引いたかのクラスラベル。"""
    labels = torch.randint(0, N_CLASSES, (n,))
    angles = labels.float() * (2 * math.pi / N_CLASSES)
    centers = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
    return centers + std * torch.randn(n, 2), labels


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        freqs = torch.exp(torch.linspace(0.0, math.log(1000.0), dim // 2))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t[:, None] * self.freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ConditionalVelocityField(nn.Module):
    def __init__(self, data_dim: int = 2, hidden: int = 256, emb_dim: int = 64, depth: int = 4):
        super().__init__()
        self.time_embed = TimeEmbedding(emb_dim)
        # null トークン込みで N_CLASSES + 1 個
        self.label_embed = nn.Embedding(N_CLASSES + 1, emb_dim)

        layers: list[nn.Module] = [nn.Linear(data_dim + 2 * emb_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, data_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, self.time_embed(t), self.label_embed(y)], dim=1)
        return self.net(h)


def cfm_loss(model: ConditionalVelocityField, x1: torch.Tensor, y: torch.Tensor, p_uncond: float) -> torch.Tensor:
    b = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = torch.rand(b, device=x1.device)

    xt = (1.0 - t[:, None]) * x0 + t[:, None] * x1
    target = x1 - x0

    # ラベルドロップアウト: 無条件モデルも同じ重みで同時に学習する
    drop = torch.rand(b, device=x1.device) < p_uncond
    y = torch.where(drop, torch.full_like(y, NULL_LABEL), y)

    return (model(xt, t, y) - target).pow(2).mean()


def train(model, steps: int, batch_size: int, lr: float, p_uncond: float, device) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    history: list[float] = []

    model.train()
    for step in range(1, steps + 1):
        x1, y = sample_8gaussians(batch_size)
        loss = cfm_loss(model, x1.to(device), y.to(device), p_uncond)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        history.append(loss.item())
        if step % max(1, steps // 10) == 0 or step == 1:
            print(f"step {step:>6d}/{steps}  loss {loss.item():.4f}")
    return history


@torch.no_grad()
def sample_ode(model, labels: torch.Tensor, steps: int, guidance: float, device) -> torch.Tensor:
    """CFG 付き midpoint 積分。guidance=0 で条件そのまま、>0 で条件を強調。"""
    model.eval()
    n = labels.shape[0]
    x = torch.randn(n, 2, device=device)
    y = labels.to(device)
    y_null = torch.full_like(y, NULL_LABEL)
    dt = 1.0 / steps

    def velocity(x_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
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
    return x


def make_figure(model, device, out_path: Path) -> None:
    real, real_y = sample_8gaussians(4000)
    labels = torch.arange(N_CLASSES).repeat_interleave(500)  # 各クラス 500 個
    guidances = [0.0, 1.0, 3.0]

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 1 + len(guidances), figsize=(5 * (1 + len(guidances)), 4.8))

    axes[0].scatter(real[:, 0], real[:, 1], s=3, alpha=0.6, c=[cmap(int(i)) for i in real_y])
    axes[0].set_title("real data (8 gaussians)")

    for ax, w in zip(axes[1:], guidances):
        x = sample_ode(model, labels, steps=50, guidance=w, device=device).cpu()
        ax.scatter(x[:, 0], x[:, 1], s=3, alpha=0.6, c=[cmap(int(i)) for i in labels])
        ax.set_title(f"generated (CFG w={w})")

    for ax in axes:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Conditional Flow Matching + CFG sample")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--p-uncond", type=float, default=0.1, help="ラベルドロップアウト率")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="flow_matching_conditional.png")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = ConditionalVelocityField().to(device)
    train(model, args.steps, args.batch_size, args.lr, args.p_uncond, device)

    # 指定クラスだけを生成できることを確認
    for cls in (0, 3):
        x = sample_ode(model, torch.full((1000,), cls), steps=50, guidance=1.0, device=device).cpu()
        angle = cls * (2 * math.pi / N_CLASSES)
        center = torch.tensor([2.0 * math.cos(angle), 2.0 * math.sin(angle)])
        err = (x - center).norm(dim=1).mean().item()
        print(f"class {cls}: 中心からの平均距離 = {err:.3f} (データ std=0.12 相当なら成功)")

    out_dir = Path(__file__).resolve().parents[2] / "out" / "conditional"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_figure(model, device, out_dir / args.out)


if __name__ == "__main__":
    main()
