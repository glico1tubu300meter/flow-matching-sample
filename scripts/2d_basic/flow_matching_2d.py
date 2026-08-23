"""Flow Matching (Conditional Flow Matching) の最小サンプル。

2次元トイデータ (two moons) を対象に、
  ノイズ分布 p0 = N(0, I)  →  データ分布 p1
への確率フロー ODE の速度場 v_theta(x, t) を学習する。

理論の要点
----------
直線補間パス (Rectified Flow / OT-CFM):
    x_t = (1 - t) * x0 + t * x1        x0 ~ N(0,I), x1 ~ data, t ~ U(0,1)
この経路に沿った条件付き速度は定数:
    u_t(x_t | x0, x1) = dx_t/dt = x1 - x0
CFM 損失は
    L = E_{t,x0,x1} || v_theta(x_t, t) - (x1 - x0) ||^2
であり、この最小化解は周辺確率経路を生成するベクトル場に一致する
(Lipman et al., 2023 / Liu et al., 2022)。

サンプリングは ODE
    dx/dt = v_theta(x, t),  x(0) ~ N(0, I)
を t: 0 → 1 で数値積分するだけ。拡散モデルと違いノイズ注入は不要。

実行:
    python flow_matching_2d.py
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


# ---------------------------------------------------------------- データ生成
def sample_two_moons(n: int, noise: float = 0.06, generator=None) -> torch.Tensor:
    """sklearn 非依存の two moons。おおよそ標準化されたスケールで返す。"""
    n_a = n // 2
    n_b = n - n_a
    t_a = torch.rand(n_a, generator=generator) * math.pi
    t_b = torch.rand(n_b, generator=generator) * math.pi

    moon_a = torch.stack([torch.cos(t_a), torch.sin(t_a)], dim=1)
    moon_b = torch.stack([1.0 - torch.cos(t_b), 0.5 - torch.sin(t_b)], dim=1)

    x = torch.cat([moon_a, moon_b], dim=0)
    x = x + noise * torch.randn(x.shape, generator=generator)
    # 平均 0・スケールを N(0,I) に近づけておくと学習が安定する
    x = (x - torch.tensor([0.5, 0.25])) / 0.75
    return x[torch.randperm(n, generator=generator)]


# ------------------------------------------------------------------ モデル
class TimeEmbedding(nn.Module):
    """t ∈ [0,1] を正弦波特徴に展開する (Transformer と同じ形式)。"""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dim は偶数にすること"
        self.dim = dim
        half = dim // 2
        # 周波数はバッファに持たせて device 移動に追随させる
        freqs = torch.exp(torch.linspace(0.0, math.log(1000.0), half))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) -> (B, dim)
        args = t[:, None] * self.freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class VelocityField(nn.Module):
    """速度場 v_theta(x, t) を表す MLP。"""

    def __init__(self, data_dim: int = 2, hidden: int = 256, time_dim: int = 64, depth: int = 4):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        layers: list[nn.Module] = [nn.Linear(data_dim + time_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, data_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, D), t: (B,)
        return self.net(torch.cat([x, self.time_embed(t)], dim=1))


# ------------------------------------------------------------------- 学習
def cfm_loss(model: VelocityField, x1: torch.Tensor) -> torch.Tensor:
    """Conditional Flow Matching 損失。x1 はデータバッチ (B, D)。"""
    b = x1.shape[0]
    x0 = torch.randn_like(x1)                       # ノイズ側の端点
    t = torch.rand(b, device=x1.device)             # t ~ U(0,1)

    xt = (1.0 - t[:, None]) * x0 + t[:, None] * x1  # 直線補間
    target = x1 - x0                                # 条件付き速度 (定数)

    return (model(xt, t) - target).pow(2).mean()


def train(model: VelocityField, steps: int, batch_size: int, lr: float, device: torch.device) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    history: list[float] = []

    model.train()
    for step in range(1, steps + 1):
        x1 = sample_two_moons(batch_size).to(device)
        loss = cfm_loss(model, x1)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        history.append(loss.item())
        if step % max(1, steps // 10) == 0 or step == 1:
            recent = sum(history[-100:]) / len(history[-100:])
            print(f"step {step:>6d}/{steps}  loss {loss.item():.4f}  (avg100 {recent:.4f})")
    return history


# --------------------------------------------------------------- サンプリング
@torch.no_grad()
def sample_ode(
    model: VelocityField,
    n: int,
    steps: int,
    device: torch.device,
    method: str = "midpoint",
    return_traj: bool = False,
):
    """dx/dt = v(x,t) を t: 0 -> 1 で積分する。

    method="euler"    : 1 次。step 数を増やせば十分。
    method="midpoint" : 2 次。同じ NFE でも精度が高い (評価回数は 2 倍/step)。
    """
    model.eval()
    x = torch.randn(n, 2, device=device)
    dt = 1.0 / steps
    traj = [x.clone()]

    for i in range(steps):
        t = torch.full((n,), i * dt, device=device)
        if method == "euler":
            x = x + dt * model(x, t)
        elif method == "midpoint":
            k1 = model(x, t)
            k2 = model(x + 0.5 * dt * k1, t + 0.5 * dt)
            x = x + dt * k2
        else:
            raise ValueError(f"unknown method: {method}")
        if return_traj:
            traj.append(x.clone())

    return (x, torch.stack(traj)) if return_traj else x


# ------------------------------------------------------------------- 可視化
def make_figure(model: VelocityField, history: list[float], device: torch.device, out_path: Path) -> None:
    real = sample_two_moons(3000)
    fake = sample_ode(model, 3000, steps=50, device=device).cpu()
    _, traj = sample_ode(model, 400, steps=50, device=device, return_traj=True)
    traj = traj.cpu()

    lim = 3.0
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))

    axes[0].plot(history, lw=0.6, color="tab:blue", alpha=0.5)
    if len(history) >= 50:
        kernel = torch.ones(50) / 50
        smooth = torch.nn.functional.conv1d(
            torch.tensor(history)[None, None, :], kernel[None, None, :]
        ).flatten()
        axes[0].plot(range(49, len(history)), smooth, color="tab:red", lw=1.5, label="moving avg (50)")
        axes[0].legend()
    axes[0].set_title("CFM loss")
    axes[0].set_xlabel("step")

    axes[1].scatter(real[:, 0], real[:, 1], s=3, alpha=0.5, color="tab:gray")
    axes[1].set_title("real data $p_1$")

    axes[2].scatter(fake[:, 0], fake[:, 1], s=3, alpha=0.5, color="tab:green")
    axes[2].set_title("generated (ODE, 50 steps)")

    # 軌跡: t=0 のガウス分布が t=1 のデータ分布へ流れていく様子
    for k in range(traj.shape[1]):
        axes[3].plot(traj[:, k, 0], traj[:, k, 1], lw=0.4, alpha=0.25, color="tab:blue")
    axes[3].scatter(traj[0, :, 0], traj[0, :, 1], s=4, color="tab:orange", label="$t=0$ (noise)")
    axes[3].scatter(traj[-1, :, 0], traj[-1, :, 1], s=4, color="tab:red", label="$t=1$ (data)")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].set_title("probability flow trajectories")

    for ax in axes[1:]:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"saved: {out_path}")


# --------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description="2D Flow Matching sample")
    p.add_argument("--steps", type=int, default=5000, help="学習ステップ数")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="flow_matching_2d.png")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = VelocityField().to(device)
    history = train(model, args.steps, args.batch_size, args.lr, device)

    # Euler と midpoint の比較 (少ない step で差が出る)
    for method in ("euler", "midpoint"):
        for n_steps in (5, 20, 100):
            x = sample_ode(model, 2000, steps=n_steps, device=device, method=method)
            print(f"{method:>8s} steps={n_steps:>3d}  mean={x.mean(0).tolist()}  std={x.std(0).tolist()}")

    out_dir = Path(__file__).resolve().parents[2] / "out" / "2d_basic"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_figure(model, history, device, out_dir / args.out)


if __name__ == "__main__":
    main()
