"""少数の特定サンプルを固定し、CFM と Reflow で軌跡がどう違うかを見る。

`compare_fields.py` はベクトル場そのものを比較したが、こちらは
「同じ初期ノイズ点をいくつか置いて、実際にどう動くか」を追う版。
点数を絞ることで、曲がる/直進する違いが 1 本 1 本の線としてはっきり見える。

出力:
  trace_points.png  静止画: 左 CFM (曲線) / 右 Reflow (直線)、点に番号付き
  trace_points.gif  同じ点が同時に動くアニメーション (2 パネル並走)

実行 (checkpoints/ に重みが必要。無ければ straightness.py を先に実行):
    python trace_points.py
    python trace_points.py --n-points 12 --seed 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

import flow_matching_2d as fm

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
OUT_DIR = ROOT / "out" / "2d_basic"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LIM = 3.0


def load(name: str, device):
    path = CKPT_DIR / name
    if not path.exists():
        raise SystemExit(f"{path} がありません。先に `python straightness.py` を実行してください。")
    model = fm.VelocityField().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def integrate(model, x0: torch.Tensor, steps: int) -> torch.Tensor:
    """midpoint 法。各時刻の状態 (steps+1, N, 2) を返す。"""
    dt = 1.0 / steps
    n = x0.shape[0]
    x = x0.clone()
    out = [x.clone()]
    for i in range(steps):
        t = torch.full((n,), i * dt, device=x.device)
        k1 = model(x, t)
        k2 = model(x + 0.5 * dt * k1, t + 0.5 * dt)
        x = x + dt * k2
        out.append(x.clone())
    return torch.stack(out)


# ------------------------------------------------------------------ 静止画
def make_static(base, reflowed, x0: torch.Tensor, device, out_path: Path, steps: int = 80) -> None:
    n = x0.shape[0]
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(n)]

    real = fm.sample_two_moons(2500)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
    fig.suptitle(f"same {n} noise points, traced through each model's ODE", fontsize=13)

    for ax, model, title in ((axes[0], base, "CFM (curved)"), (axes[1], reflowed, "after 1 reflow (straight)")):
        traj = integrate(model, x0.to(device), steps).cpu()  # (T+1, n, 2)
        ax.scatter(real[:, 0], real[:, 1], s=3, alpha=0.12, color="tab:gray", zorder=0)
        for k in range(n):
            ax.plot(traj[:, k, 0], traj[:, k, 1], color=colors[k], lw=1.4, alpha=0.85, zorder=2)
            ax.scatter(*traj[0, k], color=colors[k], marker="o", s=55, edgecolor="white", zorder=3)
            ax.scatter(*traj[-1, k], color=colors[k], marker="X", s=70, edgecolor="white", zorder=3)
            ax.annotate(str(k), traj[0, k], textcoords="offset points", xytext=(6, 6), fontsize=9, color=colors[k])
        ax.set_title(title)
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.set_aspect("equal")

    axes[0].text(
        -LIM + 0.1, LIM - 0.3,
        "○ = $t$=0 (noise)   ✕ = $t$=1 (generated)",
        fontsize=9, va="top",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved: {out_path}")


# ------------------------------------------------------------------- GIF
def make_gif(base, reflowed, x0: torch.Tensor, device, out_path: Path, steps: int = 80, fps: int = 20) -> None:
    n = x0.shape[0]
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(n)]
    real = fm.sample_two_moons(2500)

    traj_b = integrate(base, x0.to(device), steps).cpu()
    traj_r = integrate(reflowed, x0.to(device), steps).cpu()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
    fig.suptitle(f"same {n} noise points: CFM vs. after 1 reflow", fontsize=13)

    dots, tails = {}, {}
    for ax, title in zip(axes, ("CFM (curved)", "after 1 reflow (straight)")):
        ax.scatter(real[:, 0], real[:, 1], s=3, alpha=0.12, color="tab:gray", zorder=0)
        ax.set_title(title)
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.set_aspect("equal")

    for panel_key, ax, traj in (("b", axes[0], traj_b), ("r", axes[1], traj_r)):
        tails[panel_key] = [ax.plot([], [], color=colors[k], lw=1.4, alpha=0.8)[0] for k in range(n)]
        dots[panel_key] = ax.scatter(traj[0, :, 0], traj[0, :, 1], c=colors, s=55, edgecolor="white", zorder=3)

    time_text = axes[0].text(0.03, 0.95, "", transform=axes[0].transAxes, va="top", fontsize=11)

    def update(i: int):
        for key, traj in (("b", traj_b), ("r", traj_r)):
            dots[key].set_offsets(traj[i].numpy())
            for k, line in enumerate(tails[key]):
                line.set_data(traj[: i + 1, k, 0].numpy(), traj[: i + 1, k, 1].numpy())
        time_text.set_text(f"$t$ = {i / steps:.2f}")
        return [*tails["b"], *tails["r"], dots["b"], dots["r"], time_text]

    frames = list(range(steps + 1)) + [steps] * 20
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved: {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description="少数のサンプル点を追跡して CFM / Reflow を比較")
    p.add_argument("--n-points", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--no-gif", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    base = load("velocity_2d.pt", device)
    reflowed = load("velocity_2d_reflow1.pt", device)

    torch.manual_seed(args.seed)
    x0 = torch.randn(args.n_points, 2)
    print("初期ノイズ点:")
    for k, p_ in enumerate(x0):
        print(f"  {k}: ({p_[0]:.2f}, {p_[1]:.2f})")

    make_static(base, reflowed, x0, device, OUT_DIR / "trace_points.png", steps=args.steps)
    if not args.no_gif:
        make_gif(base, reflowed, x0, device, OUT_DIR / "trace_points.gif", steps=args.steps)


if __name__ == "__main__":
    main()
