"""学習済み速度場による生成過程を GIF アニメーションにまとめる (2D two moons 版)。

出力: out/2d_basic/flow_matching_2d.gif  (ノイズ→two moons の流れ + 速度場ベクトル)

チェックポイントは out/2d_basic/checkpoints/ に保存し、2 回目以降は再学習をスキップする
(`--retrain` で強制的に学習し直す)。

条件付き版 (CFG) の GIF は `scripts/conditional/make_gif_conditional.py` を参照。

実行:
    python make_gif_2d.py
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
CKPT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 20
HOLD_FRAMES = 25  # 最終状態を少し止めてループを見やすくする


def load_or_train(name: str, build, train_fn, retrain: bool, device):
    ckpt = CKPT_DIR / f"{name}.pt"
    model = build().to(device)
    if ckpt.exists() and not retrain:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"loaded checkpoint: {ckpt}")
    else:
        train_fn(model)
        torch.save(model.state_dict(), ckpt)
        print(f"saved checkpoint: {ckpt}")
    return model


@torch.no_grad()
def trajectory(velocity, x: torch.Tensor, steps: int) -> torch.Tensor:
    """midpoint 法で積分し、各時刻の状態 (steps+1, N, 2) を返す。"""
    dt = 1.0 / steps
    n = x.shape[0]
    out = [x.clone()]
    for i in range(steps):
        t = torch.full((n,), i * dt, device=x.device)
        k1 = velocity(x, t)
        k2 = velocity(x + 0.5 * dt * k1, t + 0.5 * dt)
        x = x + dt * k2
        out.append(x.clone())
    return torch.stack(out)


def frame_indices(n_steps: int) -> list[int]:
    return list(range(n_steps + 1)) + [n_steps] * HOLD_FRAMES


def make_gif_2d(model, device, out_path: Path, n_points: int = 1200, n_steps: int = 80) -> None:
    x0 = torch.randn(n_points, 2, device=device)
    traj = trajectory(lambda x, t: model(x, t), x0, n_steps).cpu()

    g = torch.linspace(-3, 3, 21)
    gx, gy = torch.meshgrid(g, g, indexing="xy")
    grid = torch.stack([gx.flatten(), gy.flatten()], dim=1).to(device)

    real = fm.sample_two_moons(3000)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 5.6))
    fig.suptitle("Flow Matching: $\\mathcal{N}(0,I)\\;\\rightarrow\\;$ two moons", fontsize=13)

    ax_l.scatter(real[:, 0], real[:, 1], s=3, alpha=0.12, color="tab:gray")
    tail_lines = [ax_l.plot([], [], lw=0.5, alpha=0.25, color="tab:blue")[0] for _ in range(150)]
    dots = ax_l.scatter([], [], s=6, color="tab:red", zorder=3)

    quiver = ax_r.quiver(
        grid[:, 0].cpu(), grid[:, 1].cpu(),
        torch.zeros(grid.shape[0]), torch.zeros(grid.shape[0]),
        color="tab:blue", alpha=0.7, scale=60,
    )
    dots_r = ax_r.scatter([], [], s=5, color="tab:red", alpha=0.7, zorder=3)

    for ax, title in ((ax_l, "samples $x_t$"), (ax_r, "velocity field $v_\\theta(x,t)$")):
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.set_title(title)

    time_text = ax_l.text(0.03, 0.95, "", transform=ax_l.transAxes, va="top", fontsize=11)

    @torch.no_grad()
    def update(i: int):
        pts = traj[i]
        dots.set_offsets(pts.numpy())
        dots_r.set_offsets(pts[:400].numpy())

        start = max(0, i - 15)
        for k, line in enumerate(tail_lines):
            line.set_data(traj[start : i + 1, k, 0].numpy(), traj[start : i + 1, k, 1].numpy())

        t_val = min(i, n_steps) / n_steps
        t_batch = torch.full((grid.shape[0],), t_val, device=device)
        v = model(grid, t_batch).cpu()
        quiver.set_UVC(v[:, 0], v[:, 1])

        time_text.set_text(f"$t$ = {t_val:.2f}")
        return [dots, dots_r, quiver, time_text, *tail_lines]

    anim = FuncAnimation(fig, update, frames=frame_indices(n_steps), interval=1000 // FPS, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"saved: {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description="2D two moons の生成過程を GIF 化する")
    p.add_argument("--train-steps", type=int, default=5000)
    p.add_argument("--ode-steps", type=int, default=80, help="GIF のフレーム数 (= ODE ステップ数)")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = load_or_train(
        "velocity_2d",
        fm.VelocityField,
        lambda m: fm.train(m, args.train_steps, 512, 2e-3, device),
        args.retrain, device,
    )
    make_gif_2d(model, device, OUT_DIR / "flow_matching_2d.gif", n_steps=args.ode_steps)


if __name__ == "__main__":
    main()
