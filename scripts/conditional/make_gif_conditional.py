"""学習済み条件付き速度場による生成過程を GIF アニメーションにまとめる (CFG 比較版)。

出力: out/conditional/flow_matching_conditional.gif
      (8 gaussians をクラス条件生成し、CFG w=0 と w=3 を並べて比較)

チェックポイントは out/conditional/checkpoints/ に保存し、2 回目以降は再学習をスキップする
(`--retrain` で強制的に学習し直す)。

2D 基本版の GIF は `scripts/2d_basic/make_gif_2d.py` を参照。

実行:
    python make_gif_conditional.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

import flow_matching_conditional as fmc

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
OUT_DIR = ROOT / "out" / "conditional"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 20
HOLD_FRAMES = 25


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


def make_gif_conditional(model, device, out_path: Path, per_class: int = 150, n_steps: int = 80) -> None:
    labels = torch.arange(fmc.N_CLASSES).repeat_interleave(per_class).to(device)
    y_null = torch.full_like(labels, fmc.NULL_LABEL)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(int(i)) for i in labels.cpu()]

    def make_velocity(w: float):
        def velocity(x, t):
            v_cond = model(x, t, labels)
            if w == 0.0:
                return v_cond
            v_uncond = model(x, t, y_null)
            return v_uncond + (1.0 + w) * (v_cond - v_uncond)
        return velocity

    guidances = [0.0, 3.0]
    x0 = torch.randn(labels.shape[0], 2, device=device)
    trajs = [trajectory(make_velocity(w), x0.clone(), n_steps).cpu() for w in guidances]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    fig.suptitle("Conditional Flow Matching: effect of classifier-free guidance", fontsize=13)

    scatters = []
    for ax, w, tr in zip(axes, guidances, trajs):
        scatters.append(ax.scatter(tr[0][:, 0], tr[0][:, 1], s=8, c=colors, alpha=0.8))
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
        ax.set_aspect("equal")
        ax.set_title(f"CFG $w$ = {w}" + ("  (conditional only)" if w == 0 else "  (guidance boosted)"))

    time_text = axes[0].text(0.03, 0.95, "", transform=axes[0].transAxes, va="top", fontsize=11)

    def update(i: int):
        for sc, tr in zip(scatters, trajs):
            sc.set_offsets(tr[i].numpy())
        time_text.set_text(f"$t$ = {min(i, n_steps) / n_steps:.2f}")
        return [*scatters, time_text]

    anim = FuncAnimation(fig, update, frames=frame_indices(n_steps), interval=1000 // FPS, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"saved: {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description="条件付き Flow Matching (CFG) の生成過程を GIF 化する")
    p.add_argument("--train-steps", type=int, default=5000)
    p.add_argument("--ode-steps", type=int, default=80)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model_c = load_or_train(
        "velocity_conditional",
        fmc.ConditionalVelocityField,
        lambda m: fmc.train(m, args.train_steps, 512, 2e-3, 0.1, device),
        args.retrain, device,
    )
    make_gif_conditional(model_c, device, OUT_DIR / "flow_matching_conditional.gif", n_steps=args.ode_steps)


if __name__ == "__main__":
    main()
