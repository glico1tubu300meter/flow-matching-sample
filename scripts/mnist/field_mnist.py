"""MNIST での「速度場」を可視化する。

2D トイデータでは空間が (x,y) の 2 次元だったので、各点に矢印 v_theta(x,t) in R^2
を描けた (compare_fields.py の quiver/streamplot)。MNIST は空間が 784 次元
(28x28 ピクセル) なので、1 点における速度ベクトルも 784 次元になり矢印には
できない。

しかし v_theta(x_t, t) は x_t と全く同じ形 (1x28x28) を持つ。つまり
「各ピクセルを今後どちらにどれだけ動かすか」をそのまま画像として
(発散カラーマップで正負を色分けして) 表示できる。これが MNIST 版の
「速度場」の可視化になる。

出力:
  field_mnist.png   1 サンプルについて、複数の t で x_t / v_theta(x_t,t) / |v| を並べる
  field_mnist.gif   同じ内容を t について連続アニメーション

実行 (checkpoints/velocity_mnist.pt が必要):
    python field_mnist.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from flow_matching_mnist import N_CLASSES, NULL_LABEL, UNetVelocity, to_img

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
OUT_DIR = ROOT / "out" / "mnist"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(device):
    ckpt = CKPT_DIR / "velocity_mnist.pt"
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} がありません。先に `python flow_matching_mnist.py` を実行してください。")
    model = UNetVelocity().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def integrate(model, label: int, steps: int, guidance: float, device, seed: int):
    """1 サンプルを midpoint 法で積分し、各時刻の (x_t, v_t) を両方保存する。"""
    torch.manual_seed(seed)
    x = torch.randn(1, 1, 28, 28, device=device)
    y = torch.full((1,), label, device=device)
    y_null = torch.full_like(y, NULL_LABEL)
    dt = 1.0 / steps

    def velocity(x_in, t_in):
        v_cond = model(x_in, t_in, y)
        if guidance == 0.0:
            return v_cond
        v_uncond = model(x_in, t_in, y_null)
        return v_uncond + (1.0 + guidance) * (v_cond - v_uncond)

    xs, vs = [x.clone()], []
    for i in range(steps):
        t = torch.full((1,), i * dt, device=device)
        v_now = velocity(x, t)          # この時刻での「今の」速度 (可視化用)
        vs.append(v_now.clone())
        k2 = velocity(x + 0.5 * dt * v_now, t + 0.5 * dt)
        x = x + dt * k2
        xs.append(x.clone())
    vs.append(vs[-1])  # 長さを xs に揃える (最後は直前の値で埋める)
    return torch.cat(xs).cpu(), torch.cat(vs).cpu()


def make_static(xs: torch.Tensor, vs: torch.Tensor, steps: int, out_path: Path) -> None:
    show_t = [0, 5, 10, 15, 20, 30, 50]
    show_t = [t for t in show_t if t <= steps]

    vmax = vs[show_t].abs().max().item()
    fig, axes = plt.subplots(2, len(show_t), figsize=(1.7 * len(show_t), 3.6))

    for col, ti in enumerate(show_t):
        ax_x, ax_v = axes[0, col], axes[1, col]
        ax_x.imshow(to_img(xs[ti, 0]).numpy(), cmap="gray", vmin=0, vmax=1)
        ax_x.axis("off")
        ax_x.set_title(f"$x_t$  $t$={ti/steps:.2f}", fontsize=9)

        im = ax_v.imshow(vs[ti, 0].numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax_v.axis("off")
        ax_v.set_title("$v_\\theta(x_t,t)$", fontsize=9)

    fig.suptitle(
        "top: noisy sample $x_t$   /   bottom: predicted velocity $v_\\theta(x_t,t)$ (same shape as image, red=+/blue=-)",
        fontsize=11,
    )
    fig.colorbar(im, ax=axes[1, :].tolist(), fraction=0.02, pad=0.01, label="velocity")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved: {out_path}")


def make_gif(xs: torch.Tensor, vs: torch.Tensor, steps: int, out_path: Path, fps: int = 15) -> None:
    vmax = vs.abs().max().item()
    fig, (ax_x, ax_v) = plt.subplots(1, 2, figsize=(6.4, 3.6))

    im_x = ax_x.imshow(to_img(xs[0, 0]).numpy(), cmap="gray", vmin=0, vmax=1)
    ax_x.axis("off")
    ax_x.set_title("$x_t$")

    im_v = ax_v.imshow(vs[0, 0].numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_v.axis("off")
    ax_v.set_title("$v_\\theta(x_t,t)$")

    time_text = fig.suptitle("t = 0.00")

    def update(i: int):
        idx = min(i, steps)
        im_x.set_data(to_img(xs[idx, 0]).numpy())
        im_v.set_data(vs[idx, 0].numpy())
        time_text.set_text(f"t = {idx/steps:.2f}")
        return [im_x, im_v, time_text]

    frames = list(range(steps + 1)) + [steps] * 15
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    print(f"saved: {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description="MNIST 版の速度場可視化")
    p.add_argument("--label", type=int, default=3, choices=range(N_CLASSES))
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-gif", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = load_model(device)
    xs, vs = integrate(model, args.label, args.steps, args.guidance, device, args.seed)

    make_static(xs, vs, args.steps, OUT_DIR / "field_mnist.png")
    if not args.no_gif:
        make_gif(xs, vs, args.steps, OUT_DIR / "field_mnist.gif")


if __name__ == "__main__":
    main()
