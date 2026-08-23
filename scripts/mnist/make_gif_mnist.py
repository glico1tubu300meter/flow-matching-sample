"""MNIST 版の生成過程 GIF。作り方は `make_gif.py` と全く同じ考え方:

  1. ODE (midpoint 法) を steps 分割し、各時刻の状態を全部保持しておく
  2. matplotlib の FuncAnimation で 1 フレーム = 1 ステップとして描画関数を書く
  3. PillowWriter で .gif に書き出す

違いは「描画するものが散布図ではなく画像 (imshow)」という点だけ。

出力:
  mnist_denoising.gif   0-9 を同時生成しながら t: 0->1 で数字が浮かび上がる

実行 (checkpoints/velocity_mnist.pt が必要。無ければ flow_matching_mnist.py を先に実行):
    python make_gif_mnist.py
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
FPS = 15
HOLD_FRAMES = 20  # 最後の数字で止めてループを見やすくする


def load_model(device):
    ckpt = CKPT_DIR / "velocity_mnist.pt"
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} がありません。先に `python flow_matching_mnist.py` を実行してください。")
    model = UNetVelocity().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def integrate_with_guidance(model, labels: torch.Tensor, steps: int, guidance: float, device) -> torch.Tensor:
    """midpoint 法。各時刻の画像 (steps+1, N, 1, 28, 28) を返す。"""
    n = labels.shape[0]
    x = torch.randn(n, 1, 28, 28, device=device)
    y = labels.to(device)
    y_null = torch.full_like(y, NULL_LABEL)
    dt = 1.0 / steps

    def velocity(x_in, t_in):
        v_cond = model(x_in, t_in, y)
        if guidance == 0.0:
            return v_cond
        v_uncond = model(x_in, t_in, y_null)
        return v_uncond + (1.0 + guidance) * (v_cond - v_uncond)

    out = [x.clone()]
    for i in range(steps):
        t = torch.full((n,), i * dt, device=device)
        k1 = velocity(x, t)
        k2 = velocity(x + 0.5 * dt * k1, t + 0.5 * dt)
        x = x + dt * k2
        out.append(x.clone())
    return torch.stack(out)


def make_gif(model, device, out_path: Path, steps: int, guidance: float, samples_per_class: int, dpi: int = 100) -> None:
    labels = torch.arange(N_CLASSES).repeat_interleave(samples_per_class)
    traj = integrate_with_guidance(model, labels, steps, guidance, device).cpu()  # (T+1, N, 1, 28, 28)

    n = labels.shape[0]
    ncols = N_CLASSES
    nrows = samples_per_class

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.1, nrows * 1.1 + 0.6))
    if nrows == 1:
        axes = axes[None, :]

    ims = []
    for k in range(n):
        r, c = divmod(k, ncols)
        im = axes[r, c].imshow(to_img(traj[0, k, 0]).numpy(), cmap="gray", vmin=0, vmax=1)
        axes[r, c].axis("off")
        ims.append(im)
        if r == 0:
            axes[r, c].set_title(str(c), fontsize=10)

    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=12)
    fig.suptitle(f"Flow Matching on MNIST: noise -> digits  (CFG w={guidance})", fontsize=13)

    def update(i: int):
        idx = min(i, steps)
        for k, im in enumerate(ims):
            im.set_data(to_img(traj[idx, k, 0]).numpy())
        time_text.set_text(f"t = {idx / steps:.2f}")
        return [*ims, time_text]

    frames = list(range(steps + 1)) + [steps] * HOLD_FRAMES
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // FPS, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=dpi)
    plt.close(fig)
    print(f"saved: {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description="MNIST の生成過程を GIF 化する")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--samples-per-class", type=int, default=3)
    p.add_argument("--dpi", type=int, default=100, help="小さくするとファイルサイズが減る")
    p.add_argument("--out", type=str, default="mnist_denoising.gif")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = load_model(device)
    make_gif(model, device, OUT_DIR / args.out, args.steps, args.guidance, args.samples_per_class, args.dpi)


if __name__ == "__main__":
    main()
