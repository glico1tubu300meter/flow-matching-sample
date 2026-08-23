"""CIFAR-10 版の生成過程 GIF。作り方は `scripts/mnist/make_gif_mnist.py` と全く同じ:

  1. ODE (midpoint 法) を steps 分割し、各時刻の状態を全部保持しておく
  2. matplotlib の FuncAnimation で 1 フレーム = 1 ステップとして描画関数を書く
  3. PillowWriter で .gif に書き出す

違いは画像が 1x28x28 (グレースケール) ではなく 3x32x32 (カラー) という点だけ。

出力:
  cifar10_denoising.gif   10 クラスを同時生成しながら t: 0->1 で絵が浮かび上がる

実行 (checkpoints/velocity_cifar10.pt が必要。無ければ flow_matching_cifar10.py を先に実行):
    python make_gif_cifar10.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from flow_matching_cifar10 import CLASS_NAMES, N_CLASSES, NULL_LABEL, ResBlock, TimeEmbedding, UNetVelocity, to_img

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
OUT_DIR = ROOT / "out" / "cifar10"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 15
HOLD_FRAMES = 20


class LegacyUNetVelocity(nn.Module):
    """v1 (attention導入前) のチェックポイントを読むための互換クラス。
    唯一の違いはボトルネックが `mid1`+`mid_attn`+`mid2` ではなく単一の `mid`
    ResBlock だった点。アーキテクチャの都合上 UNetVelocity とは state_dict の
    キーが噛み合わないため、v1 チェックポイント読み込み専用に残してある。
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 96, emb_dim: int = 128):
        super().__init__()
        self.time_embed = TimeEmbedding(emb_dim)
        self.label_embed = nn.Embedding(N_CLASSES + 1, emb_dim)

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.down1 = ResBlock(base_ch, base_ch, emb_dim)
        self.pool1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)
        self.down2 = ResBlock(base_ch, base_ch * 2, emb_dim)
        self.pool2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)

        self.mid = ResBlock(base_ch * 2, base_ch * 2, emb_dim)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)
        self.dec2 = ResBlock(base_ch * 4, base_ch, emb_dim)
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1)
        self.dec1 = ResBlock(base_ch * 2, base_ch, emb_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

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


def load_model(device, ckpt_name: str, base_ch: int, use_attn: bool, legacy: bool = False):
    ckpt = CKPT_DIR / ckpt_name
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} がありません。先に `python flow_matching_cifar10.py` を実行してください。")
    if legacy:
        model = LegacyUNetVelocity(base_ch=base_ch).to(device)
    else:
        model = UNetVelocity(base_ch=base_ch, use_attn=use_attn).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def integrate_with_guidance(model, labels: torch.Tensor, steps: int, guidance: float, device) -> torch.Tensor:
    """midpoint 法。各時刻の画像 (steps+1, N, 3, 32, 32) を返す。"""
    n = labels.shape[0]
    x = torch.randn(n, 3, 32, 32, device=device)
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


def make_gif(model, device, out_path: Path, steps: int, guidance: float, samples_per_class: int, dpi: int = 90) -> None:
    # 各行に全クラスが1個ずつ並ぶように tile する (repeat_interleave だと
    # 上段に前半のクラスだけ、下段に後半のクラスだけが並んでしまい、
    # 列見出し CLASS_NAMES[c]ともズレるバグがあった)
    labels = torch.arange(N_CLASSES).repeat(samples_per_class)
    traj = integrate_with_guidance(model, labels, steps, guidance, device).cpu()  # (T+1, N, 3, 32, 32)

    n = labels.shape[0]
    ncols = N_CLASSES
    nrows = samples_per_class

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.15, nrows * 1.15 + 0.7))
    if nrows == 1:
        axes = axes[None, :]

    ims = []
    for k in range(n):
        r, c = divmod(k, ncols)
        im = axes[r, c].imshow(to_img(traj[0, k]).permute(1, 2, 0).numpy())
        axes[r, c].axis("off")
        ims.append(im)
        if r == 0:
            axes[r, c].set_title(CLASS_NAMES[c], fontsize=7, rotation=30, ha="left")

    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=12)
    fig.suptitle(f"Flow Matching on CIFAR-10: noise -> images  (CFG w={guidance})", fontsize=13)

    def update(i: int):
        idx = min(i, steps)
        for k, im in enumerate(ims):
            im.set_data(to_img(traj[idx, k]).permute(1, 2, 0).numpy())
        time_text.set_text(f"t = {idx / steps:.2f}")
        return [*ims, time_text]

    frames = list(range(steps + 1)) + [steps] * HOLD_FRAMES
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // FPS, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=dpi)
    plt.close(fig)
    print(f"saved: {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description="CIFAR-10 の生成過程を GIF 化する")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--samples-per-class", type=int, default=2)
    p.add_argument("--out", type=str, default="cifar10_denoising.gif")
    p.add_argument("--ckpt", type=str, default="velocity_cifar10.pt")
    p.add_argument("--base-ch", type=int, default=96)
    p.add_argument("--no-attn", action="store_true", help="v1 (attention無し) のチェックポイントを読むときに指定")
    p.add_argument("--legacy", action="store_true", help="v1 (mid1/mid2導入前) のチェックポイント用互換ローダーを使う")
    p.add_argument("--dpi", type=int, default=90)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = load_model(device, args.ckpt, args.base_ch, use_attn=not args.no_attn, legacy=args.legacy)
    make_gif(model, device, OUT_DIR / args.out, args.steps, args.guidance, args.samples_per_class, args.dpi)


if __name__ == "__main__":
    main()
