"""実データの数字どうしを「ノイズ空間での補間」でモーフィングする。

Flow Matching の ODE は拡散モデルの SDE と違って決定論的かつ可逆。
つまり実データの画像 x1 から逆向きに ODE を解けば、対応するノイズ x0 が
(近似的に) 求まる (DDIM inversion と同じ発想)。

    forward:  x0 (noise) --ODE(dx/dt=v)--> x1 (data)
    backward: x1 (data)  --ODE(dx/dt=v, 時間を逆に)--> x0 (noise)

手順:
  1. 数字A (例: "3") と数字B (例: "8") の実画像をそれぞれ、自分自身のクラス
     ラベルで逆向きに ODE を解いてノイズ x0_a, x0_b を求める。
  2. x0_a と x0_b を球面線形補間 (slerp) する。
     高次元ガウスノイズは単純な線形補間 (lerp) だと中間点だけ極端に
     ノルムが小さくなり (集中現象)、崩れた画像になりやすいため slerp を使う。
  3. 補間したノイズを、2 クラスの条件付き速度場を同じ比率でブレンドした
     速度場で順方向に解く。alpha=0 で数字A、alpha=1 で数字B、間は
     ノイズ・クラス条件の両方が滑らかに混ざった「モーフィング」になる。

出力:
  morph_mnist.png   複数ペア x 複数 alpha のグリッド (静止画)
  morph_mnist.gif    1 ペアを alpha 連続で滑らかにモーフィングする GIF

実行 (checkpoints/velocity_mnist.pt が必要):
    python morph_mnist.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from flow_matching_mnist import N_CLASSES, NULL_LABEL, UNetVelocity, load_mnist_tensor, to_img

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
OUT_DIR = ROOT / "out" / "mnist"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 20
HOLD_FRAMES = 15


def load_model(device):
    ckpt = CKPT_DIR / "velocity_mnist.pt"
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} がありません。先に `python flow_matching_mnist.py` を実行してください。")
    model = UNetVelocity().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def cfg_velocity(model, x, t, y, guidance, device):
    v_cond = model(x, t, y)
    if guidance == 0.0:
        return v_cond
    y_null = torch.full_like(y, NULL_LABEL)
    v_uncond = model(x, t, y_null)
    return v_uncond + (1.0 + guidance) * (v_cond - v_uncond)


@torch.no_grad()
def integrate_forward(model, x0: torch.Tensor, labels: torch.Tensor, steps: int, guidance: float, device) -> torch.Tensor:
    """midpoint 法で t: 0 -> 1 (ノイズ -> データ)。"""
    x = x0.to(device)
    y = labels.to(device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), i * dt, device=device)
        k1 = cfg_velocity(model, x, t, y, guidance, device)
        k2 = cfg_velocity(model, x + 0.5 * dt * k1, t + 0.5 * dt, y, guidance, device)
        x = x + dt * k2
    return x.cpu()


@torch.no_grad()
def integrate_backward(model, x1: torch.Tensor, labels: torch.Tensor, steps: int, guidance: float, device) -> torch.Tensor:
    """midpoint 法で t: 1 -> 0 (データ -> ノイズ)。実データの近似的な逆変換 (inversion)。"""
    x = x1.to(device)
    y = labels.to(device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), 1.0 - i * dt, device=device)
        k1 = cfg_velocity(model, x, t, y, guidance, device)
        k2 = cfg_velocity(model, x - 0.5 * dt * k1, t - 0.5 * dt, y, guidance, device)
        x = x - dt * k2
    return x.cpu()


@torch.no_grad()
def decode_blend(model, x0: torch.Tensor, label_a: int, label_b: int, alpha: float,
                  steps: int, guidance: float, device) -> torch.Tensor:
    """2 クラスの条件付き速度場を alpha でブレンドしながら順方向に解く。"""
    x = x0.to(device)
    n = x.shape[0]
    y_a = torch.full((n,), label_a, device=device)
    y_b = torch.full((n,), label_b, device=device)
    dt = 1.0 / steps

    def velocity(x_in, t_in):
        va = cfg_velocity(model, x_in, t_in, y_a, guidance, device)
        vb = cfg_velocity(model, x_in, t_in, y_b, guidance, device)
        return (1.0 - alpha) * va + alpha * vb

    for i in range(steps):
        t = torch.full((n,), i * dt, device=device)
        k1 = velocity(x, t)
        k2 = velocity(x + 0.5 * dt * k1, t + 0.5 * dt)
        x = x + dt * k2
    return x.cpu()


def slerp(a: torch.Tensor, b: torch.Tensor, alpha: float, eps: float = 1e-6) -> torch.Tensor:
    """球面線形補間。高次元ガウスノイズを補間するときの定番テクニック
    (単純な線形補間だと中間点のノルムが縮んで画像が崩れやすい)。
    """
    a_flat, b_flat = a.reshape(-1), b.reshape(-1)
    a_n, b_n = a_flat / a_flat.norm(), b_flat / b_flat.norm()
    omega = torch.acos((a_n * b_n).sum().clamp(-1 + eps, 1 - eps))
    if omega.abs() < eps:
        out = (1 - alpha) * a_flat + alpha * b_flat
    else:
        out = (torch.sin((1 - alpha) * omega) * a_flat + torch.sin(alpha * omega) * b_flat) / torch.sin(omega)
    return out.reshape(a.shape)


def pick_example(images: torch.Tensor, labels: torch.Tensor, cls: int, seed: int) -> torch.Tensor:
    """指定クラスの実画像を1枚、乱数シードで再現可能に選ぶ。"""
    idx = (labels == cls).nonzero(as_tuple=True)[0]
    g = torch.Generator().manual_seed(seed)
    pick = idx[torch.randint(0, idx.shape[0], (1,), generator=g)]
    return images[pick]


def morph_pair(model, images, labels, cls_a: int, cls_b: int, device, steps: int = 60,
               guidance: float = 0.0, seed: int = 0):
    """1ペア分のモーフィングを計算する。戻り値: (real_a, real_b, x0_a, x0_b)。"""
    real_a = pick_example(images, labels, cls_a, seed).to(device)
    real_b = pick_example(images, labels, cls_b, seed + 1).to(device)

    x0_a = integrate_backward(model, real_a, torch.full((1,), cls_a), steps, guidance, device)
    x0_b = integrate_backward(model, real_b, torch.full((1,), cls_b), steps, guidance, device)
    return real_a.cpu(), real_b.cpu(), x0_a, x0_b


def make_static_grid(model, images, labels, device, out_path: Path, pairs=((3, 8), (4, 9), (1, 7)),
                      n_alpha: int = 9, steps: int = 60, guidance: float = 0.0, seed: int = 0):
    alphas = torch.linspace(0, 1, n_alpha)
    n_rows = len(pairs)

    fig, axes = plt.subplots(n_rows, n_alpha, figsize=(1.6 * n_alpha, 1.6 * n_rows + 0.6))
    if n_rows == 1:
        axes = axes[None, :]

    recon_errors = []
    for row, (cls_a, cls_b) in enumerate(pairs):
        real_a, real_b, x0_a, x0_b = morph_pair(model, images, labels, cls_a, cls_b, device,
                                                 steps, guidance, seed)
        for col, alpha in enumerate(alphas.tolist()):
            x0_mid = slerp(x0_a[0], x0_b[0], alpha)[None]
            img = decode_blend(model, x0_mid, cls_a, cls_b, alpha, steps, guidance, device)
            axes[row, col].imshow(to_img(img[0, 0]).numpy(), cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(f"a={alpha:.2f}", fontsize=9)
            if col == 0:
                axes[row, col].set_ylabel(f"{cls_a} -> {cls_b}", fontsize=10)

        # 復元誤差の確認: alpha=0 (=元のノイズそのまま) を実画像Aと比べる
        recon_a = decode_blend(model, x0_a, cls_a, cls_a, 0.0, steps, guidance, device)
        err = (recon_a - real_a).pow(2).mean().sqrt().item()
        recon_errors.append((cls_a, cls_b, err))

    fig.suptitle("Noise-space morphing via ODE inversion (a=0: digit A reconstructed, a=1: digit B reconstructed)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved: {out_path}")
    for cls_a, cls_b, err in recon_errors:
        print(f"  reconstruction RMSE for class {cls_a} (round-trip inversion): {err:.4f}")


def make_gif(model, images, labels, device, out_path: Path, cls_a: int = 3, cls_b: int = 8,
             steps: int = 60, guidance: float = 0.0, frames: int = 60, seed: int = 0):
    real_a, real_b, x0_a, x0_b = morph_pair(model, images, labels, cls_a, cls_b, device, steps, guidance, seed)
    alphas = torch.linspace(0, 1, frames)

    imgs = []
    for alpha in alphas.tolist():
        x0_mid = slerp(x0_a[0], x0_b[0], alpha)[None]
        img = decode_blend(model, x0_mid, cls_a, cls_b, alpha, steps, guidance, device)
        imgs.append(to_img(img[0, 0]).numpy())

    # 往復させてループを綺麗にする (0->1->0)
    imgs_full = imgs + imgs[::-1]

    fig, ax = plt.subplots(figsize=(4, 4.4))
    im = ax.imshow(imgs_full[0], cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title(f"class {cls_a}  <->  class {cls_b}   (a=0.00)")

    def update(i):
        idx = i % len(imgs_full)
        im.set_data(imgs_full[idx])
        a_val = alphas[idx] if idx < frames else alphas[len(imgs_full) - 1 - idx]
        title.set_text(f"class {cls_a}  <->  class {cls_b}   (a={a_val:.2f})")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=len(imgs_full), interval=1000 // FPS, blit=False)
    fig.tight_layout()
    anim.save(out_path, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"saved: {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description="ノイズ空間の補間による数字モーフィング")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--guidance", type=float, default=0.0, help="inversion/decode に使う CFG 強度 (既定 0 = 素の条件付き)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif-pair", type=int, nargs=2, default=[3, 8])
    p.add_argument("--no-gif", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = load_model(device)
    images, labels = load_mnist_tensor(device)
    images, labels = images.cpu(), labels.cpu()

    make_static_grid(model, images, labels, device, OUT_DIR / "morph_mnist.png",
                      steps=args.steps, guidance=args.guidance, seed=args.seed)

    if not args.no_gif:
        make_gif(model, images, labels, device, OUT_DIR / "morph_mnist.gif",
                  cls_a=args.gif_pair[0], cls_b=args.gif_pair[1],
                  steps=args.steps, guidance=args.guidance, seed=args.seed)


if __name__ == "__main__":
    main()
