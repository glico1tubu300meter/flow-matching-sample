"""三日月から離れた領域で速度場が「不安定」なことを実測する。

考え方:
  訓練データ x_t = (1-t)x0 + t*x1 は、ノイズ点とデータ点を結ぶ線分上にしか
  存在しない。三日月から離れた領域 (特に t が 1 に近いところ) はその線分が
  たまたま通過したときしか勾配を受け取らないため、v_theta(x,t) はほぼ
  「未定義」= 初期化や学習の乱数に左右される不安定な外挿になる。

検証方法:
  同じアーキテクチャを異なる乱数シードで独立に 2 回学習し (model A, model B)、
  グリッド上の各点・各時刻で両者の速度場のずれ ||v_A - v_B|| を測る。
  同時にその点における「訓練データ密度」(局所的に x_t サンプルがどれだけ
  近くにあるか) を推定し、密度が低いほどずれが大きいことを示す。
  密度が高い場所 (三日月の上) では、2 つの独立なモデルでも同じ答えに
  収束するはず (= 良く学習されている)。

出力:
  field_stability.png  density vs disagreement の散布図 + t=0.9 のずれマップ

実行:
    python field_stability.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import flow_matching_2d as fm

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
OUT_DIR = ROOT / "out" / "2d_basic"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LIM = 3.0


def load_or_train(name: str, seed: int, steps: int, device):
    path = CKPT_DIR / name
    model = fm.VelocityField().to(device)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"loaded: {path}")
        return model
    torch.manual_seed(seed)
    fm.train(model, steps, 512, 2e-3, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"saved: {path}")
    return model


@torch.no_grad()
def local_density(grid_pts: torch.Tensor, t: float, n_ref: int = 20000, k: int = 5) -> torch.Tensor:
    """時刻 t における訓練データ x_t = (1-t)x0 + t*x1 のサンプルを大量に引き、
    グリッド各点の k 近傍平均距離の逆数を「局所密度」の目安として返す。
    """
    x1 = fm.sample_two_moons(n_ref)
    x0 = torch.randn(n_ref, 2)
    xt = (1 - t) * x0 + t * x1  # (n_ref, 2)

    d = torch.cdist(grid_pts, xt)              # (G, n_ref)
    knn = d.topk(k, largest=False).values       # (G, k)
    mean_dist = knn.mean(dim=1).clamp_min(1e-3)
    return 1.0 / mean_dist


@torch.no_grad()
def field_disagreement(model_a, model_b, grid_pts: torch.Tensor, t: float, device) -> torch.Tensor:
    tt = torch.full((grid_pts.shape[0],), t, device=device)
    va = model_a(grid_pts, tt)
    vb = model_b(grid_pts, tt)
    return (va - vb).norm(dim=1).cpu()


def make_figure(model_a, model_b, device, out_path: Path, n: int = 40) -> None:
    axis = torch.linspace(-LIM, LIM, n)
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    grid = torch.stack([gx.flatten(), gy.flatten()], dim=1)
    grid_dev = grid.to(device)

    times = [0.3, 0.6, 0.9]
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, len(times))

    all_density, all_disagreement = [], []
    for col, t in enumerate(times):
        density = local_density(grid, t)
        disagree = field_disagreement(model_a, model_b, grid_dev, t, device)
        all_density.append(density)
        all_disagreement.append(disagree)

        ax = fig.add_subplot(gs[0, col])
        sc = ax.scatter(grid[:, 0], grid[:, 1], c=disagree, s=14, cmap="magma")
        real = fm.sample_two_moons(1500)
        ax.scatter(real[:, 0], real[:, 1], s=2, alpha=0.15, color="cyan")
        ax.set_title(f"$\\|v_A - v_B\\|$ at $t$={t}")
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, fraction=0.046)

    density_all = torch.cat(all_density)
    disagree_all = torch.cat(all_disagreement)

    ax = fig.add_subplot(gs[1, :])
    ax.scatter(density_all, disagree_all, s=5, alpha=0.25, color="tab:blue")
    ax.set_xscale("log")
    ax.set_xlabel("local training-data density (log scale, higher = closer to $x_t$ samples)")
    ax.set_ylabel("$\\|v_A(x,t) - v_B(x,t)\\|$  (disagreement between 2 independently trained models)")
    corr = torch.corrcoef(torch.stack([density_all.log(), disagree_all]))[0, 1].item()
    ax.set_title(f"low density -> two independently trained models disagree more  (Pearson r = {corr:.2f} in log-density)")

    fig.suptitle(
        "Same point & time, but two independently trained models give different v(x,t) where training data is sparse\n"
        "-> the field there is underdetermined by training: an unstable, seed-dependent extrapolation",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved: {out_path}")
    print(f"Pearson r (log density vs disagreement) = {corr:.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-steps", type=int, default=5000)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model_a = load_or_train("velocity_2d.pt", seed=0, steps=args.train_steps, device=device)
    model_b = load_or_train("velocity_2d_seedB.pt", seed=999, steps=args.train_steps, device=device)

    make_figure(model_a, model_b, device, OUT_DIR / "field_stability.png")


if __name__ == "__main__":
    main()
