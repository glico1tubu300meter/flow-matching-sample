"""CFM (元) と Reflow 1 回後の速度場を並べて比較する。

見たいこと:
  * 元の v_theta(x,t) は t とともに向きが変わる → 軌跡が曲がる
  * Reflow 後は t に対してほぼ不変 (定数場に近い) → 軌跡が直線になり 1 ステップ生成できる

出力:
  compare_fields.png  各時刻 t のストリームプロットを 2 段 (元 / reflow) で比較
  compare_fields.gif  同じものを t について連続アニメーション (quiver)

実行 (checkpoints/ の重みを使う。無ければ straightness.py を先に実行):
    python compare_fields.py
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


def make_grid(n: int, device):
    """streamplot 用に 1 次元軸と (n,n) メッシュを返す。"""
    axis = torch.linspace(-LIM, LIM, n)
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")  # shape (n, n), 行が y
    pts = torch.stack([gx.flatten(), gy.flatten()], dim=1).to(device)
    return axis, gx, gy, pts


@torch.no_grad()
def field_at(model, pts: torch.Tensor, t: float) -> torch.Tensor:
    tt = torch.full((pts.shape[0],), t, device=pts.device)
    return model(pts, tt).cpu()


# ------------------------------------------------------------------ 静止画
def make_static(base, reflowed, device, out_path: Path, n: int = 32) -> None:
    axis, _, _, pts = make_grid(n, device)
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    real = fm.sample_two_moons(2000)

    fig, axes = plt.subplots(2, len(times), figsize=(4.0 * len(times), 8.8), constrained_layout=True)
    fig.suptitle(
        "velocity field $v_\\theta(x,t)$ :  top = CFM (field keeps reshaping as $t$ grows)   /   "
        "bottom = after 1 reflow (almost unchanged for $t<1$; the $t=1$ panel is off-distribution)",
        fontsize=13,
    )

    # 色スケールを全パネルで揃える
    speeds = []
    for model in (base, reflowed):
        for t in times:
            speeds.append(field_at(model, pts, t).norm(dim=1).max().item())
    vmax = max(speeds)

    for row, (model, label) in enumerate(((base, "CFM"), (reflowed, "reflow 1"))):
        for col, t in enumerate(times):
            ax = axes[row, col]
            v = field_at(model, pts, t)
            u = v[:, 0].reshape(n, n).numpy()
            w = v[:, 1].reshape(n, n).numpy()
            speed = (u**2 + w**2) ** 0.5

            ax.scatter(real[:, 0], real[:, 1], s=2, alpha=0.10, color="tab:gray", zorder=0)
            ax.streamplot(
                axis.numpy(), axis.numpy(), u, w,
                color=speed, cmap="viridis", norm=plt.Normalize(0, vmax),
                density=1.1, linewidth=0.8, arrowsize=0.7,
            )
            ax.set_xlim(-LIM, LIM)
            ax.set_ylim(-LIM, LIM)
            ax.set_aspect("equal")
            ax.set_title(f"{label}   $t$={t:.2f}", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved: {out_path}")


# ------------------------------------------------------------------- GIF
def make_gif(base, reflowed, device, out_path: Path, n: int = 22, frames: int = 60, fps: int = 20) -> None:
    _, gx, gy, pts = make_grid(n, device)
    gx_np, gy_np = gx.flatten().numpy(), gy.flatten().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    fig.suptitle("velocity field over time:  CFM vs. after 1 reflow", fontsize=13)

    quivers = []
    for ax, label in zip(axes, ("CFM (time-varying)", "after 1 reflow (nearly constant)")):
        q = ax.quiver(gx_np, gy_np, torch.zeros(n * n), torch.zeros(n * n),
                      color="tab:blue", alpha=0.8, scale=55)
        quivers.append(q)
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.set_aspect("equal")
        ax.set_title(label)

    time_text = axes[0].text(0.03, 0.95, "", transform=axes[0].transAxes, va="top", fontsize=11)

    def update(i: int):
        t = i / (frames - 1)
        for q, model in zip(quivers, (base, reflowed)):
            v = field_at(model, pts, t)
            q.set_UVC(v[:, 0], v[:, 1])
        time_text.set_text(f"$t$ = {t:.2f}")
        return [*quivers, time_text]

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved: {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")


# ------------------------------------------------------------------- 指標
@torch.no_grad()
def report_time_variation(base, reflowed, device, n: int = 40, steps: int = 50) -> None:
    """場が時間方向にどれだけ変化するかを 2 通りの測り方で比較する。

    (A) 一様グリッド上: 空間全体で平均する。ただし各時刻で確率質量が無い領域
        (p_t のサポート外) の値は損失に効かず、学習で決まっていない。特に t=1 では
        データ多様体の外がほぼ全部それに当たるため、この指標は差を隠してしまう。

    (B) 実際に流れが通る点の上: x_t を ODE で運びながら v(x_t,t) を測る。Reflow が
        「直線化」するのはこの意味であり、v(x_t,t) が t によらず一定に近づく。
    """
    _, _, _, pts = make_grid(n, device)
    print("\n" + "=" * 64)
    print(f"{'速度場の時間変化':<34}{'CFM':>14}{'reflow 後':>16}")
    print("=" * 64)

    # ---- (A) 一様グリッド
    grid_stats = {}
    for label, model in (("base", base), ("reflow", reflowed)):
        fields = torch.stack([field_at(model, pts, i / steps) for i in range(steps + 1)])  # (T,G,2)
        spread = (fields - fields.mean(dim=0, keepdim=True)).norm(dim=2)
        grid_stats[label] = spread.mean().item()

    print("(A) 一様グリッド上で t 平均からのずれ")
    print(f"{'    mean_x,t ||v - mean_t v||':<34}{grid_stats['base']:>14.4f}{grid_stats['reflow']:>16.4f}")
    print("    ※ 質量の無い領域を含むため差が出にくい (下の (B) が本質)")
    print("-" * 64)

    # ---- (B) 実際の軌跡上
    traj_stats = {}
    for label, model in (("base", base), ("reflow", reflowed)):
        x = torch.randn(2000, 2, device=device)
        dt = 1.0 / steps
        vs = []
        for i in range(steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            v = model(x, t)
            vs.append(v)
            k2 = model(x + 0.5 * dt * v, t + 0.5 * dt)
            x = x + dt * k2
        vs = torch.stack(vs)                                    # (T, N, 2)
        spread = (vs - vs.mean(dim=0, keepdim=True)).norm(dim=2)  # 各軌跡の時間平均からのずれ
        rel = spread.mean().item() / vs.norm(dim=2).mean().item()
        traj_stats[label] = (spread.mean().item(), rel)

    print("(B) 実際に流れが通る点 x_t の上での ||v(x_t,t) - 時間平均||")
    print(f"{'    絶対値':<34}{traj_stats['base'][0]:>14.4f}{traj_stats['reflow'][0]:>16.4f}")
    print(f"{'    |v| に対する比':<34}{traj_stats['base'][1]:>14.4f}{traj_stats['reflow'][1]:>16.4f}")
    print("=" * 64)


def main() -> None:
    p = argparse.ArgumentParser(description="速度場の比較 (CFM vs reflow)")
    p.add_argument("--no-gif", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    base = load("velocity_2d.pt", device)
    reflowed = load("velocity_2d_reflow1.pt", device)

    report_time_variation(base, reflowed, device)
    make_static(base, reflowed, device, OUT_DIR / "compare_fields.png")
    if not args.no_gif:
        make_gif(base, reflowed, device, OUT_DIR / "compare_fields.gif")


if __name__ == "__main__":
    main()
