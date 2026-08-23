"""「Flow Matching は直線を学習するのか？」を実測で確かめるスクリプト。

要点:
  * 訓練時の直線パス x_t = (1-t)x0 + t x1 は、ペア (x0,x1) を固定したときの
    「条件付き」経路。その速度 x1 - x0 は定数。
  * だが学習される v_theta(x,t) はその周辺化   v_theta(x,t) ~ E[x1 - x0 | x_t = x]
    であり、同じ点を通る無数の直線の平均。したがって t とともに向きが変わり、
    ODE の軌跡は一般に曲がる。
  * Reflow (Liu et al. 2022) は、学習済みモデルが作ったペア (x0, ODE(x0)) を
    「決定論的カップリング」として再学習する。このペアでは経路が交差しないため、
    周辺速度場がほぼ定数になり、軌跡が直線化して 1 ステップ生成に近づく。

このスクリプトは
  (1) 元モデルの軌跡の曲がり具合を測る
  (2) Reflow を 1 回行い、同じ指標を測り直す
  (3) 1 ステップ Euler 生成の質を比較する
  (4) 比較図 straightness.png を出力する

実行 (checkpoints/velocity_2d.pt があれば再学習しない):
    python straightness.py
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
# チェックポイント・データはスクリプトと同じフォルダ配下に保存される(初回実行時に自動作成)
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ 指標
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


def chord_deviation(traj: torch.Tensor) -> tuple[float, float]:
    """始点と終点を結ぶ直線から軌跡がどれだけ外れるか。

    戻り値: (平均ずれ / 弦長, 最大ずれ / 弦長)  — 0 なら完全な直線。
    """
    n_steps = traj.shape[0] - 1
    x0, x1 = traj[0], traj[-1]
    ts = torch.linspace(0, 1, n_steps + 1, device=traj.device)[:, None, None]
    chord = x0[None] + ts * (x1 - x0)[None]          # 直線だった場合の位置
    dev = (traj - chord).norm(dim=2)                 # (T+1, N)
    length = (x1 - x0).norm(dim=1).clamp_min(1e-6)   # (N,)
    rel = dev / length[None, :]
    return rel.mean().item(), rel.max(dim=0).values.mean().item()


@torch.no_grad()
def straightness_score(model, n: int, steps: int, device) -> float:
    """S = E int_0^1 || v(x_t,t) - (x1 - x0) ||^2 dt  (0 なら完全に直線)。"""
    x0 = torch.randn(n, 2, device=device)
    traj = integrate(model, x0, steps)
    x1 = traj[-1]
    target = x1 - x0
    total = 0.0
    for i in range(steps + 1):
        t = torch.full((n,), i / steps, device=device)
        total += (model(traj[i], t) - target).pow(2).sum(dim=1).mean().item()
    return total / (steps + 1)


@torch.no_grad()
def euler_n_step(model, x0: torch.Tensor, steps: int) -> torch.Tensor:
    x = x0.clone()
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        x = x + dt * model(x, t)
    return x


def moons_distance(x: torch.Tensor) -> float:
    """生成点から実データ最近傍までの平均距離 (小さいほどデータ多様体に乗っている)。"""
    real = fm.sample_two_moons(4000).to(x.device)
    d = torch.cdist(x, real)
    return d.min(dim=1).values.mean().item()


# ----------------------------------------------------------------- Reflow
def reflow(base_model, steps: int, batch_size: int, lr: float, n_pairs: int, device):
    """(x0, ODE(x0)) のペアで新しい速度場を学習する = 1 回の Reflow。"""
    print(f"\n[reflow] 決定論的ペアを {n_pairs} 組生成中 ...")
    x0_all = torch.randn(n_pairs, 2, device=device)
    x1_all = torch.empty_like(x0_all)
    for i in range(0, n_pairs, 4096):  # メモリのためチャンク分割
        chunk = x0_all[i : i + 4096]
        x1_all[i : i + 4096] = integrate(base_model, chunk, 100)[-1]

    model = fm.VelocityField().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n_pairs, (batch_size,), device=device)
        x0, x1 = x0_all[idx], x1_all[idx]
        t = torch.rand(batch_size, device=device)

        # 独立サンプリングではなく「対応の付いたペア」を直線で結ぶのが肝
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        loss = (model(xt, t) - (x1 - x0)).pow(2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % max(1, steps // 5) == 0:
            print(f"[reflow] step {step:>6d}/{steps}  loss {loss.item():.4f}")
    return model


# ------------------------------------------------------------------ 可視化
def make_figure(base, reflowed, device, out_path: Path) -> None:
    torch.manual_seed(123)
    x0 = torch.randn(300, 2, device=device)
    traj_b = integrate(base, x0, 60).cpu()
    traj_r = integrate(reflowed, x0, 60).cpu()

    torch.manual_seed(456)
    x0_big = torch.randn(2000, 2, device=device)
    one_b = euler_n_step(base, x0_big, 1).cpu()
    one_r = euler_n_step(reflowed, x0_big, 1).cpu()

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10))

    for ax, traj, title in (
        (axes[0, 0], traj_b, "trajectories: CFM (curved)"),
        (axes[0, 1], traj_r, "trajectories: after 1 reflow (straight)"),
    ):
        for k in range(traj.shape[1]):
            ax.plot(traj[:, k, 0], traj[:, k, 1], lw=0.5, alpha=0.35, color="tab:blue")
        ax.scatter(traj[0, :, 0], traj[0, :, 1], s=5, color="tab:orange", label="$t=0$")
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=5, color="tab:red", label="$t=1$")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)

    for ax, x, title in (
        (axes[1, 0], one_b, "1-step Euler: CFM"),
        (axes[1, 1], one_r, "1-step Euler: after reflow"),
    ):
        ax.scatter(x[:, 0], x[:, 1], s=4, alpha=0.5, color="tab:green")
        ax.set_title(title)

    for ax in axes.flatten():
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\nsaved: {out_path}")


# --------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description="軌跡の直線性の実測と Reflow")
    p.add_argument("--train-steps", type=int, default=5000)
    p.add_argument("--reflow-steps", type=int, default=5000)
    p.add_argument("--n-pairs", type=int, default=60000)
    p.add_argument("--redo-reflow", action="store_true", help="保存済み reflow 重みを無視して再学習")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt = CKPT_DIR / "velocity_2d.pt"
    base = fm.VelocityField().to(device)
    if ckpt.exists():
        base.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"loaded: {ckpt}")
    else:
        fm.train(base, args.train_steps, 512, 2e-3, device)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(base.state_dict(), ckpt)

    reflow_ckpt = CKPT_DIR / "velocity_2d_reflow1.pt"
    if reflow_ckpt.exists() and not args.redo_reflow:
        reflowed = fm.VelocityField().to(device)
        reflowed.load_state_dict(torch.load(reflow_ckpt, map_location=device))
        print(f"loaded: {reflow_ckpt}  (--redo-reflow で再学習)")
    else:
        reflowed = reflow(base, args.reflow_steps, 512, 2e-3, args.n_pairs, device)
        reflow_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(reflowed.state_dict(), reflow_ckpt)
        print(f"saved: {reflow_ckpt}")

    print("\n" + "=" * 62)
    print(f"{'':<24}{'CFM (元)':>18}{'reflow 後':>18}")
    print("=" * 62)

    torch.manual_seed(999)
    results = {}
    for name, model in (("base", base), ("reflow", reflowed)):
        traj = integrate(model, torch.randn(2000, 2, device=device), 100)
        mean_dev, max_dev = chord_deviation(traj)
        results[name] = {
            "mean_dev": mean_dev,
            "max_dev": max_dev,
            "S": straightness_score(model, 2000, 50, device),
        }
        for n_steps in (1, 2, 4, 50):
            x = euler_n_step(model, torch.randn(2000, 2, device=device), n_steps)
            results[name][f"euler{n_steps}"] = moons_distance(x)

    def row(label: str, key: str, fmt: str = ".4f") -> None:
        print(f"{label:<24}{results['base'][key]:>18{fmt}}{results['reflow'][key]:>18{fmt}}")

    row("弦からの平均ずれ", "mean_dev")
    row("弦からの最大ずれ", "max_dev")
    row("直線性 S (0 が直線)", "S")
    print("-" * 62)
    print("Euler N ステップ生成 → 実データ最近傍までの平均距離")
    for n_steps in (1, 2, 4, 50):
        row(f"  N = {n_steps}", f"euler{n_steps}")
    print("=" * 62)

    make_figure(base, reflowed, device, OUT_DIR / "straightness.png")


if __name__ == "__main__":
    main()
