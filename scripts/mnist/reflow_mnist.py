"""MNIST 版 Reflow: 学習済み速度場が作った決定論的ペアで再学習し、軌跡を直線化する
(`scripts/2d_basic/straightness.py` の MNIST 拡張)。

理論は 2D 版と同じ:
  - CFM は独立サンプリング (x0,x1) の周辺化なので、学習された速度場に沿った
    軌跡は一般に曲がる。
  - Reflow は学習済みモデルの ODE が実際に作ったペア (x0, ODE(x0)) を
    「決定論的カップリング」として再学習する。このペアでは経路が交差しない
    ため周辺速度場がほぼ定数になり、軌跡が直線化する。

手順:
  (1) 元モデルでペア (x0, y, x1=ODE(x0|y)) を大量に生成
  (2) その直線パス x_t=(1-t)x0+t x1 で新しい速度場を再学習
  (3) checkpoints/velocity_mnist_reflow1.pt に保存

実行 (checkpoints/velocity_mnist.pt が必要):
    python reflow_mnist.py --n-pairs 8000 --train-steps 6000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flow_matching_mnist import N_CLASSES, NULL_LABEL, UNetVelocity, load_mnist_tensor

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
OUT_DIR = ROOT / "out" / "mnist"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def load_base_model(device):
    ckpt = CKPT_DIR / "velocity_mnist.pt"
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} がありません。先に `python flow_matching_mnist.py` を実行してください。")
    model = UNetVelocity().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def integrate_from(model, x0: torch.Tensor, labels: torch.Tensor, steps: int, guidance: float, device) -> torch.Tensor:
    """指定した x0 から midpoint 法で ODE を解き、x1 = ODE(x0) を返す。"""
    x = x0.to(device)
    y = labels.to(device)
    y_null = torch.full_like(y, NULL_LABEL)
    dt = 1.0 / steps

    def velocity(x_in, t_in):
        v_cond = model(x_in, t_in, y)
        if guidance == 0.0:
            return v_cond
        v_uncond = model(x_in, t_in, y_null)
        return v_uncond + (1.0 + guidance) * (v_cond - v_uncond)

    for i in range(steps):
        t = torch.full((x.shape[0],), i * dt, device=device)
        k1 = velocity(x, t)
        k2 = velocity(x + 0.5 * dt * k1, t + 0.5 * dt)
        x = x + dt * k2
    return x.cpu()


@torch.no_grad()
def generate_pairs(base_model, n_pairs: int, steps: int, guidance: float, device,
                    batch_size: int = 500, seed: int = 0):
    """決定論的カップリング (x0, y, x1) を n_pairs 組生成する。"""
    torch.manual_seed(seed)
    x0_all = torch.randn(n_pairs, 1, 28, 28)
    y_all = torch.randint(0, N_CLASSES, (n_pairs,))
    x1_all = torch.empty(n_pairs, 1, 28, 28)

    for i in range(0, n_pairs, batch_size):
        j = min(i + batch_size, n_pairs)
        x1_all[i:j] = integrate_from(base_model, x0_all[i:j], y_all[i:j], steps, guidance, device)
        print(f"  [reflow] pairs {j}/{n_pairs}")

    return x0_all, y_all, x1_all


def reflow_loss(model, x0, x1, y, p_uncond):
    b = x0.shape[0]
    t = torch.rand(b, device=x0.device)
    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
    target = x1 - x0

    drop = torch.rand(b, device=x0.device) < p_uncond
    y = torch.where(drop, torch.full_like(y, NULL_LABEL), y)

    return (model(xt, t, y) - target).pow(2).mean()


def reflow_train(model, x0_all, y_all, x1_all, steps: int, batch_size: int, lr: float,
                  p_uncond: float, device) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    n = x0_all.shape[0]
    x0_all, y_all, x1_all = x0_all.to(device), y_all.to(device), x1_all.to(device)

    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch_size,), device=device)
        loss = reflow_loss(model, x0_all[idx], x1_all[idx], y_all[idx], p_uncond)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % max(1, steps // 20) == 0 or step == 1:
            print(f"[reflow-train] step {step:>6d}/{steps}  loss {loss.item():.4f}")


@torch.no_grad()
def chord_deviation(traj: torch.Tensor) -> tuple[float, float]:
    """始点・終点を結ぶ弦から軌跡がどれだけ外れるか (0 なら完全な直線)。traj: (T+1,N,784)。"""
    n_steps = traj.shape[0] - 1
    x0, x1 = traj[0], traj[-1]
    ts = torch.linspace(0, 1, n_steps + 1)[:, None, None]
    chord = x0[None] + ts * (x1 - x0)[None]
    dev = (traj - chord).norm(dim=2)
    length = (x1 - x0).norm(dim=1).clamp_min(1e-6)
    rel = dev / length[None, :]
    return rel.mean().item(), rel.max(dim=0).values.mean().item()


@torch.no_grad()
def report_straightness(base_model, reflow_model, device, steps: int = 60, n: int = 300, guidance: float = 1.0):
    """base と reflow の軌跡の直線性を実測して比較する (straightness.py と同じ指標)。"""
    labels = torch.randint(0, N_CLASSES, (n,))
    torch.manual_seed(12345)
    x0 = torch.randn(n, 1, 28, 28)

    results = {}
    for name, model in (("base", base_model), ("reflow", reflow_model)):
        x = x0.clone()
        y = labels
        traj = [x.clone()]

        @torch.no_grad()
        def velocity(x_in, t_in, m=model, yy=y):
            v_cond = m(x_in.to(device), t_in.to(device), yy.to(device)).cpu()
            if guidance == 0.0:
                return v_cond
            y_null = torch.full_like(yy, NULL_LABEL)
            v_uncond = m(x_in.to(device), t_in.to(device), y_null.to(device)).cpu()
            return v_uncond + (1.0 + guidance) * (v_cond - v_uncond)

        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((n,), i * dt)
            k1 = velocity(x, t)
            k2 = velocity(x + 0.5 * dt * k1, t + 0.5 * dt)
            x = x + dt * k2
            traj.append(x.clone())
        traj = torch.stack(traj).reshape(steps + 1, n, -1)
        mean_dev, max_dev = chord_deviation(traj)
        results[name] = (mean_dev, max_dev)

    print("\n" + "=" * 56)
    print(f"{'':<28}{'base (CFM)':>14}{'reflow 後':>14}")
    print("=" * 56)
    print(f"{'弦からの平均ずれ (弦長比)':<28}{results['base'][0]:>14.4f}{results['reflow'][0]:>14.4f}")
    print(f"{'弦からの最大ずれ (弦長比)':<28}{results['base'][1]:>14.4f}{results['reflow'][1]:>14.4f}")
    print("=" * 56)


def main() -> None:
    p = argparse.ArgumentParser(description="MNIST 版 Reflow")
    p.add_argument("--n-pairs", type=int, default=8000)
    p.add_argument("--pair-steps", type=int, default=60, help="ペア生成 ODE のステップ数")
    p.add_argument("--train-steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--p-uncond", type=float, default=0.1)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--redo", action="store_true", help="保存済み reflow 重みを無視して再学習")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    base_model = load_base_model(device)

    reflow_ckpt = CKPT_DIR / "velocity_mnist_reflow1.pt"
    reflow_model = UNetVelocity().to(device)
    if reflow_ckpt.exists() and not args.redo:
        reflow_model.load_state_dict(torch.load(reflow_ckpt, map_location=device))
        print(f"loaded: {reflow_ckpt}  (--redo で再学習)")
    else:
        print(f"[reflow] generating {args.n_pairs} deterministic pairs ...")
        x0_all, y_all, x1_all = generate_pairs(base_model, args.n_pairs, args.pair_steps, args.guidance, device)
        print("[reflow] training on straight-line pairs ...")
        reflow_train(reflow_model, x0_all, y_all, x1_all, args.train_steps, args.batch_size,
                     args.lr, args.p_uncond, device)
        torch.save(reflow_model.state_dict(), reflow_ckpt)
        print(f"saved: {reflow_ckpt}")

    reflow_model.eval()
    report_straightness(base_model, reflow_model, device, guidance=args.guidance)


if __name__ == "__main__":
    main()
