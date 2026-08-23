"""Flow Matching を CIFAR-10 (32x32 カラー) に適用するサンプル。

`scripts/mnist/flow_matching_mnist.py` と損失・ODE サンプリングは完全に同一。
違うのは入力が 1x28x28 (グレースケール) ではなく 3x32x32 (カラー) になる点だけ。
U-Net の解像度パスも 32 -> 16 -> 8 -> 16 -> 32 と、同じ 2 段ダウンサンプル構成が
そのまま使える (28 の代わりに 32 なので割り切れる)。

クラスラベル (10種) 条件付け + CFG も MNIST 版と同じ仕組み。

実行:
    python flow_matching_cifar10.py --benchmark-steps 200   # まず速度計測だけ
    python flow_matching_cifar10.py --steps 8000            # 本番学習
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from torchvision.datasets import CIFAR10  # noqa: E402
from torchvision.utils import make_grid  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
OUT_DIR = ROOT / "out" / "cifar10"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

N_CLASSES = 10
NULL_LABEL = N_CLASSES
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


# ------------------------------------------------------------------ データ
def load_cifar_tensor(device) -> tuple[torch.Tensor, torch.Tensor]:
    """全画像をメモリに載せる (50000 x 3 x 32 x 32, [-1,1] 正規化)。"""
    ds = CIFAR10(root=DATA_DIR, train=True, download=True)
    x = torch.from_numpy(ds.data).float()          # (N,32,32,3), [0,255]
    x = x.permute(0, 3, 1, 2) / 127.5 - 1.0          # (N,3,32,32), [-1,1]
    y = torch.tensor(ds.targets).long()
    return x.to(device), y.to(device)


# ------------------------------------------------------------------ モデル
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(torch.linspace(0.0, math.log(1000.0), half))
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t[:, None] * self.freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch_in)
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, ch_out)
        self.norm2 = nn.GroupNorm(8, ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, padding=1)
        self.skip = nn.Conv2d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """8x8 (=64トークン) のボトルネックにだけ入れる軽量な self-attention。
    畳み込みだけでは捉えにくい大域的な構造 (物体全体の輪郭など) を学習しやすくする。
    トークン数が少ないので計算コストはごくわずか。
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv.unbind(1)  # each (b, heads, head_dim, hw)
        attn = torch.softmax((q.transpose(-1, -2) @ k) / (self.head_dim ** 0.5), dim=-1)  # (b,heads,hw,hw)
        out = v @ attn.transpose(-1, -2)  # (b, heads, head_dim, hw)
        out = out.reshape(b, c, h, w)
        return x + self.proj(out)


class UNetVelocity(nn.Module):
    """32x32 -> 16x16 -> 8x8 -> 16x16 -> 32x32 の小さな U-Net。MNIST 版と同構造だが、
    入出力チャンネルを 1 -> 3 に変更し、ボトルネック (8x8) に self-attention を追加。
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 128, emb_dim: int = 128, use_attn: bool = True):
        super().__init__()
        self.time_embed = TimeEmbedding(emb_dim)
        self.label_embed = nn.Embedding(N_CLASSES + 1, emb_dim)

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.down1 = ResBlock(base_ch, base_ch, emb_dim)
        self.pool1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)          # 32 -> 16
        self.down2 = ResBlock(base_ch, base_ch * 2, emb_dim)
        self.pool2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)  # 16 -> 8

        self.mid1 = ResBlock(base_ch * 2, base_ch * 2, emb_dim)
        self.mid_attn = SelfAttention2d(base_ch * 2) if use_attn else nn.Identity()
        self.mid2 = ResBlock(base_ch * 2, base_ch * 2, emb_dim)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)  # 8 -> 16
        self.dec2 = ResBlock(base_ch * 4, base_ch, emb_dim)
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1)          # 16 -> 32
        self.dec1 = ResBlock(base_ch * 2, base_ch, emb_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        emb = self.time_embed(t) + self.label_embed(y)

        h0 = self.in_conv(x)
        h1 = self.down1(h0, emb)
        h2 = self.down2(self.pool1(h1), emb)
        m = self.mid1(self.pool2(h2), emb)
        m = self.mid_attn(m)
        m = self.mid2(m, emb)

        u2 = self.up2(m)
        d2 = self.dec2(torch.cat([u2, h2], dim=1), emb)
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, h1], dim=1), emb)

        return self.out_conv(F.silu(self.out_norm(d1)))


class EMA:
    """重みの指数移動平均。生成品質を大きく改善することが知られている定番手法
    (学習中の重みは各ステップで揺れるが、EMAはそのノイズを平滑化する)。
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


# ------------------------------------------------------------------- 学習
def sample_t(batch_size: int, device, t_dist: str = "uniform",
             logit_normal_mean: float = 0.0, logit_normal_std: float = 1.0) -> torch.Tensor:
    """時刻 t のサンプリング分布。

    - "uniform": t ~ U(0,1) (Flow Matching の基本形。既定)
    - "logit_normal": u ~ N(mean, std) を sigmoid で (0,1) に写す。SD3 (Esser et al. 2024)
      が採用する既定スケジュールで、t=0.5 付近 (ノイズと実データの中間、最も難しく
      情報量の多い領域) を重点的にサンプリングする。
    """
    if t_dist == "uniform":
        return torch.rand(batch_size, device=device)
    if t_dist == "logit_normal":
        u = torch.randn(batch_size, device=device) * logit_normal_std + logit_normal_mean
        return torch.sigmoid(u)
    raise ValueError(f"unknown t_dist: {t_dist!r} (uniform / logit_normal)")


def cfm_loss(model, x1: torch.Tensor, y: torch.Tensor, p_uncond: float,
             t_dist: str = "uniform", logit_normal_mean: float = 0.0,
             logit_normal_std: float = 1.0) -> torch.Tensor:
    b = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = sample_t(b, x1.device, t_dist, logit_normal_mean, logit_normal_std)
    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
    target = x1 - x0

    drop = torch.rand(b, device=x1.device) < p_uncond
    y = torch.where(drop, torch.full_like(y, NULL_LABEL), y)

    return (model(xt, t, y) - target).pow(2).mean()


def train(model, images, labels, steps: int, batch_size: int, lr: float, p_uncond: float,
          device, log_every: int = 200, ema: "EMA | None" = None,
          ema_start: int = 200, flip_aug: bool = True, writer=None, step_offset: int = 0,
          t_dist: str = "uniform", logit_normal_mean: float = 0.0,
          logit_normal_std: float = 1.0) -> list[float]:
    """writer: torch.utils.tensorboard.SummaryWriter があれば毎ステップ loss を記録する
    (TensorBoard でリアルタイムに曲線を確認できる)。
    step_offset: --resume で追加学習するとき、TensorBoard 上のステップ番号を
    前回の続きから振るためのオフセット (これが無いと曲線が step=1 から
    再スタートして別のグラフに見えてしまう)。
    t_dist: "uniform" (既定) か "logit_normal" (SD3 スケジュール、t=0.5 付近を重点サンプリング)。
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    n = images.shape[0]
    history = []

    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch_size,), device=device)
        batch = images[idx]
        if flip_aug:
            # 水平反転を確率 0.5 でランダムに適用し、実質的にデータを2倍に増やす
            flip = torch.rand(batch.shape[0], device=device) < 0.5
            batch = torch.where(flip[:, None, None, None], batch.flip(dims=[3]), batch)
        loss = cfm_loss(model, batch, labels[idx], p_uncond,
                         t_dist=t_dist, logit_normal_mean=logit_normal_mean,
                         logit_normal_std=logit_normal_std)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if ema is not None and step >= ema_start:
            ema.update(model)

        history.append(loss.item())
        if writer is not None:
            writer.add_scalar("loss/step", loss.item(), step_offset + step)
            writer.add_scalar("lr", sched.get_last_lr()[0], step_offset + step)
        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            sec_per_step = elapsed / step
            eta = sec_per_step * (steps - step)
            recent = sum(history[-100:]) / len(history[-100:])
            if writer is not None:
                writer.add_scalar("loss/avg100", recent, step_offset + step)
            print(f"step {step:>6d}/{steps}  loss {loss.item():.4f}  (avg100 {recent:.4f})  "
                  f"{sec_per_step*1000:.1f} ms/step  ETA {eta/60:.1f} min")
    return history


# --------------------------------------------------------------- サンプリング
@torch.no_grad()
def sample_ode(model, labels: torch.Tensor, steps: int, guidance: float, device,
                return_traj: bool = False):
    model.eval()
    n = labels.shape[0]
    x = torch.randn(n, 3, 32, 32, device=device)
    y = labels.to(device)
    y_null = torch.full_like(y, NULL_LABEL)
    dt = 1.0 / steps
    traj = [x.clone()] if return_traj else None

    def velocity(x_in, t_in):
        v_cond = model(x_in, t_in, y)
        if guidance == 0.0:
            return v_cond
        v_uncond = model(x_in, t_in, y_null)
        return v_uncond + (1.0 + guidance) * (v_cond - v_uncond)

    for i in range(steps):
        t = torch.full((n,), i * dt, device=device)
        k1 = velocity(x, t)
        k2 = velocity(x + 0.5 * dt * k1, t + 0.5 * dt)
        x = x + dt * k2
        if return_traj:
            traj.append(x.clone())

    return (x, torch.stack(traj)) if return_traj else x


def to_img(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1) / 2).cpu()


def make_tb_writer(run_name: str, resume: bool = False):
    """TensorBoard の SummaryWriter を作る。`train(..., writer=writer)` に渡すと
    毎ステップの loss がリアルタイムに記録される。戻り値は (writer, log_dir)。

    resume=True の場合、同じ run_name で始まる既存のログフォルダのうち最新のものに
    追記する (新しいフォルダを作らない)。これにより TensorBoard 上で
    「1回目の学習の続き」として同じ1本の曲線に見える (step 番号も
    `get_last_tb_step` で前回の続きから振る必要がある。`train(..., step_offset=...)`
    と組み合わせて使う)。

    見るには別ターミナルで:
        tensorboard --logdir "<out>/cifar10/tb_logs"
    を実行し、表示される http://localhost:6006 をブラウザで開く。
    """
    import datetime
    from pathlib import Path as _Path

    from torch.utils.tensorboard import SummaryWriter

    base_dir = OUT_DIR / "tb_logs"
    if resume:
        candidates = sorted(
            (p for p in base_dir.glob(f"{run_name}_*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            log_dir = candidates[-1]
            writer = SummaryWriter(log_dir=str(log_dir))
            print(f"[tensorboard] resuming existing run (same graph): {log_dir}")
            return writer, log_dir
        print(f"[tensorboard] resume=True だが '{run_name}_*' が見つからないため新規作成する")

    log_dir = base_dir / f"{run_name}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[tensorboard] logging to: {log_dir}")
    print(f"[tensorboard] run:  tensorboard --logdir \"{base_dir}\"")
    return writer, log_dir


def tb_sidecar_path(ckpt_path: Path) -> Path:
    """チェックポイントと対になる「どの TensorBoard ログフォルダを使ったか」を
    記録するサイドカーファイルのパス。名前のパターンマッチ (前方一致) だと
    別のrunと紛らわしくマッチすることがあるため、これで確実に同じフォルダを
    指せるようにする。"""
    return ckpt_path.with_name(ckpt_path.name + ".tblogdir")


def save_tb_logdir(ckpt_path: Path, log_dir) -> None:
    tb_sidecar_path(ckpt_path).write_text(str(log_dir), encoding="utf-8")


def load_tb_logdir(ckpt_path: Path):
    p = tb_sidecar_path(ckpt_path)
    if p.exists():
        d = Path(p.read_text(encoding="utf-8").strip())
        if d.is_dir():
            return d
    return None


def get_last_tb_step(log_dir, tag: str = "loss/step") -> int:
    """指定した TensorBoard ログフォルダに記録済みの最大ステップ番号を返す
    (--resume 時にステップ番号を続きから振るために使う)。"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return 0
    events = ea.Scalars(tag)
    return events[-1].step if events else 0


def make_figures(model, device, out_dir: Path, suffix: str = "") -> None:
    labels = torch.arange(N_CLASSES).repeat_interleave(8)
    imgs = sample_ode(model, labels, steps=50, guidance=1.0, device=device)
    grid = make_grid(to_img(imgs), nrow=8, padding=2)
    plt.figure(figsize=(6, 7.5))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title("generated CIFAR-10 (rows = class, CFG w=1)")
    plt.tight_layout()
    plt.savefig(out_dir / f"cifar10_grid{suffix}.png", dpi=140)
    plt.close()
    print(f"saved: {out_dir / f'cifar10_grid{suffix}.png'}")

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
    labels8 = torch.arange(8)
    for ax, n_steps in zip(axes, [1, 2, 4, 10, 50]):
        imgs = sample_ode(model, labels8, steps=n_steps, guidance=1.0, device=device)
        grid = make_grid(to_img(imgs), nrow=8, padding=2)
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.axis("off")
        ax.set_title(f"{n_steps} step(s)")
    fig.suptitle("fewer ODE steps -> lower quality (classes 0-7, one sample each)")
    fig.tight_layout()
    fig.savefig(out_dir / f"cifar10_steps{suffix}.png", dpi=140)
    plt.close(fig)
    print(f"saved: {out_dir / f'cifar10_steps{suffix}.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description="CIFAR-10 Flow Matching sample")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--p-uncond", type=float, default=0.1)
    p.add_argument("--base-ch", type=int, default=128)
    p.add_argument("--no-attn", action="store_true", help="ボトルネックの self-attention を無効化する")
    p.add_argument("--no-flip-aug", action="store_true", help="水平反転データ拡張を無効化する")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", type=str, default="velocity_cifar10_v2.pt")
    p.add_argument("--out-suffix", type=str, default="_v2",
                    help="出力ファイル名の suffix (v1 の結果を上書きしないため既定で _v2)")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--benchmark-steps", type=int, default=0,
                    help="指定するとこのステップ数だけ計測用に走らせ、フル学習時間を推定して終了する")
    p.add_argument("--tensorboard", action="store_true", help="TensorBoard に loss をリアルタイム記録する")
    p.add_argument("--t-dist", type=str, default="uniform", choices=["uniform", "logit_normal"],
                    help="時刻 t のサンプリング分布。logit_normal は SD3 と同じ t=0.5 付近重点サンプリング")
    p.add_argument("--logit-normal-mean", type=float, default=0.0)
    p.add_argument("--logit-normal-std", type=float, default=1.0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    images, labels = load_cifar_tensor(device)
    print(f"CIFAR-10: {images.shape[0]} images, {images.shape[2]}x{images.shape[3]}x{images.shape[1]}")

    model = UNetVelocity(base_ch=args.base_ch, use_attn=not args.no_attn).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M  (attn={not args.no_attn}, flip_aug={not args.no_flip_aug}, "
          f"t_dist={args.t_dist})")

    if args.benchmark_steps > 0:
        print(f"\n[benchmark] running {args.benchmark_steps} steps to measure throughput ...")
        t0 = time.time()
        train(model, images, labels, args.benchmark_steps, args.batch_size, args.lr,
              args.p_uncond, device, log_every=max(1, args.benchmark_steps // 5),
              flip_aug=not args.no_flip_aug, t_dist=args.t_dist,
              logit_normal_mean=args.logit_normal_mean, logit_normal_std=args.logit_normal_std)
        elapsed = time.time() - t0
        sec_per_step = elapsed / args.benchmark_steps
        print(f"\n[benchmark] {elapsed:.1f}s for {args.benchmark_steps} steps "
              f"({sec_per_step*1000:.1f} ms/step)")
        for target_steps in (8000, 16000, 30000, 50000):
            print(f"  -> estimated time for {target_steps} steps: {sec_per_step*target_steps/60:.1f} min")
        return

    ckpt = CKPT_DIR / args.ckpt
    if ckpt.exists() and not args.retrain:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"loaded checkpoint: {ckpt}")
    else:
        writer, _ = make_tb_writer("unet_" + args.out_suffix.lstrip("_")) if args.tensorboard else (None, None)
        ema = EMA(model, decay=args.ema_decay)
        train(model, images, labels, args.steps, args.batch_size, args.lr, args.p_uncond, device,
              ema=ema, flip_aug=not args.no_flip_aug, writer=writer, t_dist=args.t_dist,
              logit_normal_mean=args.logit_normal_mean, logit_normal_std=args.logit_normal_std)
        if writer is not None:
            writer.close()
        ema.copy_to(model)  # 生成には EMA 重みを使う
        torch.save(model.state_dict(), ckpt)
        print(f"saved EMA checkpoint: {ckpt}")

    make_figures(model, device, OUT_DIR, suffix=args.out_suffix)


if __name__ == "__main__":
    main()
