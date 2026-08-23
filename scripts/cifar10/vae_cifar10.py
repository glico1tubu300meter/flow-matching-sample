"""CIFAR-10 用の軽量 VAE。潜在空間 Flow Matching (`flow_matching_cifar10_latent.py`)
の前段として、32x32x3 のピクセルを 16x16x4 の潜在表現に圧縮する。

SD3/Flux が使う本格的な VAE は空間方向に 8 倍圧縮するが、それはメガピクセル級の
画像 (冗長性が大量にある) が前提。32x32 の CIFAR-10 に同じ圧縮率をかけると
4x4 まで潰れてしまい情報が失われすぎるため、ここでは**空間 2 倍・チャンネル
3->4 の軽い圧縮**に留めている (次元数で見ると 32*32*3=3072 -> 16*16*4=1024,
約3分の1)。

損失は再構成誤差 (MSE) + KL ダイバージェンス (重み beta は小さめ)。
SD 系の VAE と同様、beta を小さくして「まずまず正規分布に近いが、
再構成品質を優先する」潜在空間にしている。

実行:
    python vae_cifar10.py --benchmark-steps 200   # 速度確認だけ
    python vae_cifar10.py --steps 8000            # 本番学習 + 再構成figの保存
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from torchvision.utils import make_grid  # noqa: E402

from flow_matching_cifar10 import OUT_DIR as PIXEL_OUT_DIR  # noqa: E402
from flow_matching_cifar10 import CKPT_DIR, load_cifar_tensor, make_tb_writer, to_img  # noqa: E402

LATENT_CH = 4
LATENT_SIZE = 16  # 32 -> 16 (空間2倍圧縮)


# ------------------------------------------------------------------ モデル
class ResBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch_in)
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, padding=1)
        self.skip = nn.Conv2d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Encoder(nn.Module):
    """32x32x3 -> (mean, logvar) それぞれ LATENT_CHx16x16。"""

    def __init__(self, ch: int = 64):
        super().__init__()
        self.in_conv = nn.Conv2d(3, ch, 3, padding=1)
        self.block1 = ResBlock(ch, ch)
        self.down = nn.Conv2d(ch, ch, 4, stride=2, padding=1)  # 32 -> 16
        self.block2 = ResBlock(ch, ch * 2)
        self.out_norm = nn.GroupNorm(8, ch * 2)
        self.out_conv = nn.Conv2d(ch * 2, LATENT_CH * 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.in_conv(x)
        h = self.block1(h)
        h = self.down(h)
        h = self.block2(h)
        h = self.out_conv(F.silu(self.out_norm(h)))
        mean, logvar = h.chunk(2, dim=1)
        logvar = logvar.clamp(-30.0, 20.0)
        return mean, logvar


class Decoder(nn.Module):
    """LATENT_CHx16x16 -> 32x32x3。"""

    def __init__(self, ch: int = 64):
        super().__init__()
        self.in_conv = nn.Conv2d(LATENT_CH, ch * 2, 3, padding=1)
        self.block1 = ResBlock(ch * 2, ch * 2)
        self.up = nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1)  # 16 -> 32
        self.block2 = ResBlock(ch, ch)
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(z)
        h = self.block1(h)
        h = self.up(h)
        h = self.block2(h)
        h = self.out_conv(F.silu(self.out_norm(h)))
        return torch.tanh(h)  # [-1, 1] に制約 (学習データの正規化範囲と合わせる)


class VAE(nn.Module):
    def __init__(self, ch: int = 64):
        super().__init__()
        self.encoder = Encoder(ch)
        self.decoder = Decoder(ch)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor,
             beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = F.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


# ------------------------------------------------------------------- 学習
def train_vae(model: VAE, images: torch.Tensor, steps: int, batch_size: int, lr: float,
              beta: float, device, log_every: int = 200, writer=None) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    n = images.shape[0]

    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch_size,), device=device)
        x = images[idx]
        recon, mean, logvar = model(x)
        loss, recon_loss, kl = vae_loss(recon, x, mean, logvar, beta)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if writer is not None:
            writer.add_scalar("loss/total", loss.item(), step)
            writer.add_scalar("loss/recon", recon_loss.item(), step)
            writer.add_scalar("loss/kl", kl.item(), step)

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            sec_per_step = elapsed / step
            eta = sec_per_step * (steps - step)
            print(f"step {step:>6d}/{steps}  loss {loss.item():.4f}  recon {recon_loss.item():.4f}  "
                  f"kl {kl.item():.2f}  {sec_per_step*1000:.1f} ms/step  ETA {eta/60:.1f} min")


@torch.no_grad()
def make_recon_figure(model: VAE, images: torch.Tensor, device, out_path: Path, n: int = 8) -> None:
    model.eval()
    idx = torch.randperm(images.shape[0])[:n]
    x = images[idx]
    mean, logvar = model.encode(x)
    recon = model.decode(mean)  # 可視化は平均 (サンプリングなし) で行う

    real_grid = make_grid(to_img(x), nrow=n, padding=2)
    recon_grid = make_grid(to_img(recon), nrow=n, padding=2)

    fig, axes = plt.subplots(2, 1, figsize=(1.6 * n, 3.6))
    axes[0].imshow(real_grid.permute(1, 2, 0).numpy())
    axes[0].axis("off")
    axes[0].set_title("real")
    axes[1].imshow(recon_grid.permute(1, 2, 0).numpy())
    axes[1].axis("off")
    axes[1].set_title(f"VAE reconstruction ({LATENT_SIZE}x{LATENT_SIZE}x{LATENT_CH} latent, "
                       f"{3072//(LATENT_SIZE*LATENT_SIZE*LATENT_CH)}x compression)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved: {out_path}")


@torch.no_grad()
def encode_dataset(model: VAE, images: torch.Tensor, device, batch_size: int = 1000,
                    use_mean: bool = True) -> torch.Tensor:
    """データセット全体を潜在表現にエンコードしてキャッシュする
    (毎ステップ VAE を通すと遅いので、Flow Matching の学習前に1回だけ行う)。
    """
    model.eval()
    n = images.shape[0]
    latents = torch.empty(n, LATENT_CH, LATENT_SIZE, LATENT_SIZE, device=device)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        mean, logvar = model.encode(images[i:j])
        latents[i:j] = mean if use_mean else model.reparameterize(mean, logvar)
    return latents


def main() -> None:
    p = argparse.ArgumentParser(description="CIFAR-10 用の軽量 VAE (潜在空間 Flow Matching の前段)")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta", type=float, default=1e-5, help="KL項の重み (小さいほど再構成優先)")
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", type=str, default="vae_cifar10.pt")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--benchmark-steps", type=int, default=0)
    p.add_argument("--tensorboard", action="store_true", help="TensorBoard に loss をリアルタイム記録する")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    images, _ = load_cifar_tensor(device)
    print(f"CIFAR-10: {images.shape[0]} images, {images.shape[2]}x{images.shape[3]}x{images.shape[1]}")
    print(f"latent shape: {LATENT_CH}x{LATENT_SIZE}x{LATENT_SIZE} "
          f"({3072 / (LATENT_CH*LATENT_SIZE*LATENT_SIZE):.1f}x compression)")

    model = VAE(ch=args.ch).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M")

    if args.benchmark_steps > 0:
        print(f"\n[benchmark] running {args.benchmark_steps} steps to measure throughput ...")
        t0 = time.time()
        train_vae(model, images, args.benchmark_steps, args.batch_size, args.lr, args.beta, device,
                  log_every=max(1, args.benchmark_steps // 5))
        elapsed = time.time() - t0
        sec_per_step = elapsed / args.benchmark_steps
        print(f"\n[benchmark] {elapsed:.1f}s for {args.benchmark_steps} steps "
              f"({sec_per_step*1000:.1f} ms/step)")
        for target_steps in (4000, 8000, 16000):
            print(f"  -> estimated time for {target_steps} steps: {sec_per_step*target_steps/60:.1f} min")
        return

    ckpt = CKPT_DIR / args.ckpt
    if ckpt.exists() and not args.retrain:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"loaded checkpoint: {ckpt}")
    else:
        writer, _ = make_tb_writer("vae") if args.tensorboard else (None, None)
        train_vae(model, images, args.steps, args.batch_size, args.lr, args.beta, device, writer=writer)
        if writer is not None:
            writer.close()
        torch.save(model.state_dict(), ckpt)
        print(f"saved checkpoint: {ckpt}")

    make_recon_figure(model, images, device, PIXEL_OUT_DIR / "vae_reconstruction.png")


if __name__ == "__main__":
    main()
