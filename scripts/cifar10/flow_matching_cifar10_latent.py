"""潜在空間版 Flow Matching (SD3 / Flux と同じ構成) の CIFAR-10 デモ。

`vae_cifar10.py` で学習した VAE を使い、
  1. 全画像を1回だけ潜在表現 (16x16x4) にエンコードしてキャッシュ
  2. その潜在表現の上で `flow_matching_cifar10.py` と全く同じ CFM 損失・
     学習ループ・EMA・データ拡張を再利用 (ピクセル空間版との差分はモデルの
     入出力チャンネル数だけ: 3ch/32x32 -> 4ch/16x16)
  3. 生成時は ODE で潜在表現をサンプリングし、最後に VAE デコーダで
     ピクセルに戻す

という、実際の潜在拡散/潜在Flow Matchingモデルと同じパイプラインを踏む。

CIFAR-10 のような低解像度では、VAE 自体の再構成誤差がボトルネックになりやすく
(README 参照)、ピクセル空間版より品質が上がるとは限らない — その比較も含めて
実験する。

前提: 先に `python vae_cifar10.py --steps 8000` 等で VAE を学習しておくこと。

実行:
    python flow_matching_cifar10_latent.py --benchmark-steps 200
    python flow_matching_cifar10_latent.py --steps 30000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from torchvision.utils import make_grid  # noqa: E402

from flow_matching_cifar10 import (
    CKPT_DIR,
    CLASS_NAMES,
    N_CLASSES,
    NULL_LABEL,
    OUT_DIR,
    EMA,
    UNetVelocity,
    get_last_tb_step,
    load_cifar_tensor,
    load_tb_logdir,
    make_tb_writer,
    save_tb_logdir,
    to_img,
    train,
)
from vae_cifar10 import VAE, LATENT_CH, LATENT_SIZE, encode_dataset


def load_vae(device, ckpt_name: str = "vae_cifar10.pt", ch: int = 64) -> VAE:
    ckpt = CKPT_DIR / ckpt_name
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} がありません。先に `python vae_cifar10.py --steps 8000` を実行してください。")
    vae = VAE(ch=ch).to(device)
    vae.load_state_dict(torch.load(ckpt, map_location=device))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


# --------------------------------------------------------------- サンプリング
@torch.no_grad()
def sample_latent_ode(model, labels: torch.Tensor, steps: int, guidance: float, device) -> torch.Tensor:
    """潜在空間 (LATENT_CH x LATENT_SIZE x LATENT_SIZE) で ODE を解く。
    ピクセル空間版 `flow_matching_cifar10.sample_ode` と同じ midpoint 法だが、
    テンソル形状だけ潜在表現のものになっている。
    """
    n = labels.shape[0]
    z = torch.randn(n, LATENT_CH, LATENT_SIZE, LATENT_SIZE, device=device)
    y = labels.to(device)
    y_null = torch.full_like(y, NULL_LABEL)
    dt = 1.0 / steps

    def velocity(z_in, t_in):
        v_cond = model(z_in, t_in, y)
        if guidance == 0.0:
            return v_cond
        v_uncond = model(z_in, t_in, y_null)
        return v_uncond + (1.0 + guidance) * (v_cond - v_uncond)

    for i in range(steps):
        t = torch.full((n,), i * dt, device=device)
        k1 = velocity(z, t)
        k2 = velocity(z + 0.5 * dt * k1, t + 0.5 * dt)
        z = z + dt * k2
    return z


def make_figures(model, vae: VAE, device, out_dir: Path, suffix: str = "_latent",
                  class_idx: int | None = None) -> None:
    if class_idx is None:
        labels = torch.arange(N_CLASSES).repeat_interleave(8)
        nrow = 8
        title = f"generated CIFAR-10 via latent-space Flow Matching ({LATENT_CH}x{LATENT_SIZE}x{LATENT_SIZE} latent, CFG w=1)"
    else:
        labels = torch.full((64,), class_idx)
        nrow = 8
        title = (f"generated CIFAR-10 [{CLASS_NAMES[class_idx]} only] via latent-space Flow Matching "
                  f"({LATENT_CH}x{LATENT_SIZE}x{LATENT_SIZE} latent, CFG w=1)")

    z = sample_latent_ode(model, labels, steps=50, guidance=1.0, device=device)
    imgs = vae.decode(z)  # 潜在 -> ピクセル
    grid = make_grid(to_img(imgs), nrow=nrow, padding=2)
    plt.figure(figsize=(6, 7.5 if class_idx is None else 6))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / f"cifar10_grid{suffix}.png", dpi=140)
    plt.close()
    print(f"saved: {out_dir / f'cifar10_grid{suffix}.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description="潜在空間版 CIFAR-10 Flow Matching (VAE + U-Net)")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=128,
                    help="1サンプルあたりの処理効率は256の方が約9%良いが、同じ壁時計時間で比べると"
                         "更新回数が減る分だけ loss 低下は128の方が速いことを実測済み (README参照)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--p-uncond", type=float, default=0.1)
    p.add_argument("--base-ch", type=int, default=128)
    p.add_argument("--no-flip-aug", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--vae-ckpt", type=str, default="vae_cifar10.pt")
    p.add_argument("--vae-ch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", type=str, default=None,
                    help="既定は class-filter に応じて自動設定 (例: velocity_cifar10_latent_bird.pt)")
    p.add_argument("--out-suffix", type=str, default=None,
                    help="既定は class-filter に応じて自動設定 (例: _latent_bird)")
    p.add_argument("--class-filter", type=str, default=None,
                    choices=CLASS_NAMES, help="指定するとそのクラスの画像だけで学習する (VAEは共通で再学習しない)")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--resume", action="store_true",
                    help="既存チェックポイントの重みから続きを --steps 分だけ追加学習する "
                         "(optimizer/lrスケジュールは新しく作り直す)")
    p.add_argument("--benchmark-steps", type=int, default=0)
    p.add_argument("--tensorboard", action="store_true", help="TensorBoard に loss をリアルタイム記録する")
    p.add_argument("--t-dist", type=str, default="uniform", choices=["uniform", "logit_normal"],
                    help="時刻 t のサンプリング分布。logit_normal は SD3 と同じ t=0.5 付近重点サンプリング")
    p.add_argument("--logit-normal-mean", type=float, default=0.0)
    p.add_argument("--logit-normal-std", type=float, default=1.0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    class_idx = CLASS_NAMES.index(args.class_filter) if args.class_filter else None
    ckpt_name = args.ckpt or (f"velocity_cifar10_latent_{args.class_filter}.pt" if args.class_filter
                               else "velocity_cifar10_latent.pt")
    out_suffix = args.out_suffix or (f"_latent_{args.class_filter}" if args.class_filter else "_latent")

    # VAE は共通の学習済みチェックポイントをそのまま使う (再学習しない)
    vae = load_vae(device, args.vae_ckpt, args.vae_ch)
    print(f"VAE loaded (frozen, not retrained). latent shape: {LATENT_CH}x{LATENT_SIZE}x{LATENT_SIZE}")

    pixel_images, labels = load_cifar_tensor(device)
    if class_idx is not None:
        mask = labels == class_idx
        pixel_images, labels = pixel_images[mask], labels[mask]
        print(f"[class-filter] '{args.class_filter}' only: {pixel_images.shape[0]} images")

    print("[latent] encoding dataset once (cached for the rest of training) ...")
    latents = encode_dataset(vae, pixel_images, device)
    print(f"latents: {latents.shape}, mean={latents.mean().item():.3f}, std={latents.std().item():.3f}")
    del pixel_images  # ピクセルデータはもう不要 (VRAM節約)
    torch.cuda.empty_cache()

    # U-Net はチャンネル数以外そのまま流用できる (32->16->8->16->32 の代わりに
    # 16->8->4->8->16 で動く。カーネルサイズ等はハードコードされていないため)
    model = UNetVelocity(in_ch=LATENT_CH, base_ch=args.base_ch, use_attn=True).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"velocity field params: {n_params / 1e6:.2f}M")

    if args.benchmark_steps > 0:
        print(f"\n[benchmark] running {args.benchmark_steps} steps to measure throughput ...")
        t0 = time.time()
        train(model, latents, labels, args.benchmark_steps, args.batch_size, args.lr,
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

    ckpt = CKPT_DIR / ckpt_name
    if args.resume:
        if not ckpt.exists():
            raise SystemExit(f"{ckpt} がありません。--resume には既存チェックポイントが必要です。")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"[resume] loaded: {ckpt}  (+{args.steps} 追加ステップ、optimizer/lrスケジュールは新規)")
        run_name = "latent_" + out_suffix.lstrip("_")
        writer, log_dir = (None, None)
        if args.tensorboard:
            # サイドカーファイルがあれば、それが指すフォルダを確実に再利用する
            # (名前の前方一致だけだと別runと紛らわしくマッチすることがあるため)
            sidecar_dir = load_tb_logdir(ckpt)
            if sidecar_dir is not None:
                from torch.utils.tensorboard import SummaryWriter
                writer, log_dir = SummaryWriter(log_dir=str(sidecar_dir)), sidecar_dir
                print(f"[tensorboard] resuming via sidecar: {log_dir}")
            else:
                writer, log_dir = make_tb_writer(run_name, resume=True)
        step_offset = get_last_tb_step(log_dir) if writer is not None else 0
        if writer is not None:
            print(f"[tensorboard] continuing from step {step_offset} (同じグラフに追記)")
        ema = EMA(model, decay=args.ema_decay)
        train(model, latents, labels, args.steps, args.batch_size, args.lr, args.p_uncond, device,
              ema=ema, flip_aug=not args.no_flip_aug, writer=writer, ema_start=1, step_offset=step_offset,
              t_dist=args.t_dist, logit_normal_mean=args.logit_normal_mean,
              logit_normal_std=args.logit_normal_std)
        if writer is not None:
            writer.close()
            save_tb_logdir(ckpt, log_dir)
        ema.copy_to(model)
        torch.save(model.state_dict(), ckpt)
        print(f"saved EMA checkpoint (resumed): {ckpt}")
    elif ckpt.exists() and not args.retrain:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"loaded checkpoint: {ckpt}")
    else:
        writer, log_dir = make_tb_writer("latent_" + out_suffix.lstrip("_")) if args.tensorboard else (None, None)
        ema = EMA(model, decay=args.ema_decay)
        train(model, latents, labels, args.steps, args.batch_size, args.lr, args.p_uncond, device,
              ema=ema, flip_aug=not args.no_flip_aug, writer=writer, t_dist=args.t_dist,
              logit_normal_mean=args.logit_normal_mean, logit_normal_std=args.logit_normal_std)
        if writer is not None:
            writer.close()
            save_tb_logdir(ckpt, log_dir)
        ema.copy_to(model)
        torch.save(model.state_dict(), ckpt)
        print(f"saved EMA checkpoint: {ckpt}")

    make_figures(model, vae, device, OUT_DIR, suffix=out_suffix, class_idx=class_idx)


if __name__ == "__main__":
    main()
