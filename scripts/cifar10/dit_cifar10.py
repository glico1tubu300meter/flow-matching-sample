"""CIFAR-10 用 DiT (Diffusion Transformer) 版 Flow Matching。

`flow_matching_cifar10.py` の U-Net をそのまま Vision Transformer に置き換えたもの。
損失・ODE サンプリング・学習ループなどのインフラは完全に共通なので、
`flow_matching_cifar10` からそのままインポートして再利用する。差分はネットワークの
中身 (畳み込み U-Net vs パッチベースの Transformer) だけ。

アーキテクチャ (DiT, Peebles & Xie 2022 のミニ版):
  1. 32x32x3 画像を 4x4 パッチに分割 -> 8x8=64 トークン
  2. 各パッチを線形射影 (Conv2d kernel=stride=patch と等価) して埋め込みベクトルに
  3. 学習可能な位置埋め込みを加算 (Attention 自体には位置の概念が無いため)
  4. 時刻+クラス条件を adaLN-Zero (適応的な LayerNorm 変調: シフト・スケール・
     ゲートの3つをブロックごとに条件から生成) で各 Transformer ブロックに注入。
     U-Net 版の「特徴マップに単純加算」より表現力が高い一方、パラメータ数は増える
  5. Self-Attention + MLP の Transformer ブロックを N 層
  6. 最終層で adaLN 変調 + 線形層によりパッチをピクセルに戻す (unpatchify)

U-Net 版 (base_ch=128, attention付き) が約7.93Mパラメータなので、dim=256, depth=6
で近いパラメータ数 (約7.4M) になるよう揃えてあり、同条件で比較しやすい。

実行:
    python dit_cifar10.py --benchmark-steps 200   # まず速度計測だけ
    python dit_cifar10.py --steps 30000           # 本番学習
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from flow_matching_cifar10 import (
    CKPT_DIR,
    N_CLASSES,
    NULL_LABEL,
    OUT_DIR,
    EMA,
    TimeEmbedding,
    load_cifar_tensor,
    make_figures,
    make_tb_writer,
    train,
)


# ------------------------------------------------------------------ モデル
class AdaLNModulation(nn.Module):
    """条件ベクトル c から (shift, scale, gate, ...) を生成する小さな MLP。
    最終層をゼロ初期化することで、学習開始時点では各ブロックが恒等写像に近い
    振る舞いをするようにする (DiT 論文の "adaLN-Zero" のトリック)。
    """

    def __init__(self, cond_dim: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, n_out))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.ada = AdaLNModulation(dim, 6 * dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1.unsqueeze(1) * attn_out
        h = modulate(self.norm2(x), shift2, scale2)
        x = x + gate2.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, patch_size: int, out_ch: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_ch)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.ada = AdaLNModulation(dim, 2 * dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class DiTVelocity(nn.Module):
    """CIFAR-10 (32x32x3) 用の小さな DiT。forward の入出力は UNetVelocity と
    同じ (x, t, y) -> velocity なので、既存の学習・サンプリングコードをそのまま使える。
    """

    def __init__(self, img_size: int = 32, patch_size: int = 4, in_ch: int = 3,
                 dim: int = 256, depth: int = 6, num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.in_ch = in_ch
        num_patches = self.grid ** 2

        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.time_embed = TimeEmbedding(dim)
        self.label_embed = nn.Embedding(N_CLASSES + 1, dim)

        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.final = FinalLayer(dim, patch_size, in_ch)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        p, g, c = self.patch_size, self.grid, self.in_ch
        x = x.reshape(b, g, g, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, g, p, g, p)
        return x.reshape(b, c, g * p, g * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        tok = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, dim)
        tok = tok + self.pos_embed
        c = self.time_embed(t) + self.label_embed(y)
        for blk in self.blocks:
            tok = blk(tok, c)
        out = self.final(tok, c)  # (B, N, patch*patch*C)
        return self.unpatchify(out)


# --------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description="CIFAR-10 DiT (Diffusion Transformer) Flow Matching")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--p-uncond", type=float, default=0.1)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--no-flip-aug", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", type=str, default="velocity_cifar10_dit.pt")
    p.add_argument("--out-suffix", type=str, default="_dit")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--benchmark-steps", type=int, default=0)
    p.add_argument("--tensorboard", action="store_true", help="TensorBoard に loss をリアルタイム記録する")
    p.add_argument("--t-dist", type=str, default="uniform", choices=["uniform", "logit_normal"])
    p.add_argument("--logit-normal-mean", type=float, default=0.0)
    p.add_argument("--logit-normal-std", type=float, default=1.0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    images, labels = load_cifar_tensor(device)
    print(f"CIFAR-10: {images.shape[0]} images, {images.shape[2]}x{images.shape[3]}x{images.shape[1]}")

    model = DiTVelocity(patch_size=args.patch_size, dim=args.dim, depth=args.depth,
                         num_heads=args.num_heads).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M  (dim={args.dim}, depth={args.depth}, "
          f"heads={args.num_heads}, patch={args.patch_size})")

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
        for target_min in (30, 60, 82, 120):
            print(f"  -> steps that fit in {target_min} min: {int(target_min*60/sec_per_step)}")
        return

    ckpt = CKPT_DIR / args.ckpt
    if ckpt.exists() and not args.retrain:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"loaded checkpoint: {ckpt}")
    else:
        writer, _ = make_tb_writer("dit_" + args.out_suffix.lstrip("_")) if args.tensorboard else (None, None)
        ema = EMA(model, decay=args.ema_decay)
        train(model, images, labels, args.steps, args.batch_size, args.lr, args.p_uncond, device,
              ema=ema, flip_aug=not args.no_flip_aug, writer=writer, t_dist=args.t_dist,
              logit_normal_mean=args.logit_normal_mean, logit_normal_std=args.logit_normal_std)
        if writer is not None:
            writer.close()
        ema.copy_to(model)
        torch.save(model.state_dict(), ckpt)
        print(f"saved EMA checkpoint: {ckpt}")

    make_figures(model, device, OUT_DIR, suffix=args.out_suffix)


if __name__ == "__main__":
    main()
