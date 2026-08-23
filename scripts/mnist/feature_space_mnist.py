"""MNIST (784 次元) の速度場を低次元に写像して眺める。

なぜ PCA と t-SNE で扱いが違うか
--------------------------------
PCA は線形写像 y = P(x - mean) (P: 784 -> 2 の直交射影)。線形なので

    dy/dt = P (dx/dt) = P v_theta(x, t)

が厳密に成り立つ。つまり速度ベクトルをそのまま射影でき、2D トイデータの
ときと同じ streamplot / quiver を「ちゃんとした矢印場」として描ける。
さらに P の疑似逆写像 (x ≈ mean + y @ P^T) でグリッド点を画像へ持ち上げ、
そこでモデルを評価すれば、PCA 平面全体の場を可視化できる (固有digit的な絵)。

t-SNE は非線形なので上の関係が成り立たず、速度ベクトルを射影する数学的
根拠がない。できるのは「実際の軌跡上の点」を埋め込んで位置を追うことだけ
(矢印場は作れない)。ただし可視化としては情報量が多く、ノイズが強い
序盤 (t が小さい) はクラスに関係なく団子状に固まり、t が進むとクラスごとに
分離していく様子がよく見える。

出力 (静止画):
  pca_trajectories_mnist.png   実データ (PCA 平面) + 生成軌跡 + 生成終点
  pca_field_mnist.png          PCA 平面上に持ち上げた画像で評価した速度場 (streamplot)
  tsne_trajectories_mnist.png  t-SNE 平面上の軌跡 (矢印場なし、位置のみ)

出力 (GIF, `--gif` 指定時):
  pca_trajectories_mnist.gif   軌跡が実際に動く様子 (trace_points.gif の MNIST 版)
  pca_field_mnist.gif          速度場 (quiver) が t とともに変化する様子
  tsne_trajectories_mnist.gif  t-SNE 平面上を軌跡が動く様子

実行 (checkpoints/velocity_mnist.pt が必要):
    python feature_space_mnist.py            # 静止画のみ
    python feature_space_mnist.py --gif       # GIF も作る
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

FPS = 15
HOLD_FRAMES = 15  # 最後で少し止めてループを見やすくする

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
OUT_DIR = ROOT / "out" / "mnist"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(device):
    ckpt = CKPT_DIR / "velocity_mnist.pt"
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} がありません。先に `python flow_matching_mnist.py` を実行してください。")
    model = UNetVelocity().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


# ------------------------------------------------------------------ PCA
def fit_pca(images: torch.Tensor, n_components: int = 2):
    """images: (N,1,28,28). 戻り値: mean (784,), components (784, n_components) 直交列。"""
    x = images.reshape(images.shape[0], -1)
    mean = x.mean(dim=0)
    xc = x - mean
    _, _, v = torch.pca_lowrank(xc, q=n_components + 4, niter=4)
    return mean, v[:, :n_components]


def pca_project(images: torch.Tensor, mean: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    x = images.reshape(images.shape[0], -1)
    return (x - mean) @ components


def pca_lift(y: torch.Tensor, mean: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    """y: (N,2) -> 784 次元へ持ち上げ、画像形状 (N,1,28,28) にして返す。"""
    x = mean[None, :] + y @ components.T
    return x.reshape(-1, 1, 28, 28)


# ------------------------------------------------------------------- ODE
@torch.no_grad()
def integrate(model, labels: torch.Tensor, steps: int, guidance: float, device, seed: int) -> torch.Tensor:
    """midpoint 法。各時刻の画像 (steps+1, N, 1, 28, 28) を返す。"""
    torch.manual_seed(seed)
    n = labels.shape[0]
    x = torch.randn(n, 1, 28, 28, device=device)
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


@torch.no_grad()
def velocity_at(model, x: torch.Tensor, t: float, label: int, guidance: float, device) -> torch.Tensor:
    n = x.shape[0]
    y = torch.full((n,), label, device=device)
    tt = torch.full((n,), t, device=device)
    v_cond = model(x, tt, y)
    if guidance == 0.0:
        return v_cond
    y_null = torch.full_like(y, NULL_LABEL)
    v_uncond = model(x, tt, y_null)
    return v_uncond + (1.0 + guidance) * (v_cond - v_uncond)


# ------------------------------------------------------------- 図1: 軌跡
def make_pca_trajectories(model, real_images, real_labels, mean, comps, device, out_path: Path,
                           classes=(0, 3, 6, 8), per_class: int = 4, steps: int = 60, guidance: float = 1.0):
    real_2d = pca_project(real_images, mean, comps)

    fig, ax = plt.subplots(figsize=(8, 7.5))
    sc = ax.scatter(real_2d[:, 0], real_2d[:, 1], c=real_labels, cmap="tab10", s=4, alpha=0.25, vmin=0, vmax=9)
    cbar = plt.colorbar(sc, ax=ax, ticks=range(10), fraction=0.046)
    cbar.set_label("real digit label")

    cmap = plt.get_cmap("tab10")
    for cls in classes:
        labels = torch.full((per_class,), cls)
        traj = integrate(model, labels, steps, guidance, device, seed=cls).cpu()  # (T+1, per_class, 1,28,28)
        for k in range(per_class):
            path2d = pca_project(traj[:, k], mean, comps)
            ax.plot(path2d[:, 0], path2d[:, 1], color=cmap(cls), lw=1.2, alpha=0.85,
                     label=f"class {cls}" if k == 0 else None)
            ax.scatter(*path2d[0], color=cmap(cls), marker="o", s=30, edgecolor="white", zorder=3)
            ax.scatter(*path2d[-1], color=cmap(cls), marker="X", s=45, edgecolor="white", zorder=3)

    ax.legend(loc="upper right", fontsize=8, title="generation trajectory\n○=noise($t$=0)  ✕=generated($t$=1)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of MNIST pixel space: real digits (background) + generation trajectories")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved: {out_path}")


def make_pca_trajectories_gif(model, real_images, real_labels, mean, comps, device, out_path: Path,
                               classes=(0, 3, 6, 8), per_class: int = 4, steps: int = 60, guidance: float = 1.0):
    """`make_pca_trajectories` と同じ軌跡を、t: 0->1 で実際に動かして見せる版。"""
    real_2d = pca_project(real_images, mean, comps)
    cmap = plt.get_cmap("tab10")

    # 全クラス分の軌跡をまとめて (T+1, N, 2) に投影しておく
    all_paths = []
    for cls in classes:
        labels = torch.full((per_class,), cls)
        traj = integrate(model, labels, steps, guidance, device, seed=cls).cpu()
        for k in range(per_class):
            all_paths.append(pca_project(traj[:, k], mean, comps))
    all_paths = torch.stack(all_paths, dim=1)  # (T+1, N_total, 2)
    n_total = all_paths.shape[1]
    colors = [cmap(cls) for cls in classes for _ in range(per_class)]

    fig, ax = plt.subplots(figsize=(8, 7.5))
    sc = ax.scatter(real_2d[:, 0], real_2d[:, 1], c=real_labels, cmap="tab10", s=4, alpha=0.2, vmin=0, vmax=9)
    plt.colorbar(sc, ax=ax, ticks=range(10), fraction=0.046, label="real digit label")

    tail_lines = [ax.plot([], [], color=colors[k], lw=1.2, alpha=0.85)[0] for k in range(n_total)]
    dots = ax.scatter(all_paths[0, :, 0], all_paths[0, :, 1], c=colors, s=30, edgecolor="white", zorder=3)
    time_text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=11)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA plane: real digits (background) + generation trajectories in motion")

    def update(i: int):
        idx = min(i, steps)
        dots.set_offsets(all_paths[idx].numpy())
        for k, line in enumerate(tail_lines):
            line.set_data(all_paths[: idx + 1, k, 0].numpy(), all_paths[: idx + 1, k, 1].numpy())
        time_text.set_text(f"$t$ = {idx / steps:.2f}")
        return [dots, *tail_lines, time_text]

    frames = list(range(steps + 1)) + [steps] * HOLD_FRAMES
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // FPS, blit=False)
    fig.tight_layout()
    anim.save(out_path, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"saved: {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")


def make_pca_trajectories_with_images(model, real_images, real_labels, mean, comps, device, out_path: Path,
                                       classes=(0, 3, 6, 8), steps: int = 60, guidance: float = 1.0,
                                       thumb_marks=(0, 10, 20, 30, 45, 60)):
    """サンプル数を減らす代わりに、各サンプルが PCA 平面上をどう動くかに加えて
    その時点の実画像がどう変化しているかをフィルムストリップとして併記する。

    上段: PCA 平面上の軌跡 (実データを背景に、選んだ時刻に丸印)
    下段: サンプルごとに 1 行、選んだ時刻ごとの実画像を並べる (行の枠色が軌跡の色と対応)
    """
    real_2d = pca_project(real_images, mean, comps)
    cmap = plt.get_cmap("tab10")
    n_samples = len(classes)
    thumb_marks = [m for m in thumb_marks if m <= steps]

    fig = plt.figure(figsize=(2.0 * len(thumb_marks) + 3, 7.5 + 1.7 * n_samples))
    gs = fig.add_gridspec(n_samples + 3, len(thumb_marks), height_ratios=[1] * 3 + [1] * n_samples)
    ax_main = fig.add_subplot(gs[0:3, :])

    sc = ax_main.scatter(real_2d[:, 0], real_2d[:, 1], c=real_labels, cmap="tab10", s=4, alpha=0.22, vmin=0, vmax=9)
    plt.colorbar(sc, ax=ax_main, ticks=range(10), fraction=0.03, pad=0.01, label="real digit label")

    trajs = []
    for cls in classes:
        traj = integrate(model, torch.full((1,), cls), steps, guidance, device, seed=cls).cpu()  # (T+1,1,1,28,28)
        trajs.append(traj)
        path2d = pca_project(traj[:, 0], mean, comps)
        color = cmap(cls)
        ax_main.plot(path2d[:, 0], path2d[:, 1], color=color, lw=1.6, alpha=0.9, label=f"class {cls}")
        ax_main.scatter(*path2d[0], color=color, marker="o", s=45, edgecolor="white", zorder=4)
        ax_main.scatter(*path2d[-1], color=color, marker="X", s=65, edgecolor="white", zorder=4)
        for j, m in enumerate(thumb_marks):
            ax_main.scatter(*path2d[m], color=color, marker="s", s=28, edgecolor="white", zorder=5)
            ax_main.annotate(str(j), path2d[m], fontsize=7, color="white", ha="center", va="center", zorder=6)

    ax_main.legend(loc="upper right", fontsize=8,
                   title="○=$t$=0 (noise)  ✕=$t$=1 (generated)\n□=numbered marks (see thumbnails below)")
    ax_main.set_xlabel("PC1")
    ax_main.set_ylabel("PC2")
    ax_main.set_title("PCA plane: trajectory position (top) vs. actual image at that moment (bottom)")

    for row, (cls, traj) in enumerate(zip(classes, trajs)):
        color = cmap(cls)
        for col, m in enumerate(thumb_marks):
            ax = fig.add_subplot(gs[3 + row, col])
            ax.imshow(to_img(traj[m, 0, 0]).numpy(), cmap="gray", vmin=0, vmax=1)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"[{col}] $t$={m/steps:.2f}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"class {cls}", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved: {out_path}")


def make_pca_trajectories_with_images_gif(model, real_images, real_labels, mean, comps, device, out_path: Path,
                                           classes=(0, 3, 6, 8), steps: int = 60, guidance: float = 1.0):
    """`make_pca_trajectories_with_images` の静止画 (フィルムストリップ) の代わりに、
    上段の PCA 平面上の動きと、下段の「今の画像」をリアルタイムで同期させて動かす版。
    """
    real_2d = pca_project(real_images, mean, comps)
    cmap = plt.get_cmap("tab10")
    n_samples = len(classes)

    trajs, paths = [], []
    for cls in classes:
        traj = integrate(model, torch.full((1,), cls), steps, guidance, device, seed=cls).cpu()  # (T+1,1,1,28,28)
        trajs.append(traj)
        paths.append(pca_project(traj[:, 0], mean, comps))
    paths = torch.stack(paths, dim=1)  # (T+1, n_samples, 2)
    colors = [cmap(cls) for cls in classes]

    fig = plt.figure(figsize=(8, 10.5))
    gs = fig.add_gridspec(4, n_samples, height_ratios=[3, 3, 3, 1.6])
    ax_main = fig.add_subplot(gs[0:3, :])

    sc = ax_main.scatter(real_2d[:, 0], real_2d[:, 1], c=real_labels, cmap="tab10", s=4, alpha=0.22, vmin=0, vmax=9)
    plt.colorbar(sc, ax=ax_main, ticks=range(10), fraction=0.03, pad=0.01, label="real digit label")

    tail_lines = [ax_main.plot([], [], color=colors[k], lw=1.6, alpha=0.9,
                                label=f"class {classes[k]}")[0] for k in range(n_samples)]
    dots = ax_main.scatter(paths[0, :, 0], paths[0, :, 1], c=colors, s=55, edgecolor="white", zorder=4)
    time_text = ax_main.text(0.02, 0.97, "", transform=ax_main.transAxes, va="top", fontsize=12)

    ax_main.legend(loc="upper right", fontsize=8)
    ax_main.set_xlabel("PC1")
    ax_main.set_ylabel("PC2")
    ax_main.set_title("PCA plane position (top) in sync with the actual image (bottom)")

    img_axes, img_arts = [], []
    for col, cls in enumerate(classes):
        ax = fig.add_subplot(gs[3, col])
        im = ax.imshow(to_img(trajs[col][0, 0, 0]).numpy(), cmap="gray", vmin=0, vmax=1)
        for spine in ax.spines.values():
            spine.set_edgecolor(colors[col])
            spine.set_linewidth(2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"class {cls}", fontsize=9)
        img_axes.append(ax)
        img_arts.append(im)

    def update(i: int):
        idx = min(i, steps)
        dots.set_offsets(paths[idx].numpy())
        for k, line in enumerate(tail_lines):
            line.set_data(paths[: idx + 1, k, 0].numpy(), paths[: idx + 1, k, 1].numpy())
        for k, im in enumerate(img_arts):
            im.set_data(to_img(trajs[k][idx, 0, 0]).numpy())
        time_text.set_text(f"$t$ = {idx / steps:.2f}")
        return [dots, *tail_lines, *img_arts, time_text]

    frames = list(range(steps + 1)) + [steps] * HOLD_FRAMES
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // FPS, blit=False)
    fig.tight_layout()
    anim.save(out_path, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"saved: {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")


# --------------------------------------------------------------- 図2: 場
def make_pca_field(model, real_images, real_labels, mean, comps, device, out_path: Path,
                    label: int = 3, times=(0.2, 0.5, 0.8, 0.99), grid_n: int = 22, guidance: float = 1.0,
                    n_traj: int = 6, steps: int = 60):
    real_2d = pca_project(real_images, mean, comps)
    same_class = real_2d[real_labels == label]

    # グリッド範囲: そのクラスの実データ分布を少し広げた範囲にする
    lo = same_class.min(dim=0).values - 4.0
    hi = same_class.max(dim=0).values + 4.0
    gx = torch.linspace(lo[0].item(), hi[0].item(), grid_n)
    gy = torch.linspace(lo[1].item(), hi[1].item(), grid_n)
    mesh_x, mesh_y = torch.meshgrid(gx, gy, indexing="xy")
    grid_2d = torch.stack([mesh_x.flatten(), mesh_y.flatten()], dim=1)
    grid_imgs = pca_lift(grid_2d, mean, comps).to(device)

    traj_labels = torch.full((n_traj,), label)
    traj = integrate(model, traj_labels, steps, guidance, device, seed=label).cpu()

    fig, axes = plt.subplots(1, len(times), figsize=(4.3 * len(times), 4.6))
    for ax, t in zip(axes, times):
        v = velocity_at(model, grid_imgs, t, label, guidance, device).cpu()  # (G,1,28,28)
        # v は接ベクトル (位置ではない) なので mean は引かず、そのまま P を掛けるだけでよい
        v_2d = v.reshape(v.shape[0], -1) @ comps
        u = v_2d[:, 0].reshape(grid_n, grid_n).numpy()
        w = v_2d[:, 1].reshape(grid_n, grid_n).numpy()
        speed = (u**2 + w**2) ** 0.5

        ax.scatter(same_class[:, 0], same_class[:, 1], s=4, alpha=0.15, color="gray", zorder=0)
        ax.streamplot(gx.numpy(), gy.numpy(), u, w, color=speed, cmap="viridis", density=1.1,
                      linewidth=0.8, arrowsize=0.8)

        near = (traj_labels == label).nonzero(as_tuple=True)[0]
        step_idx = min(int(t * steps), steps)
        pts = pca_project(traj[step_idx, near], mean, comps)
        ax.scatter(pts[:, 0], pts[:, 1], color="red", s=25, zorder=4, edgecolor="white",
                    label="actual samples at this $t$" if t == times[0] else None)

        ax.set_title(f"$t$={t}")
        ax.set_xlim(lo[0].item(), hi[0].item())
        ax.set_ylim(lo[1].item(), hi[1].item())

    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"velocity field $v_\\theta(x,t)$ projected onto PCA plane (class={label}, grid lifted via inverse PCA)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved: {out_path}")


def make_pca_field_gif(model, real_images, real_labels, mean, comps, device, out_path: Path,
                        label: int = 3, grid_n: int = 22, guidance: float = 1.0,
                        n_traj: int = 6, steps: int = 60):
    """`make_pca_field` の streamplot は毎フレーム再計算すると重いので、
    アニメーションでは quiver (矢印) に切り替えて t とともに動かす。
    """
    real_2d = pca_project(real_images, mean, comps)
    same_class = real_2d[real_labels == label]

    lo = same_class.min(dim=0).values - 4.0
    hi = same_class.max(dim=0).values + 4.0
    gx = torch.linspace(lo[0].item(), hi[0].item(), grid_n)
    gy = torch.linspace(lo[1].item(), hi[1].item(), grid_n)
    mesh_x, mesh_y = torch.meshgrid(gx, gy, indexing="xy")
    grid_2d = torch.stack([mesh_x.flatten(), mesh_y.flatten()], dim=1)
    grid_imgs = pca_lift(grid_2d, mean, comps).to(device)

    traj_labels = torch.full((n_traj,), label)
    traj = integrate(model, traj_labels, steps, guidance, device, seed=label).cpu()
    traj_2d = torch.stack([pca_project(traj[i], mean, comps) for i in range(traj.shape[0])])  # (T+1,n_traj,2)

    # グリッド間隔の 8 割くらいの長さで矢印を揃える (scale_units="xy" なら
    # U=1 のとき描画長 = 1/scale データ単位になる)
    spacing = ((hi - lo) / (grid_n - 1)).mean().item()
    arrow_len = spacing * 0.8

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(same_class[:, 0], same_class[:, 1], s=4, alpha=0.15, color="gray", zorder=0)
    # 実データから離れた領域 (特に t->1) は訓練信号が乏しく速度の「大きさ」が
    # 暴れやすい (field_stability.py の議論と同じ)。方向は意味があるので、
    # 矢印は向きだけ揃えて長さ一定にし、大きさは色 (log スケール) で表す。
    quiver = ax.quiver(grid_2d[:, 0].numpy(), grid_2d[:, 1].numpy(),
                        torch.zeros(grid_2d.shape[0]), torch.zeros(grid_2d.shape[0]),
                        torch.zeros(grid_2d.shape[0]),
                        cmap="viridis", alpha=0.85, scale=1.0 / arrow_len, scale_units="xy",
                        pivot="mid", width=0.004)
    quiver.set_clim(0, 1)
    dots = ax.scatter(traj_2d[0, :, 0], traj_2d[0, :, 1], color="red", s=30, edgecolor="white", zorder=4)
    time_text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=11)
    fig.colorbar(quiver, ax=ax, fraction=0.046, label="log-scaled speed (direction is exact; length is normalized)")

    ax.set_xlim(lo[0].item(), hi[0].item())
    ax.set_ylim(lo[1].item(), hi[1].item())
    ax.set_title(f"velocity field direction on PCA plane over time (class={label})")

    def update(i: int):
        idx = min(i, steps)
        t = idx / steps
        v = velocity_at(model, grid_imgs, t, label, guidance, device).cpu()
        v_2d = v.reshape(v.shape[0], -1) @ comps
        speed = v_2d.norm(dim=1)
        unit = v_2d / speed.clamp_min(1e-6)[:, None]
        log_speed = torch.log1p(speed)
        log_speed_norm = log_speed / log_speed.max().clamp_min(1e-6)
        quiver.set_UVC(unit[:, 0], unit[:, 1], log_speed_norm)
        dots.set_offsets(traj_2d[idx].numpy())
        time_text.set_text(f"$t$ = {t:.2f}")
        return [quiver, dots, time_text]

    frames = list(range(steps + 1)) + [steps] * HOLD_FRAMES
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // FPS, blit=False)
    fig.tight_layout()
    anim.save(out_path, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"saved: {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")


# ------------------------------------------------------------- 図3: t-SNE
def make_tsne_trajectories(model, real_images, real_labels, device, out_path: Path,
                            classes=(0, 3, 6, 8), per_class: int = 3, steps: int = 60,
                            t_marks=(0, 5, 10, 15, 20, 30, 45, 60), guidance: float = 1.0,
                            n_real_ref: int = 600, seed: int = 0):
    from sklearn.manifold import TSNE

    torch.manual_seed(seed)
    idx = torch.randperm(real_images.shape[0])[:n_real_ref]
    ref_imgs = real_images[idx].cpu()
    ref_labels = real_labels[idx].cpu()

    traj_points, traj_meta = [], []  # meta: (class, sample_idx, t_index)
    for cls in classes:
        labels = torch.full((per_class,), cls)
        traj = integrate(model, labels, steps, guidance, device, seed=cls + 1000).cpu()
        for k in range(per_class):
            for ti in t_marks:
                traj_points.append(traj[ti, k].reshape(-1))
                traj_meta.append((cls, k, ti))

    traj_points = torch.stack(traj_points)
    all_points = torch.cat([ref_imgs.reshape(n_real_ref, -1), traj_points], dim=0).numpy()

    print(f"[t-SNE] embedding {all_points.shape[0]} points in {all_points.shape[1]} dims ...")
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(all_points)
    ref_emb = emb[:n_real_ref]
    traj_emb = emb[n_real_ref:]

    fig, ax = plt.subplots(figsize=(8, 7.5))
    sc = ax.scatter(ref_emb[:, 0], ref_emb[:, 1], c=ref_labels.numpy(), cmap="tab10", s=8, alpha=0.35, vmin=0, vmax=9)
    plt.colorbar(sc, ax=ax, ticks=range(10), fraction=0.046, label="real digit label")

    cmap = plt.get_cmap("tab10")
    by_sample: dict[tuple[int, int], list[tuple[int, float, float]]] = {}
    for (cls, k, ti), (px, py) in zip(traj_meta, traj_emb):
        by_sample.setdefault((cls, k), []).append((ti, px, py))

    for (cls, k), pts in by_sample.items():
        pts.sort(key=lambda p: p[0])
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        ax.plot(xs, ys, color=cmap(cls), lw=1.3, alpha=0.85, marker="o", markersize=3,
                 label=f"class {cls}" if k == 0 else None)
        ax.scatter(xs[-1], ys[-1], color=cmap(cls), marker="X", s=60, edgecolor="white", zorder=4)

    ax.legend(loc="upper right", fontsize=8, title="○...○✕ = $t$: 0 -> 1 (marks at\n" + str(t_marks) + f" of {steps})")
    ax.set_title("t-SNE of MNIST pixel space: real digits + generation trajectories\n(nonlinear map: positions only, no valid vector field)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved: {out_path}")


def make_tsne_trajectories_gif(model, real_images, real_labels, device, out_path: Path,
                                classes=(0, 3, 6, 8), per_class: int = 3, steps: int = 60,
                                guidance: float = 1.0, n_real_ref: int = 400, seed: int = 0,
                                frame_stride: int = 2):
    """t-SNE は非線形なので毎フレーム再埋め込みはできない (位置の連続性が保証されない)。
    そのため 1 回だけ、密な t マークで軌跡上の点を同時埋め込みしておき、
    その「固定された」2D 位置の上をアニメーションで動かす。
    """
    from sklearn.manifold import TSNE

    torch.manual_seed(seed)
    idx = torch.randperm(real_images.shape[0])[:n_real_ref]
    ref_imgs = real_images[idx].cpu()
    ref_labels = real_labels[idx].cpu()

    t_marks = list(range(0, steps + 1, frame_stride))
    if t_marks[-1] != steps:
        t_marks.append(steps)

    traj_by_sample = []  # [(cls, traj[T+1,1,28,28])]
    for cls in classes:
        labels = torch.full((per_class,), cls)
        traj = integrate(model, labels, steps, guidance, device, seed=cls + 1000).cpu()
        for k in range(per_class):
            traj_by_sample.append((cls, traj[:, k]))

    traj_points, traj_meta = [], []
    for cls, traj in traj_by_sample:
        for ti in t_marks:
            traj_points.append(traj[ti].reshape(-1))
            traj_meta.append((cls, ti))
    traj_points = torch.stack(traj_points)

    all_points = torch.cat([ref_imgs.reshape(n_real_ref, -1), traj_points], dim=0).numpy()
    print(f"[t-SNE gif] embedding {all_points.shape[0]} points ({len(t_marks)} time marks x "
          f"{len(traj_by_sample)} samples + {n_real_ref} real) ...")
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(all_points)
    ref_emb = emb[:n_real_ref]
    traj_emb = emb[n_real_ref:]

    n_samples = len(traj_by_sample)
    n_marks = len(t_marks)
    # (n_marks, n_samples, 2) に並べ直す
    paths = torch.zeros(n_marks, n_samples, 2)
    sample_index = {}
    for s_idx, (cls, _) in enumerate(traj_by_sample):
        sample_index[s_idx] = cls
    for i, (cls, ti) in enumerate(traj_meta):
        s_idx = i // n_marks
        m_idx = i % n_marks
        paths[m_idx, s_idx] = torch.from_numpy(traj_emb[i])

    cmap = plt.get_cmap("tab10")
    colors = [cmap(sample_index[s]) for s in range(n_samples)]

    fig, ax = plt.subplots(figsize=(8, 7.5))
    sc = ax.scatter(ref_emb[:, 0], ref_emb[:, 1], c=ref_labels.numpy(), cmap="tab10", s=8, alpha=0.3, vmin=0, vmax=9)
    plt.colorbar(sc, ax=ax, ticks=range(10), fraction=0.046, label="real digit label")

    tail_lines = [ax.plot([], [], color=colors[k], lw=1.3, alpha=0.85)[0] for k in range(n_samples)]
    dots = ax.scatter(paths[0, :, 0], paths[0, :, 1], c=colors, s=45, edgecolor="white", zorder=4)
    time_text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("t-SNE plane: real digits (background) + generation trajectories in motion\n"
                  "(embedding fixed once; only the dot position moves)")

    def update(i: int):
        idx = min(i, n_marks - 1)
        dots.set_offsets(paths[idx].numpy())
        for k, line in enumerate(tail_lines):
            line.set_data(paths[: idx + 1, k, 0].numpy(), paths[: idx + 1, k, 1].numpy())
        time_text.set_text(f"$t$ = {t_marks[idx] / steps:.2f}")
        return [dots, *tail_lines, time_text]

    frames = list(range(n_marks)) + [n_marks - 1] * HOLD_FRAMES
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // FPS, blit=False)
    fig.tight_layout()
    anim.save(out_path, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"saved: {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description="MNIST の速度場を PCA / t-SNE 平面に写像して可視化する")
    p.add_argument("--pca-fit-n", type=int, default=8000)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--no-tsne", action="store_true", help="t-SNE は重いので省略したい場合")
    p.add_argument("--gif", action="store_true", help="GIF アニメーションも作る")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = load_model(device)
    images, labels = load_mnist_tensor(device)

    idx = torch.randperm(images.shape[0])[: args.pca_fit_n]
    mean, comps = fit_pca(images[idx], n_components=2)
    print("PCA components fitted.")

    real_sub_idx = torch.randperm(images.shape[0])[:4000]
    real_sub = images[real_sub_idx].cpu()
    real_sub_labels = labels[real_sub_idx].cpu()
    mean_cpu, comps_cpu = mean.cpu(), comps.cpu()

    make_pca_trajectories(model, real_sub, real_sub_labels, mean_cpu, comps_cpu, device,
                           OUT_DIR / "pca_trajectories_mnist.png", guidance=args.guidance)
    make_pca_trajectories_with_images(model, real_sub, real_sub_labels, mean_cpu, comps_cpu, device,
                                       OUT_DIR / "pca_trajectories_with_images_mnist.png", guidance=args.guidance)
    make_pca_field(model, real_sub, real_sub_labels, mean_cpu, comps_cpu, device,
                    OUT_DIR / "pca_field_mnist.png", guidance=args.guidance)

    if not args.no_tsne:
        make_tsne_trajectories(model, images, labels, device, OUT_DIR / "tsne_trajectories_mnist.png",
                                guidance=args.guidance)

    if args.gif:
        make_pca_trajectories_gif(model, real_sub, real_sub_labels, mean_cpu, comps_cpu, device,
                                   OUT_DIR / "pca_trajectories_mnist.gif", guidance=args.guidance)
        make_pca_trajectories_with_images_gif(model, real_sub, real_sub_labels, mean_cpu, comps_cpu, device,
                                               OUT_DIR / "pca_trajectories_with_images_mnist.gif",
                                               guidance=args.guidance)
        make_pca_field_gif(model, real_sub, real_sub_labels, mean_cpu, comps_cpu, device,
                            OUT_DIR / "pca_field_mnist.gif", guidance=args.guidance)
        if not args.no_tsne:
            make_tsne_trajectories_gif(model, images, labels, device, OUT_DIR / "tsne_trajectories_mnist.gif",
                                        guidance=args.guidance)


if __name__ == "__main__":
    main()
