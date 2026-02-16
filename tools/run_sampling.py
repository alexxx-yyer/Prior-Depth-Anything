#!/usr/bin/env python3
"""
仅做稀疏采样：从 RGB + 稠密 prior 深度 生成指定 pattern 的稀疏深度，并保存。

支持的 pattern（与 sparse_sampler.get_sparse_depth 一致）：
  - 数字（如 1000）     : 随机保留 N 个有效 prior 点
  - sift / orb         : SfM 风格，用特征点位置采样
  - LiDAR_N（如 LiDAR_8）: N 线 LiDAR 模拟
  - downsample_ / downsample_N : 降采样网格（down_fill_mode=linear 时无需 completion 模型）

用法（在 Prior-Depth-Anything 目录下）：
  # 安装后 CLI 推理时顺带采样（会跑完整模型）：
  priorda test --image_path assets/sample-1/rgb.jpg --prior_path assets/sample-1/gt_depth.png --pattern 1000 --visualize 1

  # 仅采样、不跑模型（本脚本）：
  # 单个 pattern
  python run_sampling.py --image_path assets/sample-1/rgb.jpg --prior_path assets/sample-1/gt_depth.png --pattern 1000 --out_dir out_sampling
  # 多个 pattern，每个单独保存
  python run_sampling.py --image_path assets/sample-1/rgb.jpg --prior_path assets/sample-1/gt_depth.png --pattern LiDAR_8 sift 1000 downsample_4 --out_dir out_sampling
  # 总图模式：2 行 x 3 列 (RGB, GT_depth, LiDAR_8 | 100pts, sift, 8xdownsample)，并分别保存每张子图
  python run_sampling.py --image_path ... --prior_path ... --out_dir sample --summary
"""

import argparse
import os
import numpy as np
import torch
import cv2
from PIL import Image
from matplotlib.colors import Normalize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 仅做采样不需要加载完整 PriorDepthAnything，只用到 SparseSampler
from prior_depth_anything.sparse_sampler import SparseSampler


def render_depth(depth_np: np.ndarray, mask: np.ndarray, vmin: float, vmax: float,
                 dilate: int = 0) -> np.ndarray:
    """与 eval/vis_pattern.py 一致：深度 colormap，无效为黑；dilate>0 时膨胀稀疏点便于可见。"""
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    H, W = depth_np.shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    if not mask.any():
        return canvas
    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        depth_dilated = cv2.dilate(depth_np.astype(np.float32), kernel)
        mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
        vals = np.where(mask_dilated, depth_dilated, 0)
        colored = (plt.cm.turbo(norm(vals))[:, :, :3] * 255).astype(np.uint8)
        canvas[mask_dilated] = colored[mask_dilated]
    else:
        vals = np.where(mask, depth_np, 0)
        colored = (plt.cm.turbo(norm(vals))[:, :, :3] * 255).astype(np.uint8)
        canvas[mask] = colored[mask]
    return canvas


def auto_dilate(n_points: int, H: int, W: int) -> int:
    """稀疏时适当膨胀半径，与 vis_pattern 一致。"""
    if n_points <= 0:
        return 0
    target_area = 0.005 * H * W
    r = max(1, int(np.sqrt(target_area / (n_points * np.pi))))
    return min(r, 15)


def load_image(path: str) -> np.ndarray:
    """[H, W, 3] uint8."""
    if path.endswith('.npy'):
        x = np.load(path)
    else:
        x = np.asarray(Image.open(path))
    if x.ndim == 2:
        x = np.stack([x] * 3, axis=-1)
    return x


def load_prior(path: str) -> torch.Tensor:
    """[H, W] float32."""
    if path.endswith('.npy'):
        x = np.load(path).astype(np.float32)
    else:
        x = np.asarray(Image.open(path)).astype(np.float32)
    return torch.from_numpy(x)


# benchmark 固定 pattern
BENCHMARK_PATTERNS = ['LiDAR_32', 'sift', '100', 'downsample_8']


def run_sampling_from_args(args):
    """根据已解析的 args 执行稀疏采样（可供 run_sampling_benchmark 等调用）。"""
    os.makedirs(args.out_dir, exist_ok=True)

    image = load_image(args.image_path)
    prior = load_prior(args.prior_path)
    # prior 与 image 尺寸一致
    if prior.shape[:2] != image.shape[:2]:
        h, w = image.shape[0], image.shape[1]
        prior = torch.nn.functional.interpolate(
            prior.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest-exact'
        ).squeeze()

    base = os.path.splitext(os.path.basename(args.prior_path))[0]
    prior_np = prior.cpu().numpy()
    valid = prior_np > 1e-4
    vmin = float(np.percentile(prior_np[valid], 2)) if valid.any() else 0.0
    vmax = float(np.percentile(prior_np[valid], 98)) if valid.any() else 1.0
    H, W = prior_np.shape

    # completion=None：仅采样不跑深度补全；downsample 时请用 down_fill_mode=linear
    sampler = SparseSampler(device=args.device, completion=None)

    # ---------- 总图模式：2 行 x 3 列 (RGB, GT_depth, LiDAR_8 | 100, sift, 8xdownsample)，并分别保存 ----------
    if args.summary:
        summary_patterns = ['LiDAR_8', '100', 'sift', 'downscale_8']
        panels = []  # (title, img_rgb_array, n_pts or None)

        # 1. RGB
        panels.append(('RGB', image.copy(), None))
        # 2. GT_depth
        gt_vis = render_depth(prior_np, valid, vmin, vmax, dilate=0)
        panels.append(('GT_depth', gt_vis, None))

        for pat in summary_patterns:
            try:
                data = sampler(image=image, prior=prior, pattern=pat, down_fill_mode=args.down_fill_mode)
            except Exception as e:
                print(f'[SKIP] {pat}: {e}')
                panels.append((pat, np.zeros((H, W, 3), dtype=np.uint8), 0))
                continue
            sparse_depth = data['sparse_depth'].squeeze().cpu().numpy()
            sparse_mask = data['sparse_mask'].squeeze().cpu().numpy()
            n_pts = int(sparse_mask.sum())
            dilate_r = auto_dilate(n_pts, H, W)
            depth_vis = render_depth(sparse_depth, sparse_mask, vmin, vmax, dilate=dilate_r)
            panels.append((pat, depth_vis, n_pts))

        # 分别保存每张子图
        for title, img, n_pts in panels:
            pat_safe = title.replace('/', '_')
            if title == 'RGB':
                path = os.path.join(args.out_dir, 'rgb.png')
                Image.fromarray(img).save(path)
            elif title == 'GT_depth':
                path = os.path.join(args.out_dir, 'gt_depth.png')
                Image.fromarray(img).save(path)
            else:
                path = os.path.join(args.out_dir, f'prior_vis_{pat_safe}.png')
                fig, ax = plt.subplots(1, 1, figsize=(6, 6 * H / W))
                ax.imshow(img)
                ax.set_title(f'{title} ({n_pts} pts)', fontsize=12, color='white')
                ax.axis('off')
                ax.set_facecolor('black')
                fig.patch.set_facecolor('black')
                plt.tight_layout()
                plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='black')
                plt.close()
            print(f'  保存: {path}')

        # 保存 2x3 总图
        fig, axes = plt.subplots(2, 3, figsize=(12, 8 * H / W))
        fig.patch.set_facecolor('black')
        for idx, (title, img, n_pts) in enumerate(panels):
            ax = axes.flatten()[idx]
            ax.imshow(img)
            ax.set_title(title if n_pts is None else f'{title} ({n_pts} pts)', fontsize=10, color='white')
            ax.axis('off')
            ax.set_facecolor('black')
        plt.tight_layout()
        summary_path = os.path.join(args.out_dir, 'vis_summary.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
        print(f'总图: {summary_path}')
        return

    # ---------- 按 pattern 逐个采样并保存 ----------
    # RGB 与 GT (prior) 也保存
    if args.save_vis:
        rgb_path = os.path.join(args.out_dir, 'rgb.png')
        Image.fromarray(image).save(rgb_path)
        print(f'  RGB: {rgb_path}')
        gt_vis = render_depth(prior_np, valid, vmin, vmax, dilate=0)
        gt_vis_path = os.path.join(args.out_dir, f'{base}_prior_vis_gt.png')
        fig, ax = plt.subplots(1, 1, figsize=(6, 6 * H / W))
        ax.imshow(gt_vis)
        ax.set_title('GT depth', fontsize=12, color='white')
        ax.axis('off')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        plt.tight_layout()
        plt.savefig(gt_vis_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
        print(f'  GT 可视化: {gt_vis_path}')

    for pattern in args.pattern:
        pat_safe = pattern.replace('/', '_')
        try:
            data = sampler(
                image=image,
                prior=prior,
                pattern=pattern,
                down_fill_mode=args.down_fill_mode,
            )
        except Exception as e:
            print(f'[SKIP] {pattern}: {e}')
            continue

        sparse_depth = data['sparse_depth'].squeeze().cpu().numpy()
        sparse_mask = data['sparse_mask'].squeeze().cpu().numpy()
        n_pts = int(sparse_mask.sum())

        if args.save_npy:
            out_sparse = os.path.join(args.out_dir, f'{base}_sparse_{pat_safe}.npy')
            out_mask = os.path.join(args.out_dir, f'{base}_mask_{pat_safe}.npy')
            np.save(out_sparse, sparse_depth)
            np.save(out_mask, sparse_mask)
            print(f'{pattern}: 稀疏深度 {out_sparse}, 掩码 {out_mask}, 有效点数 {n_pts}')
        else:
            print(f'{pattern}: 有效点数 {n_pts}')

        if args.save_vis:
            # 与 eval/vis_pattern.png 一致：turbo 深度图、无效黑、稀疏点膨胀
            dilate_r = auto_dilate(n_pts, H, W)
            depth_vis = render_depth(sparse_depth, sparse_mask, vmin, vmax, dilate=dilate_r)
            vis_path = os.path.join(args.out_dir, f'{base}_prior_vis_{pat_safe}.png')
            fig, ax = plt.subplots(1, 1, figsize=(6, 6 * H / W))
            ax.imshow(depth_vis)
            ax.set_title(f'{pattern} ({n_pts} pts)', fontsize=12)
            ax.axis('off')
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            ax.title.set_color('white')
            plt.tight_layout()
            plt.savefig(vis_path, dpi=150, bbox_inches='tight', facecolor='black')
            plt.close()
            print(f'  可视化: {vis_path}')


def main():
    parser = argparse.ArgumentParser(description='Prior-Depth-Anything 仅稀疏采样（不跑深度补全模型）')
    parser.add_argument('--image_path', required=True, help='RGB 图像路径，如 assets/sample-1/rgb.jpg')
    parser.add_argument('--prior_path', required=True, help='Prior 深度图路径（稠密或已有稀疏），如 assets/sample-1/gt_depth.png')
    parser.add_argument('--pattern', nargs='+', default=None,
                        help='一个或多个 pattern，每个单独保存；与 --summary 二选一')
    parser.add_argument('--summary', action='store_true',
                        help='总图模式：2x3 (RGB, GT_depth, LiDAR_8, 100pts, sift, 8xdownsample)，并分别保存每张')
    parser.add_argument('--out_dir', default='out_sampling', help='输出目录')
    parser.add_argument('--down_fill_mode', default='linear',
                        choices=('linear', 'knn', 'global'),
                        help='仅 downsample_* 时有效；选 knn/global 需完整模型，此处用 linear')
    parser.add_argument('--device', default='cpu', help='cpu 或 cuda:0')
    parser.add_argument('--save_vis', type=int, default=1, help='是否保存稀疏深度可视化图')
    parser.add_argument('--save_npy', type=int, default=0, help='是否保存 .npy 稀疏深度/掩码（默认仅保存可视化）')
    args = parser.parse_args()

    if not args.summary and not args.pattern:
        parser.error('请指定 --pattern 或 --summary')

    run_sampling_from_args(args)


if __name__ == '__main__':
    main()
