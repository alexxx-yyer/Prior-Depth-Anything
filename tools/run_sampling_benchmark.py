#!/usr/bin/env python3
"""
Benchmark 场景的固定结构采样：只需传入场景目录，自动使用 image.jpg / depth.png / sample/。

目录结构约定：
  {scene_dir}/
    image.jpg   # RGB
    depth.png   # 稠密 prior 深度
    sample/     # 输出目录（自动创建）

固定 pattern：LiDAR_8, sift, 1000, downsample_4

用法（在 Prior-Depth-Anything 目录下）：
  # 单场景
  python run_sampling_benchmark.py ../datasets/eval/benchmark/DDAD/val/000007/CAMERA_06/

  # 多场景
  python run_sampling_benchmark.py ../datasets/eval/benchmark/DDAD/val/000007/CAMERA_06/ ../datasets/eval/benchmark/DDAD/val/000008/CAMERA_06/

  # 同时保存 2x3 总图 vis_summary.png
  python run_sampling_benchmark.py --summary ../datasets/eval/benchmark/DDAD/val/000007/CAMERA_06/
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from run_sampling import BENCHMARK_PATTERNS, run_sampling_from_args


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark 场景稀疏采样：传入场景目录，使用固定结构 image.jpg / depth.png / sample/'
    )
    parser.add_argument(
        'scene_dirs',
        nargs='+',
        help='一个或多个场景目录，如 ../datasets/eval/benchmark/DDAD/val/000007/CAMERA_06/',
    )
    parser.add_argument('--summary', action='store_true', help='额外保存 2x3 总图 vis_summary.png')
    parser.add_argument('--device', default='cpu', help='cpu 或 cuda:0')
    parser.add_argument('--save_vis', type=int, default=1, help='是否保存稀疏深度可视化图')
    parser.add_argument('--save_npy', type=int, default=0, help='是否保存 .npy 稀疏深度/掩码')
    args = parser.parse_args()

    for scene_dir in args.scene_dirs:
        scene_dir = os.path.normpath(os.path.abspath(scene_dir))
        image_path = os.path.join(scene_dir, 'image.jpg')
        prior_path = os.path.join(scene_dir, 'depth.png')
        out_dir = os.path.join(scene_dir, 'sample')

        if not os.path.isfile(image_path):
            print(f'[SKIP] 无图像: {image_path}')
            continue
        if not os.path.isfile(prior_path):
            print(f'[SKIP] 无深度: {prior_path}')
            continue

        print(f'\n--- {scene_dir} ---')
        bench_args = argparse.Namespace(
            image_path=image_path,
            prior_path=prior_path,
            pattern=BENCHMARK_PATTERNS,
            summary=False,
            out_dir=out_dir,
            down_fill_mode='linear',
            device=args.device,
            save_vis=args.save_vis,
            save_npy=args.save_npy,
        )
        run_sampling_from_args(bench_args)

        if args.summary and args.save_vis:
            base = os.path.splitext(os.path.basename(prior_path))[0]
            titles = ['RGB', 'GT_depth'] + BENCHMARK_PATTERNS
            names = ['rgb.png', f'{base}_prior_vis_gt.png'] + [f'{base}_prior_vis_{p}.png' for p in BENCHMARK_PATTERNS]
            paths = [os.path.join(out_dir, n) for n in names]
            if all(os.path.isfile(p) for p in paths):
                imgs = [plt.imread(p) for p in paths]
                if imgs[0].ndim == 2:
                    imgs = [plt.cm.gray(g)[:, :, :3] if g.ndim == 2 else g for g in imgs]
                fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                fig.patch.set_facecolor('black')
                for ax, title, img in zip(axes.flatten(), titles, imgs):
                    ax.imshow(img)
                    ax.set_title(title, fontsize=10, color='white')
                    ax.axis('off')
                    ax.set_facecolor('black')
                plt.tight_layout()
                summary_path = os.path.join(out_dir, 'vis_summary.png')
                plt.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='black')
                plt.close()
                print(f'  总图: {summary_path}')
            else:
                print('[SKIP] --summary 需要先有各可视化图，请保证 --save_vis 1')


if __name__ == '__main__':
    main()
