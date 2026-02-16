import argparse
import os
import torch
import numpy as np
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import torch.nn.functional as F
from PIL import Image
from prior_depth_anything.utils import (
    depth2disparity, 
    disparity2depth
)
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    
def project_point_map_to_depth_map(
    point_map: torch.Tensor, extrinsics_cam: torch.Tensor, intrinsics_cam: torch.Tensor, size
) -> torch.Tensor:
    depth_map_list = []
    for frame_idx in range(point_map.shape[0]):
        curr_depth_map = world_coords_points_to_depth(
            point_map[frame_idx], extrinsics_cam[frame_idx], intrinsics_cam[frame_idx], size
        )
        
        depth_map_list.append(curr_depth_map)
        
    depth_map_array = torch.stack(depth_map_list)
    
    return depth_map_array

def world_coords_points_to_depth(
    point_map: torch.Tensor,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    size,
    eps=1e-8,
) -> torch.Tensor:
    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3:]
    
    cam_point_map = point_map @ R.T + T.T
    points_3d_homo = cam_point_map.T
    projected_points = intrinsic @ points_3d_homo
    
    if size[0] * size[1] == point_map.shape[0]:
        depth_map = projected_points[2, :].reshape(size)
    else:
        projected_points /= projected_points[2, :]
        
        height, width = size
        depth_map = torch.full((height, width), torch.inf).to(point_map.device)
        u, v = torch.round(projected_points[0, :]).to(torch.int), torch.round(projected_points[1, :]).to(torch.int)
        valid_mask = (0 <= u) & (u < width) & (0 <= v) & (v < height)
        
        for i in range(len(u)):
            if valid_mask[i] and cam_point_map[i, 2] < depth_map[v[i], u[i]]:
                    depth_map[v[i], u[i]] = cam_point_map[i, 2]
                    
        depth_map[depth_map > 1e5] = 0
    return depth_map

def parse_args():
    parser = argparse.ArgumentParser(description='Depth enhancement with VGGT + Prior-Depth-Anything')
    # VGGT 模型：不传则使用 HuggingFace 默认 "facebook/VGGT-1B"；传本地目录或 hub id 则从该路径加载
    parser.add_argument('--vggt_path', type=str, default=None,
                        help='VGGT 模型路径：本地目录或 HuggingFace repo id，默认从 facebook/VGGT-1B 加载')
    # Prior-Depth-Anything 模型
    parser.add_argument('--prior_mde_dir', type=str, default=None,
                        help='Prior-Depth 的 MDE 权重目录，需包含 depth_anything_v2_{size}.pth；不传则从 HF 下载')
    parser.add_argument('--prior_ckpt_dir', type=str, default=None,
                        help='Prior-Depth 的 checkpoint 目录，需包含 prior_depth_anything_{size}{postfix}.pth；不传则从 HF 下载')
    parser.add_argument('--prior_version', type=str, default='1.0', choices=['1.0', '1.1'],
                        help='Prior-Depth 版本，影响 checkpoint 文件名后缀')
    parser.add_argument('--prior_frozen_model_size', type=str, default=None,
                        help='Prior-Depth frozen MDE 尺寸，如 vits/vitb/vitl；不传用默认 vitb')
    parser.add_argument('--prior_conditioned_model_size', type=str, default=None,
                        help='Prior-Depth conditioned 模型尺寸，如 vits/vitb/vitl；不传用默认 vitb')
    # 输入
    parser.add_argument('--image', type=str, default='assets/sample-4/rgb.jpg',
                        help='输入 RGB 图像路径')
    parser.add_argument('--depth', type=str, default='assets/sample-4/gt_depth.png',
                        help='GT 深度图路径（用于评估）')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Initialize VGGT：支持本地路径或 HuggingFace repo id
    vggt_path = args.vggt_path if args.vggt_path else "facebook/VGGT-1B"
    print(f"Initialize VGGT and load weights from: {vggt_path}")
    if vggt_path and os.path.isdir(vggt_path):
        vggt = VGGT.from_pretrained(vggt_path, local_files_only=True).to(device)
    else:
        vggt = VGGT.from_pretrained(vggt_path).to(device)
    print("VGGT loaded!")

    # Initialize prior-depth-anything：支持本地 mde_dir / ckpt_dir
    from prior_depth_anything.plugin import PriorDARefiner, PriorDARefinerMetrics
    refiner_kw = dict(device=device, version=args.prior_version, coarse_only=False)
    if args.prior_mde_dir is not None:
        refiner_kw["mde_dir"] = args.prior_mde_dir
    if args.prior_ckpt_dir is not None:
        refiner_kw["ckpt_dir"] = args.prior_ckpt_dir
    if args.prior_frozen_model_size is not None:
        refiner_kw["frozen_model_size"] = args.prior_frozen_model_size
    if args.prior_conditioned_model_size is not None:
        refiner_kw["conditioned_model_size"] = args.prior_conditioned_model_size
    Refiner = PriorDARefiner(**refiner_kw)

    with torch.no_grad():
        ########## Depth-Estimation stage.
        image_names = [args.image]
        depth_name = args.depth
        images = load_and_preprocess_images(image_names).to(device)

        # Predict attributes including cameras, depth maps, and point maps.
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = vggt(images)
            
        ########## Refine stage.
        # Load gt for evaluation.
        gt_depth = torch.from_numpy(
            np.asarray(Image.open(depth_name)).astype(np.float32)).to(device)
        priorda_image = torch.from_numpy(np.asarray(Image.open(image_names[0])).astype(np.uint8))
        
        ### Refine depth 
        depth_map, depth_conf = predictions['depth'], predictions['depth_conf']
        refined_depth, meview_depth_map = Refiner.predict(
            image=priorda_image, depth_map=depth_map.squeeze(), confidence=depth_conf.squeeze())
        predictions['refined_depth'] = F.interpolate(
            refined_depth[None, None, ...], size=(depth_map.shape[-3], depth_map.shape[-2]), 
            mode='bilinear', align_corners=True
        )
        
        raw_metrics, refined_metrics = Refiner.raw_refined_metrics(
            gt_depth=gt_depth, raw_depth=meview_depth_map, refined_depth=refined_depth
        )
        print("############# Depth map refinement #############")
        print("raw_metrics: ", raw_metrics)
        print("refined_metrics: ", refined_metrics)
        
        ### Refine point_map.
        world_points, world_points_conf = predictions['world_points'], predictions['world_points_conf']
        pose_enc = predictions['pose_enc']
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth_by_project = project_point_map_to_depth_map(
            world_points.view(-1, 3).unsqueeze(0), extrinsics_cam=extrinsic.squeeze(0), 
            intrinsics_cam=intrinsic.squeeze(0), size=images.shape[-2:]
        )
        
        refined_projected, meview_depth_by_project = Refiner.predict(
            image=priorda_image, depth_map=depth_by_project.squeeze(), confidence=world_points_conf.squeeze())
        inview_refined_projected = F.interpolate(
            refined_projected[None, None, ...], size=(depth_map.shape[-3], depth_map.shape[-2]), 
            mode='bilinear', align_corners=True
        ).squeeze()
        refined_world_points = unproject_depth_map_to_point_map(
            inview_refined_projected[None, ..., None], extrinsic.squeeze(0), intrinsic.squeeze(0))
        predictions['refined_points'] = refined_world_points
        
        raw_metrics_point, refined_metrics_point = Refiner.raw_refined_metrics(
            gt_depth=gt_depth, raw_depth=meview_depth_by_project, refined_depth=refined_projected
        )
        
        print("############# Point map refinement #############")
        print("raw_metrics: ", raw_metrics_point)
        print("refined_metrics: ", refined_metrics_point)
        
        