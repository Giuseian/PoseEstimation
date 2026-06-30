import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import yaml as pyyaml
import numpy as np
import trimesh
import imageio

FOUNDATIONPOSE_ROOT = Path(__file__).resolve().parents[1] / "submodules" / "FoundationPose"
if str(FOUNDATIONPOSE_ROOT) not in sys.path:
    sys.path.insert(0, str(FOUNDATIONPOSE_ROOT))

from estimater import *
from datareader import *


def load_intrinsics(intrinsics_path: Path) -> np.ndarray:
    with open(intrinsics_path, "r") as f:
        data = pyyaml.safe_load(f)

    if "K" in data:
        return np.array(data["K"], dtype=np.float32)

    return np.array(
        [
            [data["fx"], 0.0, data["cx"]],
            [0.0, data["fy"], data["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def load_rgb(rgb_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {rgb_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_depth(depth_path: Path) -> np.ndarray:
    if depth_path.suffix == ".npy":
        depth = np.load(depth_path).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        return depth

    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_path}")

    if depth_raw.dtype == np.uint16:
        depth = depth_raw.astype(np.float32) / 1000.0
    else:
        depth = depth_raw.astype(np.float32)

    depth[~np.isfinite(depth)] = 0.0
    return depth


def load_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask image: {mask_path}")
    return mask > 0


def first_existing_path(paths: List[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path

    formatted = "\n".join(f"  - {path}" for path in paths)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n{formatted}")


def get_depth_path(data_root: Path, timestamp: str, frame_id: str) -> Path:
    return first_existing_path(
        [
            data_root / "depth" / timestamp / "png" / f"{frame_id}.png",
            data_root / "depth" / timestamp / "npy" / f"{frame_id}.npy",
        ],
        f"depth for frame {frame_id}",
    )


def get_rgb_path(data_root: Path, timestamp: str, frame_id: str) -> Path:
    return first_existing_path(
        [data_root / "rgb" / timestamp / f"{frame_id}.png"],
        f"RGB for frame {frame_id}",
    )


def get_mask_path(data_root: Path, timestamp: str, frame_id: str) -> Path:
    return first_existing_path(
        [data_root / "masks" / timestamp / f"{frame_id}.png"],
        f"mask for frame {frame_id}",
    )


def list_frame_ids(data_root: Path, timestamp: str) -> List[str]:
    rgb_dir = data_root / "rgb" / timestamp
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory does not exist: {rgb_dir}")

    frame_ids = sorted(path.stem for path in rgb_dir.glob("*.png"))
    if len(frame_ids) == 0:
        raise RuntimeError(f"No RGB frames found in: {rgb_dir}")

    return frame_ids


def save_pose(pose: np.ndarray, pose_dir: Path, frame_id: str) -> None:
    pose_path = pose_dir / f"{frame_id}.txt"
    np.savetxt(pose_path, pose.reshape(4, 4))
    print(f"[FP] Saved pose: {pose_path}")


def save_visualization(
    color: np.ndarray,
    K: np.ndarray,
    pose: np.ndarray,
    to_origin: np.ndarray,
    bbox: np.ndarray,
    vis_dir: Path,
    frame_id: str,
) -> None:
    center_pose = pose @ np.linalg.inv(to_origin)

    vis = draw_posed_3d_box(
        K,
        img=color,
        ob_in_cam=center_pose,
        bbox=bbox,
    )

    vis = draw_xyz_axis(
        vis,
        ob_in_cam=center_pose,
        scale=0.1,
        K=K,
        thickness=3,
        transparency=0,
        is_input_rgb=True,
    )

    vis_path = vis_dir / f"{frame_id}.png"
    imageio.imwrite(vis_path, vis)
    print(f"[FP] Saved visualization: {vis_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--mesh_file", type=str, required=True)

    parser.add_argument(
        "--start_frame",
        type=str,
        default="000000",
        help="Frame used for initial registration.",
    )

    parser.add_argument(
        "--end_frame",
        type=str,
        default=None,
        help="Optional last frame id to process.",
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process.",
    )

    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=1)

    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    data_root = Path(args.data_root)
    timestamp = args.timestamp

    intrinsics_path = first_existing_path(
        [
            data_root / "camera" / timestamp / "intrinsic.yaml",
            data_root / "camera" / timestamp / "intrinsics.yaml",
        ],
        "camera intrinsics",
    )

    mesh_path = Path(args.mesh_file)
    if not mesh_path.is_absolute():
        mesh_path = data_root / mesh_path

    output_dir = data_root / "outputs" / timestamp
    debug_dir = output_dir / "foundationpose_debug"
    pose_dir = output_dir / "ob_in_cam"
    vis_dir = output_dir / "vis"

    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    print(f"[FP] Data root: {data_root}")
    print(f"[FP] Timestamp: {timestamp}")
    print(f"[FP] Intrinsics: {intrinsics_path}")
    print(f"[FP] Mesh: {mesh_path}")
    print(f"[FP] Output: {output_dir}")

    K = load_intrinsics(intrinsics_path).astype(np.float32)

    mesh = trimesh.load(str(mesh_path))
    mesh.vertices = mesh.vertices.astype(np.float32)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(debug_dir),
        debug=args.debug,
        glctx=glctx,
    )

    est.diameter = float(est.diameter)

    all_frame_ids = list_frame_ids(data_root, timestamp)

    if args.start_frame not in all_frame_ids:
        raise ValueError(f"start_frame {args.start_frame} not found in RGB frames.")

    start_idx = all_frame_ids.index(args.start_frame)
    frame_ids = all_frame_ids[start_idx:]

    if args.end_frame is not None:
        if args.end_frame not in frame_ids:
            raise ValueError(f"end_frame {args.end_frame} not found after start_frame.")
        end_idx = frame_ids.index(args.end_frame)
        frame_ids = frame_ids[: end_idx + 1]

    if args.max_frames is not None:
        frame_ids = frame_ids[: args.max_frames]

    print(f"[FP] Processing {len(frame_ids)} frames")
    print(f"[FP] First frame: {frame_ids[0]}")
    print(f"[FP] Last frame: {frame_ids[-1]}")

    pose = None

    for i, frame_id in enumerate(frame_ids):
        print(f"\n[FP] Frame {i + 1}/{len(frame_ids)}: {frame_id}")

        rgb_path = get_rgb_path(data_root, timestamp, frame_id)
        depth_path = get_depth_path(data_root, timestamp, frame_id)

        color = load_rgb(rgb_path)
        depth = load_depth(depth_path)

        if color.shape[:2] != depth.shape:
            raise ValueError(
                f"RGB/depth shape mismatch for frame {frame_id}: "
                f"rgb={color.shape}, depth={depth.shape}"
            )

        if i == 0:
            mask_path = get_mask_path(data_root, timestamp, frame_id)
            mask = load_mask(mask_path)

            if color.shape[:2] != mask.shape:
                raise ValueError(
                    f"RGB/mask shape mismatch for frame {frame_id}: "
                    f"rgb={color.shape}, mask={mask.shape}"
                )

            print(f"[FP] Running initial registration with mask: {mask_path}")

            pose = est.register(
                K=K,
                rgb=color,
                depth=depth,
                ob_mask=mask,
                iteration=args.est_refine_iter,
            )
        else:
            print("[FP] Running tracking")

            pose = est.track_one(
                rgb=color,
                depth=depth,
                K=K,
                iteration=args.track_refine_iter,
            )

        save_pose(pose, pose_dir, frame_id)

        if args.debug >= 1:
            save_visualization(
                color=color,
                K=K,
                pose=pose,
                to_origin=to_origin,
                bbox=bbox,
                vis_dir=vis_dir,
                frame_id=frame_id,
            )

    print("\n[FP] Sequence processing completed")


if __name__ == "__main__":
    main()








# import os
# import sys
# import argparse
# from pathlib import Path
# from typing import List

# import cv2
# import yaml as pyyaml
# import numpy as np
# import trimesh
# import imageio

# FOUNDATIONPOSE_ROOT = Path(__file__).resolve().parents[1] / "submodules" / "FoundationPose"
# if str(FOUNDATIONPOSE_ROOT) not in sys.path:
#     sys.path.insert(0, str(FOUNDATIONPOSE_ROOT))

# from estimater import *
# from datareader import *


# def load_intrinsics(intrinsics_path: Path) -> np.ndarray:
#     with open(intrinsics_path, "r") as f:
#         data = pyyaml.safe_load(f)

#     if "K" in data:
#         K = np.array(data["K"], dtype=np.float32)
#     else:
#         K = np.array(
#             [
#                 [data["fx"], 0.0, data["cx"]],
#                 [0.0, data["fy"], data["cy"]],
#                 [0.0, 0.0, 1.0],
#             ],
#             dtype=np.float32,
#         )

#     return K


# def load_rgb(rgb_path: Path) -> np.ndarray:
#     # cv2 loads BGR, FoundationPose visualization expects RGB when is_input_rgb=True.
#     bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
#     if bgr is None:
#         raise FileNotFoundError(f"Could not read RGB image: {rgb_path}")

#     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#     return rgb


# def load_depth(depth_path: Path) -> np.ndarray:
#     if depth_path.suffix == ".npy":
#         depth = np.load(depth_path).astype(np.float32)
#         depth[~np.isfinite(depth)] = 0.0
#         return depth

#     # Depth PNG is saved as uint16 millimeters.
#     depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
#     if depth_raw is None:
#         raise FileNotFoundError(f"Could not read depth image: {depth_path}")

#     if depth_raw.dtype == np.uint16:
#         depth = depth_raw.astype(np.float32) / 1000.0
#     else:
#         depth = depth_raw.astype(np.float32)

#     depth[~np.isfinite(depth)] = 0.0
#     return depth


# def load_mask(mask_path: Path) -> np.ndarray:
#     mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
#     if mask is None:
#         raise FileNotFoundError(f"Could not read mask image: {mask_path}")

#     return mask > 0


# def first_existing_path(paths: List[Path], label: str) -> Path:
#     for path in paths:
#         if path.exists():
#             return path

#     formatted_paths = "\n".join(f"  - {path}" for path in paths)
#     raise FileNotFoundError(f"Could not find {label}. Tried:\n{formatted_paths}")


# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--data_root",
#         type=str,
#         required=True,
#         help="Root data directory containing rgb/depth/camera/masks/meshes/outputs.",
#     )
#     parser.add_argument(
#         "--timestamp",
#         type=str,
#         required=True,
#         help="Timestamp of the run to process.",
#     )
#     parser.add_argument(
#         "--frame_id",
#         type=str,
#         default="000000",
#         help="Frame id without extension.",
#     )
#     parser.add_argument(
#         "--mesh_file",
#         type=str,
#         required=True,
#         help="Path to the object mesh file, e.g. meshes/box_green_model/meshes/green_box.obj.",
#     )
#     parser.add_argument(
#         "--est_refine_iter",
#         type=int,
#         default=5,
#         help="Number of refinement iterations for initial registration.",
#     )
#     parser.add_argument(
#         "--debug",
#         type=int,
#         default=1,
#     )

#     args = parser.parse_args()

#     set_logging_format()
#     set_seed(0)

#     data_root = Path(args.data_root)
#     timestamp = args.timestamp
#     frame_id = args.frame_id

#     rgb_path = first_existing_path(
#         [data_root / "rgb" / timestamp / f"{frame_id}.png"],
#         "RGB image",
#     )
#     depth_path = first_existing_path(
#         [
#             data_root / "depth" / timestamp / "png" / f"{frame_id}.png",
#             data_root / "depth" / timestamp / "npy" / f"{frame_id}.npy",
#         ],
#         "depth image",
#     )
#     mask_path = first_existing_path(
#         [data_root / "masks" / timestamp / f"{frame_id}.png"],
#         "mask image",
#     )
#     intrinsics_path = first_existing_path(
#         [
#             data_root / "camera" / timestamp / "intrinsic.yaml",
#             data_root / "camera" / timestamp / "intrinsics.yaml",
#         ],
#         "camera intrinsics",
#     )

#     mesh_path = Path(args.mesh_file)
#     if not mesh_path.is_absolute():
#         mesh_path = data_root / mesh_path

#     output_dir = data_root / "outputs" / timestamp
#     debug_dir = output_dir / "foundationpose_debug"
#     pose_dir = output_dir / "ob_in_cam"
#     vis_dir = output_dir / "vis"

#     os.makedirs(debug_dir, exist_ok=True)
#     os.makedirs(pose_dir, exist_ok=True)
#     os.makedirs(vis_dir, exist_ok=True)

#     print(f"[FP] RGB: {rgb_path}")
#     print(f"[FP] Depth: {depth_path}")
#     print(f"[FP] Mask: {mask_path}")
#     print(f"[FP] Intrinsics: {intrinsics_path}")
#     print(f"[FP] Mesh: {mesh_path}")
#     print(f"[FP] Output: {output_dir}")

#     color = load_rgb(rgb_path)
#     depth = load_depth(depth_path)
#     mask = load_mask(mask_path)
#     K = load_intrinsics(intrinsics_path).astype(np.float32)

#     if color.shape[:2] != depth.shape:
#         raise ValueError(f"RGB/depth shape mismatch: rgb={color.shape}, depth={depth.shape}")

#     if color.shape[:2] != mask.shape:
#         raise ValueError(f"RGB/mask shape mismatch: rgb={color.shape}, mask={mask.shape}")

#     mesh = trimesh.load(str(mesh_path))
#     mesh.vertices = mesh.vertices.astype(np.float32)

#     to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
#     bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

#     scorer = ScorePredictor()
#     refiner = PoseRefinePredictor()
#     glctx = dr.RasterizeCudaContext()

#     est = FoundationPose(
#         model_pts=mesh.vertices,
#         model_normals=mesh.vertex_normals,
#         mesh=mesh,
#         scorer=scorer,
#         refiner=refiner,
#         debug_dir=str(debug_dir),
#         debug=args.debug,
#         glctx=glctx,
#     )
#     est.diameter = float(est.diameter)

#     pose = est.register(
#         K=K,
#         rgb=color,
#         depth=depth,
#         ob_mask=mask,
#         iteration=args.est_refine_iter,
#     )

#     pose_path = pose_dir / f"{frame_id}.txt"
#     np.savetxt(pose_path, pose.reshape(4, 4))
#     print(f"[FP] Saved pose to: {pose_path}")

#     center_pose = pose @ np.linalg.inv(to_origin)
#     vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
#     vis = draw_xyz_axis(
#         vis,
#         ob_in_cam=center_pose,
#         scale=0.1,
#         K=K,
#         thickness=3,
#         transparency=0,
#         is_input_rgb=True,
#     )

#     vis_path = vis_dir / f"{frame_id}.png"
#     imageio.imwrite(vis_path, vis)
#     print(f"[FP] Saved visualization to: {vis_path}")


# if __name__ == "__main__":
#     main()
