import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import imageio
import numpy as np
import trimesh
import yaml as pyyaml

FOUNDATIONPOSE_ROOT = Path(__file__).resolve().parents[1] / "submodules" / "FoundationPose"
if str(FOUNDATIONPOSE_ROOT) not in sys.path:
    sys.path.insert(0, str(FOUNDATIONPOSE_ROOT))

from Utils import draw_posed_3d_box, draw_xyz_axis


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


def first_existing_path(paths, label: str) -> Path:
    for path in paths:
        if path.exists():
            return path

    formatted = "\n".join(f"  - {path}" for path in paths)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n{formatted}")


def list_frame_ids(pose_dir: Path, requested_frame: Optional[str]):
    if requested_frame is not None:
        return [requested_frame]

    frame_ids = sorted(path.stem for path in pose_dir.glob("*.txt"))
    if not frame_ids:
        raise RuntimeError(f"No pose files found in: {pose_dir}")

    return frame_ids


def load_pose(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64).reshape(4, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--mesh_file", type=str, required=True)
    parser.add_argument("--frame_id", type=str, default=None)
    parser.add_argument(
        "--draw_pred",
        action="store_true",
        help="Also draw FoundationPose prediction in green when available.",
    )
    parser.add_argument("--axis_scale", type=float, default=0.1)

    args = parser.parse_args()

    data_root = Path(args.data_root)
    timestamp = args.timestamp

    mesh_path = Path(args.mesh_file)
    if not mesh_path.is_absolute():
        mesh_path = data_root / mesh_path

    intrinsics_path = first_existing_path(
        [
            data_root / "camera" / timestamp / "intrinsic.yaml",
            data_root / "camera" / timestamp / "intrinsics.yaml",
        ],
        "camera intrinsics",
    )

    gt_pose_dir = data_root / "gt" / timestamp / "ob_in_cam"
    pred_pose_dir = data_root / "outputs" / timestamp / "ob_in_cam"
    vis_dir = data_root / "outputs" / timestamp / "metrics_debug" / "gt_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    K = load_intrinsics(intrinsics_path).astype(np.float32)

    mesh = trimesh.load(str(mesh_path))
    mesh.vertices = mesh.vertices.astype(np.float32)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    frame_ids = list_frame_ids(gt_pose_dir, args.frame_id)

    for frame_id in frame_ids:
        rgb_path = first_existing_path(
            [data_root / "rgb" / timestamp / f"{frame_id}.png"],
            f"RGB for frame {frame_id}",
        )
        gt_pose_path = first_existing_path(
            [gt_pose_dir / f"{frame_id}.txt"],
            f"GT pose for frame {frame_id}",
        )

        color = load_rgb(rgb_path)
        gt_pose = load_pose(gt_pose_path)
        gt_center_pose = gt_pose @ np.linalg.inv(to_origin)

        vis = color.copy()

        if args.draw_pred:
            pred_pose_path = pred_pose_dir / f"{frame_id}.txt"
            if pred_pose_path.exists():
                pred_pose = load_pose(pred_pose_path)
                pred_center_pose = pred_pose @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(
                    K,
                    img=vis,
                    ob_in_cam=pred_center_pose,
                    bbox=bbox,
                    line_color=(0, 255, 0),
                    linewidth=2,
                )

        vis = draw_posed_3d_box(
            K,
            img=vis,
            ob_in_cam=gt_center_pose,
            bbox=bbox,
            line_color=(255, 0, 0),
            linewidth=2,
        )
        vis = draw_xyz_axis(
            vis,
            ob_in_cam=gt_center_pose,
            scale=args.axis_scale,
            K=K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )

        vis_path = vis_dir / f"{frame_id}.png"
        imageio.imwrite(vis_path, vis)
        print(f"[GT_VIS] Saved visualization: {vis_path}")


if __name__ == "__main__":
    main()
