import os
import sys
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def get_latest_timestamp(root: Path) -> str:
    if not root.exists():
        raise FileNotFoundError(f"Timestamp root does not exist: {root}")

    timestamp_dirs = sorted(
        [path for path in root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )

    if not timestamp_dirs:
        raise RuntimeError(f"No timestamp folders found in: {root}")

    return timestamp_dirs[-1].name


def list_object_mask_paths(
    data_root: Path,
    timestamp: str,
    frame_id: str,
) -> List[Tuple[str, Path]]:
    candidate_roots = [
        data_root / "masks_by_object" / timestamp,
        data_root / "masks_by_objects" / timestamp,
    ]

    masks_root = None
    for candidate in candidate_roots:
        if candidate.exists():
            masks_root = candidate
            break

    if masks_root is None:
        formatted = "\n".join(f"  - {path}" for path in candidate_roots)
        raise FileNotFoundError(
            f"Object masks timestamp directory not found. Tried:\n{formatted}"
        )

    object_masks = []
    for object_dir in sorted(path for path in masks_root.iterdir() if path.is_dir()):
        mask_path = object_dir / f"{frame_id}.png"
        if mask_path.exists():
            object_masks.append((object_dir.name, mask_path))

    if not object_masks:
        raise RuntimeError(
            f"No object masks found for frame {frame_id} under: {masks_root}"
        )

    return object_masks


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


def pose_to_scene_position(pose: np.ndarray) -> List[float]:
    pose_matrix = pose.reshape(4, 4)

    fp_x = float(pose_matrix[0, 3])
    fp_y = float(pose_matrix[1, 3])
    fp_z = float(pose_matrix[2, 3])

    # FoundationPose:
    #   x = right
    #   y = down
    #   z = depth
    #
    # Real robot camera frame D435_head_camera_link:
    #   x = depth
    #   y = left
    #   z = up
    #
    # Mapping:
    #   robot x =  fp_z
    #   robot y = -fp_x
    #   robot z = -fp_y
    return [round(fp_z, 6), round(-fp_x, 6), round(-fp_y, 6)]


def save_poses_json(
    output_root: Path,
    frame_id: str,
    object_positions: Dict[str, List[float]],
) -> Path:
    payload = {
        f"{frame_id}.png": object_positions,
    }

    poses_path = output_root / "poses.json"
    poses_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[FP] Saved poses JSON: {poses_path}")
    return poses_path


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
    vis = color.copy()

    vis = draw_posed_3d_box(
        K,
        img=vis,
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


def list_available_meshes(data_root: Path) -> List[Path]:
    meshes_root = data_root / "meshes"
    if not meshes_root.exists():
        return []

    mesh_paths = []
    for suffix in ["*.obj", "*.ply", "*.stl"]:
        mesh_paths.extend(meshes_root.rglob(suffix))

    return sorted(mesh_paths)


def resolve_mesh_path(data_root: Path, mesh_input: str) -> Path:
    mesh_path = Path(mesh_input).expanduser()

    if not mesh_path.is_absolute():
        mesh_path = data_root / mesh_path

    mesh_path = mesh_path.resolve()

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    if not mesh_path.is_file():
        raise ValueError(f"Mesh path is not a file: {mesh_path}")

    return mesh_path


def path_relative_to_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def mesh_id_from_mesh_path(data_root: Path, mesh_path: Path) -> str:
    """
    Examples:
      /workspace/shared_data/realsense/meshes/box_big/box_big.obj
        -> box_big

      /workspace/shared_data/realsense/meshes/box_amazon_model/meshes/box_amazon_model.obj
        -> box_amazon_model
    """
    meshes_root = (data_root / "meshes").resolve()
    mesh_path = mesh_path.resolve()

    try:
        relative_to_meshes = mesh_path.relative_to(meshes_root)
        if len(relative_to_meshes.parts) >= 1:
            return relative_to_meshes.parts[0]
    except ValueError:
        pass

    return mesh_path.stem


def load_mesh_metadata(data_root: Path) -> Dict[str, Any]:
    metadata_path = data_root / "meshes" / "mesh_metadata.json"

    if not metadata_path.exists():
        print(f"[FP][WARN] mesh_metadata.json not found: {metadata_path}")
        print("[FP][WARN] mesh_assignments.json will still be saved.")
        return {}

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[FP][WARN] Could not read mesh metadata: {metadata_path}")
        print(f"[FP][WARN] Reason: {exc}")
        return {}

    if not isinstance(metadata, dict):
        print(f"[FP][WARN] Invalid mesh metadata format: {metadata_path}")
        return {}

    meshes = metadata.get("meshes")
    if not isinstance(meshes, dict):
        print(
            "[FP][WARN] mesh_metadata.json does not contain a valid "
            f"'meshes' dictionary: {metadata_path}"
        )
        return {}

    print(f"[FP] Loaded mesh metadata: {metadata_path}")
    return metadata


def save_mesh_assignments_json(
    output_root: Path,
    data_root: Path,
    timestamp: str,
    mesh_by_object: Dict[str, Path],
    mesh_metadata: Dict[str, Any],
) -> Path:
    metadata_path = data_root / "meshes" / "mesh_metadata.json"

    known_mesh_ids = set()
    meshes_metadata = mesh_metadata.get("meshes", {})
    if isinstance(meshes_metadata, dict):
        known_mesh_ids = set(meshes_metadata.keys())

    objects_payload = {}

    for object_name, mesh_path in sorted(mesh_by_object.items()):
        mesh_id = mesh_id_from_mesh_path(data_root, mesh_path)
        mesh_relative_path = path_relative_to_or_absolute(mesh_path, data_root)
        mesh_metadata_found = mesh_id in known_mesh_ids

        if known_mesh_ids and not mesh_metadata_found:
            print()
            print("[FP][WARN] Mesh id not found in mesh_metadata.json")
            print(f"[FP][WARN] Object:  {object_name}")
            print(f"[FP][WARN] Mesh id: {mesh_id}")
            print(f"[FP][WARN] Mesh:    {mesh_path}")
            print()

        objects_payload[object_name] = {
            "mesh_path": str(mesh_path.resolve()),
            "mesh_relative_path": mesh_relative_path,
            "mesh_id": mesh_id,
            "mesh_metadata_found": mesh_metadata_found,
        }

    payload = {
        "timestamp": timestamp,
        "data_root": str(data_root.resolve()),
        "mesh_metadata_path": str(metadata_path.resolve()),
        "objects": objects_payload,
    }

    assignment_path = output_root / "mesh_assignments.json"
    assignment_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[FP] Saved mesh assignments JSON: {assignment_path}")
    print()
    print("[FP] Mesh assignments:")
    for object_name, item in objects_payload.items():
        print(
            f"  - {object_name}: "
            f"mesh_id={item['mesh_id']}, "
            f"mesh={item['mesh_relative_path']}, "
            f"metadata_found={item['mesh_metadata_found']}"
        )

    return assignment_path


def prompt_mesh_for_object(
    data_root: Path,
    object_name: str,
    available_meshes: List[Path],
) -> Path:
    while True:
        print("\n======================================================")
        print(f"[FP] Select CAD mesh for object: {object_name}")

        if available_meshes:
            for index, mesh_path in enumerate(available_meshes, start=1):
                try:
                    display_path = mesh_path.relative_to(data_root)
                except ValueError:
                    display_path = mesh_path

                print(f"  {index}. {display_path}")

            print("Insert a number from the list, or insert a mesh path.")
        else:
            print(f"No meshes found under: {data_root / 'meshes'}")
            print("Insert an absolute mesh path, or a path relative to data_root.")

        user_input = input(f"CAD for {object_name}: ").strip()

        if not user_input:
            print("[FP][ERROR] Empty input. Please try again.")
            continue

        if user_input.isdigit() and available_meshes:
            selected_index = int(user_input)

            if 1 <= selected_index <= len(available_meshes):
                mesh_path = available_meshes[selected_index - 1]
                print(f"[FP] Selected mesh for {object_name}: {mesh_path}")
                return mesh_path

            print("[FP][ERROR] Invalid mesh number. Please try again.")
            continue

        try:
            mesh_path = resolve_mesh_path(data_root, user_input)
            print(f"[FP] Selected mesh for {object_name}: {mesh_path}")
            return mesh_path
        except Exception as exc:
            print(f"[FP][ERROR] {exc}")
            print("Please try again.")


def load_mesh_geometry(mesh_path: Path) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    mesh = trimesh.load(str(mesh_path))
    mesh.vertices = mesh.vertices.astype(np.float32)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    return mesh, to_origin, extents, bbox


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/workspace/shared_data/realsense")

    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help=(
            "Timestamp to process. If omitted, the latest folder under "
            "data_root/masks_by_object is used."
        ),
    )

    parser.add_argument(
        "--mesh_file",
        type=str,
        default=None,
        help=(
            "Optional mesh file used for all objects. If omitted, the script "
            "asks interactively which CAD to use for each object mask."
        ),
    )

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

    data_root = Path(args.data_root).resolve()

    timestamp = args.timestamp
    if timestamp is None:
        if (data_root / "masks_by_object").exists():
            timestamp = get_latest_timestamp(data_root / "masks_by_object")
        elif (data_root / "masks_by_objects").exists():
            timestamp = get_latest_timestamp(data_root / "masks_by_objects")
        else:
            raise FileNotFoundError(
                "Neither masks_by_object nor masks_by_objects exists under data_root."
            )

    intrinsics_path = first_existing_path(
        [
            data_root / "camera" / timestamp / "intrinsic.yaml",
            data_root / "camera" / timestamp / "intrinsics.yaml",
        ],
        "camera intrinsics",
    )

    output_root = data_root / "outputs_by_object" / timestamp
    os.makedirs(output_root, exist_ok=True)

    mesh_metadata = load_mesh_metadata(data_root)

    print(f"[FP] Data root: {data_root}")
    print(f"[FP] Timestamp: {timestamp}")
    print(f"[FP] Intrinsics: {intrinsics_path}")
    print(f"[FP] Output root: {output_root}")

    K = load_intrinsics(intrinsics_path).astype(np.float32)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

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

    object_masks = list_object_mask_paths(
        data_root=data_root,
        timestamp=timestamp,
        frame_id=frame_ids[0],
    )

    print(f"[FP] Processing {len(frame_ids)} frames")
    print(f"[FP] First frame: {frame_ids[0]}")
    print(f"[FP] Last frame: {frame_ids[-1]}")
    print(f"[FP] Objects: {', '.join(object_name for object_name, _ in object_masks)}")

    available_meshes = list_available_meshes(data_root)
    mesh_by_object = {}

    if args.mesh_file is not None:
        shared_mesh_path = resolve_mesh_path(data_root, args.mesh_file)
        print(f"[FP] Using same mesh for all objects: {shared_mesh_path}")

        for object_name, _ in object_masks:
            mesh_by_object[object_name] = shared_mesh_path
    else:
        for object_name, _ in object_masks:
            mesh_by_object[object_name] = prompt_mesh_for_object(
                data_root=data_root,
                object_name=object_name,
                available_meshes=available_meshes,
            )

    save_mesh_assignments_json(
        output_root=output_root,
        data_root=data_root,
        timestamp=timestamp,
        mesh_by_object=mesh_by_object,
        mesh_metadata=mesh_metadata,
    )

    frame_data = []

    for frame_id in frame_ids:
        print(f"\n[FP] Loading frame: {frame_id}")

        rgb_path = get_rgb_path(data_root, timestamp, frame_id)
        depth_path = get_depth_path(data_root, timestamp, frame_id)

        color = load_rgb(rgb_path)
        depth = load_depth(depth_path)

        if color.shape[:2] != depth.shape:
            raise ValueError(
                f"RGB/depth shape mismatch for frame {frame_id}: "
                f"rgb={color.shape}, depth={depth.shape}"
            )

        frame_data.append((frame_id, color, depth))

    object_positions = {}

    for object_name, first_mask_path in object_masks:
        print("\n======================================================")
        print(f"[FP] Object: {object_name}")
        print(f"[FP] Initial mask: {first_mask_path}")
        print(f"[FP] Mesh: {mesh_by_object[object_name]}")
        print("======================================================")

        mesh, to_origin, _extents, bbox = load_mesh_geometry(mesh_by_object[object_name])

        object_output_dir = output_root / object_name
        debug_dir = object_output_dir / "foundationpose_debug"
        pose_dir = object_output_dir / "ob_in_cam"
        vis_dir = object_output_dir / "vis"

        os.makedirs(debug_dir, exist_ok=True)
        os.makedirs(pose_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

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

        pose = None

        for i, (frame_id, color, depth) in enumerate(frame_data):
            print(f"\n[FP][{object_name}] Frame {i + 1}/{len(frame_data)}: {frame_id}")

            if i == 0:
                mask = load_mask(first_mask_path)

                if color.shape[:2] != mask.shape:
                    raise ValueError(
                        f"RGB/mask shape mismatch for object {object_name}, frame {frame_id}: "
                        f"rgb={color.shape}, mask={mask.shape}"
                    )

                print(f"[FP][{object_name}] Running initial registration")

                pose = est.register(
                    K=K,
                    rgb=color,
                    depth=depth,
                    ob_mask=mask,
                    iteration=args.est_refine_iter,
                )
            else:
                print(f"[FP][{object_name}] Running tracking")

                pose = est.track_one(
                    rgb=color,
                    depth=depth,
                    K=K,
                    iteration=args.track_refine_iter,
                )

            save_pose(pose, pose_dir, frame_id)

            if frame_id == frame_data[0][0]:
                object_positions[object_name] = pose_to_scene_position(pose)

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

    save_poses_json(
        output_root=output_root,
        frame_id=frame_data[0][0],
        object_positions=object_positions,
    )

    print("\n[FP] Multi-object sequence processing completed")


if __name__ == "__main__":
    main()