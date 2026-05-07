import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def load_pose(path: Path) -> np.ndarray:
    pose = np.loadtxt(path, dtype=np.float64)
    pose = pose.reshape(4, 4)
    return pose


def project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt

    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt

    return rotation


def rotation_error_angle_deg(rotation_error: np.ndarray) -> float:
    value = (np.trace(rotation_error) - 1.0) / 2.0
    value = float(np.clip(value, -1.0, 1.0))
    return math.degrees(math.acos(value))


def rpy_zyx_deg(rotation: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract roll, pitch, yaw using R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    sy = math.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
    singular = sy < 1e-9

    if not singular:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def compute_metrics(pred_pose: np.ndarray, gt_pose: np.ndarray) -> Dict[str, float]:
    translation_error_m = pred_pose[:3, 3] - gt_pose[:3, 3]
    translation_error_cm = translation_error_m * 100.0
    translation_norm_cm = float(np.linalg.norm(translation_error_cm))

    pred_rotation = project_to_rotation(pred_pose[:3, :3])
    gt_rotation = project_to_rotation(gt_pose[:3, :3])

    rotation_error = gt_rotation.T @ pred_rotation
    rotation_error = project_to_rotation(rotation_error)

    roll_err_deg, pitch_err_deg, yaw_err_deg = rpy_zyx_deg(rotation_error)
    rot_angle_deg = rotation_error_angle_deg(rotation_error)

    return {
        "tx_err_cm": float(translation_error_cm[0]),
        "ty_err_cm": float(translation_error_cm[1]),
        "tz_err_cm": float(translation_error_cm[2]),
        "trans_norm_cm": translation_norm_cm,
        "roll_err_deg": roll_err_deg,
        "pitch_err_deg": pitch_err_deg,
        "yaw_err_deg": yaw_err_deg,
        "rot_angle_deg": rot_angle_deg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional CSV output path. Defaults to outputs/<timestamp>/metrics_debug/pose_errors.csv.",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    timestamp = args.timestamp

    pred_dir = data_root / "outputs" / timestamp / "ob_in_cam"
    gt_dir = data_root / "gt" / timestamp / "ob_in_cam"

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction pose directory does not exist: {pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT pose directory does not exist: {gt_dir}")

    if args.output_csv is None:
        output_csv = data_root / "outputs" / timestamp / "metrics_debug" / "pose_errors.csv"
    else:
        output_csv = Path(args.output_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    pred_frames = {path.stem for path in pred_dir.glob("*.txt")}
    gt_frames = {path.stem for path in gt_dir.glob("*.txt")}
    frame_ids = sorted(pred_frames & gt_frames)

    if not frame_ids:
        raise RuntimeError(f"No matching pose files found between {pred_dir} and {gt_dir}")

    missing_gt = sorted(pred_frames - gt_frames)
    missing_pred = sorted(gt_frames - pred_frames)

    fieldnames = [
        "frame",
        "tx_err_cm",
        "ty_err_cm",
        "tz_err_cm",
        "trans_norm_cm",
        "roll_err_deg",
        "pitch_err_deg",
        "yaw_err_deg",
        "rot_angle_deg",
    ]

    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for frame_id in frame_ids:
            pred_pose = load_pose(pred_dir / f"{frame_id}.txt")
            gt_pose = load_pose(gt_dir / f"{frame_id}.txt")

            row = {"frame": frame_id}
            row.update(compute_metrics(pred_pose, gt_pose))
            writer.writerow(row)

    print(f"[METRICS] Compared {len(frame_ids)} frames")
    print(f"[METRICS] Saved CSV: {output_csv}")

    if missing_gt:
        print(f"[METRICS] Frames with prediction but no GT: {len(missing_gt)}")
    if missing_pred:
        print(f"[METRICS] Frames with GT but no prediction: {len(missing_pred)}")


if __name__ == "__main__":
    main()
