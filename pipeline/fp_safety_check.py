#!/usr/bin/env python3

"""

"""



import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_latest_timestamp(data_dir: Path) -> str:
    masks_dir = data_dir / "masks"

    if not masks_dir.exists():
        raise FileNotFoundError(f"Directory masks not found: {masks_dir}")

    candidates = [p for p in masks_dir.iterdir() if p.is_dir()]

    if not candidates:
        raise FileNotFoundError(f"No timestamp folders found in: {masks_dir}")

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.name


def read_pose_file(pose_path: Path) -> str:
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_path}")

    return pose_path.read_text().strip()


def show_images(mask_path: Path, vis_path: Path, timestamp: str) -> None:
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask image not found: {mask_path}")

    if not vis_path.exists():
        raise FileNotFoundError(f"Visualization image not found: {vis_path}")

    mask_img = mpimg.imread(mask_path)
    vis_img = mpimg.imread(vis_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"FoundationPose safety check - {timestamp}")

    axes[0].imshow(mask_img, cmap="gray")
    axes[0].set_title("SAM3 mask - frame 000000")
    axes[0].axis("off")

    axes[1].imshow(vis_img)
    axes[1].set_title("FoundationPose vis - frame 000000")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safety check for SAM3 mask, FoundationPose visualization and first pose."
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Base directory containing masks/ and outputs/ folders.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest timestamp folder inside masks/.",
    )
    mode.add_argument(
        "--timestamp",
        type=str,
        help="Specific timestamp to check, e.g. 2026-06-12_15-44-38.",
    )

    parser.add_argument(
        "--frame",
        default="000000",
        help="Frame id to check. Default: 000000.",
    )

    parser.add_argument(
        "--no-popup",
        action="store_true",
        help="Do not open matplotlib popup, only print paths and pose.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()

    if args.latest:
        timestamp = get_latest_timestamp(data_dir)
    else:
        timestamp = args.timestamp

    frame = args.frame

    mask_path = data_dir / "masks" / timestamp / f"{frame}.png"
    vis_path = data_dir / "outputs" / timestamp / "vis" / f"{frame}.png"
    pose_path = data_dir / "outputs" / timestamp / "ob_in_cam" / f"{frame}.txt"

    print("\n========== FoundationPose Safety Check ==========")
    print(f"Data dir:   {data_dir}")
    print(f"Timestamp:  {timestamp}")
    print(f"Frame:      {frame}")
    print(f"Mask:       {mask_path}")
    print(f"Vis:        {vis_path}")
    print(f"Pose:       {pose_path}")

    pose_text = read_pose_file(pose_path)

    print("\n---------- First frame pose: ob_in_cam ----------")
    print(pose_text)
    print("-------------------------------------------------\n")

    if not args.no_popup:
        show_images(mask_path, vis_path, timestamp)

    answer = input("Safety check OK? Continue pipeline? [y/N]: ").strip().lower()

    if answer not in ["y", "yes"]:
        print("Safety check rejected. Stop pipeline.")
        return 1

    print("Safety check accepted. Continue pipeline.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)