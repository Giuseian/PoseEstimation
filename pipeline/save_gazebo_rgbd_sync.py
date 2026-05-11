import argparse
import os
import select
import subprocess
import threading
import time
from collections import deque
from datetime import datetime

import yaml
import cv2
import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

import tf2_ros
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def gz_stamp_to_sec(stamp: dict) -> float:
    return float(stamp["sec"]) + float(stamp["nsec"]) * 1e-9


def quaternion_to_rotation_matrix(x, y, z, w):
    q = np.array([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)

    x, y, z, w = q / norm

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_transform_matrix(position, orientation):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_to_rotation_matrix(
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    )
    matrix[:3, 3] = [position.x, position.y, position.z]
    return matrix


def pose_msg_to_matrix(msg):
    return make_transform_matrix(msg.pose.position, msg.pose.orientation)


def transform_msg_to_matrix(msg):
    return make_transform_matrix(msg.transform.translation, msg.transform.rotation)


def parse_gz_pose_block(block):
    pose = {
        "name": None,
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }
    section = None

    for raw_line in block:
        line = raw_line.strip()

        if line.startswith("name:"):
            pose["name"] = line.split(":", 1)[1].strip().strip('"')
        elif line == "position {":
            section = "position"
        elif line == "orientation {":
            section = "orientation"
        elif line == "}":
            section = None
        elif section in ("position", "orientation") and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            if key in pose[section]:
                pose[section][key] = float(value.strip())

    return pose


def gz_pose_to_matrix(pose):
    class Vector:
        pass

    position = Vector()
    orientation = Vector()

    position.x = pose["position"]["x"]
    position.y = pose["position"]["y"]
    position.z = pose["position"]["z"]

    orientation.x = pose["orientation"]["x"]
    orientation.y = pose["orientation"]["y"]
    orientation.z = pose["orientation"]["z"]
    orientation.w = pose["orientation"]["w"]

    return make_transform_matrix(position, orientation)


def parse_gz_header_stamp(block):
    stamp = {"sec": None, "nsec": 0}
    in_stamp = False

    for raw_line in block:
        line = raw_line.strip()

        if line == "stamp {":
            in_stamp = True
        elif in_stamp and line == "}":
            in_stamp = False
        elif in_stamp and line.startswith("sec:"):
            stamp["sec"] = int(line.split(":", 1)[1].strip())
        elif in_stamp and line.startswith("nsec:"):
            stamp["nsec"] = int(line.split(":", 1)[1].strip())

    if stamp["sec"] is None:
        return None

    return stamp


def nearest_by_time(buffer, target_time: float):
    if not buffer:
        return None, None

    best = min(buffer, key=lambda item: abs(item[0] - target_time))
    return best, abs(best[0] - target_time)


class RGBDDataSaverSync(Node):
    def __init__(
        self,
        capture_hz: float = 20.0,
        buffer_seconds: float = 3.0,
        max_sync_dt: float = 0.05,
        sync_delay: float = 0.2,
        use_latest_tf: bool = False,
    ):
        super().__init__("rgbd_data_saver_sync")

        if capture_hz <= 0.0:
            raise ValueError("capture_hz must be greater than 0")

        self.bridge = CvBridge()
        self.capture_period = 1.0 / capture_hz
        self.buffer_seconds = buffer_seconds
        self.max_sync_dt = max_sync_dt
        self.sync_delay = sync_delay
        self.use_latest_tf = use_latest_tf
        self.last_saved_rgb_time = None

        self.save_dir = os.path.expanduser("~/pose_estimation_data")
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.rgb_dir = os.path.join(self.save_dir, "rgb", self.run_timestamp)
        self.depth_npy_dir = os.path.join(self.save_dir, "depth", self.run_timestamp, "npy")
        self.depth_png_dir = os.path.join(self.save_dir, "depth", self.run_timestamp, "png")
        self.camera_dir = os.path.join(self.save_dir, "camera", self.run_timestamp)
        self.meshes_dir = os.path.join(self.save_dir, "meshes")
        self.gt_pose_dir = os.path.join(self.save_dir, "gt", self.run_timestamp, "ob_in_cam")

        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_npy_dir, exist_ok=True)
        os.makedirs(self.depth_png_dir, exist_ok=True)
        os.makedirs(self.camera_dir, exist_ok=True)
        os.makedirs(self.meshes_dir, exist_ok=True)
        os.makedirs(self.gt_pose_dir, exist_ok=True)

        self.rgb_buffer = deque()
        self.depth_buffer = deque()
        self.pelvis_buffer = deque()
        self.box_pose_buffer = deque()
        self.buffer_lock = threading.Lock()

        self.K_saved = False
        self.frame_id = 0

        self.rgb_topic = "/D435_head_camera/color/image_raw"
        self.depth_topic = "/D435_head_camera/aligned_depth_to_color/image_raw"
        self.camera_info_topic = "/D435_head_camera/color/camera_info"
        self.pelvis_pose_topic = "/xbotcore/link_state/pelvis/pose"
        self.gazebo_pose_topic = "/world/default/pose/info"
        self.gt_object_name = "box_red_001"
        self.pelvis_frame = "pelvis"
        self.camera_optical_frame = "D435_head_camera_gz_optical_frame"

        self.gz_pose_proc = None
        self.gz_pose_stop = threading.Event()
        self.gz_pose_thread = None

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=buffer_seconds))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 100)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 100)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(PoseStamped, self.pelvis_pose_topic, self.pelvis_pose_callback, 100)

        self.start_gazebo_pose_reader()
        self.timer = self.create_timer(self.capture_period, self.save_frame)

        self.get_logger().info(f"Saving data to: {self.save_dir}")
        self.get_logger().info(f"Run timestamp: {self.run_timestamp}")
        self.get_logger().info(f"Capture rate: {capture_hz:.3f} Hz")
        self.get_logger().info(f"Max sync delta: {max_sync_dt:.3f} s")
        self.get_logger().info(f"Sync delay: {sync_delay:.3f} s")
        self.get_logger().info(f"Saving GT poses to: {self.gt_pose_dir}")

    def destroy_node(self):
        self.gz_pose_stop.set()

        if self.gz_pose_proc is not None:
            self.gz_pose_proc.terminate()
            try:
                self.gz_pose_proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self.gz_pose_proc.kill()

        if self.gz_pose_thread is not None:
            self.gz_pose_thread.join(timeout=0.5)

        super().destroy_node()

    def append_buffer(self, buffer, item):
        buffer.append(item)
        newest_time = item[0]

        while buffer and newest_time - buffer[0][0] > self.buffer_seconds:
            buffer.popleft()

    def rgb_callback(self, msg):
        with self.buffer_lock:
            self.append_buffer(self.rgb_buffer, (stamp_to_sec(msg.header.stamp), msg))

    def depth_callback(self, msg):
        with self.buffer_lock:
            self.append_buffer(self.depth_buffer, (stamp_to_sec(msg.header.stamp), msg))

    def pelvis_pose_callback(self, msg):
        with self.buffer_lock:
            self.append_buffer(self.pelvis_buffer, (stamp_to_sec(msg.header.stamp), msg))

    def camera_info_callback(self, msg):
        if self.K_saved:
            return

        K = np.array(msg.k).reshape(3, 3)

        intrinsics = {
            "width": int(msg.width),
            "height": int(msg.height),
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "K": K.tolist(),
            "rgb_topic": self.rgb_topic,
            "depth_topic": self.depth_topic,
            "camera_info_topic": self.camera_info_topic,
            "pelvis_pose_topic": self.pelvis_pose_topic,
            "gazebo_pose_topic": self.gazebo_pose_topic,
            "gt_object_name": self.gt_object_name,
            "camera_optical_frame": self.camera_optical_frame,
            "run_timestamp": self.run_timestamp,
            "save_root": self.save_dir,
            "max_sync_dt": self.max_sync_dt,
            "sync_delay": self.sync_delay,
        }

        out_path = os.path.join(self.camera_dir, "intrinsics.yaml")
        with open(out_path, "w") as f:
            yaml.dump(intrinsics, f)

        self.get_logger().info(f"Saved camera intrinsics to {out_path}")
        self.K_saved = True

    def start_gazebo_pose_reader(self):
        cmd = ["gz", "topic", "-e", "-t", self.gazebo_pose_topic]
        try:
            self.gz_pose_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self.get_logger().warn(f"Could not run {' '.join(cmd)}: {exc}")
            return

        self.gz_pose_thread = threading.Thread(target=self.gazebo_pose_reader_loop, daemon=True)
        self.gz_pose_thread.start()

    def gazebo_pose_reader_loop(self):
        header_block = []
        current_stamp = None
        in_header = False
        in_pose = False
        depth = 0
        block = []

        while not self.gz_pose_stop.is_set():
            try:
                ready, _, _ = select.select([self.gz_pose_proc.stdout], [], [], 0.1)
                if not ready:
                    if self.gz_pose_proc.poll() is not None:
                        self.get_logger().warn(f"Gazebo pose command stopped: {self.gazebo_pose_topic}")
                        return
                    continue

                line = self.gz_pose_proc.stdout.readline()
                if line == "":
                    if self.gz_pose_proc.poll() is not None:
                        self.get_logger().warn(f"Gazebo pose command stopped: {self.gazebo_pose_topic}")
                        return
                    continue

                stripped = line.strip()

                if stripped == "header {":
                    in_header = True
                    depth = 1
                    header_block = [line]
                    continue

                if in_header:
                    header_block.append(line)
                    depth += stripped.count("{")
                    depth -= stripped.count("}")
                    if depth == 0:
                        stamp = parse_gz_header_stamp(header_block)
                        current_stamp = gz_stamp_to_sec(stamp) if stamp is not None else None
                        in_header = False
                    continue

                if not in_pose and stripped == "pose {":
                    in_pose = True
                    depth = 1
                    block = [line]
                    continue

                if not in_pose:
                    continue

                block.append(line)
                depth += stripped.count("{")
                depth -= stripped.count("}")

                if depth == 0:
                    pose = parse_gz_pose_block(block)
                    if pose["name"] == self.gt_object_name and current_stamp is not None:
                        with self.buffer_lock:
                            self.append_buffer(self.box_pose_buffer, (current_stamp, pose))

                    in_pose = False
                    block = []
            except Exception as exc:
                self.get_logger().warn(f"Gazebo pose reader failed: {exc}")
                return

    def select_synced_messages(self):
        with self.buffer_lock:
            if not self.rgb_buffer:
                return None

            latest_rgb_time = self.rgb_buffer[-1][0]
            target_time = latest_rgb_time - self.sync_delay
            candidates = [item for item in self.rgb_buffer if item[0] <= target_time]

            if self.last_saved_rgb_time is not None:
                candidates = [item for item in candidates if item[0] > self.last_saved_rgb_time]

            if not candidates:
                return None

            rgb_time, rgb_msg = candidates[-1]
            if self.last_saved_rgb_time is not None and rgb_time <= self.last_saved_rgb_time:
                return None

            depth_item, depth_dt = nearest_by_time(self.depth_buffer, rgb_time)
            pelvis_item, pelvis_dt = nearest_by_time(self.pelvis_buffer, rgb_time)
            box_item, box_dt = nearest_by_time(self.box_pose_buffer, rgb_time)

        if depth_item is None or pelvis_item is None or box_item is None:
            self.get_logger().warn("Waiting for synchronized RGB/depth/pelvis/box data...")
            return None

        if depth_dt > self.max_sync_dt or pelvis_dt > self.max_sync_dt or box_dt > self.max_sync_dt:
            self.get_logger().warn(
                "Skipping frame due to sync delta: "
                f"depth={depth_dt:.4f}s, pelvis={pelvis_dt:.4f}s, box={box_dt:.4f}s"
            )
            return None

        self.last_saved_rgb_time = rgb_time

        return {
            "rgb_time": rgb_time,
            "rgb_msg": rgb_msg,
            "depth_msg": depth_item[1],
            "pelvis_msg": pelvis_item[1],
            "box_pose": box_item[1],
            "depth_dt": depth_dt,
            "pelvis_dt": pelvis_dt,
            "box_dt": box_dt,
        }

    def lookup_pelvis_to_camera(self, rgb_msg):
        if self.use_latest_tf:
            tf_time = Time()
        else:
            tf_time = Time.from_msg(rgb_msg.header.stamp)

        return self.tf_buffer.lookup_transform(
            self.pelvis_frame,
            self.camera_optical_frame,
            tf_time,
            timeout=Duration(seconds=0.02),
        )

    def get_gt_pose_in_camera(self, synced):
        pelvis_msg = synced["pelvis_msg"]

        if pelvis_msg.header.frame_id != "world":
            self.get_logger().warn(f"Expected pelvis pose in world, got frame_id={pelvis_msg.header.frame_id}")
            return None

        try:
            tf_pelvis_cam = self.lookup_pelvis_to_camera(synced["rgb_msg"])
        except Exception as exc:
            self.get_logger().warn(
                f"Could not lookup TF {self.pelvis_frame} -> {self.camera_optical_frame}: {exc}"
            )
            return None

        t_world_pelvis = pose_msg_to_matrix(pelvis_msg)
        t_pelvis_cam = transform_msg_to_matrix(tf_pelvis_cam)
        t_world_cam = t_world_pelvis @ t_pelvis_cam
        t_world_box = gz_pose_to_matrix(synced["box_pose"])

        return np.linalg.inv(t_world_cam) @ t_world_box

    def save_frame(self):
        synced = self.select_synced_messages()
        if synced is None:
            return

        gt_pose = self.get_gt_pose_in_camera(synced)
        if gt_pose is None:
            return

        rgb = self.bridge.imgmsg_to_cv2(synced["rgb_msg"], desired_encoding="rgb8")
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        depth = self.bridge.imgmsg_to_cv2(synced["depth_msg"], desired_encoding="passthrough")
        depth = depth.astype(np.float32)

        if synced["depth_msg"].encoding == "16UC1":
            depth = depth / 1000.0

        frame_name = f"{self.frame_id:06d}"

        rgb_path = os.path.join(self.rgb_dir, f"{frame_name}.png")
        depth_npy_path = os.path.join(self.depth_npy_dir, f"{frame_name}.npy")
        depth_png_path = os.path.join(self.depth_png_dir, f"{frame_name}.png")
        gt_pose_path = os.path.join(self.gt_pose_dir, f"{frame_name}.txt")

        cv2.imwrite(rgb_path, rgb_bgr)
        np.save(depth_npy_path, depth)
        depth_png = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_png = np.clip(depth_png * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        cv2.imwrite(depth_png_path, depth_png)
        np.savetxt(gt_pose_path, gt_pose)

        finite_depth = depth[np.isfinite(depth)]
        if finite_depth.size > 0:
            depth_min = float(np.min(finite_depth))
            depth_max = float(np.max(finite_depth))
        else:
            depth_min = float("nan")
            depth_max = float("nan")

        self.get_logger().info(
            f"Saved frame {frame_name} | "
            f"rgb={rgb.shape}, depth={depth.shape}, "
            f"depth_encoding={synced['depth_msg'].encoding}, "
            f"depth_min={depth_min:.3f}, depth_max={depth_max:.3f}, "
            f"sync_dt(depth/pelvis/box)="
            f"{synced['depth_dt']:.4f}/{synced['pelvis_dt']:.4f}/{synced['box_dt']:.4f}s, "
            f"gt_pose={gt_pose_path}"
        )

        self.frame_id += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture_hz", type=float, default=20.0)
    parser.add_argument("--buffer_seconds", type=float, default=3.0)
    parser.add_argument("--max_sync_dt", type=float, default=0.05)
    parser.add_argument(
        "--sync_delay",
        type=float,
        default=0.2,
        help="Save RGB frames this many seconds after acquisition so GT/TF buffers can catch up.",
    )
    parser.add_argument(
        "--use_latest_tf",
        action="store_true",
        help="Use latest camera TF instead of lookup at RGB timestamp.",
    )
    args = parser.parse_args()

    rclpy.init()
    node = RGBDDataSaverSync(
        capture_hz=args.capture_hz,
        buffer_seconds=args.buffer_seconds,
        max_sync_dt=args.max_sync_dt,
        sync_delay=args.sync_delay,
        use_latest_tf=args.use_latest_tf,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
