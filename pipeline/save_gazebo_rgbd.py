import os
import select
import subprocess
import time
from datetime import datetime

import yaml
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time

import tf2_ros
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


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


class RGBDDataSaver(Node):
    def __init__(self):
        super().__init__("rgbd_data_saver")

        self.bridge = CvBridge()

        self.save_dir = os.path.expanduser("~/pose_estimation_data")
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.rgb_dir = os.path.join(self.save_dir, "rgb", self.run_timestamp)
        self.depth_npy_dir = os.path.join(self.save_dir, "depth", self.run_timestamp, "npy")
        self.depth_png_dir = os.path.join(self.save_dir, "depth", self.run_timestamp, "png")
        self.camera_dir = os.path.join(self.save_dir, "camera", self.run_timestamp)
        self.meshes_dir = os.path.join(self.save_dir, "meshes")
        self.gt_pose_dir = os.path.join(
            self.save_dir,
            "gt",
            self.run_timestamp,
            "ob_in_cam",
        )

        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_npy_dir, exist_ok=True)
        os.makedirs(self.depth_png_dir, exist_ok=True)
        os.makedirs(self.camera_dir, exist_ok=True)
        os.makedirs(self.meshes_dir, exist_ok=True)
        os.makedirs(self.gt_pose_dir, exist_ok=True)

        self.rgb_msg = None
        self.depth_msg = None
        self.pelvis_pose_msg = None
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
        self.gz_pose_timeout = 1.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(PoseStamped, self.pelvis_pose_topic, self.pelvis_pose_callback, 10)

        self.timer = self.create_timer(1.0, self.save_frame)

        self.get_logger().info(f"Saving data to: {self.save_dir}")
        self.get_logger().info(f"Run timestamp: {self.run_timestamp}")
        self.get_logger().info(f"Saving GT poses to: {self.gt_pose_dir}")

    def rgb_callback(self, msg):
        self.rgb_msg = msg

    def depth_callback(self, msg):
        self.depth_msg = msg

    def pelvis_pose_callback(self, msg):
        self.pelvis_pose_msg = msg

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
        }

        out_path = os.path.join(self.camera_dir, "intrinsics.yaml")
        with open(out_path, "w") as f:
            yaml.dump(intrinsics, f)

        self.get_logger().info(f"Saved camera intrinsics to {out_path}")
        self.K_saved = True

    def read_gazebo_pose(self, target_name):
        cmd = ["gz", "topic", "-e", "-t", self.gazebo_pose_topic]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            self.get_logger().warn(f"Could not run {' '.join(cmd)}: {exc}")
            return None

        deadline = time.monotonic() + self.gz_pose_timeout
        in_pose = False
        depth = 0
        block = []

        try:
            while time.monotonic() < deadline:
                remaining = max(0.0, deadline - time.monotonic())
                ready, _, _ = select.select([proc.stdout], [], [], remaining)
                if not ready:
                    break

                line = proc.stdout.readline()
                if line == "":
                    break

                stripped = line.strip()

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
                    if pose["name"] == target_name:
                        return pose

                    in_pose = False
                    block = []

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                proc.kill()

        return None

    def get_gt_pose_in_camera(self):
        if self.pelvis_pose_msg is None:
            self.get_logger().warn("Waiting for pelvis pose...")
            return None

        if self.pelvis_pose_msg.header.frame_id != "world":
            self.get_logger().warn(
                f"Expected pelvis pose in world, got frame_id={self.pelvis_pose_msg.header.frame_id}"
            )
            return None

        box_pose = self.read_gazebo_pose(self.gt_object_name)
        if box_pose is None:
            self.get_logger().warn(
                f"Could not read {self.gt_object_name} from {self.gazebo_pose_topic}"
            )
            return None

        try:
            tf_pelvis_cam = self.tf_buffer.lookup_transform(
                self.pelvis_frame,
                self.camera_optical_frame,
                Time(),
            )
        except Exception as exc:
            self.get_logger().warn(
                f"Could not lookup TF {self.pelvis_frame} -> {self.camera_optical_frame}: {exc}"
            )
            return None

        t_world_pelvis = pose_msg_to_matrix(self.pelvis_pose_msg)
        t_pelvis_cam = transform_msg_to_matrix(tf_pelvis_cam)
        t_world_cam = t_world_pelvis @ t_pelvis_cam
        t_world_box = gz_pose_to_matrix(box_pose)

        return np.linalg.inv(t_world_cam) @ t_world_box

    def save_frame(self):
        if self.rgb_msg is None or self.depth_msg is None:
            self.get_logger().warn("Waiting for RGB and depth messages...")
            return

        gt_pose = self.get_gt_pose_in_camera()
        if gt_pose is None:
            self.get_logger().warn("Skipping frame because GT pose is not available.")
            return

        rgb = self.bridge.imgmsg_to_cv2(self.rgb_msg, desired_encoding="rgb8")
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        depth = self.bridge.imgmsg_to_cv2(self.depth_msg, desired_encoding="passthrough")
        depth = depth.astype(np.float32)

        if self.depth_msg.encoding == "16UC1":
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
            f"depth_encoding={self.depth_msg.encoding}, "
            f"depth_min={depth_min:.3f}, depth_max={depth_max:.3f}, "
            f"gt_pose={gt_pose_path}"
        )

        self.frame_id += 1


def main():
    rclpy.init()
    node = RGBDDataSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()






### OLD VERSION WORKING W/OUT GT 

# import os
# from datetime import datetime

# import yaml
# import cv2
# import numpy as np

# import rclpy
# from rclpy.node import Node

# from sensor_msgs.msg import Image, CameraInfo
# from cv_bridge import CvBridge


# class RGBDDataSaver(Node):
#     def __init__(self):
#         super().__init__("rgbd_data_saver")

#         self.bridge = CvBridge()

#         self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#         self.save_root = os.path.expanduser("~/pose_estimation_data")
#         self.save_dir = self.save_root

#         self.rgb_dir = os.path.join(self.save_root, "rgb", self.timestamp)

#         self.depth_dir = os.path.join(self.save_root, "depth", self.timestamp)
#         self.depth_npy_dir = os.path.join(self.depth_dir, "npy")
#         self.depth_png_dir = os.path.join(self.depth_dir, "png")

#         self.camera_dir = os.path.join(self.save_root, "camera", self.timestamp)
#         self.masks_dir = os.path.join(self.save_root, "masks", self.timestamp)
#         self.meshes_dir = os.path.join(self.save_root, "meshes", self.timestamp)
#         self.outputs_dir = os.path.join(self.save_root, "outputs", self.timestamp)

#         os.makedirs(self.rgb_dir, exist_ok=True)
#         os.makedirs(self.depth_npy_dir, exist_ok=True)
#         os.makedirs(self.depth_png_dir, exist_ok=True)
#         os.makedirs(self.camera_dir, exist_ok=True)
#         os.makedirs(self.masks_dir, exist_ok=True)
#         os.makedirs(self.meshes_dir, exist_ok=True)
#         os.makedirs(self.outputs_dir, exist_ok=True)

#         self.rgb_msg = None
#         self.depth_msg = None
#         self.K_saved = False
#         self.frame_id = 0

#         self.rgb_topic = "/D435_head_camera/color/image_raw"
#         self.depth_topic = "/D435_head_camera/aligned_depth_to_color/image_raw"
#         self.camera_info_topic = "/D435_head_camera/color/camera_info"

#         self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10)
#         self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
#         self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

#         self.timer = self.create_timer(1.0, self.save_frame)

#         self.get_logger().info(f"Saving data to: {self.save_root} with run timestamp: {self.timestamp}")

#     def rgb_callback(self, msg):
#         self.rgb_msg = msg

#     def depth_callback(self, msg):
#         self.depth_msg = msg

#     def camera_info_callback(self, msg):
#         if self.K_saved:
#             return

#         K = np.array(msg.k).reshape(3, 3)

#         intrinsics = {
#             "width": int(msg.width),
#             "height": int(msg.height),
#             "fx": float(K[0, 0]),
#             "fy": float(K[1, 1]),
#             "cx": float(K[0, 2]),
#             "cy": float(K[1, 2]),
#             "K": K.tolist(),
#             "rgb_topic": self.rgb_topic,
#             "depth_topic": self.depth_topic,
#             "camera_info_topic": self.camera_info_topic,
#             "save_root": self.save_root,
#             "run_timestamp": self.timestamp,
#         }

#         out_path = os.path.join(self.camera_dir, "intrinsic.yaml")
#         with open(out_path, "w") as f:
#             yaml.dump(intrinsics, f)

#         self.get_logger().info(f"Saved camera intrinsics to {out_path}")
#         self.K_saved = True

#     def save_frame(self):
#         if self.rgb_msg is None or self.depth_msg is None:
#             self.get_logger().warn("Waiting for RGB and depth messages...")
#             return

#         rgb = self.bridge.imgmsg_to_cv2(self.rgb_msg, desired_encoding="rgb8")
#         rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

#         depth = self.bridge.imgmsg_to_cv2(
#             self.depth_msg,
#             desired_encoding="passthrough",
#         )
#         depth = depth.astype(np.float32)

#         # Se un giorno il topic fosse 16UC1, lo convertiamo da mm a metri.
#         # Nel tuo caso attuale è 32FC1, quindi dovrebbe già essere in metri.
#         if self.depth_msg.encoding == "16UC1":
#             depth = depth / 1000.0

#         frame_name = f"{self.frame_id:06d}"

#         rgb_path = os.path.join(self.rgb_dir, f"{frame_name}.png")
#         depth_npy_path = os.path.join(self.depth_npy_dir, f"{frame_name}.npy")
#         depth_png_path = os.path.join(self.depth_png_dir, f"{frame_name}.png")

#         # RGB
#         cv2.imwrite(rgb_path, rgb_bgr)

#         # Depth NPY: float32 in metri
#         np.save(depth_npy_path, depth)

#         # Depth PNG: uint16 in millimetri, compatibile con molti esempi FoundationPose
#         depth_mm = depth * 1000.0
#         depth_mm[~np.isfinite(depth_mm)] = 0.0
#         depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
#         cv2.imwrite(depth_png_path, depth_mm)

#         finite_depth = depth[np.isfinite(depth) & (depth > 0)]

#         if finite_depth.size > 0:
#             depth_min = float(np.min(finite_depth))
#             depth_max = float(np.max(finite_depth))
#         else:
#             depth_min = float("nan")
#             depth_max = float("nan")

#         self.get_logger().info(
#             f"Saved frame {frame_name} | "
#             f"rgb={rgb.shape}, depth={depth.shape}, "
#             f"depth_encoding={self.depth_msg.encoding}, "
#             f"depth_min={depth_min:.3f} m, depth_max={depth_max:.3f} m"
#         )

#         self.frame_id += 1


# def main():
#     rclpy.init()
#     node = RGBDDataSaver()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == "__main__":
#     main()


