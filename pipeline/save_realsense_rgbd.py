#python /home/user/PoseEstimation/pipeline/save_realsense_rgbd.py --save_dir ~/shared_data --max_frames 1

import argparse
import os
from datetime import datetime

import cv2
import numpy as np
import yaml

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class RealSenseRGBDDataSaver(Node):
    def __init__(
        self,
        capture_hz,
        save_dir,
        rgb_topic,
        depth_topic,
        camera_info_topic,
        max_rgb_depth_dt,
        resize_width,
        resize_height,
        max_frames,
    ):
        super().__init__("realsense_rgbd_data_saver")

        if capture_hz <= 0.0:
            raise ValueError("capture_hz must be greater than 0")

        self.bridge = CvBridge()
        self.save_root = os.path.expanduser(save_dir)
        self.save_dir = os.path.join(self.save_root, "realsense")
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.rgb_dir = os.path.join(self.save_dir, "rgb", self.run_timestamp)
        self.depth_npy_dir = os.path.join(self.save_dir, "depth", self.run_timestamp, "npy")
        self.depth_png_dir = os.path.join(self.save_dir, "depth", self.run_timestamp, "png")
        self.camera_dir = os.path.join(self.save_dir, "camera", self.run_timestamp)

        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_npy_dir, exist_ok=True)
        os.makedirs(self.depth_png_dir, exist_ok=True)
        os.makedirs(self.camera_dir, exist_ok=True)

        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.camera_info_topic = camera_info_topic
        self.max_rgb_depth_dt = max_rgb_depth_dt
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.max_frames = max_frames

        self.rgb_msg = None
        self.depth_msg = None
        self.camera_info_msg = None
        self.last_saved_rgb_stamp = None
        self.K_saved = False
        self.frame_id = 0
        self.stop_requested = False

        self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        self.timer = self.create_timer(1.0 / capture_hz, self.save_frame)

        self.get_logger().info(f"Saving data to: {self.save_dir}")
        self.get_logger().info(f"Run timestamp: {self.run_timestamp}")
        self.get_logger().info(f"Capture rate: {capture_hz:.3f} Hz")
        self.get_logger().info(f"RGB topic: {self.rgb_topic}")
        self.get_logger().info(f"Depth topic: {self.depth_topic}")
        self.get_logger().info(f"Camera info topic: {self.camera_info_topic}")

        if self.max_frames is None:
            self.get_logger().info("Capture mode: continuous")
        else:
            self.get_logger().info(f"Capture mode: stop after {self.max_frames} valid frame(s)")

        if self.resize_width is not None and self.resize_height is not None:
            self.get_logger().info(f"Saving resized frames: {self.resize_width}x{self.resize_height}")

    def rgb_callback(self, msg):
        self.rgb_msg = msg

    def depth_callback(self, msg):
        self.depth_msg = msg

    def camera_info_callback(self, msg):
        self.camera_info_msg = msg

        if self.K_saved:
            return

        K = np.array(msg.k).reshape(3, 3)
        width = int(msg.width)
        height = int(msg.height)

        if self.resize_width is not None and self.resize_height is not None:
            scale_x = float(self.resize_width) / float(width)
            scale_y = float(self.resize_height) / float(height)

            K = K.copy()
            K[0, 0] *= scale_x
            K[1, 1] *= scale_y
            K[0, 2] *= scale_x
            K[1, 2] *= scale_y

            width = self.resize_width
            height = self.resize_height

        intrinsics = {
            "width": width,
            "height": height,
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "K": K.tolist(),
            "rgb_topic": self.rgb_topic,
            "depth_topic": self.depth_topic,
            "camera_info_topic": self.camera_info_topic,
            "camera_frame_id": msg.header.frame_id,
            "run_timestamp": self.run_timestamp,
            "save_root": self.save_root,
            "dataset_root": self.save_dir,
            "source_width": int(msg.width),
            "source_height": int(msg.height),
        }

        if self.resize_width is not None and self.resize_height is not None:
            intrinsics["resize_width"] = self.resize_width
            intrinsics["resize_height"] = self.resize_height

        out_path = os.path.join(self.camera_dir, "intrinsics.yaml")

        with open(out_path, "w") as f:
            yaml.dump(intrinsics, f)

        self.get_logger().info(f"Saved camera intrinsics to {out_path}")
        self.K_saved = True

    def save_frame(self):
        if self.stop_requested:
            return

        if self.rgb_msg is None or self.depth_msg is None:
            self.get_logger().warn("Waiting for RGB and aligned depth messages...")
            return

        if self.camera_info_msg is None:
            self.get_logger().warn("Waiting for camera info...")
            return

        rgb_stamp = stamp_to_sec(self.rgb_msg.header.stamp)
        depth_stamp = stamp_to_sec(self.depth_msg.header.stamp)
        rgb_depth_dt = abs(rgb_stamp - depth_stamp)

        if self.last_saved_rgb_stamp is not None and rgb_stamp <= self.last_saved_rgb_stamp:
            return

        if rgb_depth_dt > self.max_rgb_depth_dt:
            self.get_logger().warn(
                f"Skipping frame because RGB/depth stamps differ by {rgb_depth_dt:.4f}s"
            )
            return

        rgb = self.bridge.imgmsg_to_cv2(self.rgb_msg, desired_encoding="rgb8")
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        depth = self.bridge.imgmsg_to_cv2(self.depth_msg, desired_encoding="passthrough")
        depth = depth.astype(np.float32)

        if self.depth_msg.encoding == "16UC1":
            depth = depth / 1000.0

        if self.resize_width is not None and self.resize_height is not None:
            target_size = (self.resize_width, self.resize_height)
            rgb_bgr = cv2.resize(rgb_bgr, target_size, interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)

        frame_name = f"{self.frame_id:06d}"

        rgb_path = os.path.join(self.rgb_dir, f"{frame_name}.png")
        depth_npy_path = os.path.join(self.depth_npy_dir, f"{frame_name}.npy")
        depth_png_path = os.path.join(self.depth_png_dir, f"{frame_name}.png")

        cv2.imwrite(rgb_path, rgb_bgr)
        np.save(depth_npy_path, depth)

        depth_png = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_png = np.clip(depth_png * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        cv2.imwrite(depth_png_path, depth_png)

        finite_depth = depth[np.isfinite(depth) & (depth > 0.0)]

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
            f"rgb_depth_dt={rgb_depth_dt:.4f}s, "
            f"depth_min={depth_min:.3f}, depth_max={depth_max:.3f}"
        )

        self.last_saved_rgb_stamp = rgb_stamp
        self.frame_id += 1

        if self.max_frames is not None and self.frame_id >= self.max_frames:
            self.get_logger().info(
                f"Reached max_frames={self.max_frames}. Stop requested."
            )
            self.stop_requested = True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--capture_hz",
        type=float,
        default=10.0,
        help="Frame capture frequency in Hz.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="~/shared_data",
        help="Root directory where rgb/depth/camera folders are saved.",
    )
    parser.add_argument(
        "--rgb_topic",
        type=str,
        default="/head_cam/color/image_raw",
    )
    parser.add_argument(
        "--depth_topic",
        type=str,
        default="/head_cam/aligned_depth_to_color/image_raw",
    )
    parser.add_argument(
        "--camera_info_topic",
        type=str,
        default="/head_cam/color/camera_info",
    )
    parser.add_argument(
        "--max_rgb_depth_dt",
        type=float,
        default=0.05,
        help="Maximum allowed RGB/depth timestamp difference in seconds.",
    )
    parser.add_argument(
        "--resize_width",
        type=int,
        default=None,
        help="Optional output RGB/depth width.",
    )
    parser.add_argument(
        "--resize_height",
        type=int,
        default=None,
        help="Optional output RGB/depth height.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help=(
            "Maximum number of valid RGB-D frames to save. "
            "If omitted, capture continues until stopped."
        ),
    )

    args = parser.parse_args()

    if (args.resize_width is None) != (args.resize_height is None):
        parser.error("--resize_width and --resize_height must be provided together")

    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max_frames must be greater than 0")

    rclpy.init()

    node = RealSenseRGBDDataSaver(
        capture_hz=args.capture_hz,
        save_dir=args.save_dir,
        rgb_topic=args.rgb_topic,
        depth_topic=args.depth_topic,
        camera_info_topic=args.camera_info_topic,
        max_rgb_depth_dt=args.max_rgb_depth_dt,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        max_frames=args.max_frames,
    )

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            if node.stop_requested:
                break

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()






### WORKING VERSION 
# import argparse
# import os
# from datetime import datetime

# import cv2
# import numpy as np
# import yaml

# import rclpy
# from cv_bridge import CvBridge
# from rclpy.node import Node
# from sensor_msgs.msg import CameraInfo, Image


# def stamp_to_sec(stamp):
#     return float(stamp.sec) + float(stamp.nanosec) * 1e-9


# class RealSenseRGBDDataSaver(Node):
#     def __init__(
#         self,
#         capture_hz,
#         save_dir,
#         rgb_topic,
#         depth_topic,
#         camera_info_topic,
#         max_rgb_depth_dt,
#         resize_width,
#         resize_height,
#     ):
#         super().__init__("realsense_rgbd_data_saver")

#         if capture_hz <= 0.0:
#             raise ValueError("capture_hz must be greater than 0")

#         self.bridge = CvBridge()
#         self.save_root = os.path.expanduser(save_dir)
#         self.save_dir = os.path.join(self.save_root, "realsense")
#         self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#         self.rgb_dir = os.path.join(self.save_dir, "rgb", self.run_timestamp)
#         self.depth_npy_dir = os.path.join(self.save_dir, "depth", self.run_timestamp, "npy")
#         self.depth_png_dir = os.path.join(self.save_dir, "depth", self.run_timestamp, "png")
#         self.camera_dir = os.path.join(self.save_dir, "camera", self.run_timestamp)

#         os.makedirs(self.rgb_dir, exist_ok=True)
#         os.makedirs(self.depth_npy_dir, exist_ok=True)
#         os.makedirs(self.depth_png_dir, exist_ok=True)
#         os.makedirs(self.camera_dir, exist_ok=True)

#         self.rgb_topic = rgb_topic
#         self.depth_topic = depth_topic
#         self.camera_info_topic = camera_info_topic
#         self.max_rgb_depth_dt = max_rgb_depth_dt
#         self.resize_width = resize_width
#         self.resize_height = resize_height

#         self.rgb_msg = None
#         self.depth_msg = None
#         self.camera_info_msg = None
#         self.last_saved_rgb_stamp = None
#         self.K_saved = False
#         self.frame_id = 0

#         self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10)
#         self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
#         self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

#         self.timer = self.create_timer(1.0 / capture_hz, self.save_frame)

#         self.get_logger().info(f"Saving data to: {self.save_dir}")
#         self.get_logger().info(f"Run timestamp: {self.run_timestamp}")
#         self.get_logger().info(f"Capture rate: {capture_hz:.3f} Hz")
#         self.get_logger().info(f"RGB topic: {self.rgb_topic}")
#         self.get_logger().info(f"Depth topic: {self.depth_topic}")
#         self.get_logger().info(f"Camera info topic: {self.camera_info_topic}")
#         if self.resize_width is not None and self.resize_height is not None:
#             self.get_logger().info(f"Saving resized frames: {self.resize_width}x{self.resize_height}")

#     def rgb_callback(self, msg):
#         self.rgb_msg = msg

#     def depth_callback(self, msg):
#         self.depth_msg = msg

#     def camera_info_callback(self, msg):
#         self.camera_info_msg = msg

#         if self.K_saved:
#             return

#         K = np.array(msg.k).reshape(3, 3)
#         width = int(msg.width)
#         height = int(msg.height)

#         if self.resize_width is not None and self.resize_height is not None:
#             scale_x = float(self.resize_width) / float(width)
#             scale_y = float(self.resize_height) / float(height)
#             K = K.copy()
#             K[0, 0] *= scale_x
#             K[1, 1] *= scale_y
#             K[0, 2] *= scale_x
#             K[1, 2] *= scale_y
#             width = self.resize_width
#             height = self.resize_height

#         intrinsics = {
#             "width": width,
#             "height": height,
#             "fx": float(K[0, 0]),
#             "fy": float(K[1, 1]),
#             "cx": float(K[0, 2]),
#             "cy": float(K[1, 2]),
#             "K": K.tolist(),
#             "rgb_topic": self.rgb_topic,
#             "depth_topic": self.depth_topic,
#             "camera_info_topic": self.camera_info_topic,
#             "camera_frame_id": msg.header.frame_id,
#             "run_timestamp": self.run_timestamp,
#             "save_root": self.save_root,
#             "dataset_root": self.save_dir,
#             "source_width": int(msg.width),
#             "source_height": int(msg.height),
#         }

#         if self.resize_width is not None and self.resize_height is not None:
#             intrinsics["resize_width"] = self.resize_width
#             intrinsics["resize_height"] = self.resize_height

#         out_path = os.path.join(self.camera_dir, "intrinsics.yaml")
#         with open(out_path, "w") as f:
#             yaml.dump(intrinsics, f)

#         self.get_logger().info(f"Saved camera intrinsics to {out_path}")
#         self.K_saved = True

#     def save_frame(self):
#         if self.rgb_msg is None or self.depth_msg is None:
#             self.get_logger().warn("Waiting for RGB and aligned depth messages...")
#             return

#         if self.camera_info_msg is None:
#             self.get_logger().warn("Waiting for camera info...")
#             return

#         rgb_stamp = stamp_to_sec(self.rgb_msg.header.stamp)
#         depth_stamp = stamp_to_sec(self.depth_msg.header.stamp)
#         rgb_depth_dt = abs(rgb_stamp - depth_stamp)

#         if self.last_saved_rgb_stamp is not None and rgb_stamp <= self.last_saved_rgb_stamp:
#             return

#         if rgb_depth_dt > self.max_rgb_depth_dt:
#             self.get_logger().warn(
#                 f"Skipping frame because RGB/depth stamps differ by {rgb_depth_dt:.4f}s"
#             )
#             return

#         rgb = self.bridge.imgmsg_to_cv2(self.rgb_msg, desired_encoding="rgb8")
#         rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

#         depth = self.bridge.imgmsg_to_cv2(self.depth_msg, desired_encoding="passthrough")
#         depth = depth.astype(np.float32)

#         if self.depth_msg.encoding == "16UC1":
#             depth = depth / 1000.0

#         if self.resize_width is not None and self.resize_height is not None:
#             target_size = (self.resize_width, self.resize_height)
#             rgb_bgr = cv2.resize(rgb_bgr, target_size, interpolation=cv2.INTER_AREA)
#             depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)

#         frame_name = f"{self.frame_id:06d}"

#         rgb_path = os.path.join(self.rgb_dir, f"{frame_name}.png")
#         depth_npy_path = os.path.join(self.depth_npy_dir, f"{frame_name}.npy")
#         depth_png_path = os.path.join(self.depth_png_dir, f"{frame_name}.png")

#         cv2.imwrite(rgb_path, rgb_bgr)
#         np.save(depth_npy_path, depth)

#         depth_png = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
#         depth_png = np.clip(depth_png * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
#         cv2.imwrite(depth_png_path, depth_png)

#         finite_depth = depth[np.isfinite(depth) & (depth > 0.0)]
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
#             f"rgb_depth_dt={rgb_depth_dt:.4f}s, "
#             f"depth_min={depth_min:.3f}, depth_max={depth_max:.3f}"
#         )

#         self.last_saved_rgb_stamp = rgb_stamp
#         self.frame_id += 1


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--capture_hz",
#         type=float,
#         default=10.0,
#         help="Frame capture frequency in Hz.",
#     )
#     parser.add_argument(
#         "--save_dir",
#         type=str,
#         #default="~/pose_estimation_data",
#         default="~/shared_data",
#         help="Root directory where rgb/depth/camera folders are saved.",
#     )
#     parser.add_argument(
#         "--rgb_topic",
#         type=str,
#         #default="/camera/camera/color/image_raw",
#         default="/head_cam/color/image_raw",
#     )
#     parser.add_argument(
#         "--depth_topic",
#         type=str,
#         #default="/camera/camera/aligned_depth_to_color/image_raw",
#         default="/head_cam/aligned_depth_to_color/image_raw",
#     )
#     parser.add_argument(
#         "--camera_info_topic",
#         type=str,
#         #default="/camera/camera/color/camera_info",
#         default="/head_cam/color/camera_info",
#     )
#     parser.add_argument(
#         "--max_rgb_depth_dt",
#         type=float,
#         default=0.05,
#         help="Maximum allowed RGB/depth timestamp difference in seconds.",
#     )
#     parser.add_argument(
#         "--resize_width",
#         type=int,
#         default=None,
#         help="Optional output RGB/depth width.",
#     )
#     parser.add_argument(
#         "--resize_height",
#         type=int,
#         default=None,
#         help="Optional output RGB/depth height.",
#     )
#     args = parser.parse_args()

#     if (args.resize_width is None) != (args.resize_height is None):
#         parser.error("--resize_width and --resize_height must be provided together")

#     rclpy.init()
#     node = RealSenseRGBDDataSaver(
#         capture_hz=args.capture_hz,
#         save_dir=args.save_dir,
#         rgb_topic=args.rgb_topic,
#         depth_topic=args.depth_topic,
#         camera_info_topic=args.camera_info_topic,
#         max_rgb_depth_dt=args.max_rgb_depth_dt,
#         resize_width=args.resize_width,
#         resize_height=args.resize_height,
#     )

#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()
