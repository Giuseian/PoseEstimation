import argparse
import os
import select
import subprocess
import threading

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


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


class GazeboBoxPoseBridge(Node):
    def __init__(self, gz_topic: str, object_name: str, ros_topic: str):
        super().__init__("gz_box_pose_bridge")

        self.gz_topic = gz_topic
        self.object_name = object_name
        self.ros_topic = ros_topic

        self.proc = None
        self.stop_event = threading.Event()
        self.thread = None

        self.publisher = self.create_publisher(PoseStamped, ros_topic, 100)
        self.start_reader()

        self.get_logger().info(f"Reading Gazebo topic: {gz_topic}")
        self.get_logger().info(f"Object name: {object_name}")
        self.get_logger().info(f"Publishing ROS2 topic: {ros_topic}")

    def destroy_node(self):
        self.stop_event.set()

        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

        if self.thread is not None:
            self.thread.join(timeout=0.5)

        super().destroy_node()

    def start_reader(self):
        cmd = ["gz", "topic", "-e", "-t", self.gz_topic]

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self.get_logger().error(f"Could not run {' '.join(cmd)}: {exc}")
            return

        self.thread = threading.Thread(target=self.reader_loop, daemon=True)
        self.thread.start()

    def publish_pose(self, stamp, pose):
        msg = PoseStamped()
        msg.header.frame_id = "world"

        if stamp is not None:
            msg.header.stamp.sec = int(stamp["sec"])
            msg.header.stamp.nanosec = int(stamp["nsec"])
        else:
            msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.position.x = pose["position"]["x"]
        msg.pose.position.y = pose["position"]["y"]
        msg.pose.position.z = pose["position"]["z"]

        msg.pose.orientation.x = pose["orientation"]["x"]
        msg.pose.orientation.y = pose["orientation"]["y"]
        msg.pose.orientation.z = pose["orientation"]["z"]
        msg.pose.orientation.w = pose["orientation"]["w"]

        self.publisher.publish(msg)

    def reader_loop(self):
        header_block = []
        current_stamp = None
        in_header = False
        in_pose = False
        depth = 0
        block = []

        while not self.stop_event.is_set():
            try:
                ready, _, _ = select.select([self.proc.stdout], [], [], 0.1)
                if not ready:
                    if self.proc.poll() is not None:
                        self.get_logger().warn(f"Gazebo command stopped: {self.gz_topic}")
                        return
                    continue

                line = self.proc.stdout.readline()
                if line == "":
                    if self.proc.poll() is not None:
                        self.get_logger().warn(f"Gazebo command stopped: {self.gz_topic}")
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
                        current_stamp = parse_gz_header_stamp(header_block)
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
                    if pose["name"] == self.object_name:
                        self.publish_pose(current_stamp, pose)

                    in_pose = False
                    block = []
            except Exception as exc:
                self.get_logger().warn(f"Gazebo reader failed: {exc}")
                return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gz_topic", type=str, default="/world/default/pose/info")
    parser.add_argument("--object_name", type=str, default="box_red_001")
    parser.add_argument("--ros_topic", type=str, default="/gt/box_red_001/world_pose")
    args = parser.parse_args()

    rclpy.init()
    node = GazeboBoxPoseBridge(
        gz_topic=args.gz_topic,
        object_name=args.object_name,
        ros_topic=args.ros_topic,
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
