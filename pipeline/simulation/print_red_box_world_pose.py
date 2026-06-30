import argparse
import select
import subprocess


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


def format_stamp(stamp):
    if stamp is None:
        return "unknown"
    return f"{stamp['sec']}.{stamp['nsec']:09d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="/world/default/pose/info")
    parser.add_argument("--object_name", type=str, default="box_red_001")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    cmd = ["gz", "topic", "-e", "-t", args.topic]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    header_block = []
    current_stamp = None
    in_header = False
    in_pose = False
    depth = 0
    block = []

    try:
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], 0.5)
            if not ready:
                if proc.poll() is not None:
                    raise RuntimeError(f"Gazebo command stopped: {' '.join(cmd)}")
                continue

            line = proc.stdout.readline()
            if line == "":
                if proc.poll() is not None:
                    raise RuntimeError(f"Gazebo command stopped: {' '.join(cmd)}")
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
                if pose["name"] == args.object_name:
                    position = pose["position"]
                    orientation = pose["orientation"]
                    print(
                        f"stamp={format_stamp(current_stamp)} "
                        f"name={args.object_name} "
                        f"position_xyz=({position['x']:.9f}, {position['y']:.9f}, {position['z']:.9f}) "
                        f"orientation_xyzw=({orientation['x']:.9f}, {orientation['y']:.9f}, "
                        f"{orientation['z']:.9f}, {orientation['w']:.9f})",
                        flush=True,
                    )

                    if args.once:
                        return

                in_pose = False
                block = []
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
