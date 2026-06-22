import argparse
import io
import math
import time
from pathlib import Path

import airsim
from PIL import Image

from memory_graph import MemoryGraph


def warn(message):
    print(f"[WARNING] {message}")


def get_pose_yaw(client, vehicle_name):
    try:
        pose = client.simGetVehiclePose(vehicle_name=vehicle_name)
    except TypeError as exc:
        warn(f"simGetVehiclePose(vehicle_name=...) failed: {exc}; retrying without vehicle_name")
        pose = client.simGetVehiclePose()

    position = pose.position
    roll, pitch, yaw = airsim.to_eularian_angles(pose.orientation)
    return position.x_val, position.y_val, position.z_val, math.degrees(yaw)


def print_pose(client, vehicle_name, prefix="[POSE]"):
    x, y, z, yaw_deg = get_pose_yaw(client, vehicle_name)
    print(f"{prefix} x={x:.3f}, y={y:.3f}, z={z:.3f}, yaw_deg={yaw_deg:.2f}")


def pose_distance(pose_a, pose_b):
    ax, ay, az, _ = pose_a
    bx, by, bz, _ = pose_b
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def yaw_delta_deg(yaw_a, yaw_b):
    return abs((yaw_a - yaw_b + 180.0) % 360.0 - 180.0)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value}")


def join_async_call_with_vehicle_fallback(client, method_name, args, vehicle_name):
    method = getattr(client, method_name)
    try:
        method(*args, vehicle_name=vehicle_name).join()
    except TypeError as exc:
        warn(f"{method_name}(..., vehicle_name=...) failed: {exc}; retrying without vehicle_name")
        method(*args).join()


def initialize_drone(client, vehicle_name):
    client.enableApiControl(True, vehicle_name=vehicle_name)
    client.armDisarm(True, vehicle_name=vehicle_name)
    join_async_call_with_vehicle_fallback(client, "takeoffAsync", (), vehicle_name)


def move_body_frame(client, vx, vy, vz, duration, vehicle_name):
    join_async_call_with_vehicle_fallback(
        client,
        "moveByVelocityBodyFrameAsync",
        (vx, vy, vz, duration),
        vehicle_name,
    )


def rotate_to_yaw(client, target_yaw_deg, vehicle_name):
    join_async_call_with_vehicle_fallback(
        client,
        "rotateToYawAsync",
        (target_yaw_deg,),
        vehicle_name,
    )


def save_rgb_image(client, camera_name, image_path, success_prefix="[IMAGE]"):
    try:
        responses = client.simGetImages(
            [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)]
        )
        if not responses:
            raise RuntimeError("simGetImages returned no responses")

        response = responses[0]
        image_data = response.image_data_uint8
        if not image_data:
            raise RuntimeError(
                "empty RGB image response; try rerunning with --camera_name 0"
            )

        raw = bytes(image_data)
        expected_rgb = response.width * response.height * 3
        expected_rgba = response.width * response.height * 4

        if response.width > 0 and response.height > 0 and len(raw) in (expected_rgb, expected_rgba):
            mode = "RGB" if len(raw) == expected_rgb else "RGBA"
            image = Image.frombytes(mode, (response.width, response.height), raw)
            image = image.convert("RGB")
        else:
            image = Image.open(io.BytesIO(raw)).convert("RGB")

        image.save(image_path, quality=95)
    except Exception as exc:
        print(
            f"[IMAGE ERROR] camera_name={camera_name}, error={exc}. "
            "Try rerunning with --camera_name 0."
        )
        return False

    print(f"{success_prefix} saved to {image_path}")
    return True


def save_current_rgb_image(client, camera_name, image_dir, image_index):
    image_path = image_dir / f"manual_view_{image_index:04d}.jpg"
    if not save_rgb_image(client, camera_name, image_path):
        return image_index
    return image_index + 1


def add_map_node(client, args, graph, image_dir, output_dir, reason, pose=None):
    node_id = len(graph.nodes)
    image_path = image_dir / f"node_{node_id:04d}_rgb.jpg"
    if pose is None:
        pose = get_pose_yaw(client, args.vehicle_name)
    pose_x, pose_y, pose_z, yaw_deg = pose
    if not save_rgb_image(client, args.camera_name, image_path, success_prefix="[IMAGE]"):
        return None

    node = graph.add_node(
        env_name=args.env_name,
        pose_x=pose_x,
        pose_y=pose_y,
        pose_z=pose_z,
        yaw_deg=yaw_deg,
        rgb_path=image_path,
        auto_tag=reason,
        reason=reason,
    )
    print(
        "[MAP] add node "
        f"id={node['node_id']}, "
        f"reason={reason}, "
        f"pose=({pose_x:.3f}, {pose_y:.3f}, {pose_z:.3f}), "
        f"yaw={yaw_deg:.2f}, image={image_path}"
    )
    save_memory_graph(graph, output_dir)
    return node


def save_memory_graph(graph, output_dir):
    json_path = output_dir / "memory_graph.json"
    csv_path = output_dir / "memory_nodes.csv"
    graph.save_json(json_path)
    graph.save_nodes_csv(csv_path)
    print(f"[MAP] saved graph to {json_path} and {csv_path}")


def maybe_auto_add_node(client, args, graph, image_dir, output_dir):
    if not args.auto_sample or not graph.nodes:
        return None

    current_pose = get_pose_yaw(client, args.vehicle_name)
    last_node = graph.nodes[-1]
    last_pose = (
        last_node["pose_x"],
        last_node["pose_y"],
        last_node["pose_z"],
        last_node["yaw_deg"],
    )
    distance = pose_distance(current_pose, last_pose)
    yaw_delta = yaw_delta_deg(current_pose[3], last_pose[3])

    if distance >= args.sample_dist:
        return add_map_node(
            client,
            args,
            graph,
            image_dir,
            output_dir,
            reason="distance",
            pose=current_pose,
        )
    if yaw_delta >= args.sample_yaw_deg:
        return add_map_node(
            client,
            args,
            graph,
            image_dir,
            output_dir,
            reason="yaw",
            pose=current_pose,
        )
    if args.verbose_map_check:
        print(
            "[MAP] no auto node: "
            f"dist_since_last={distance:.2f}, yaw_delta={yaw_delta:.2f}"
        )
    return None


def print_recent_nodes(graph, limit=5):
    if not graph.nodes:
        print("[MAP] no nodes")
        return
    for node in graph.nodes[-limit:]:
        print(
            "[MAP] node "
            f"id={node['node_id']}, "
            f"reason={node['reason']}, "
            f"auto_tag={node['auto_tag']}, "
            f"manual_tag={node['manual_tag']}, "
            f"pose=({node['pose_x']:.3f}, {node['pose_y']:.3f}, {node['pose_z']:.3f}), "
            f"yaw={node['yaw_deg']:.2f}"
        )


def print_available_vehicles(client):
    try:
        vehicles = client.listVehicles()
    except AttributeError as exc:
        warn(f"client.listVehicles() is not available: {exc}")
    except Exception as exc:
        warn(f"client.listVehicles() failed: {exc}")
    else:
        print(f"vehicles={vehicles}")


def connect_airsim(sim_ip, sim_port):
    candidate_ports = [sim_port]
    for fallback_port in (41452, 41451):
        if fallback_port not in candidate_ports:
            candidate_ports.append(fallback_port)

    errors = []
    for port in candidate_ports:
        print(f"[INFO] Connecting AirSim RPC at {sim_ip}:{port}")
        client = airsim.MultirotorClient(ip=sim_ip, port=port)
        try:
            client.confirmConnection()
        except Exception as exc:
            warn(f"AirSim RPC connection failed at {sim_ip}:{port}: {exc}")
            errors.append(f"{sim_ip}:{port} -> {exc}")
            continue
        return client, port

    error_text = "; ".join(errors)
    raise RuntimeError(
        "Unable to connect to AirSim RPC. Make sure env_airsim_16 is still running "
        "in another terminal and that its ApiServerPort is reachable. "
        f"Tried: {error_text}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Manual AirSim control for Phase 1 connection checks.")
    parser.add_argument("--env_name", default="env_airsim_16")
    parser.add_argument("--vehicle_name", default="drone_1")
    parser.add_argument("--forward_step_m", type=float, default=6.0)
    parser.add_argument("--vertical_step_m", type=float, default=3.0)
    parser.add_argument("--yaw_step_deg", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=3.0)
    parser.add_argument("--sim_ip", default="127.0.0.1")
    parser.add_argument("--sim_port", type=int, default=41452)
    parser.add_argument("--camera_name", default="front_custom")
    parser.add_argument("--output_dir", default="./memory_outputs/env_airsim_16_manual_map")
    parser.add_argument("--sample_dist", type=float, default=15.0)
    parser.add_argument("--sample_yaw_deg", type=float, default=45.0)
    parser.add_argument("--auto_sample", type=parse_bool, default=True)
    parser.add_argument("--save_final_node", type=parse_bool, default=True)
    parser.add_argument("--verbose_map_check", type=parse_bool, default=False)
    return parser.parse_args()


def require_positive(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def run_command(client, args, command, image_dir, image_index, graph, output_dir):
    command_key = command.lower()
    speed = args.speed
    forward_duration = args.forward_step_m / speed
    vertical_duration = args.vertical_step_m / speed
    movement_command = False

    if command_key == "w":
        move_body_frame(client, speed, 0.0, 0.0, forward_duration, args.vehicle_name)
        movement_command = True
    elif command_key == "s":
        move_body_frame(client, -speed, 0.0, 0.0, forward_duration, args.vehicle_name)
        movement_command = True
    elif command_key == "a":
        move_body_frame(client, 0.0, -speed, 0.0, forward_duration, args.vehicle_name)
        movement_command = True
    elif command_key == "d":
        move_body_frame(client, 0.0, speed, 0.0, forward_duration, args.vehicle_name)
        movement_command = True
    elif command_key == "q":
        move_body_frame(client, 0.0, 0.0, -speed, vertical_duration, args.vehicle_name)
        movement_command = True
    elif command_key == "e":
        move_body_frame(client, 0.0, 0.0, speed, vertical_duration, args.vehicle_name)
        movement_command = True
    elif command_key in ("j", "l"):
        _, _, _, current_yaw_deg = get_pose_yaw(client, args.vehicle_name)
        delta = args.yaw_step_deg if command_key == "l" else -args.yaw_step_deg
        rotate_to_yaw(client, current_yaw_deg + delta, args.vehicle_name)
        movement_command = True
    elif command_key == "p":
        pass
    elif command_key == "v":
        return save_current_rgb_image(client, args.camera_name, image_dir, image_index)
    elif command_key == "m":
        add_map_node(client, args, graph, image_dir, output_dir, reason="manual")
    elif command_key == "t":
        if not graph.nodes:
            print("[MAP] no node to tag")
            return image_index
        tag = input("tag> ")
        node = graph.update_node_tag(graph.nodes[-1]["node_id"], tag)
        print(f"[MAP] node id={node['node_id']} tag={tag}")
        save_memory_graph(graph, output_dir)
    elif command_key == "x":
        save_memory_graph(graph, output_dir)
    elif command_key == "list":
        print_recent_nodes(graph)
    elif command_key.startswith("tag "):
        parts = command.split(maxsplit=2)
        if len(parts) < 3:
            print("[MAP] usage: tag <node_id> <tag_text>")
            return image_index
        node_id = int(parts[1])
        tag = parts[2]
        node = graph.update_node_tag(node_id, tag)
        print(f"[MAP] node id={node['node_id']} tag={tag}")
        save_memory_graph(graph, output_dir)
    else:
        print(f"[INFO] Unknown command: {command}")
        return image_index

    time.sleep(0.3)
    print_pose(client, args.vehicle_name)
    if movement_command:
        maybe_auto_add_node(client, args, graph, image_dir, output_dir)
    return image_index


def main():
    args = parse_args()
    require_positive(args.forward_step_m, "forward_step_m")
    require_positive(args.vertical_step_m, "vertical_step_m")
    require_positive(args.speed, "speed")
    require_positive(args.sample_dist, "sample_dist")
    require_positive(args.sample_yaw_deg, "sample_yaw_deg")
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    image_index = 0
    graph = MemoryGraph()

    client, connected_port = connect_airsim(args.sim_ip, args.sim_port)

    print(f"env_name={args.env_name}")
    print(f"vehicle_name={args.vehicle_name}")
    print(f"airsim_rpc={args.sim_ip}:{connected_port}")
    print(f"camera_name={args.camera_name}")
    print(f"image_dir={image_dir}")
    print(f"[CONFIG] forward_step_m={args.forward_step_m}")
    print(f"[CONFIG] vertical_step_m={args.vertical_step_m}")
    print(f"[CONFIG] yaw_step_deg={args.yaw_step_deg}")
    print(f"[CONFIG] speed={args.speed}")
    print(f"[CONFIG] sample_dist={args.sample_dist}")
    print(f"[CONFIG] sample_yaw_deg={args.sample_yaw_deg}")
    print(f"[CONFIG] auto_sample={args.auto_sample}")
    print_available_vehicles(client)

    initialize_drone(client, args.vehicle_name)
    time.sleep(0.3)
    print_pose(client, args.vehicle_name)
    add_map_node(client, args, graph, image_dir, output_dir, reason="start")

    print("[INFO] Commands: w/s/a/d move, q/e up/down, j/l yaw, p pose, v save image, m add node, t tag latest, tag <id> <text>, list, x save graph, quit exit")
    while True:
        command = input("cmd> ").strip()
        if command == "":
            continue
        if command.lower() == "quit":
            print_pose(client, args.vehicle_name, prefix="[FINAL POSE]")
            if args.save_final_node:
                add_map_node(client, args, graph, image_dir, output_dir, reason="final")
            save_memory_graph(graph, output_dir)
            try:
                join_async_call_with_vehicle_fallback(client, "hoverAsync", (), args.vehicle_name)
            except Exception as exc:
                warn(f"hoverAsync failed during exit: {exc}")
            break

        try:
            image_index = run_command(
                client,
                args,
                command,
                image_dir,
                image_index,
                graph,
                output_dir,
            )
        except Exception as exc:
            print(f"[ERROR] Command '{command}' failed: {exc}")


if __name__ == "__main__":
    main()
