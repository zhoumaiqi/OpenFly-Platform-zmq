
import argparse
import csv
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
import os, json
from extern.hf.configuration_prismatic import OpenFlyConfig
from extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


AutoConfig.register("openvla", OpenFlyConfig)
AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)


from unrealcv import Client  
import cv2  
import numpy as np  
import io
import time
import math
import re
import subprocess, threading
import airsim
from common import *
import psutil
import requests
import random
from ours_spf_agent import OurSPFAgent
from memory_graph import MemoryGraph




def kill_env_process(keyword):
    result = subprocess.run(['pgrep', '-n', keyword], stdout=subprocess.PIPE)
    cr_pid = result.stdout.decode().strip()
    if len(cr_pid) > 0:
        subprocess.run(['kill', '-9', cr_pid])

class AirsimBridge:
    def __init__(self, env_name):
        self.env_name = env_name
        self._sim_thread = threading.Thread(target=self._init_airsim_sim)
        self._sim_thread.start()
        time.sleep(10)

        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()
        self._client.enableApiControl(True)
        self._client.armDisarm(True)

    def _init_airsim_sim(self):
        env_dir = "/workspace/smbu_zzh/OpenFly-Platform/envs/airsim/" + self.env_name 

        if not os.path.exists(env_dir):
            raise ValueError(f"Specified directory {env_dir} does not exist")
        
        command = ["bash", f"{env_dir}/LinuxNoEditor/start.sh"]
        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()
        # print("Command output:\n", stdout)

    def set_camera_pose(self, x, y, z, pitch, yaw, roll):
        target_pose = airsim.Pose(airsim.Vector3r(x, -y, -z),
                                  airsim.to_quaternion(math.radians(pitch), 0, math.radians(-yaw)))
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        self._client.simSetVehiclePose(target_pose, True)

    def set_drone_pos(self, x, y, z, pitch, yaw, roll):
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        qua = euler_to_quaternion(pitch, -yaw, roll)
        target_pose = airsim.Pose(airsim.Vector3r(x, y, z),
                                  airsim.Quaternionr(qua[0], qua[1], qua[2], qua[3]))
        self._client.simSetVehiclePose(target_pose, True)
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        time.sleep(0.1)

    def _camera_init(self):
        '''Camera initialization'''
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(15), 0, 0))
        self._client.simSetCameraPose("0", camera_pose)
        time.sleep(1)

    def _drone_init(self):
        '''Drone initialization'''
        self.set_drone_pos(0, 0, 0, 0, 0, 0)
        time.sleep(1)

    def get_camera_data(self, camera_type = 'color'):
        valid_types = {'color', 'object_mask', 'depth'}
        if camera_type not in valid_types:
            raise ValueError(f"Invalid camera type. Expected one of {valid_types}, but got '{camera_type}'.")

        if camera_type == 'color':
            image_type = airsim.ImageType.Scene
        elif camera_type == 'depth':
            image_type = airsim.ImageType.DepthPlanar
        else:
            image_type = airsim.ImageType.Segmentation

        responses = self._client.simGetImages([airsim.ImageRequest('front_custom', image_type, False, False)])
        response = responses[0]
        if response.pixels_as_float:
            img_data = np.array(response.image_data_float, dtype=np.float32)
            img_data = np.reshape(img_data, (response.height, response.width))
        else:
            img_data = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_data = img_data.reshape(response.height, response.width, 3)

        return img_data

    def save_image(self, image_data, file_path):
        cv2.imwrite(file_path, image_data)

    def process_camera_data(self, file_path, camera_type='color'):
        img = self.get_camera_data(camera_type)
        self.save_image(img, file_path)
        print("Image saved")

class UEBridge:
    def __init__(self, ue_ip, ue_port, env_name):
        self.kill_failed_process()
        time.sleep(10)

        # port = self.find_available_port()

        port = random.randint(9000, 9100)
        print(f"Available port: {port}")
        self.modify_port_in_ini(port, env_name)
        ue_port = port

        self.env_name = env_name
        self._sim_thread = threading.Thread(target=self._init_ue_sim)
        self._sim_thread.start()
        time.sleep(15)

        self._client = Client((ue_ip, ue_port))
        self._connection_check()

        self._camera_init()
        print("cam init")

    def find_available_port(self):
        port = 9000
        while True:
            result = subprocess.run(['lsof', f'-i:{port}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            netstat_output = result.stdout.decode()

            if f'PID' not in netstat_output:
                return port
            port += 1

    def modify_port_in_ini(self, port, ue_env_name):
        ini_file = f"envs/ue/{ue_env_name}/City_UE52/Binaries/Linux/unrealcv.ini"
        with open(ini_file, 'r') as file:
            lines = file.readlines()

        with open(ini_file, 'w') as file:
            for line in lines:
                if line.startswith("Port="):
                    file.write(f"Port={port}\n")
                else:
                    file.write(line)

    def kill_failed_process(self):
        result = subprocess.run(['pgrep', '-n', 'CrashReport'], stdout=subprocess.PIPE)
        cr_pid = result.stdout.decode().strip()
        if len(cr_pid) > 0:
            subprocess.run(['kill', '-9', cr_pid])

        result = subprocess.run(['pgrep', '-n', 'CitySample'], stdout=subprocess.PIPE)
        cr_pid = result.stdout.decode().strip()
        if len(cr_pid) > 0:
            subprocess.run(['kill', '-9', cr_pid])

    def _init_ue_sim(self):
        env_dir = "envs/ue/" + self.env_name
        if not os.path.exists(env_dir):
            raise ValueError(f"Specified directory {env_dir} does not exist")

        command = ["bash", f"{env_dir}/CitySample.sh"]

        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()
        # print("Command output:\n", stdout)
        time.sleep(2)

    def __del__(self):
        self._client.disconnect()

    def _connection_check(self):
        '''Check if connected'''
        if self._client.connect():
            print('UnrealCV connected successfully')
        else:
            print('UnrealCV is not connected')
            exit()

    def set_camera_pose(self, x, y, z, pitch, yaw, roll):
        '''Set camera position'''
        x = x * 100
        y = - y * 100
        z = z * 100
        camera_settings = {
            'location': {'x': x, 'y': y, 'z': z},
            'rotation': {'pitch': pitch, 'yaw': -yaw, 'roll': roll}
        }

        self._client.request('vset /camera/0/location {x} {y} {z}'.format(**camera_settings['location']))
        self._client.request('vset /camera/1/location {x} {y} {z}'.format(**camera_settings['location']))
        self._client.request('vset /camera/0/rotation {pitch} {yaw} {roll}'.format(**camera_settings['rotation']))
        self._client.request('vset /camera/1/rotation {pitch} {yaw} {roll}'.format(**camera_settings['rotation']))
        print('camera_settings', camera_settings)

    def _camera_init(self):
        '''Camera initialization'''
        time.sleep(2)
        self._client.request('vset /cameras/spawn')
        self._client.request('vset /camera/1/size 1920 1080')
        time.sleep(2)
        self.set_camera_pose(150, 400, 15, 0, 0, 0)  # Initial position
        time.sleep(2)

    def get_camera_data(self, camera_type = 'lit'):
        valid_types = {'lit', 'object_mask', 'depth'}
        if camera_type not in valid_types:
            raise ValueError(f"Invalid camera type. Expected one of {valid_types}, but got '{camera_type}'.")

        if camera_type == 'lit':
            data = self._client.request('vget /camera/1/lit png')
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        elif camera_type == 'object_mask':
            data = self._client.request('vget /camera/1/object_mask png')
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        elif camera_type == 'depth':
            data = self._client.request('vget /camera/1/depth npy')
            depth_np = np.load(io.BytesIO(data))
            return depth_np  # Return depth data

    def save_image(self, image_data, file_path):
        cv2.imwrite(file_path, image_data)

    def process_camera_data(self, file_path, camera_type='lit'):
        img = self.get_camera_data(camera_type)
        self.save_image(img, file_path)

class GSBridge:  
    def __init__(self, env_name):  
        self.env_name = env_name
        self._sim_thread = threading.Thread(target=self._init_gs_sim)
        self._sim_thread.start()
        self.url = "http://localhost:18080/render"
        time.sleep(10)
  
    def _init_gs_sim(self):
        # dataset_dir = "envs/gs/" + self.env_name  
        dataset_dir = "/media/pjlabrl/hdd/all_files_relate_to_3dgs/reconstruction_result/nwpu02"
        gs_vis_tool_dir = "envs/gs/SIBR_viewers/"  
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Specified directory {dataset_dir} does not exist")
        command = [
            gs_vis_tool_dir + "install/bin/SIBR_gaussianHierarchyViewer_app",
            "--path", f"{dataset_dir}/camera_calibration/aligned",
            "--scaffold", f"{dataset_dir}/output/scaffold/point_cloud/iteration_30000",
            "--model-path", f"{dataset_dir}/output/merged.hier",
            "--images-path", f"{dataset_dir}/camera_calibration/rectified/images"
        ]
        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()
        print("Command output:\n", stdout)

    def transform_euler_to_new_frame(self, roll, pitch, yaw):
        R = euler_to_rotation_matrix(roll, pitch, yaw)
        transformation_matrix = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])
        new_R = np.dot(transformation_matrix, R)
        new_roll, new_pitch, new_yaw = rotation_matrix_to_euler_angles(new_R)
        return new_roll, new_pitch, new_yaw
    
    def rotation_matrix_roll(self, roll):
        return np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

    def rotation_matrix_pitch(self, pitch):
        return np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

    def rotation_matrix_yaw(self, yaw):
        return np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

    def transform_to_camera_frame(self, roll, pitch, yaw):
        R_roll = self.rotation_matrix_roll(roll)
        R_pitch = self.rotation_matrix_pitch(pitch)
        R_yaw = self.rotation_matrix_yaw(yaw)
        R_combined = np.dot(R_pitch, np.dot(R_yaw, R_roll))
        QW, QX, QY, QZ = rotation_matrix_to_quaternion(R_combined)
        print(f"QW: {QW}, QX: {QX}, QY: {QY}, QZ: {QZ}")
        transformation_matrix = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        new_R = np.dot(transformation_matrix, R_combined)
        QW_new, QX_new, QY_new, QZ_new = rotation_matrix_to_quaternion(new_R)
        return QW_new, QX_new, QY_new, QZ_new

    def set_camera_pose(self, x, y, z, pitch, yaw, roll, path_params):
        yaw = -yaw
        pitch = -40
        QW, QX, QY, QZ = self.transform_to_camera_frame(math.radians(roll), math.radians(pitch), math.radians(yaw))
        camera_position = world2cam_WXYZ(x, y, z, QW, QX, QY, QZ)
        quat = [QW, QX, QY, QZ]
        camera_id = 0
        image_name = "00000000.png"
        image_data = f"{camera_id} {' '.join(map(str, quat))} {' '.join(map(str, [camera_position[0], camera_position[1], camera_position[2]]))} {0} {image_name}"
        camera_params = f"0 PINHOLE 1436 1077 718.861 718.861 718 538.5"
        data = {
            "camera": camera_params,
            "image": image_data,
            "path": path_params
        }
        print(data)
        try:
            response = requests.post(self.url, data=data)
            if response.status_code == 200:
                print("Request successful!")
                print(response.text) 
            else:
                print(f"Request failed, status code: {response.status_code}")
                print(response.text)
            memory = psutil.virtual_memory()
            print(memory.percent)
            if memory.percent >= 90:
                print("Memory usage is above 90%")
                self.process.terminate()
                self.__init__()
        except requests.RequestException as e:
            print(f"Error during request: {e}")
            time.sleep(20)

    def process_camera_data(self, file_path):
        pass





def get_images(lst,if_his,step):
    if if_his is False:
        return lst[-1]
    else:
        if step == 1:
            if len(lst) >= 2:
                return [lst[-2], lst[-1]]
            elif len(lst) == 1:
                return [lst[0], lst[0]]
        elif step == 2:
            if len(lst) >= 3:
                return lst[-3:]
            elif len(lst) == 2:
                return [lst[0], lst[0], lst[1]]
            elif len(lst) == 1:
                return [lst[0],lst[0], lst[0]]

def convert_to_action_id(action):
    action_dict = {
        "0": np.array([1, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # stop
        "1": np.array([0, 3, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward
        "2": np.array([0, 0, 15, 0, 0, 0, 0, 0]).astype(np.float32),  # turn left 30
        "3": np.array([0, 0, 0, 15, 0, 0, 0, 0]).astype(np.float32),  # turn right 30
        "4": np.array([0, 0, 0, 0, 2, 0, 0, 0]).astype(np.float32),  # go up
        "5": np.array([0, 0, 0, 0, 0, 2, 0, 0]).astype(np.float32),  # go down
        "6": np.array([0, 0, 0, 0, 0, 0, 5, 0]).astype(np.float32),  # move left
        "7": np.array([0, 0, 0, 0, 0, 0, 0, 5]).astype(np.float32),  # move right
        "8": np.array([0, 6, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward 6
        "9": np.array([0, 9, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward 9
    }
    action_values = list(action_dict.values())
    result = 0

    matched = False
    for idx, value in enumerate(action_values):
        if np.array_equal(action, value):
            result = idx
            matched = True
            break
    # If no match is found, default to 0
    if not matched:
        result = 0
    return result


def action_id_to_name(action_id):
    if action_id == "":
        return ""
    action_names = {
        0: "stop",
        1: "forward_3",
        2: "turn_left_30",
        3: "turn_right_30",
        4: "up",
        5: "down",
        6: "left",
        7: "right",
        8: "forward_6",
        9: "forward_9",
    }
    return action_names.get(int(action_id), f"unknown_{action_id}")


def pose_distance(pose_a, pose_b):
    return math.sqrt(
        (pose_a[0] - pose_b[0]) ** 2
        + (pose_a[1] - pose_b[1]) ** 2
        + (pose_a[2] - pose_b[2]) ** 2
    )


def yaw_delta_deg(yaw_a, yaw_b):
    return abs((math.degrees(yaw_a) - math.degrees(yaw_b) + 180.0) % 360.0 - 180.0)


def choice_group(choice):
    choice = str(choice).upper()
    if choice in {"P1", "P6", "P11"}:
        return "left"
    if choice in {"P5", "P10", "P15"}:
        return "right"
    if choice:
        return "center"
    return ""


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value}")


def memory_output_dir(base_dir, env_name, sample_id):
    return os.path.join(base_dir, "memory_graphs", env_name, f"sample_{sample_id:04d}")


STAGE_SPLIT_RE = re.compile(
    r"\b(then|finally|next|after that|afterward|afterwards)\b",
    flags=re.IGNORECASE,
)


def extract_stage_plan(instruction):
    parts = []
    last = 0
    for match in STAGE_SPLIT_RE.finditer(instruction):
        prefix = instruction[last:match.start()].strip(" ,.;")
        if prefix:
            parts.append(prefix)
        last = match.start()
    tail = instruction[last:].strip(" ,.;")
    if tail:
        parts.append(tail)

    stages = []
    for idx, stage_text in enumerate(parts or [instruction]):
        lower = stage_text.lower()
        motion_hints = []
        for word in ("left", "right", "straight", "forward", "up", "down", "turn"):
            if word in lower:
                motion_hints.append(word)
        landmark_phrase = stage_text
        for token in ("toward", "to", "until", "near"):
            marker = f" {token} "
            if marker in lower:
                start = lower.find(marker) + len(marker)
                landmark_phrase = stage_text[start:].strip(" ,.;")
                break
        stages.append(
            {
                "stage_id": idx,
                "stage_text": stage_text,
                "motion_hint": ", ".join(motion_hints),
                "landmark_phrase": landmark_phrase,
                "status": "pending",
            }
        )
    return stages


class StageMemoryTracker:
    CLOSE_MEMORY_VALID_WINDOW = 3

    def __init__(self, stage_plan):
        self.stage_plan = stage_plan
        self.active_stage = 0
        self.recent_evidence = []
        self.transitions = []
        self.stage_stats = {}
        for stage in stage_plan:
            stage_id = stage["stage_id"]
            self.stage_stats[stage_id] = {
                "reliable_observed_count": 0,
                "reliable_active_visible_count": 0,
                "reliable_centered_count": 0,
                "reliable_close_count": 0,
                "reliable_passed_count": 0,
                "consecutive_active_visible_count": 0,
                "consecutive_centered_count": 0,
                "consecutive_close_count": 0,
                "consecutive_passed_count": 0,
                "consecutive_complete_count": 0,
                "consecutive_tracker_completion_count": 0,
                "weak_completion_candidate_count": 0,
                "tracker_completion_candidate_count": 0,
                "recent_observed_stage_ids": [],
                "recent_progress": [],
                "transition_candidates": [],
                "weak_transition_candidate_steps": [],
                "transition_candidate_steps": [],
                "ignored_fallback_count": 0,
                "close_memory_activation_count": 0,
                "close_memory_candidate_count": 0,
                "close_memory_expired_count": 0,
                "close_memory_cleared_count": 0,
                "last_close_step": None,
                "last_close_confidence": None,
                "last_near_step": None,
                "close_memory_active": False,
                "close_memory_age": "",
                "close_memory_source": "",
                "close_memory_valid_window": self.CLOSE_MEMORY_VALID_WINDOW,
                "close_memory_blocked_count": 0,
                "stage_first_reliable_step": None,
                "long_approach_candidate_count": 0,
                "long_approach_candidate_steps": [],
                "long_approach_max_score": 0,
                "long_approach_first_step": None,
                "long_approach_last_step": None,
                "long_approach_confirmed_count": 0,
                "long_approach_confirmed_steps": [],
                "consecutive_long_approach_confirmed_count": 0,
            }

    def _close_memory_age(self, stats, step_id):
        steps = [
            value for value in (stats["last_close_step"], stats["last_near_step"])
            if value is not None
        ]
        if not steps:
            return ""
        return step_id - max(steps)

    def _activate_close_memory(self, stats, step_id, confidence, source):
        if not stats["close_memory_active"]:
            stats["close_memory_activation_count"] += 1
        stats["close_memory_active"] = True
        stats["close_memory_source"] = source
        if source == "close_true":
            stats["last_close_step"] = step_id
            stats["last_close_confidence"] = confidence
        elif source == "progress_near":
            stats["last_near_step"] = step_id
        elif source == "vlm_complete":
            if stats["last_near_step"] is None:
                stats["last_near_step"] = step_id

    def _clear_close_memory(self, stats):
        stats["close_memory_active"] = False
        stats["close_memory_age"] = ""
        stats["close_memory_source"] = ""

    def _reset_stage_runtime_counts(self, stats):
        stats["consecutive_active_visible_count"] = 0
        stats["consecutive_centered_count"] = 0
        stats["consecutive_close_count"] = 0
        stats["consecutive_passed_count"] = 0
        stats["consecutive_complete_count"] = 0
        stats["consecutive_tracker_completion_count"] = 0
        stats["consecutive_long_approach_confirmed_count"] = 0
        stats["close_memory_active"] = False
        stats["close_memory_age"] = ""
        stats["close_memory_source"] = ""

    def update(self, step_id, ours_info):
        observed_stage_id = ours_info.get("observed_stage_id", -1)
        active_stage_target_visible = bool(
            ours_info.get("active_stage_target_visible", ours_info.get("stage_target_visible", False))
        )
        active_stage_target_centered = bool(ours_info.get("active_stage_target_centered", False))
        active_stage_target_close = bool(ours_info.get("active_stage_target_close", False))
        active_stage_target_passed = bool(ours_info.get("active_stage_target_passed", False))
        stage_progress = ours_info.get("stage_progress", "unclear")
        stage_complete_candidate = bool(ours_info.get("stage_complete_candidate", False))
        confidence = float(ours_info.get("confidence", 0.0) or 0.0)
        fallback_used = bool(ours_info.get("fallback_used", False))
        evaluated_stage = self.active_stage
        active_stats = self.stage_stats[self.active_stage]
        reliable = not fallback_used and confidence >= 0.8
        ignored_due_to_fallback = False
        stage_memory_note = ""
        weak_completion_candidate = False
        weak_completion_reason = ""
        tracker_completion_candidate = False
        tracker_completion_reason = ""
        close_memory_event_note = ""
        long_approach_candidate = False
        long_approach_reason = ""
        long_approach_score = 0
        long_approach_visible_count = active_stats["reliable_active_visible_count"]
        long_approach_centered_count = active_stats["reliable_centered_count"]
        long_approach_weak_count = active_stats["weak_completion_candidate_count"]
        long_approach_step_span = 0
        long_approach_confirmed = False
        long_approach_confirmed_reason = ""
        long_approach_confirmed_count = active_stats["long_approach_confirmed_count"]

        if fallback_used:
            active_stats["ignored_fallback_count"] += 1
            active_stats["consecutive_tracker_completion_count"] = 0
            active_stats["consecutive_long_approach_confirmed_count"] = 0
            ignored_due_to_fallback = True
            stage_memory_note = "ignored_due_to_fallback"
        elif reliable:
            if active_stats["stage_first_reliable_step"] is None:
                active_stats["stage_first_reliable_step"] = step_id
            if observed_stage_id >= 0:
                active_stats["recent_observed_stage_ids"].append(observed_stage_id)
                active_stats["recent_observed_stage_ids"] = active_stats["recent_observed_stage_ids"][-20:]
            active_stats["recent_progress"].append(stage_progress)
            active_stats["recent_progress"] = active_stats["recent_progress"][-20:]
            if observed_stage_id == self.active_stage:
                active_stats["reliable_observed_count"] += 1
            if active_stage_target_visible:
                active_stats["reliable_active_visible_count"] += 1
                active_stats["consecutive_active_visible_count"] += 1
            else:
                active_stats["consecutive_active_visible_count"] = 0
            if active_stage_target_centered:
                active_stats["reliable_centered_count"] += 1
                active_stats["consecutive_centered_count"] += 1
            else:
                active_stats["consecutive_centered_count"] = 0
            if active_stage_target_close:
                active_stats["reliable_close_count"] += 1
                active_stats["consecutive_close_count"] += 1
            else:
                active_stats["consecutive_close_count"] = 0
            if active_stage_target_passed:
                active_stats["reliable_passed_count"] += 1
                active_stats["consecutive_passed_count"] += 1
            else:
                active_stats["consecutive_passed_count"] = 0
            if stage_complete_candidate:
                active_stats["consecutive_complete_count"] += 1
                active_stats["transition_candidates"].append(
                    {
                        "step_id": step_id,
                        "confidence": confidence,
                        "stage_progress": stage_progress,
                        "observed_stage_id": observed_stage_id,
                    }
                )
            else:
                active_stats["consecutive_complete_count"] = 0

            if active_stage_target_close:
                self._activate_close_memory(active_stats, step_id, confidence, "close_true")
            elif stage_progress == "near":
                self._activate_close_memory(active_stats, step_id, confidence, "progress_near")
            elif stage_complete_candidate:
                self._activate_close_memory(active_stats, step_id, confidence, "vlm_complete")

            clear_close_memory = (
                (not active_stage_target_visible and observed_stage_id != self.active_stage)
                or stage_progress == "lost"
                or active_stage_target_passed
                or (observed_stage_id >= 0 and observed_stage_id != self.active_stage and not active_stage_target_visible)
            )
            if clear_close_memory and active_stats["close_memory_active"]:
                self._clear_close_memory(active_stats)
                active_stats["close_memory_cleared_count"] += 1
                active_stats["close_memory_blocked_count"] += 1
                close_memory_event_note = "close_memory_cleared=lost_or_mismatch"
                stage_memory_note = close_memory_event_note
                print(f"[STAGE] close memory cleared stage={self.active_stage} step={step_id} reason=lost_or_mismatch")

            close_memory_age = self._close_memory_age(active_stats, step_id)
            active_stats["close_memory_age"] = close_memory_age
            if (
                active_stats["close_memory_active"]
                and close_memory_age != ""
                and close_memory_age > active_stats["close_memory_valid_window"]
            ):
                last_close_step = active_stats["last_close_step"]
                self._clear_close_memory(active_stats)
                active_stats["close_memory_expired_count"] += 1
                close_memory_event_note = "close_memory_expired"
                stage_memory_note = close_memory_event_note
                print(f"[STAGE] close memory expired stage={self.active_stage} step={step_id} last_close_step={last_close_step}")

            if (
                active_stats["consecutive_centered_count"] >= 3
                and active_stats["consecutive_active_visible_count"] >= 3
            ):
                weak_completion_candidate = True
                weak_completion_reason = "centered_visible_count"
                active_stats["weak_completion_candidate_count"] += 1
                active_stats["weak_transition_candidate_steps"].append(
                    {
                        "step_id": step_id,
                        "reason": weak_completion_reason,
                        "confidence": confidence,
                    }
                )
                stage_memory_note = f"weak_completion_candidate={weak_completion_reason}"
                print(
                    f"[STAGE] weak completion candidate stage={self.active_stage} "
                    f"step={step_id} reason={weak_completion_reason}"
                )

            if active_stats["consecutive_close_count"] >= 2:
                tracker_completion_candidate = True
                tracker_completion_reason = "close_count"
            elif active_stage_target_passed:
                tracker_completion_candidate = True
                tracker_completion_reason = "passed"
            elif active_stats["consecutive_complete_count"] >= 2:
                tracker_completion_candidate = True
                tracker_completion_reason = "vlm_complete_count"
            elif (
                active_stats["close_memory_active"]
                and active_stats["close_memory_age"] != ""
                and active_stats["close_memory_age"] <= active_stats["close_memory_valid_window"]
                and (active_stage_target_visible or active_stage_target_centered)
            ):
                tracker_completion_candidate = True
                tracker_completion_reason = "smoothed_close_memory"

            recent_observed = active_stats["recent_observed_stage_ids"]
            active_observed_count = sum(1 for item in recent_observed if item == self.active_stage)
            majority_observed_active = bool(recent_observed) and active_observed_count > len(recent_observed) / 2
            first_reliable_step = active_stats["stage_first_reliable_step"]
            long_approach_step_span = 0 if first_reliable_step is None else step_id - first_reliable_step
            long_approach_visible_count = active_stats["reliable_active_visible_count"]
            long_approach_centered_count = active_stats["reliable_centered_count"]
            long_approach_weak_count = active_stats["weak_completion_candidate_count"]

            if long_approach_visible_count >= 6:
                long_approach_score += 1
            if long_approach_centered_count >= 4:
                long_approach_score += 1
            if long_approach_weak_count >= 3:
                long_approach_score += 1
            if majority_observed_active:
                long_approach_score += 1
            if long_approach_step_span >= 10:
                long_approach_score += 1
            if active_stats["reliable_close_count"] == 0 and active_stats["reliable_passed_count"] == 0:
                long_approach_score += 1

            long_approach_candidate = (
                long_approach_score >= 5
                and active_stage_target_visible
                and long_approach_centered_count >= 4
                and long_approach_weak_count >= 3
                and majority_observed_active
                and active_stats["reliable_close_count"] == 0
                and active_stats["reliable_passed_count"] == 0
                and (long_approach_step_span >= 8 or long_approach_visible_count >= 6)
            )
            if long_approach_candidate:
                long_approach_reason = "stable_visible_centered_but_no_close"
                active_stats["long_approach_candidate_count"] += 1
                active_stats["long_approach_max_score"] = max(
                    active_stats["long_approach_max_score"],
                    long_approach_score,
                )
                if active_stats["long_approach_first_step"] is None:
                    active_stats["long_approach_first_step"] = step_id
                active_stats["long_approach_last_step"] = step_id
                active_stats["long_approach_candidate_steps"].append(
                    {
                        "step_id": step_id,
                        "score": long_approach_score,
                        "reason": long_approach_reason,
                        "visible_count": long_approach_visible_count,
                        "centered_count": long_approach_centered_count,
                        "weak_count": long_approach_weak_count,
                        "step_span": long_approach_step_span,
                    }
                )
                if stage_memory_note:
                    stage_memory_note = f"{stage_memory_note}; long_approach_candidate={long_approach_reason}"
                else:
                    stage_memory_note = f"long_approach_candidate={long_approach_reason}"
                print(
                    f"[STAGE] long approach candidate stage={self.active_stage} "
                    f"step={step_id} score={long_approach_score} reason={long_approach_reason}"
                )

            long_approach_confirmed = (
                long_approach_candidate
                and long_approach_score >= 5
                and active_stage_target_visible
                and observed_stage_id == self.active_stage
                and active_stats["long_approach_candidate_count"] >= 2
                and self.active_stage < len(self.stage_plan) - 1
                and (long_approach_step_span >= 10 or long_approach_visible_count >= 8)
                and active_stats["reliable_close_count"] == 0
                and active_stats["reliable_passed_count"] == 0
                and not tracker_completion_candidate
            )
            if long_approach_confirmed:
                long_approach_confirmed_reason = "repeated_long_approach_without_close"
                active_stats["long_approach_confirmed_count"] += 1
                active_stats["consecutive_long_approach_confirmed_count"] += 1
                long_approach_confirmed_count = active_stats["long_approach_confirmed_count"]
                active_stats["long_approach_confirmed_steps"].append(
                    {
                        "step_id": step_id,
                        "score": long_approach_score,
                        "reason": long_approach_confirmed_reason,
                        "visible_count": long_approach_visible_count,
                        "centered_count": long_approach_centered_count,
                        "weak_count": long_approach_weak_count,
                        "step_span": long_approach_step_span,
                    }
                )
                tracker_completion_candidate = True
                tracker_completion_reason = "long_approach_confirmed"
                if stage_memory_note:
                    stage_memory_note = f"{stage_memory_note}; long_approach_confirmed={long_approach_confirmed_reason}"
                else:
                    stage_memory_note = f"long_approach_confirmed={long_approach_confirmed_reason}"
            elif not fallback_used:
                active_stats["consecutive_long_approach_confirmed_count"] = 0

            if tracker_completion_candidate:
                active_stats["tracker_completion_candidate_count"] += 1
                active_stats["consecutive_tracker_completion_count"] += 1
                if tracker_completion_reason == "smoothed_close_memory":
                    active_stats["close_memory_candidate_count"] += 1
                transition_candidate = {
                    "step_id": step_id,
                    "reason": tracker_completion_reason,
                    "confidence": confidence,
                }
                if tracker_completion_reason == "long_approach_confirmed":
                    transition_candidate["score"] = long_approach_score
                active_stats["transition_candidate_steps"].append(transition_candidate)
                tracker_note = f"tracker_completion_candidate={tracker_completion_reason}"
                stage_memory_note = f"{stage_memory_note}; {tracker_note}" if stage_memory_note else tracker_note
                if tracker_completion_reason == "long_approach_confirmed":
                    print(
                        f"[STAGE] strong completion candidate stage={self.active_stage} "
                        f"step={step_id} reason=long_approach_confirmed score={long_approach_score}"
                    )
                else:
                    print(
                        f"[STAGE] strong completion candidate stage={self.active_stage} "
                        f"step={step_id} reason={tracker_completion_reason}"
                    )
                if (
                    active_stats["consecutive_tracker_completion_count"] < 2
                    and self.active_stage < len(self.stage_plan) - 1
                ):
                    print(
                        f"[STAGE] strong candidate held stage={self.active_stage} "
                        f"step={step_id} reason={tracker_completion_reason} "
                        f"consecutive={active_stats['consecutive_tracker_completion_count']} need=2"
                    )
            else:
                active_stats["consecutive_tracker_completion_count"] = 0
        else:
            active_stats["consecutive_active_visible_count"] = 0
            active_stats["consecutive_centered_count"] = 0
            active_stats["consecutive_close_count"] = 0
            active_stats["consecutive_passed_count"] = 0
            active_stats["consecutive_complete_count"] = 0
            active_stats["consecutive_tracker_completion_count"] = 0
            active_stats["consecutive_long_approach_confirmed_count"] = 0

        if fallback_used:
            close_memory_age = self._close_memory_age(active_stats, step_id)
            active_stats["close_memory_age"] = close_memory_age
            if (
                active_stats["close_memory_active"]
                and close_memory_age != ""
                and close_memory_age > active_stats["close_memory_valid_window"]
            ):
                last_close_step = active_stats["last_close_step"]
                self._clear_close_memory(active_stats)
                active_stats["close_memory_expired_count"] += 1
                close_memory_event_note = "close_memory_expired"
                stage_memory_note = f"{stage_memory_note}; {close_memory_event_note}"
                print(f"[STAGE] close memory expired stage={self.active_stage} step={step_id} last_close_step={last_close_step}")

        previous_stage = self.active_stage
        stage_transition = ""
        stage_transition_candidate = tracker_completion_candidate
        tracker_consecutive_count = active_stats["consecutive_tracker_completion_count"]
        long_approach_consecutive_count = active_stats["consecutive_long_approach_confirmed_count"]
        if (
            tracker_completion_candidate
            and reliable
            and tracker_consecutive_count >= 2
            and self.active_stage < len(self.stage_plan) - 1
        ):
            self.stage_plan[self.active_stage]["status"] = "completed"
            self.active_stage += 1
            self.stage_plan[self.active_stage]["status"] = "active"
            active_stats["consecutive_tracker_completion_count"] = 0
            self._reset_stage_runtime_counts(self.stage_stats[self.active_stage])
            stage_transition = f"stage_{previous_stage}_to_{self.active_stage}:{tracker_completion_reason}"
            transition_note = (
                "stage_transition=long_approach_confirmed"
                if tracker_completion_reason == "long_approach_confirmed"
                else "transition_by_strong_tracker_completion"
            )
            stage_memory_note = f"{stage_memory_note}; {transition_note}" if stage_memory_note else transition_note
            self.transitions.append(
                {
                    "step_id": step_id,
                    "from_stage": previous_stage,
                    "to_stage": self.active_stage,
                    "reason": tracker_completion_reason,
                    "tracker_completion_reason": tracker_completion_reason,
                    "long_approach_score": long_approach_score if tracker_completion_reason == "long_approach_confirmed" else "",
                    "visible_count": long_approach_visible_count if tracker_completion_reason == "long_approach_confirmed" else "",
                    "centered_count": long_approach_centered_count if tracker_completion_reason == "long_approach_confirmed" else "",
                    "weak_count": long_approach_weak_count if tracker_completion_reason == "long_approach_confirmed" else "",
                    "step_span": long_approach_step_span if tracker_completion_reason == "long_approach_confirmed" else "",
                    "consecutive_tracker_completion_count": tracker_consecutive_count,
                }
            )
            print(
                f"[STAGE] transition stage {previous_stage} -> {self.active_stage} "
                f"at step {step_id} reason={tracker_completion_reason}"
            )

        evidence = {
            "step_id": step_id,
            "active_stage": evaluated_stage,
            "observed_stage_id": observed_stage_id,
            "active_stage_target_visible": active_stage_target_visible,
            "active_stage_target_centered": active_stage_target_centered,
            "active_stage_target_close": active_stage_target_close,
            "active_stage_target_passed": active_stage_target_passed,
            "stage_target_visible": active_stage_target_visible,
            "stage_progress": stage_progress,
            "stage_complete_candidate": stage_complete_candidate,
            "weak_stage_completion_candidate": weak_completion_candidate,
            "weak_stage_completion_reason": weak_completion_reason,
            "stage_completion_candidate_by_tracker": tracker_completion_candidate,
            "tracker_completion_reason": tracker_completion_reason,
            "stage_transition_candidate": stage_transition_candidate,
            "long_approach_candidate": long_approach_candidate,
            "long_approach_reason": long_approach_reason,
            "long_approach_score": long_approach_score,
            "long_approach_visible_count": long_approach_visible_count,
            "long_approach_centered_count": long_approach_centered_count,
            "long_approach_weak_count": long_approach_weak_count,
            "long_approach_step_span": long_approach_step_span,
            "long_approach_confirmed": long_approach_confirmed,
            "long_approach_confirmed_reason": long_approach_confirmed_reason,
            "long_approach_confirmed_count": long_approach_confirmed_count,
            "consecutive_tracker_completion_count": tracker_consecutive_count,
            "consecutive_long_approach_confirmed_count": long_approach_consecutive_count,
            "close_memory_active": active_stats["close_memory_active"],
            "close_memory_age": active_stats["close_memory_age"],
            "close_memory_source": active_stats["close_memory_source"],
            "last_close_step": active_stats["last_close_step"],
            "last_near_step": active_stats["last_near_step"],
            "confidence": confidence,
            "fallback_used": fallback_used,
            "ignored_due_to_fallback": ignored_due_to_fallback,
            "stage_transition": stage_transition,
            "stage_memory_note": stage_memory_note,
        }
        self.recent_evidence.append(evidence)
        if len(self.recent_evidence) > 20:
            self.recent_evidence = self.recent_evidence[-20:]

        print(
            "[STAGE] "
            f"active={evaluated_stage} observed={observed_stage_id} "
            f"visible={active_stage_target_visible} centered={active_stage_target_centered} "
            f"close={active_stage_target_close} passed={active_stage_target_passed} "
            f"progress={stage_progress} complete={stage_complete_candidate} "
            f"weak={weak_completion_candidate} strong={tracker_completion_candidate} "
            f"long_approach={long_approach_candidate} score={long_approach_score} "
            f"long_approach_confirmed={long_approach_confirmed} tracker_reason={tracker_completion_reason} "
            f"close_mem={active_stats['close_memory_active']} age={active_stats['close_memory_age']} "
            f"source={active_stats['close_memory_source']} fallback={fallback_used} "
            f"ignored={ignored_due_to_fallback}"
        )
        return evidence

    def to_json(self):
        return {
            "active_stage": self.active_stage,
            "stage_plan": self.stage_plan,
            "recent_evidence": self.recent_evidence,
            "stage_stats": self.stage_stats,
            "transitions": self.transitions,
        }


def save_stage_outputs(stage_plan, stage_tracker, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "stage_plan.json"), "w", encoding="utf-8") as f:
        json.dump(stage_plan, f, indent=2)
    if stage_tracker is not None:
        with open(os.path.join(output_dir, "stage_memory.json"), "w", encoding="utf-8") as f:
            json.dump(stage_tracker.to_json(), f, indent=2)


def save_memory_graph(graph, output_dir, stage_plan=None, stage_tracker=None):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "memory_graph.json")
    nodes_path = os.path.join(output_dir, "memory_nodes.csv")
    edges_path = os.path.join(output_dir, "memory_edges.csv")
    graph.save_json(json_path)
    graph.save_nodes_csv(nodes_path)
    graph.save_edges_csv(edges_path)
    if stage_plan is not None:
        save_stage_outputs(stage_plan, stage_tracker, output_dir)
    print(f"[MEMORY] saved graph to {json_path}")


def maybe_add_memory_node(
    graph,
    output_dir,
    args,
    env_name,
    sample_id,
    instruction,
    pose,
    step_id,
    ours_info,
    action_id,
    end_position,
    reason=None,
    stage_evidence=None,
    stage_plan=None,
    stage_tracker=None,
):
    if not args.enable_memory_record:
        return None

    selected_choice = ours_info.get("choice", "")
    selected_group = choice_group(selected_choice)
    visible = ours_info.get("target_visible", "")
    fallback_used = ours_info.get("fallback_used", "")
    action_name = action_id_to_name(action_id)
    distance_to_goal = calculate_distance(end_position, pose[:3])
    stage_evidence = stage_evidence or {}

    add_reason = reason
    if add_reason is None:
        if not graph.nodes:
            add_reason = "start"
        else:
            last_node = graph.nodes[-1]
            last_pose = (
                last_node["pose_x"],
                last_node["pose_y"],
                last_node["pose_z"],
                last_node["yaw"],
            )
            if pose_distance(pose, last_pose) >= args.memory_sample_dist:
                add_reason = "distance"
            elif yaw_delta_deg(pose[3], last_pose[3]) >= args.memory_sample_yaw_deg:
                add_reason = "yaw"
            elif str(action_id) != str(last_node.get("action_id", "")):
                add_reason = "action_change"
            elif selected_group != last_node.get("choice_group", ""):
                add_reason = "choice_group_change"
            elif str(visible) != str(last_node.get("target_visible", "")):
                add_reason = "target_visible_change"
            elif str(fallback_used) != str(last_node.get("fallback_used", "")):
                add_reason = "fallback_change"

    if add_reason is None:
        return None

    previous_node = graph.nodes[-1] if graph.nodes else None
    is_revisit = False
    revisit_from_node = None
    for node in graph.nodes[:-1]:
        node_pose = (node["pose_x"], node["pose_y"], node["pose_z"], node["yaw"])
        if pose_distance(pose, node_pose) <= args.memory_revisit_dist:
            is_revisit = True
            revisit_from_node = node
            break

    node = graph.add_node(
        env_name=env_name,
        sample_id=sample_id,
        step_id=step_id,
        source="online_task_run",
        pose_x=pose[0],
        pose_y=pose[1],
        pose_z=pose[2],
        yaw_deg=pose[3],
        rgb_path=ours_info.get("input_image_path", ""),
        instruction=instruction,
        selected_image_path=ours_info.get("selected_image_path", ""),
        choice=selected_choice,
        choice_group=selected_group,
        action_id=action_id,
        action_name=action_name,
        target_visible=visible,
        confidence=ours_info.get("confidence", ""),
        reason=add_reason,
        raw_response=ours_info.get("raw_response", ""),
        fallback_used=fallback_used,
        distance_to_goal=distance_to_goal,
        auto_tag=add_reason,
        is_revisit=is_revisit,
        active_stage=stage_evidence.get("active_stage", ""),
        observed_stage_id=stage_evidence.get("observed_stage_id", ""),
        active_stage_target_visible=stage_evidence.get("active_stage_target_visible", ""),
        active_stage_target_centered=stage_evidence.get("active_stage_target_centered", ""),
        active_stage_target_close=stage_evidence.get("active_stage_target_close", ""),
        active_stage_target_passed=stage_evidence.get("active_stage_target_passed", ""),
        stage_target_visible=stage_evidence.get("active_stage_target_visible", ""),
        stage_progress=stage_evidence.get("stage_progress", ""),
        stage_complete_candidate=stage_evidence.get("stage_complete_candidate", ""),
        weak_stage_completion_candidate=stage_evidence.get("weak_stage_completion_candidate", ""),
        weak_stage_completion_reason=stage_evidence.get("weak_stage_completion_reason", ""),
        stage_completion_candidate_by_tracker=stage_evidence.get("stage_completion_candidate_by_tracker", ""),
        tracker_completion_reason=stage_evidence.get("tracker_completion_reason", ""),
        stage_transition_candidate=stage_evidence.get("stage_transition_candidate", ""),
        stage_transition=stage_evidence.get("stage_transition", ""),
        long_approach_candidate=stage_evidence.get("long_approach_candidate", ""),
        long_approach_reason=stage_evidence.get("long_approach_reason", ""),
        long_approach_score=stage_evidence.get("long_approach_score", ""),
        long_approach_visible_count=stage_evidence.get("long_approach_visible_count", ""),
        long_approach_centered_count=stage_evidence.get("long_approach_centered_count", ""),
        long_approach_weak_count=stage_evidence.get("long_approach_weak_count", ""),
        long_approach_step_span=stage_evidence.get("long_approach_step_span", ""),
        long_approach_confirmed=stage_evidence.get("long_approach_confirmed", ""),
        long_approach_confirmed_reason=stage_evidence.get("long_approach_confirmed_reason", ""),
        long_approach_confirmed_count=stage_evidence.get("long_approach_confirmed_count", ""),
        consecutive_tracker_completion_count=stage_evidence.get("consecutive_tracker_completion_count", ""),
        consecutive_long_approach_confirmed_count=stage_evidence.get("consecutive_long_approach_confirmed_count", ""),
        close_memory_active=stage_evidence.get("close_memory_active", ""),
        close_memory_age=stage_evidence.get("close_memory_age", ""),
        close_memory_source=stage_evidence.get("close_memory_source", ""),
        last_close_step=stage_evidence.get("last_close_step", ""),
        last_near_step=stage_evidence.get("last_near_step", ""),
        stage_memory_note=stage_evidence.get("stage_memory_note") or (
            f"active_stage={stage_evidence.get('active_stage', '')}; "
            f"observed_stage_id={stage_evidence.get('observed_stage_id', '')}; "
            f"progress={stage_evidence.get('stage_progress', '')}"
        ),
    )

    if previous_node is not None:
        previous_pose = (
            previous_node["pose_x"],
            previous_node["pose_y"],
            previous_node["pose_z"],
            previous_node["yaw"],
        )
        graph.add_edge(
            from_node=previous_node["node_id"],
            to_node=node["node_id"],
            edge_type="trajectory",
            distance=pose_distance(pose, previous_pose),
            yaw_delta=yaw_delta_deg(pose[3], previous_pose[3]),
            step_start=previous_node["step_id"],
            step_end=step_id,
            actions_between=[action_id],
            choices_between=[selected_choice],
        )

    if revisit_from_node is not None:
        revisit_pose = (
            revisit_from_node["pose_x"],
            revisit_from_node["pose_y"],
            revisit_from_node["pose_z"],
            revisit_from_node["yaw"],
        )
        graph.add_edge(
            from_node=revisit_from_node["node_id"],
            to_node=node["node_id"],
            edge_type="revisit",
            distance=pose_distance(pose, revisit_pose),
            yaw_delta=yaw_delta_deg(pose[3], revisit_pose[3]),
            step_start=revisit_from_node["step_id"],
            step_end=step_id,
            actions_between=[],
            choices_between=[],
        )

    print(
        "[MEMORY] add node "
        f"id={node['node_id']}, reason={add_reason}, step={step_id}, "
        f"pose=({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}), "
        f"choice={selected_choice}, action={action_id}"
    )
    save_memory_graph(graph, output_dir, stage_plan=stage_plan, stage_tracker=stage_tracker)
    return node

def get_action(policy, processor, image_list, text, his, if_his=False, his_step=0):

    # Otherwise, generate new actions using the policy
    image_list = get_images(image_list, if_his, his_step)

    if isinstance(image_list, np.ndarray):
        img = image_list
        img = Image.fromarray(img)
        images = [img, img, img]
    else:
        images = []
        for img in image_list:
            img = Image.fromarray(img)
            images.append(img)
        
    prompt = text
    inputs = processor(prompt, images).to("cuda:0", dtype=torch.bfloat16)
    action = policy.predict_action(**inputs, unnorm_key="vlnv1", do_sample=False)
    print("raw action:", action)
    action = action.round().astype(int)

    # Convert action_chunk to action IDs
    action_id = convert_to_action_id(action)

    cur_action = action_id
    print("Action:", action_id)
    return cur_action

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + 
                     (point2[1] - point1[1])**2 + 
                     (point2[2] - point1[2])**2)

def getPoseAfterMakeAction(new_pose, action):
    x, y, z, yaw = new_pose

    # Define step size
    step_size = 3.0  # Translation step size (units can be adjusted as needed)

    # Update new_pose based on action value
    if action == 0:
        pass
    elif action == 1:
        x += step_size * math.cos(yaw)
        y += step_size * math.sin(yaw)
    elif action == 2:
        yaw += math.radians(30)
    elif action == 3:
        yaw -= math.radians(30)
    elif action == 4:
        z += step_size
    elif action == 5:
        z -= step_size
    elif action == 6:
        x -= step_size * math.sin(yaw)
        y += step_size * math.cos(yaw)
    elif action == 7:
        x += step_size * math.sin(yaw)
        y -= step_size * math.cos(yaw)
    elif action == 8:
        x += step_size * math.cos(yaw) *2
        y += step_size * math.sin(yaw) *2
    elif action == 9:
        x += step_size * math.cos(yaw) *3
        y += step_size * math.sin(yaw) *3

    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

    return [x, y, z, yaw]

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--agent_type", choices=["openfly", "ours"], default="openfly")
    parser.add_argument("--ours_base_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--ours_model", default="qwen3.6-27b")
    parser.add_argument("--ours_output_dir", default="./ours_eval_outputs")
    parser.add_argument("--max_envs", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--enable_memory_record", type=parse_bool, default=True)
    parser.add_argument("--memory_sample_dist", type=float, default=12.0)
    parser.add_argument("--memory_sample_yaw_deg", type=float, default=35.0)
    parser.add_argument("--memory_revisit_dist", type=float, default=8.0)
    args, _unknown = parser.parse_known_args()

    eval_info = "/workspace/smbu_zzh/OpenFly-Platform/configs/eval_test0.json"
    f = open(eval_info, 'r')
    all_eval_info = json.loads(f.read())
    f.close()
    
    processor = None
    policy = None
    ours_agent = None
    csv_file = None
    csv_writer = None
    if args.agent_type == "openfly":
        # Load model
        #model_name_or_path="IPEC-COMMUNITY/openfly-agent-7b"
        model_name_or_path="/workspace/smbu_zzh/OpenFly-Platform/model/model/openfly-agent-7b"

        processor = AutoProcessor.from_pretrained(model_name_or_path)
        policy = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to("cuda:0")
    else:
        os.makedirs(args.ours_output_dir, exist_ok=True)
        ours_agent = OurSPFAgent(
            base_url=args.ours_base_url,
            model_name=args.ours_model,
            output_dir=args.ours_output_dir,
        )
        csv_file = open(os.path.join(args.ours_output_dir, "ours_steps.csv"), "a", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "env_name",
                "sample_id",
                "step_id",
                "instruction",
                "start_x",
                "start_y",
                "start_z",
                "goal_x",
                "goal_y",
                "goal_z",
                "initial_yaw",
                "choice",
                "target_visible",
                "confidence",
                "action_id",
                "action_name",
                "reason",
                "fallback_used",
                "raw_response",
                "pose_x",
                "pose_y",
                "pose_z",
                "yaw",
                "input_image_path",
                "selected_image_path",
                "observed_stage_id",
                "active_stage_target_visible",
                "active_stage_target_centered",
                "active_stage_target_close",
                "active_stage_target_passed",
                "stage_progress",
                "stage_complete_candidate",
            ],
        )
        if csv_file.tell() == 0:
            csv_writer.writeheader()
            csv_file.flush()

    # Test metrics
    acc = 0
    stop = 0
    data_num = 0
    MAX_STEP = 100

    # Group by environment type
    env_groups = {}
    for item in all_eval_info:
        env_type = item["image_path"].split("/")[0]  # Get environment type
        if env_type not in env_groups:
            env_groups[env_type] = []
        env_groups[env_type].append(item)
    
    # Process each environment type sequentially
    env_items = list(env_groups.items())
    if args.max_envs is not None:
        env_items = env_items[:args.max_envs]

    for env_name, eval_info in env_items:
        print(f"Starting evaluation of environment: {env_name}, with {len(eval_info)} data entries")
        time.sleep(5)
        
        # Create appropriate environment bridge based on environment type
        if "airsim" in env_name:
            env_bridge = AirsimBridge(env_name)
            pos_ratio = 1.0
        elif "ue" in env_name:
            env_bridge = UEBridge(ue_ip="127.0.0.1", ue_port="9000", env_name=env_name)
            pos_ratio = 1.0
        elif "gs" in env_name:
            env_bridge = GSBridge(env_name)
            pos_ratio = 5.15
        else:
            print(f"Unknown environment type: {env_name}, skipping")
            continue
        
        # Evaluate all data for current environment
        if args.max_samples is not None:
            eval_info = eval_info[:args.max_samples]

        for idx, item in enumerate(eval_info):
            acts = []  # Reset action list
            
            pos_list = item['pos']
            text = item['gpt_instruction']
            start_postion = pos_list[0]
            start_yaw = item['yaw'][0]
            new_pose = [start_postion[0], start_postion[1], start_postion[2], start_yaw]
            end_position = pos_list[-1]
            print(f"Sample {idx}: {start_postion} -> {end_position}, initial heading: {start_yaw}")
            print("[OPENFLY TASK]")
            print(f"env_name: {env_name}")
            print(f"sample_id: {idx}")
            print(f"start_position: {start_postion}")
            print(f"goal_position: {end_position}")
            print(f"initial_yaw: {start_yaw}")
            print(f"gpt_instruction: {text}")
            memory_graph = None
            memory_dir = None
            stage_plan = None
            stage_tracker = None
            last_ours_info = None
            last_stage_evidence = None
            last_model_action = None
            last_step_id = None
            if args.agent_type == "ours" and args.enable_memory_record:
                memory_dir = memory_output_dir(args.ours_output_dir, env_name, idx)
                stage_plan = extract_stage_plan(text)
                if stage_plan:
                    stage_plan[0]["status"] = "active"
                stage_tracker = StageMemoryTracker(stage_plan)
                save_stage_outputs(stage_plan, stage_tracker, memory_dir)
                memory_graph = MemoryGraph(
                    source="online_task_run",
                    env_name=env_name,
                    sample_id=idx,
                    instruction=text,
                )
            
            stop_error = 1
            image_error = False
            
            # Set camera pose
            pitch = -45.0 if 'high' in item['image_path'] else 0.0
            env_bridge.set_camera_pose(
                start_postion[0]/pos_ratio, 
                start_postion[1]/pos_ratio, 
                start_postion[2]/pos_ratio, 
                pitch, 
                np.rad2deg(start_yaw), 
                0
            )
            
            image_list = []
            step = 0
            
            while step < MAX_STEP:
                try:
                    if args.max_steps is not None and step >= args.max_steps:
                        break
                    raw_image = env_bridge.get_camera_data()
                    cv2.imwrite("test/cur_img.jpg", raw_image)
                    image = raw_image

                    image_list.append(image)
                    if args.agent_type == "openfly":
                        model_action = get_action(policy, processor, image_list, text, acts, if_his=True, his_step=2)
                    else:
                        pose_before = new_pose.copy()
                        model_action, ours_info = ours_agent.act(
                            image=raw_image,
                            instruction=text,
                            sample_id=idx,
                            step_id=step,
                            pose=pose_before,
                            history=image_list,
                            stage_plan=stage_plan,
                            active_stage=stage_tracker.active_stage if stage_tracker is not None else 0,
                        )
                        stage_evidence = None
                        if stage_tracker is not None:
                            stage_evidence = stage_tracker.update(step, ours_info)
                        if memory_graph is not None:
                            maybe_add_memory_node(
                                graph=memory_graph,
                                output_dir=memory_dir,
                                args=args,
                                env_name=env_name,
                                sample_id=idx,
                                instruction=text,
                                pose=pose_before,
                                step_id=step,
                                ours_info=ours_info,
                                action_id=model_action,
                                end_position=end_position,
                                stage_evidence=stage_evidence,
                                stage_plan=stage_plan,
                                stage_tracker=stage_tracker,
                            )
                        last_ours_info = ours_info
                        last_stage_evidence = stage_evidence
                        last_model_action = model_action
                        last_step_id = step
                        csv_writer.writerow(
                            {
                                "env_name": env_name,
                                "sample_id": idx,
                                "step_id": step,
                                "instruction": text,
                                "start_x": start_postion[0],
                                "start_y": start_postion[1],
                                "start_z": start_postion[2],
                                "goal_x": end_position[0],
                                "goal_y": end_position[1],
                                "goal_z": end_position[2],
                                "initial_yaw": start_yaw,
                                "choice": ours_info.get("choice", ""),
                                "target_visible": ours_info.get("target_visible", ""),
                                "confidence": ours_info.get("confidence", ""),
                                "action_id": model_action,
                                "action_name": action_id_to_name(model_action),
                                "reason": ours_info.get("reason", ""),
                                "fallback_used": ours_info.get("fallback_used", ""),
                                "raw_response": ours_info.get("raw_response", ""),
                                "pose_x": pose_before[0],
                                "pose_y": pose_before[1],
                                "pose_z": pose_before[2],
                                "yaw": pose_before[3],
                                "input_image_path": ours_info.get("input_image_path", ""),
                                "selected_image_path": ours_info.get("selected_image_path", ""),
                                "observed_stage_id": ours_info.get("observed_stage_id", ""),
                                "active_stage_target_visible": ours_info.get("active_stage_target_visible", ""),
                                "active_stage_target_centered": ours_info.get("active_stage_target_centered", ""),
                                "active_stage_target_close": ours_info.get("active_stage_target_close", ""),
                                "active_stage_target_passed": ours_info.get("active_stage_target_passed", ""),
                                "stage_progress": ours_info.get("stage_progress", ""),
                                "stage_complete_candidate": ours_info.get("stage_complete_candidate", ""),
                            }
                        )
                        csv_file.flush()
                    acts.append(model_action)
                    new_pose = getPoseAfterMakeAction(new_pose, model_action)
                    print(f"Environment: {env_name}, Sample: {idx}, Step: {step}, Action: {model_action}, New position: {new_pose}")
                    env_bridge.set_camera_pose(
                        new_pose[0]/pos_ratio, 
                        new_pose[1]/pos_ratio, 
                        new_pose[2]/pos_ratio, 
                        pitch, 
                        np.rad2deg(new_pose[3]), 
                        0
                    )
                    
                    if model_action == 0:
                        stop_error = 0
                        break
                    step += 1
                except Exception as e:
                    print(f"Error processing image: {e}")
                    image_error = True
                    break
            
            if image_error:
                if memory_graph is not None:
                    save_memory_graph(
                        memory_graph,
                        memory_dir,
                        stage_plan=stage_plan,
                        stage_tracker=stage_tracker,
                    )
                continue

            if memory_graph is not None:
                final_info = last_ours_info or {
                    "choice": "",
                    "target_visible": "",
                    "confidence": "",
                    "reason": "",
                    "fallback_used": "",
                    "raw_response": "",
                    "input_image_path": "",
                    "selected_image_path": "",
                }
                maybe_add_memory_node(
                    graph=memory_graph,
                    output_dir=memory_dir,
                    args=args,
                    env_name=env_name,
                    sample_id=idx,
                    instruction=text,
                    pose=new_pose,
                    step_id=last_step_id if last_step_id is not None else step,
                    ours_info=final_info,
                    action_id=last_model_action if last_model_action is not None else "",
                    end_position=end_position,
                    reason="final",
                    stage_evidence=last_stage_evidence,
                    stage_plan=stage_plan,
                    stage_tracker=stage_tracker,
                )
                save_memory_graph(
                    memory_graph,
                    memory_dir,
                    stage_plan=stage_plan,
                    stage_tracker=stage_tracker,
                )
                
            model_end_position = new_pose
            dis = calculate_distance(end_position, model_end_position)
            
            if dis <= 20:
                acc += 1
            
            stop += stop_error
            data_num += 1
            
            print(f"Current accuracy: {acc/data_num:.4f}, Stop rate: {1-stop/data_num:.4f}, Evaluated: {data_num} samples")
        
        # Clean up environment resources
        print(f"Completed evaluation of environment {env_name}")
        kill_env_process("AirVLN")
        kill_env_process("guangzhou")
        kill_env_process("shanghai")
        kill_env_process("CitySample")
        kill_env_process("CrashReport")

        del env_bridge
        import gc
        gc.collect()
    
    # Final results
    final_acc = acc / data_num if data_num > 0 else 0
    final_stop = 1 - stop / data_num if data_num > 0 else 0
    
    print(f"\nEvaluation complete!")
    print(f"Total samples: {data_num}")
    print(f"Final accuracy: {final_acc:.4f}")
    print(f"Final stop rate: {final_stop:.4f}")

    if csv_file is not None:
        csv_file.close()


if __name__ == '__main__':
    main()
