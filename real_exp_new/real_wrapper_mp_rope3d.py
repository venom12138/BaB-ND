import sys
import os
from xarm import version
from xarm.wrapper import XArmAPI
import ctypes
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from copy import deepcopy

# ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(ROOT_DIR)
# os.chdir(ROOT_DIR)
file_dir = os.path.dirname(os.path.abspath(__file__))
print(file_dir)
sys.path.append(file_dir)
import cv2
import json
import copy
import time
import numpy as np
import multiprocessing as mp
from threadpoolctl import threadpool_limits
from tqdm import tqdm
import pickle
import random
from queue import Queue, Empty
import pyrealsense2 as rs
import open3d as o3d
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Value, Manager
from threading import Lock, Thread, Event
from camera.multi_realsense import MultiRealsense, SingleRealsense
from camera.video_recorder import VideoRecorder
from xarm6 import XARM6
import torch
from dgl.geometry import farthest_point_sampler
from real_utils import rpy_to_rotation_matrix
from common.kinematics_utils import KinHelper
from perception_for_rope3d import PerceptionModule

FLOAT_TYPE = np.float16
file_dir = os.path.dirname(os.path.abspath(__file__))

COLOR_PALETTE = np.array([
    (255, 0, 0),      # Warm Red
    (255, 69, 0),     # Warm Orange
    (255, 140, 0),    # Warm Yellow
    (255, 223, 0),    # Warm Light Yellow
    (173, 255, 47),   # Cool Light Green
    (0, 255, 0),      # Cool Green
    (0, 223, 175),    # Cool Green
    (0, 255, 255),    # Cool Cyan
    (0, 191, 255),    # Cool Light Blue
    (0, 0, 255),      # Cool Blue
])


class RealWrapper:
    def __init__(
        self, param_dict: dict, serial_numbers, ):
        # super().__init__()
        self.FLOAT_TYPE = FLOAT_TYPE
        # arm part
        ip = "192.168.1.209"
        self.arm = XArmAPI(ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1)
        self.arm.set_state(state=0)
        
        self.kin_helper = KinHelper(robot_name="xarm6")
        # self.reset_robot()
        # initial position
        x, y, z, roll, pitch, yaw = [196.2, -1.6, 434, 179.2, 0, 90]
        self.xyz = np.array([x, y, z]) / 1000
        self.rpy = np.array([roll, pitch, yaw]) / 180.0 * np.pi
        
        # realsense part
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        self.realsense = MultiRealsense(
            shm_manager=self.shm_manager,
            serial_numbers=serial_numbers,
            resolution=(640, 360),
            # resolution=(1280, 720),
            capture_fps=param_dict.get("capture_fps", 15),
            enable_depth=True,
            enable_pcd=True,
            verbose=False,
            visible_areas=None,
        )
        self.realsense.set_exposure(exposure=100, gain=60) # 200
        # self.realsense.set_white_balance(white_balance=3700)
        # self.realsense.set_exposure(exposure=None, gain=None) # 200
        exposure_time = 5
        rec_start_time = time.time() + exposure_time
        self.realsense.restart_put(start_time=rec_start_time)

        self.last_realsense_data = None
        self.camera_indices = [i for i in range(len(serial_numbers))]

        # camera calibration part
        calibrate_result_dir = f"{file_dir}/calibration_result"
        with open(f"{calibrate_result_dir}/rvecs.pkl", "rb") as f:
            rvecs = pickle.load(f)
        with open(f"{calibrate_result_dir}/tvecs.pkl", "rb") as f:
            tvecs = pickle.load(f)
        with open(f"{calibrate_result_dir}/calibration_handeye_result.pkl", "rb") as f:
            handeye_result = pickle.load(f)
        R_base2world, t_base2world = handeye_result["R_base2world"], handeye_result["t_base2world"]        
        base2world = np.eye(4)
        base2world[:3, :3] = R_base2world
        base2world[:3, 3] = t_base2world
        # adjust world coordinate in camera coordinate to world coordinate for task
        adjust_matrix = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        base2world = adjust_matrix @ base2world
        world2base = np.linalg.inv(base2world)
        workspace2world = np.eye(4)
        workspace2world[:3, 3] = np.array([-0.05, 0.2, 0]) + np.array([0.2, 0.2, 0]) # Original point is at the center of the workspace
        workspace2world[:3, :3] = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) # rotate 90 degrees to align with simulation
        self.world2base = world2base
        self.workspace2world = workspace2world
        self.workspace2base = world2base @ workspace2world

        # cam2world
        R_list, T_list = [], []
        for i in range(len(serial_numbers)):
            serial_number = serial_numbers[i]
            R_cam2board = rvecs[serial_number]  # (3, 1)
            # convert to rotation matrix
            R_cam2board = cv2.Rodrigues(R_cam2board)[0]
            t_cam2board = tvecs[serial_number]  # (3, 1)
            cam2world = np.eye(4)
            cam2world[:3, :3] = R_cam2board
            cam2world[:3, 3] = t_cam2board.flatten()
            cam2world = adjust_matrix @ np.linalg.inv(cam2world)
            R_list.append(cam2world[:3, :3].T.astype(FLOAT_TYPE))
            T_list.append(cam2world[:3, 3].astype(FLOAT_TYPE))
        self.R_list = R_list
        self.T_list = T_list

        self.rope_perception = PerceptionModule()
        self.alive = False

        # self.pose_queue = Queue()
        # self.pose_thread = Thread(target=self.process_poses)
        # self.pose_thread.daemon = True
        
        # self.arm = ArmController(self.shm_manager)
        # initialize object pcd from last frame
        self.last_mesh_transform = None
        self.rope_scale = 6 # 3.0 -> 0.5 
        self.rope_fixed_end_coord = None
        self.grasp_end = None
        self.target_offest = param_dict.get('target_offest', 0.15)
        self.obs_gap = param_dict.get('obs_gap', 0.15)
        self.last_pusher_position = self.get_pusher_position()
        self.target_pose = None
    
    def update_world_workspace_coord(self, offset):
        world2workspace = np.linalg.inv(self.workspace2world)
        world2workspace[:3, 3] += offset
        self.workspace2world = np.linalg.inv(world2workspace)
        self.workspace2base = self.world2base @ self.workspace2world
        self.target_pose = None
        self.rope_fixed_end_coord = None
        self.grasp_end = None
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.realsense.is_ready

    def start(self, wait=True, exposure_time=5):
        self.realsense.start(wait=False, put_start_time=time.time() + exposure_time)
        if wait:
            self.start_wait()
        self.alive = True
        # self.arm.start()

    def stop(self, wait=True):
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()
        self.alive = False
        # self.arm.stop()

    def start_wait(self):
        self.realsense.start_wait()

    def stop_wait(self):
        self.realsense.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def set_target_pose(self, target_pose, forbidden_area):
        self.target_pose = target_pose
        self.left_forbidden_area = forbidden_area[:2]
        self.right_forbidden_area = forbidden_area[2:]

    def get_target_pose(self, ):
        # import pdb; pdb.set_trace()
        if self.target_pose is not None:
            return self.target_pose
        x_obj_radius = 0.3
        y_obj_radius = 0.3
        
        target_center = np.array([self.rope_fixed_end_coord[0], self.grasp_end[1]+self.target_offest])
        # set the obstacle's orientation to be vertical to the line of rope fixed end and target
        fixed_end = self.rope_fixed_end_coord[0], self.rope_fixed_end_coord[1]
        vector_dir_2D = (np.array(target_center) - np.array(fixed_end)) / np.linalg.norm((np.array(target_center) - np.array(fixed_end)))
        orn_dir_2D = np.array([-vector_dir_2D[1], vector_dir_2D[0]])
        left_obj_center = target_center - orn_dir_2D * (self.obs_gap + x_obj_radius) # [-0.5, 0.3] # (x, y)
        right_obj_center = target_center + orn_dir_2D * (self.obs_gap + x_obj_radius)
        
        self.left_forbidden_area = np.array([[left_obj_center[0]-x_obj_radius*orn_dir_2D[0]+y_obj_radius*vector_dir_2D[0], \
                                            left_obj_center[1]-x_obj_radius*orn_dir_2D[1]+y_obj_radius*vector_dir_2D[1], \
                                            0.5], \
                                            [left_obj_center[0]+x_obj_radius*orn_dir_2D[0]-y_obj_radius*vector_dir_2D[0], \
                                            left_obj_center[1]+x_obj_radius*orn_dir_2D[1]-y_obj_radius*vector_dir_2D[1], \
                                            0.8+0.1], \
                                            ]) # [vector_dir_2D[0], vector_dir_2D[1], 0.0]
        self.right_forbidden_area = np.array([[right_obj_center[0]-x_obj_radius*orn_dir_2D[0]+y_obj_radius*vector_dir_2D[0], \
                                            right_obj_center[1]-x_obj_radius*orn_dir_2D[1]+y_obj_radius*vector_dir_2D[1], \
                                            0.5], \
                                            [right_obj_center[0]+x_obj_radius*orn_dir_2D[0]-y_obj_radius*vector_dir_2D[0], \
                                            right_obj_center[1]+x_obj_radius*orn_dir_2D[1]-y_obj_radius*vector_dir_2D[1], \
                                            0.8+0.1], \
                                            ]) # [vector_dir_2D[0], vector_dir_2D[1], 0.0]
        
        rope_fixed_coord_in_normal_frame = self.rope_fixed_end_coord
        rope_indices_y = np.linspace(0, self.get_rope_length(), num=10)
        target_rope = rope_fixed_coord_in_normal_frame[np.newaxis, :] + np.array([[0, -dy, 0] for dy in rope_indices_y])
        self.target_pose = target_rope[::-1] # (x, y, z)
        # self.get_fixed_action_sequence()
        return self.target_pose # (x, y, z)

    def reorder_particles(self, particle_pos, fps_idx):
        # reorder the particles based on the distance to the end effector
        particle_pos = particle_pos.reshape(-1, 3)
        # scaler = StandardScaler()
        # particle_pos_scaled = scaler.fit_transform(particle_pos)
        particle_pos_scaled = particle_pos - np.mean(particle_pos, axis=0)
        pca = PCA(n_components=1)  # We want to reduce the data to 2 principal components
        particle_pos_pca = pca.fit_transform(particle_pos_scaled).reshape(-1)
        # import pdb; pdb.set_trace()
        sorted_idx = np.argsort(particle_pos_pca)
        fps_idx = fps_idx[sorted_idx]
        
        return fps_idx
    
    def get_state(self, ):
        # keypoint_offsets_2d_list: in mm
        self.last_realsense_data = self.realsense.get(out=self.last_realsense_data)
        # change to hsv

        for i in self.camera_indices:
            # cv2.imwrite(f"rgb_image_{i}.png", self.last_realsense_data[i]["color"])
            # print(f"self.last_realsense_data[i]['color'].shape: {self.last_realsense_data[i]['color']}")
            self.last_realsense_data[i]["color"] = cv2.cvtColor(self.last_realsense_data[i]["color"], cv2.COLOR_BGR2RGB) # cv2.cvtColor(self.last_realsense_data[i]["color"], cv2.COLOR_RGB2HSV)
            # self.last_realsense_data[i]["color"] = cv2.cvtColor(self.last_realsense_data[i]["color"], cv2.COLOR_RGB2HSV)
            # self.last_realsense_data[i]["color"] = self.last_realsense_data[i]["color"]

        pusher_position = self.get_pusher_position_in_real()
        print(f"last_realsense_data: {[float(self.last_realsense_data[0]['timestamp']) for i in range(len(self.last_realsense_data))]}")
        color_list = [self.last_realsense_data[i]["color"] for i in self.camera_indices]
        depth_list = [self.last_realsense_data[i]["depth"]/1000. for i in self.camera_indices]
        bbox = np.array([[-0.1, 0.8], \
                        [-0.1, 0.8], \
                        [-0.05, pusher_position[2]-0.1]])
        # depth_threshold = [0, pusher_position[2]-0.1] # pusher_position[2]
        rope_pcd = self.rope_perception.get_tabletop_points(color_list, depth_list, self.R_list, self.T_list, self.realsense.get_intrinsics(), \
                                                bbox, ) # depth_threshold=depth_threshold
        rope_points = np.array(rope_pcd.points)
        rope_points = np.concatenate([rope_points, np.ones((rope_points.shape[0], 1))], axis=1) @ np.linalg.inv(self.workspace2world).T
        rope_points = rope_points[:, :3]
        fps_idx = farthest_point_sampler(torch.tensor(rope_points).unsqueeze(0), 10, start_idx=0)[0]
        fps_idx = fps_idx.numpy().astype(np.int32)
        fps_idx = self.reorder_particles(rope_points[fps_idx], fps_idx)
        
        # correct the fps_idx
        pusher_rope_dis = np.linalg.norm(rope_points[fps_idx] - pusher_position, axis=1)
        Is_reverse_fps_idx = (np.argmin(pusher_rope_dis) > len(fps_idx) // 2)
        if Is_reverse_fps_idx:
            fps_idx = fps_idx[::-1]
        
        # correct along the z-axis
        selected_points_z = rope_points[fps_idx][:, 2]
        selected_z_mask = selected_points_z >= 0.015
        reordered_points_idx = np.argsort(selected_points_z[selected_z_mask], )[::-1]
        fps_idx[selected_z_mask] = fps_idx[selected_z_mask][reordered_points_idx]



        # rope_pcd = o3d.geometry.PointCloud()
        # rope_pcd.points = o3d.utility.Vector3dVector(rope_points)
        # rope_colors = np.array([[0, 0, 0] for i in range(rope_points.shape[0])])
        # rope_colors[fps_idx] = COLOR_PALETTE[:len(fps_idx)] / 255.0
        # rope_pcd.colors = o3d.utility.Vector3dVector(rope_colors)
        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        # o3d.visualization.draw_geometries([rope_pcd, coordinate])



        downsampled_rope_points = rope_points[fps_idx]
        # transform to sim
        downsampled_rope_points = downsampled_rope_points * self.rope_scale
        downsampled_rope_points += np.array([0, 0, 0.5]) # 0.5 is table height in sim
        downsampled_rope_points[:, 2] = np.clip(downsampled_rope_points[:, 2], 0.55, 1e9)
        pusher_pos = self.get_pusher_position()
        # revise the first point position
        if np.linalg.norm(downsampled_rope_points[0] - (pusher_pos - np.array([0, 0, 0.5]))) >= 0.35:
            print(f"downsampled_rope_points[0]: {downsampled_rope_points[0]}, pusher_pos: {pusher_pos}")
        downsampled_rope_points[0] = pusher_pos - np.array([0, 0, 0.5])
        length = pusher_pos - self.last_pusher_position
        if self.rope_fixed_end_coord is None:
            self.rope_fixed_end_coord = deepcopy(downsampled_rope_points[-1])
        if self.grasp_end is None:
            self.grasp_end = deepcopy(downsampled_rope_points[0])
        
        return np.concatenate([downsampled_rope_points, [pusher_pos], [length]]) # this is transformed to sim

    # in simulation
    # this is the finger joint position
    def get_pusher_position(self,):
        # position in workspace in mm
        raw_pose_in_base = np.array(self.get_robot_pose())
        raw_pose_in_workspace = np.linalg.inv(self.workspace2base) @ np.concatenate([raw_pose_in_base[:3] / 1000, [1]])
        # 0.175m is the height of the gripper, this is the position of the rope gripper end in real world
        raw_pose_in_workspace += np.array([0, 0, -0.175, 0])
        raw_pose_in_workspace = raw_pose_in_workspace * self.rope_scale # this is the position of the rope gripper end in sim
        # 0.5 is table height in sim, 0.5 is the distance between the finger joint and the gripper position
        raw_pose_in_workspace += np.array([0, 0, 0.5, 0]) + np.array([0, 0, 0.5, 0]) 
        return raw_pose_in_workspace[:3]
    
    # in simulation
    # this is the end effector position
    def get_end_effector_position(self,):
        # position in workspace in mm
        raw_pose_in_base = np.array(self.get_robot_pose())
        raw_pose_in_workspace = np.linalg.inv(self.workspace2base) @ np.concatenate([raw_pose_in_base[:3] / 1000, [1]])
        # 0.175m is the height of the gripper, this is the position of the rope gripper end in real world
        raw_pose_in_workspace += np.array([0, 0, -0.175, 0])
        raw_pose_in_workspace = raw_pose_in_workspace * self.rope_scale # this is the position of the rope gripper end in sim
        raw_pose_in_workspace += np.array([0, 0, 1.616, 0]) + np.array([0, 0, 0.5, 0]) # 0.5 is table height in sim, 1.616 is the height of the gripper in sim
        return raw_pose_in_workspace[:3]
    
    # this is the real end effector position
    def get_pusher_position_in_real(self):
        # position in workspace in mm
        raw_pose_in_base = np.array(self.get_robot_pose())
        raw_pose_in_workspace = np.linalg.inv(self.workspace2base) @ np.concatenate([raw_pose_in_base[:3] / 1000, [1]])
        return raw_pose_in_workspace[:3]

    def get_robot_pose(self):
        code, raw_pose_in_base = self.arm.get_position()
        return raw_pose_in_base
    
    # z= 175
    def init_pusher_pos(self, x, y, z=118):
        # x, y, z in mm (pixel in sim)
        self.update_real(x, y, z + 100, wait=True)
        self.update_real(x, y, z, wait=True)
        return
    
    # take in real scale action
    # perform action given pusher position
    def update_real(self, x, y, z=118, wait=False):
        self.last_pusher_position = self.get_pusher_position() # in sim
        # x, y, z in mm (pixel in sim), workspace coord
        start_time = time.time()
        next_position_in_workspace = np.array([x, y, z, 1000]) / 1000
        next_position_in_base = (self.workspace2base @ next_position_in_workspace)[:3]
        next_position_in_base[2] = max(0.118, next_position_in_base[2])
        current_position_in_base = np.array(self.get_robot_pose())[:3] / 1000
        max_offset = (next_position_in_base - current_position_in_base).__abs__().max()
        # steps = max(30, int(max_offset * 10000))
        steps = int(max_offset * 1500)
        # print(f"steps: {steps}")
        next_position_in_base = np.concatenate([next_position_in_base, self.rpy])
        # print(f"next_position_in_base time: {time.time() - start_time}")
        # print(f"\nstart the action update")
        # print(f"curr_pos in base:{current_position_in_base}")
        # print(f"next_pos in base:{next_position_in_base}")
        # print(f"curr_pos in workspace:{(np.linalg.inv(self.workspace2base) @ np.concatenate([current_position_in_base, [1]]))[:3]}")
        # print(f"next_pos in workspace:{next_position_in_workspace}")
        self.set_robot_pose(next_position_in_base, steps=steps, wait=wait, sleep_time=0.03)
    
    # take in simulation scale action
    def update(self, x, y, z, wait=False):
        self.last_pusher_position = self.get_pusher_position() # in sim
        curr_eef_pos_in_sim = self.get_end_effector_position()
        delta_eef_pos_in_sim = np.array([x, y, z]) - curr_eef_pos_in_sim
        delta_eef_pos_in_real = delta_eef_pos_in_sim / self.rope_scale
        curr_eef_pos_in_real = self.get_pusher_position_in_real()
        x, y, z = delta_eef_pos_in_real + curr_eef_pos_in_real
        # x, y, z is in sim scale
        next_position_in_workspace = np.array([x, y, z, 1]) / 1
        next_position_in_base = (self.workspace2base @ next_position_in_workspace)[:3]
        next_position_in_base[2] = max(0.118, next_position_in_base[2])
        current_position_in_base = np.array(self.get_robot_pose())[:3] / 1000
        max_offset = (next_position_in_base - current_position_in_base).__abs__().max()
        # steps = max(30, int(max_offset * 10000))
        steps = int(max_offset * 1500)
        # print(f"steps: {steps}")
        next_position_in_base = np.concatenate([next_position_in_base, self.rpy])
        # print(f"next_position_in_base time: {time.time() - start_time}")
        # print(f"\nstart the action update")
        # print(f"curr_pos in base:{current_position_in_base}")
        # print(f"next_pos in base:{next_position_in_base}")
        # print(f"curr_pos in workspace:{(np.linalg.inv(self.workspace2base) @ np.concatenate([current_position_in_base, [1]]))[:3]}")
        # print(f"next_pos in workspace:{next_position_in_workspace}")
        self.set_robot_pose(next_position_in_base, steps=steps, wait=wait, sleep_time=0.03)
    
    # my sequential code current version
    def set_robot_pose(self, pose, steps=1000, wait=False, sleep_time=0.02):
        initial_qpos = np.array(self.arm.get_servo_angle()[1][0:6]) / 180.0 * np.pi
        next_servo_angle = self.kin_helper.compute_ik_sapien(initial_qpos, pose)
        angles = np.linspace(initial_qpos, next_servo_angle, steps)
        
        # Queue the new pose steps
        for angle in angles:
            self.arm.set_servo_angle_j(angles=angle, is_radian=True)
            time.sleep(0.02)
        self.arm.clean_error()
        self.arm.clean_warn()

        return

    def reset_robot(self):
        self.arm.reset(wait=True)
    
    def get_rope_length(self,):
        return 3.0
    
    def get_rope_fixed_end_coord(self,):
        return self.rope_fixed_end_coord


def main():
    # serial_numbers = ["246322301893", "246322303954", "311322303615", "311322300308"]
    # ["311322303615", "215122252880", "151422251501", "246322303954"]
    # 311322303615 215122252880 151422251501 213622251153
    serial_numbers = ["311322303615", "215122252880", "213622251153",] # "213622251153" 
    # centered by robot base: RF, LB, LF, RB
    # visible_areas = get_visible_areas(serial_numbers)
    visible_areas = None

    param_dict = {
        "capture_fps": 15,
        "record_fps": 15,
        "record_time": 10,
        "reset": False,
        "target_offest": 0.15,
        "obs_gap": 0.15,
    }

    real_wrapper = RealWrapper(param_dict, serial_numbers, )
    real_wrapper.start()
    # real_wrapper.update_real(0, 0, 200)
    # perception_model = PerceptionModule()

    time.sleep(2)
    realsense_data = None
    while True:
        # realsense_data = real_wrapper.realsense.get(out=realsense_data)
        # for i in real_wrapper.camera_indices:
        #     # cv2.imwrite(f"rgb_image_{i}.png", self.last_realsense_data[i]["color"])
        #     # print(f"self.last_realsense_data[i]['color'].shape: {self.last_realsense_data[i]['color']}")
        #     realsense_data[i]["color"] = cv2.cvtColor(realsense_data[i]["color"], cv2.COLOR_BGR2RGB)
        # rgb_list = [realsense_data[i]["color"] for i in real_wrapper.camera_indices]
        # depth_list = [realsense_data[i]["depth"]/1000. for i in real_wrapper.camera_indices]
        # R_list = real_wrapper.R_list
        # t_list = real_wrapper.T_list
        # intr_list = real_wrapper.realsense.get_intrinsics()
        # bbox = np.array([[-0.1, 0.8], \
        #                 [-0.1, 0.8], \
        #                 [-0.05, 1.0]])
        # perception_model.get_tabletop_points(rgb_list, depth_list, R_list, t_list, intr_list, \
        #                                     bbox)
        real_wrapper.get_state()
        input_key = input("Press x to break...")

    # real_wrapper.stop()


if __name__ == "__main__":
    main()
