import sys
import os
from xarm import version
from xarm.wrapper import XArmAPI
import ctypes
import matplotlib.pyplot as plt

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
import transforms3d
from queue import Queue, Empty
import pyrealsense2 as rs
import open3d as o3d
from camera.realsense import RS_Camera
from common.kinematics_utils import KinHelper
from real_exp_new.utils import rpy_to_rotation_matrix, get_pose_from_front_up_end_effector, quat2rpy # real_exp_new
# from arm_controller import ArmController
from dino_wrapper import DinoWrapper
FLOAT_TYPE = np.float16
file_dir = os.path.dirname(os.path.abspath(__file__))

class RealWrapper:
    def __init__(
        self, serial_number, dino):
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
        x, y, z, roll, pitch, yaw = [196.2, -1.6, 434, 179.2, 0, 0.3]
        self.xyz = np.array([x, y, z]) / 1000
        self.rpy = np.array([roll, pitch, yaw]) / 180.0 * np.pi

        # realsense part
        # serial_number = "311322300308"
        self.camera = RS_Camera(serial_number)
        
        # camera calibration part
        calibrate_result_dir = f"{file_dir}/calibration_result"
        with open(f"{calibrate_result_dir}/calibration_handeye_result.pkl", "rb") as f:
            handeye_result = pickle.load(f)

        R_base2world, t_base2world = handeye_result["R_base2world"], handeye_result["t_base2world"]        
        base2world = np.eye(4)
        base2world[:3, :3] = R_base2world
        base2world[:3, 3] = t_base2world
        
        R_gripper2cam = handeye_result["R_gripper2cam"]
        t_gripper2cam = handeye_result["t_gripper2cam"]
        gripper2cam = np.eye(4)
        gripper2cam[:3, :3] = R_gripper2cam
        gripper2cam[:3, 3] = t_gripper2cam

        # adjust world coordinate in camera coordinate to world coordinate for task
        adjust_matrix = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        base2world = adjust_matrix @ base2world
        # gripper2cam = adjust_matrix @ gripper2cam # TODO: need to check
        self.cam2gripper = np.linalg.inv(gripper2cam)

        world2base = np.linalg.inv(base2world)
        workspace2world = np.eye(4)
        workspace2world[:3, 3] = np.array([-0.05, 0.2, 0])
        # self.world2base = world2base
        # self.workspace2world = workspace2world
        
        self.workspace2base = world2base @ workspace2world
        self.base2workspace = np.linalg.inv(self.workspace2base)
        
        self.observation_coordinate = np.array([0.392, -0.0574, 0.472])
        self.base_iter = 1000
        self.object_height = 0.01
        self.offset = np.array([0.0128,0.0097,0.01565]) # np.array([0.032-0.0172, 0.023-0.0056, 0.01565])
        self.alive = False

        # dino part
        self.dino = dino

    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.realsense.is_ready
    
    def start(self, wait=True, exposure_time=5):
        self.alive = True
        # self.arm.start()

    def stop(self, wait=True):
        self.alive = False
        # self.arm.stop()

    def _get_cam2base(self):
        current_pose = self.get_robot_pose()
        # print("Current pose: ", current_pose)
        R_cur_gripper2base = rpy_to_rotation_matrix(current_pose[3], current_pose[4], current_pose[5])
        # Need to make srue the unit of the pose is in meters
        t_cur_gripper2base = np.array(current_pose[:3]) / 1000
        current_gripper2base = np.eye(4)
        current_gripper2base[:3, :3] = R_cur_gripper2base
        current_gripper2base[:3, 3] = t_cur_gripper2base
        # Get the camera pose in the base frame
        cam2base = np.dot(current_gripper2base, self.cam2gripper)
        # Improve the calibration based on the offset
        cam2base[:3, 3] += self.offset
        return cam2base

    # Only run one action each time, to keep the same interface as the simulation environment
    # To keep the same interface, there are some extra code to do the conversion
    def run_action(self, action_code=0, action_parameters=[], for_camera=False, speed=75, **kwargs):
        # print(f"Running action {action_code} with parameters {action_parameters}")
        if action_code == 1:
            # move the end effector, parameters: qpos
            xyz = action_parameters[:3] * 1000
            rpy = quat2rpy(action_parameters[3:])
            # Need to do a 30-degree rotation in pitch if the movement is for the camera
            if for_camera:
                rpy[1] += 30
            # pose = list(xyz) + list(rpy)
            # code = self.arm.set_position(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], speed=100)
            initial_qpos = np.array(self.arm.get_servo_angle()[1][0:6]) / 180.0 * np.pi
            xyz /= 1000
            rpy = rpy /180 * np.pi
            pose = np.array(list(xyz) + list(rpy))
            next_servo_angle = self.kin_helper.compute_ik_sapien(initial_qpos, pose)
            angles = np.linspace(initial_qpos, next_servo_angle, 1000)
            for i, angle in enumerate(angles):
                self.arm.set_servo_angle_j(angles=angle, is_radian=True)
                time.sleep(0.003)
                # if i % 100 == 0:
                #     pass
            # if for_camera:
            #     time.sleep(5)
        elif action_code == 2:
            assert False, "Not implemented"
            # Open the gripper
            half_open = False
            if len(action_parameters) > 0:
                half_open = True
            if half_open:
                self.arm.set_gripper_position(460, wait=True)
            else:
                self.arm.set_gripper_position(830, wait=True)
        elif action_code == 3:
            assert False, "Not implemented"
            # Close the gripper
            self.arm.set_gripper_position(0, wait=True)
        elif action_code == 4:
            # Make the robot back to the default position
            assert False, "Not implemented"
        else:
            raise ValueError("Invalid action code")
    
    def get_observations(self, visualize=False, **kwargs):
        # Get the observations
        points, colors, depths, mask = self.camera.get_observations()
        # Get the camera2base transformation for the current pose
        cam2base = self._get_cam2base()
        # Save all the observations
        observations = {
            "wrist": {
                "position": points,
                "rgb": colors,
                "depths": depths,
                "mask": mask,
                "c2b": cam2base,
                "intrinsic": self.camera.intrinsic_matrix,
                "dist_coef": self.camera.dist_coef,
            }
        }
        if visualize:
            # Get current valid points
            valid_points = points[mask]
            valid_colors = colors[mask]
            valid_points = np.concatenate([valid_points, np.ones([valid_points.shape[0], 1])], axis=1)
            valid_points = np.dot(cam2base, valid_points.T).T[:, :3]
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(valid_points)
            cloud.colors = o3d.utility.Vector3dVector(valid_colors)

            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            o3d.visualization.draw_geometries([cloud, coordinate])
        return observations

    def get_state(self, visualize=False):
        # Move to the predefined position and take a top-down observation
        end_effector_position = self.observation_coordinate
        front_direction = np.array([0, 0, -1])
        up_direction = np.array([0, 1, 0])
        end_effector_rotation = get_pose_from_front_up_end_effector(front=front_direction, up=up_direction)
        self.run_action(
            action_code=1,
            iteration=self.base_iter,
            action_parameters=np.concatenate(
                [
                    end_effector_position,
                    end_effector_rotation,
                ]
            ),
            for_camera=True,
        )

        observation = self.get_observations(visualize, wrist_only=True)["wrist"]
        box_centers = self.dino.predict(observation["rgb"].copy())
        c2b = observation["c2b"]
        box_centers = box_centers.numpy().astype(np.int32)
        position = observation["position"][box_centers[:,0], box_centers[:,1]]

        position_base = (position @ c2b[:3, :3].T + c2b[:3, 3])
        print(f"position_base:{position_base}")
        position_workspace = position_base @ self.base2workspace[:3, :3].T + self.base2workspace[:3, 3]
        obj_num = position_workspace.shape[0]
        print(f"position_workspace:{position_workspace}")
        
        state = np.zeros((obj_num+2, 2))
        for i in range(obj_num):
            state[i] = position_workspace[i, :2] * 1000
        
        state[-2] = self.pusher_end[:2] * 1000
        state[-1] = (self.pusher_end[:2] - self.pusher_start[:2]) * 1000
        return state # in mm/ pixel

    def get_pusher_position(self):
        # position in workspace in mm
        raw_pose_in_base = np.array(self.get_robot_pose())
        raw_pose_in_workspace = np.linalg.inv(self.workspace2base) @ np.concatenate([raw_pose_in_base[:3] / 1000, [1]])
        return raw_pose_in_workspace[:3]

    def get_robot_pose(self):
        if not self.alive:
            raise ValueError("Robot is not alive!")
        # directly use get_position
        code, raw_pose_in_base = self.arm.get_position()
        if code != 0:
            raise ValueError("get_robot_pose Error")
        
        # use qpos to compute the pose
        # cur_qpos = np.array(self.arm.get_servo_angle()[1][0:6]) / 180. * np.pi
        # fk_trans_mat = self.kin_helper.compute_fk_sapien_links(cur_qpos, [self.kin_helper.sapien_eef_idx])[0]
        # cur_position = np.zeros(6)
        # cur_position[:3] = fk_trans_mat[:3,3]*1000
        # cur_position[3:] = transforms3d.euler.mat2euler(fk_trans_mat[:3,:3],)
        # cur_position[3:] = cur_position[3:] / np.pi * 180
        # raw_pose_in_base = np.array(cur_position)

        return raw_pose_in_base
    
    # z= 175
    def init_pusher_pos(self, x, y, z=50):
        # x, y, z in mm (pixel in sim)
        self.update(x, y, z + 150)
        self.update(x, y, z)
        return

    # perform action given pusher position
    def update(self, x, y, z=50, fd=[0, 0, -1], ud=[0, 1, 0]):
        self.pusher_start = self.get_pusher_position()
        # x, y, z in mm (pixel in sim), workspace coord
        next_position_in_workspace = np.array([x, y, z, 1000]) / 1000
        next_position_in_base = (self.workspace2base @ next_position_in_workspace)[:3]
        next_position_in_base[2] = max(0.05, next_position_in_base[2]) # 0.112
        # print(f"next_position_in_base time: {time.time() - start_time}")
        # print(f"\nstart the action update")
        # print(f"curr_pos in base:{current_position_in_base}")
        # print(f"next_pos in base:{next_position_in_base}")
        # print(f"curr_pos in workspace:{(np.linalg.inv(self.workspace2base) @ np.concatenate([current_position_in_base, [1]]))[:3]}")
        # print(f"next_pos in workspace:{next_position_in_workspace}")
        ude = (self.workspace2base @ np.concatenate([ud, [1]], axis=0))[:3]
        uds = (self.workspace2base @ np.concatenate([[0,0,0], [1]], axis=0))[:3]
        ud = ude - uds
        ud[2] = 0
        ud = ud / np.linalg.norm(ud)
        front_direction = np.array(fd)
        up_direction = np.array(ud) + np.array([1e-4, 1e-4, 1e-4])
        end_effector_rotation = get_pose_from_front_up_end_effector(front=front_direction, up=up_direction)
        end_effector_position = next_position_in_base
        curr_z_in_base = np.array(self.get_robot_pose())[2]/1000
        first_eef_position = np.array([next_position_in_base[0], next_position_in_base[1], min(curr_z_in_base,0.164)])
        # first move in the xy plane
        self.run_action(
            action_code=1,
            iteration=self.base_iter,
            action_parameters=np.concatenate(
                [
                    first_eef_position,
                    end_effector_rotation,
                ]
            ),
            speed=50,
        )
        if first_eef_position[2] - end_effector_position[2] > 0.015:
            self.run_action(
                action_code=1,
                iteration=self.base_iter,
                action_parameters=np.concatenate(
                    [
                        end_effector_position,
                        end_effector_rotation,
                    ]
                ),
                speed=50,
            )
        self.pusher_end = self.get_pusher_position()

    def reset_robot(self):
        self.arm.reset(wait=True)

def main():
    # serial_numbers = ["246322301893", "246322303954", "311322303615", "311322300308"]
    # ["311322303615", "215122252880", "151422251501", "246322303954"]
    # 311322303615 215122252880 151422251501 213622251153
    serial_number = "311322300308"
    dino = DinoWrapper(obj_num=6, cls_num=2)
    real_wrapper = RealWrapper(serial_number, dino)
    real_wrapper.start()
    real_wrapper.init_pusher_pos(*[50,50,200])
    state = real_wrapper.get_state()
    print(f"state:{state}\n")
    print(f"state in base")
    time.sleep(2)
    while True:
        pusher_pos = real_wrapper.get_pusher_position()
        # real_wrapper.init_pusher_pos(*[50,50,150])
        print(f"pusher_pos:{pusher_pos}\n")
        # for big T
        # state = real_wrapper.get_state()
        # import pdb; pdb.set_trace()
        # print(f"state:{state}\n")
        # input_key = input("Press x to break...")


if __name__ == "__main__":
    main()
