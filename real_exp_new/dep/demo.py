import sys
import os
from xarm import version
from xarm.wrapper import XArmAPI
import ctypes
# ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(ROOT_DIR)
# os.chdir(ROOT_DIR)

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
import pyrealsense2 as rs
import open3d as o3d
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Value, Manager
from threading import Lock
from camera.multi_realsense import MultiRealsense, SingleRealsense
from camera.video_recorder import VideoRecorder
from xarm6 import XARM6

from common.kinematics_utils import KinHelper



def main():
    calibrate_result_dir = 'real_world/calibration_result'
    with open(f'{calibrate_result_dir}/calibration_handeye_result.pkl', 'rb') as f:
        handeye_result = pickle.load(f)
    R_base2world, t_board2base = handeye_result['R_base2world'], handeye_result['t_base2world']
    base2world = np.eye(4)
    base2world[:3, :3] = R_base2world
    base2world[:3, 3] = t_board2base
    world2base = np.linalg.inv(base2world)

    position_in_world = np.array([0.2, 0.2, -0.4, 1])
    position_in_base = world2base @ position_in_world


    workspace2world = np.eye(4)
    workspace2world[:3, 3] = np.array([0.2, -0.05, -0.2])
    positions_in_workspace = np.array([[0, 0, 0, 1],
                                        [0.4, 0, 0, 1],
                                        [0.4, 0.4, 0, 1],
                                        [0, 0.4, 0, 1],
                                        [0.2, 0.2, 0, 1],
                                        [0.05, 0.05, 0, 1]])
    ip = "192.168.1.209"
    arm = XArmAPI(ip)
    arm.motion_enable(enable=True)
    arm.set_mode(1)
    arm.set_state(state=0)
    kin_helper = KinHelper(robot_name='xarm6')

    time.sleep(1)

    # initial position
    x, y, z, roll, pitch, yaw = [196.2,-1.6,434,179.2,0,0.3]
    rpy = np.array([roll, pitch, yaw]) / 180. * np.pi
    # next_position = np.array([x, y, z, roll, pitch, yaw])

    for j in range(positions_in_workspace.shape[0]):
        position_in_workspace = positions_in_workspace[j]
        position_in_base = world2base @ workspace2world @ position_in_workspace
        initial_qpos = np.array(arm.get_servo_angle()[1][0:6]) / 180. * np.pi
        position_in_base = np.concatenate([position_in_base[:3], rpy])
        next_servo_angle = kin_helper.compute_ik_sapien(initial_qpos, position_in_base)
        print("nesxt servo angle: ", next_servo_angle)
        for i in range(1000):
            angle = initial_qpos + (next_servo_angle - initial_qpos) * i / 1000.
            # print(angle)
            # print(arm.get_position())
            arm.set_servo_angle_j(angles=angle,is_radian=True)
            time.sleep(0.005)
        input("Press Enter to continue...")

    pass

if __name__ == '__main__':
    main()