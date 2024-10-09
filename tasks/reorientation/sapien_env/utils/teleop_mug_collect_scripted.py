import os
import time

import h5py
import numpy as np
import cv2
from pathlib import Path

import sys

from sapien_env.rl_env.mug_collect_env import MugCollectRLEnv
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.gui.teleop_gui_trossen import GUIBase, DEFAULT_TABLE_TOP_CAMERAS, YX_TABLE_TOP_CAMERAS
from sapien_env.teleop.teleop_robot import TeleopRobot
from sapien_env.teleop.mug_collect_scripted_policy import SingleArmPolicy
from sapien_env.utils.data_utils import save_dict_to_hdf5
from sapien_env.utils.misc_utils import get_current_YYYY_MM_DD_hh_mm_ss_ms
MIN_VALUE = 0.0  # Minimum value of the contact sensor data
MAX_VALUE = 0.05  # Maximum value of the contact sensor data
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 300

def stack_dict(dic):
    # stack list of numpy arrays into a single numpy array inside a nested dict
    for key, item in dic.items():
        if isinstance(item, dict):
            dic[key] = stack_dict(item)
        elif isinstance(item, list):
            dic[key] = np.stack(item, axis=0)
    return dic

def main_env():
    env = MugCollectRLEnv(use_gui=True, robot_name="panda", frame_skip=10, use_visual_obs=False)
    env.reset()
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer)
    for name, params in YX_TABLE_TOP_CAMERAS.items():
        gui.create_camera(**params)
    gui.viewer.set_camera_rpy(r=0, p=-0.5, y=np.pi/2)
    gui.viewer.set_camera_xyz(x=0, y=0.5, z=0.5)
    scene = env.scene
    scene.step()
    
    teleop = TeleopRobot(robot_name="panda")
    
    for link in env.robot.get_links():
        print(link.get_name(),link.get_pose())
    for joint in env.robot.get_active_joints():
        print(joint.get_name())

    # set up data saving hyperparameters
    dataset_dir = "data/teleop_data/pick_place_soda"
    data_dict = {
        'observations': 
            {'qpos': [],
             'qvel': [],
             'images': {},},
        'action': [],
        'info':
            {'init_pose': env.manipulated_object.get_pose().to_transformation_matrix(),}
    }
    cams = gui.cams
    for cam in cams:
        data_dict['observations']['images'][f'{cam.name}_color'] = []
        data_dict['observations']['images'][f'{cam.name}_depth'] = []
        data_dict['observations']['images'][f'{cam.name}_intrinsic'] = []
        data_dict['observations']['images'][f'{cam.name}_extrinsic'] = []
    attr_dict = {
        'sim': True,
    }
    config_dict = {
        'observations':
            {
                'images': {}
            }
    }
    for cam_idx, cam in enumerate(gui.cams):
        color_save_kwargs = {
            'chunks': (1, cam.height, cam.width, 3), # (1, 480, 640, 3)
            'compression': 'gzip',
            'compression_opts': 9,
            'dtype': 'uint8',
        }
        depth_save_kwargs = {
            'chunks': (1, cam.height, cam.width), # (1, 480, 640)
            'compression': 'gzip',
            'compression_opts': 9,
            'dtype': 'uint16',
        }
        config_dict['observations']['images'][f'{cam.name}_color'] = color_save_kwargs
        config_dict['observations']['images'][f'{cam.name}_depth'] = depth_save_kwargs



    # copy current repo
    save_repo_path = f'sapien_env_{get_current_YYYY_MM_DD_hh_mm_ss_ms()}'
    save_repo_dir = os.path.join(dataset_dir, save_repo_path)
    os.system(f'mkdir -p {save_repo_dir}')
    
    curr_repo_dir = Path(__file__).parent.parent
    ignore_list = ['.git', '__pycache__', 'data', 'assets']
    for sub_dir in os.listdir(curr_repo_dir):
        if sub_dir not in ignore_list:
            os.system(f'cp -r {os.path.join(curr_repo_dir, sub_dir)} {save_repo_dir}')



    # collect data
    num_episodes = 50
    for episode_idx in range(num_episodes):
        timesteps = 0
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.h5py')
        
        env.reset()
        scripted_policy = SingleArmPolicy()
        cartisen_action =teleop.init_cartisen_action_franka(env.robot.get_qpos()[:9])
        
        arm_dof = env.arm_dof
        action = np.zeros(arm_dof+1)
        while True:
            cartisen_action = scripted_policy.single_trajectory(env,teleop.ee_link_pose)
            quit = False
            if quit:
                break
            action[:arm_dof] = teleop.ik_panda(env.robot.get_qpos()[:],cartisen_action)
            action[arm_dof:] = cartisen_action[6]
            # print(action)
            obs, reward, done, _ = env.step(action[:arm_dof+1])
            rgbs, depths = gui.render(depth=True)
            
            data_dict['observations']['qpos'].append(env.robot.get_qpos()[:-1])
            data_dict['observations']['qvel'].append(env.robot.get_qvel()[:-1])
            data_dict['action'].append(action.copy())
            for cam_idx, cam in enumerate(gui.cams):
                data_dict['observations']['images'][f'{cam.name}_color'].append(rgbs[cam_idx])
                data_dict['observations']['images'][f'{cam.name}_depth'].append(depths[cam_idx])
                data_dict['observations']['images'][f'{cam.name}_intrinsic'].append(cam.get_intrinsic_matrix())
                data_dict['observations']['images'][f'{cam.name}_extrinsic'].append(cam.get_extrinsic_matrix())
            timesteps += 1
        data_dict = stack_dict(data_dict)
        save_dict_to_hdf5(data_dict, config_dict, dataset_path, attr_dict=attr_dict)

if __name__ == '__main__':
    main_env()