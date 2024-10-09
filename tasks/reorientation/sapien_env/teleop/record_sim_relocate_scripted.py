import numpy as np
import os
import h5py
import time
import cv2
import sys
import torch
import sapien.core as sapien
from pathlib import Path
sys.path.append(os.environ['SAPIEN_ROOT'])
from omegaconf import OmegaConf
import hydra
import transforms3d
from sapien_env.rl_env.base import BaseRLEnv
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.gui.teleop_gui_trossen import GUIBase, DEFAULT_TABLE_TOP_CAMERAS
from sapien_env.teleop.teleop_robot import TeleopRobot
from sapien_env.teleop.generate_scripted_policy import SingleArmPolicy
from sapien_env.utils.data_utils import save_dict_to_hdf5
from sapien_env.utils.misc_utils import get_current_YYYY_MM_DD_hh_mm_ss_ms

MIN_VALUE = 0.0  # Minimum value of the contact sensor data
MAX_VALUE = 0.05  # Maximum value of the contact sensor data
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 300

def transform_action_from_world_to_robot(action : np.ndarray, pose : sapien.Pose):
    # :param action: (7,) np.ndarray in world frame. action[:3] is xyz, action[3:6] is euler angle, action[6] is gripper
    # :param pose: sapien.Pose of the robot base in world frame
    # :return: (7,) np.ndarray in robot frame. action[:3] is xyz, action[3:6] is euler angle, action[6] is gripper
    # transform action from world to robot frame
    action_mat = np.zeros((4,4))
    action_mat[:3,:3] = transforms3d.euler.euler2mat(action[3], action[4], action[5])
    action_mat[:3,3] = action[:3]
    action_mat[3,3] = 1
    action_mat_in_robot = np.matmul(np.linalg.inv(pose.to_transformation_matrix()),action_mat)
    action_robot = np.zeros(7)
    action_robot[:3] = action_mat_in_robot[:3,3]
    action_robot[3:6] = transforms3d.euler.mat2euler(action_mat_in_robot[:3,:3],axes='sxyz')
    action_robot[6] = action[6]
    return action_robot


def stack_dict(dic):
    # stack list of numpy arrays into a single numpy array inside a nested dict
    for key, item in dic.items():
        if isinstance(item, dict):
            dic[key] = stack_dict(item)
        elif isinstance(item, list):
            dic[key] = np.stack(item, axis=0)
    return dic


def main_env():
    teleop = TeleopRobot()
    max_timesteps = 200
    num_episodes = 5
    # set up data saving hyperparameters
    dataset_dir = "data/scripted_data/scripted_relocate_02_08"
    
    # copy current repo
    save_repo_path = f'sapien_env_{get_current_YYYY_MM_DD_hh_mm_ss_ms()}'
    save_repo_dir = os.path.join(dataset_dir, save_repo_path)
    os.system(f'mkdir -p {save_repo_dir}')
    
    curr_repo_dir = Path(__file__).parent.parent
    ignore_list = ['.git', '__pycache__', 'data', 'assets']
    for sub_dir in os.listdir(curr_repo_dir):
        if sub_dir not in ignore_list:
            os.system(f'cp -r {os.path.join(curr_repo_dir, sub_dir)} {save_repo_dir}')

    cfg = OmegaConf.create(
        {
            '_target_': 'sapien_env.rl_env.relocate_env.RelocateRLEnv',
            'use_gui': True,
            'robot_name': 'trossen_vx300s_tactile_pad',
            'frame_skip': 10,
            'use_visual_obs': False,
            'use_ray_tracing': False,
        }
    )
    with open(os.path.join(dataset_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f.name)

    success = 0
    success_rate =0
    total_trail =0
    for episode_idx in range(num_episodes):
        env : BaseRLEnv = hydra.utils.instantiate(cfg)
        episode_idx += 0
        env.seed(episode_idx)
        env.reset()
        # Setup viewer and camera
        add_default_scene_light(env.scene, env.renderer)
        gui = GUIBase(env.scene, env.renderer,headless=True)
        for cam_name, params in DEFAULT_TABLE_TOP_CAMERAS.items():
            gui.create_camera(**params)
        if not gui.headless:
            gui.viewer.set_camera_rpy(0, -0.7, 0.01)
            gui.viewer.set_camera_xyz(-0.4, 0, 0.45)
            cv2.namedWindow("Left finger and Right finger Contact Data", cv2.WINDOW_NORMAL)
            sensor_number_dim = int(env.sensors.shape[0]/2)
        scene = env.scene
        scene.step()

        # setup data saving dict
        init_poses = np.stack([env.manipulated_object.get_pose().to_transformation_matrix(),
                                    env.target.get_pose().to_transformation_matrix()])
        data_dict = {
            'observations': 
                {'joint_pos': [],
                 'joint_vel': [],
                 'ee_pos': [],
                #  'ee_vel': [],
                 'left_contact': [], 
                 'right_contact': [],
                 'left_contact_points': [],
                 'right_contact_points': [],
                 'finger_pos': {},
                 'images': {},
                 'robot_base_pose_in_world': [],
                 },
            'joint_action': [],
            'cartesian_action': [],
            'info':
                {'init_poses': init_poses,
                 }
        }
        cams = gui.cams
        finger_names = ['left_finger_link','right_finger_link']
        for finger in finger_names:
            data_dict['observations']['finger_pos'][finger] = []
        for cam in cams:
            data_dict['observations']['images'][f'{cam.name}_color'] = []
            data_dict['observations']['images'][f'{cam.name}_depth'] = []
            data_dict['observations']['images'][f'{cam.name}_intrinsics'] = []
            data_dict['observations']['images'][f'{cam.name}_extrinsics'] = []
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

        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')

        # setup robot scripted policy
        scripted_policy = SingleArmPolicy()
        cartisen_action=teleop.init_cartisen_action(env.robot.get_qpos()[:])
        action = np.zeros(7)
        arm_dof = env.arm_dof
        t1 = time.time()
        for i in range(max_timesteps):
            cartisen_action = scripted_policy.single_trajectory(env,env.palm_link.get_pose())
            
            cartisen_action_in_rob = transform_action_from_world_to_robot(cartisen_action,env.robot.get_pose())
            action[:arm_dof] = teleop.teleop_ik(env.robot.get_qpos()[:],cartisen_action_in_rob)
            action[arm_dof:] = cartisen_action_in_rob[6]
            # print(action)
            obs, reward, done, _ = env.step(action[:arm_dof+1])
            rgbs, depths = gui.render(depth=True)

            data_dict['observations']['joint_pos'].append(env.robot.get_qpos()[:-1])
            data_dict['observations']['joint_vel'].append(env.robot.get_qvel()[:-1])
            data_dict['observations']['robot_base_pose_in_world'].append(env.robot.get_pose().to_transformation_matrix())
            ee_translation = env.palm_link.get_pose().p
            ee_rotation = transforms3d.euler.quat2euler(env.palm_link.get_pose().q,axes='sxyz')
            ee_gripper = env.robot.get_qpos()[arm_dof]
            ee_pos = np.concatenate([ee_translation,ee_rotation,[ee_gripper]])
            # ee_vel = np.concatenate([env.palm_link.get_velocity(),env.palm_link.get_angular_velocity(),env.robot.get_qvel()[arm_dof:arm_dof+1]])
            # transform ee_pos and ee_vel from world to robot frame
            ee_pos = transform_action_from_world_to_robot(ee_pos,env.robot.get_pose())
            data_dict['observations']['ee_pos'].append(ee_pos)
            # data_dict['observations']['ee_vel'].append(ee_vel)
            data_dict['joint_action'].append(action.copy())
            data_dict['cartesian_action'].append(cartisen_action_in_rob.copy())

            for cam_idx, cam in enumerate(gui.cams):
                data_dict['observations']['images'][f'{cam.name}_color'].append(rgbs[cam_idx])
                data_dict['observations']['images'][f'{cam.name}_depth'].append(depths[cam_idx])
                data_dict['observations']['images'][f'{cam.name}_intrinsics'].append(cam.get_intrinsic_matrix())
                data_dict['observations']['images'][f'{cam.name}_extrinsics'].append(cam.get_extrinsic_matrix())
        total_trail += 1
        if reward ==1:
            #compute success rate
            print('reward:',reward,'success!')
            success += 1
            success_rate = success/total_trail
        print(success_rate)
        data_dict = stack_dict(data_dict)
        save_dict_to_hdf5(data_dict, config_dict, dataset_path, attr_dict=attr_dict)
        t2 = time.time()
        if not gui.headless:
            gui.viewer.close()
            cv2.destroyAllWindows()
        print(f"Episode {episode_idx} finished","time:",t2-t1)

if __name__ == '__main__':
    main_env()