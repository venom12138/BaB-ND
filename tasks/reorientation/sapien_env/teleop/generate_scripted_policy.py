import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import IPython
import transforms3d
e = IPython.embed
import os
import cv2
import sys
import sapien.core as sapien
cwd = os.getcwd()
sys.path.append(cwd)
from sapien_env.rl_env.relocate_rlenv import RelocateRLEnv
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.gui.teleop_gui_trossen import GUIBase, DEFAULT_TABLE_TOP_CAMERAS,WRTHCH_USING_CAMERAS
from sapien_env.teleop.teleop_robot import TeleopRobot
from sapien_env.utils.robot_sample import RobotPcSampler
from sapien_env.rl_env.para import ARM_INIT
import copy
import open3d as o3d
import time
MIN_VALUE = 0.0  # Minimum value of the contact sensor data
MAX_VALUE = 3.0  # Maximum value of the contact sensor data
WINDOW_WIDTH = 200
WINDOW_HEIGHT = 150

class SingleArmPolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.trajectory = None

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def single_trajectory(self,env, ee_link_pose):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(env, ee_link_pose)


        if self.trajectory[0]['t'] == self.step_count:
            self.curr_waypoint = self.trajectory.pop(0)
        next_waypoint = self.trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        xyz, quat, gripper = self.interpolate(self.curr_waypoint, next_waypoint, self.step_count)


        # Inject noise
        if self.inject_noise:
            scale = 0.01
            xyz = xyz + np.random.uniform(-scale, scale, xyz.shape)

        self.step_count += 1
        cartisen_action_dim =6
        grip_dim = 1
        cartisen_action = np.zeros(cartisen_action_dim+grip_dim)
        eluer = transforms3d.euler.quat2euler(quat,axes='sxyz')
        cartisen_action[0:3] = xyz
        cartisen_action[3:6] = eluer
        cartisen_action[6] = gripper
        return cartisen_action
    
    def generate_trajectory(self, env, ee_link_pose):
        cube_position= env.manipulated_object.get_pose().p 
        cude_offest_1 = np.array([-0.2,0.0,0])
        cude_offest_2 = cude_offest_1 + [0.07, 0, 0]
        target_position= env.target.get_pose().p + cude_offest_1 + [0.07, 0, 0.1]
        gripper_pick_quat = Quaternion(ee_link_pose.q[:4]) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=30)
        open_gripper = 0.0
        close_gripper = 0.52
        self.trajectory = [
            {"t": 0, "xyz": ee_link_pose.p, "quat": ee_link_pose.q, "gripper": open_gripper}, #init    
            # {"t": 200, "xyz": ee_link_pose.p, "quat": ee_link_pose.q, "gripper": 0.021}, #init 
            {"t": 20, "xyz": ee_link_pose.p, "quat": ee_link_pose.q, "gripper": open_gripper}, # open gripper
            {"t": 50, "xyz": cube_position+cude_offest_2, "quat": gripper_pick_quat.elements, "gripper": open_gripper}, 
            {"t": 70, "xyz": cube_position+cude_offest_2, "quat": gripper_pick_quat.elements, "gripper": open_gripper}, # move to pick
            {"t": 80, "xyz": cube_position+cude_offest_2, "quat": gripper_pick_quat.elements, "gripper": close_gripper}, #gripper close
            {"t": 90, "xyz": cube_position+cude_offest_2, "quat": gripper_pick_quat.elements, "gripper": close_gripper}, # stop for a while
            {"t": 110, "xyz": cube_position+cude_offest_2+[0,0,0.1], "quat": gripper_pick_quat.elements, "gripper": close_gripper}, # move up
            {"t": 160, "xyz": target_position, "quat": gripper_pick_quat.elements, "gripper": close_gripper},# move to target
            {"t": 170, "xyz": target_position, "quat": gripper_pick_quat.elements, "gripper": close_gripper},# stop for a while
            {"t": 200, "xyz": target_position, "quat": gripper_pick_quat.elements, "gripper": open_gripper}, # open gripper
            # {"t": 250, "xyz": target_position+np.array([-0.2,0,0]), "quat": gripper_pick_quat.elements, "gripper": 0.07}, # go back to init
        ]


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


def relocate_env():
    robot_name = 'xarm6_with_gripper'
    teleop = TeleopRobot(robot_name)
    max_timesteps = 200
    num_episodes = 50
    onscreen = True
    for episode_idx in range(num_episodes):
        # trossen_vx300s_tactile_thin, trossen_vx300s_new_gripper
        env = RelocateRLEnv(use_gui=True, robot_name=robot_name,
                            object_name="tomato_soup_can", frame_skip=10, use_visual_obs=False, use_ray_tracing=False)
        env.seed(episode_idx)
        env.reset()

        # Setup viewer and camera
        add_default_scene_light(env.scene, env.renderer)
        gui = GUIBase(env.scene, env.renderer)
        for cam_name, params in DEFAULT_TABLE_TOP_CAMERAS.items():
            gui.create_camera(**params)
        gui.viewer.set_camera_rpy(0, -0.7, 0.01)
        gui.viewer.set_camera_xyz(-0.4, 0, 0.45)
        scene = env.scene
        steps = 0
        gui.viewer.toggle_axes(False)
        scene.step()

        env.seed(episode_idx)
        # env.reset_env()
        scene.step()
        # env.step([0,0,0,0,0,0,0.85])
        scripted_policy = SingleArmPolicy()
        cartisen_action=teleop.init_cartisen_action(env.robot.get_qpos()[:])
        action = np.zeros(7)
        arm_dof = env.arm_dof
        # Set data dir
        
        for i in range(max_timesteps):
            cartisen_action = scripted_policy.single_trajectory(env,env.palm_link.get_pose())
            cartisen_action_in_rob = transform_action_from_world_to_robot(cartisen_action,env.robot.get_pose())
            action[:arm_dof] = teleop.teleop_ik(env.robot.get_qpos()[:],cartisen_action_in_rob)
            action[arm_dof:] = cartisen_action[6]
            # action[arm_dof:] = i*0.85/max_timesteps
            obs, reward, done, _ = env.step(action[:7])

            if onscreen:
                gui.render()
        gui.viewer.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    relocate_env()
