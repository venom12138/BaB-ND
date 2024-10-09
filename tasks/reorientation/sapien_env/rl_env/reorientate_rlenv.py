from functools import cached_property
from typing import Optional

import numpy as np
import sapien.core as sapien
import transforms3d
from sapien.utils import Viewer

import sys
import os
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.append(cwd)
sys.path.append(parent_dir)
# /home/venom/projects/RobotCrown/neural-dynamics-crown/tasks/reorientation/sapien_env/rl_env/reorientate_rlenv.py
from tasks.reorientation.sapien_env.rl_env.base import BaseRLEnv
from tasks.reorientation.sapien_env.sim_env.reorientate_env import ReorientateEnv
from tasks.reorientation.sapien_env.rl_env.para import ARM_INIT
import cv2
OBJECT_LIFT_LOWER_LIMIT = -0.03

class ReorientateRLEnv(ReorientateEnv, BaseRLEnv):
    def __init__(self, use_gui=False, frame_skip=5, robot_name="xarm6_with_gripper", constant_object_state=False,
                 rotation_reward_weight=0, 
                 randomness_scale=1, friction=1, object_pose_noise=0.01,use_ray_tracing=False, 
                 save_img=False, **renderer_kwargs):
        super().__init__(use_gui, frame_skip, randomness_scale, friction,use_ray_tracing=use_ray_tracing,
                         **renderer_kwargs)
        self.setup(robot_name)
        self.task_name = "reorientate"
        self.constant_object_state = constant_object_state
        self.rotation_reward_weight = rotation_reward_weight
        self.object_pose_noise = object_pose_noise
        
        # Parse link name
        self.palm_link_name = self.robot_info.palm_name # link6
        self.palm_link = [link for link in self.robot.get_links() if link.get_name() == self.palm_link_name][0] # link6

        # Object init pose
        self.object_episode_init_pose = sapien.Pose()

        # Object, palm, target pose
        self.object_pose = self.manipulated_object.get_pose() # this is cube loaded by load_cube
        self.palm_pose = self.palm_link.get_pose()
        self.palm_pos_in_base = np.zeros(3)
        self.target_in_object = np.zeros([3])
        self.target_in_object_angle = np.zeros([1])
        self.object_lift = 0
        self.gui = None
        self.save_img = save_img
        self.count = 0
        self.property_params = {
            "object_size": self.object_size,
            "well_size": self.well_size,
        }

    def update_cached_state(self):
        self.object_pose = self.manipulated_object.get_pose()
        self.palm_pose = self.palm_link.get_pose() # link6

    # this is self.get_observation 
    def get_oracle_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        object_pose = self.object_episode_init_pose if self.constant_object_state else self.manipulated_object.get_pose()
        object_pose_vec = np.concatenate([object_pose.p, object_pose.q])
        palm_pose = self.palm_link.get_pose()
        target_in_object = self.target_pose.p - object_pose.p
        target_in_palm = self.target_pose.p - palm_pose.p
        object_in_palm = object_pose.p - palm_pose.p
        palm_v = self.palm_link.get_velocity()
        palm_w = self.palm_link.get_angular_velocity()
        theta = np.arccos(np.clip(np.power(np.sum(object_pose.q * self.target_pose.q), 2) * 2 - 1, -1 + 1e-8, 1 - 1e-8))
        return np.concatenate(
            [robot_qpos_vec, object_pose_vec, palm_v, palm_w, object_in_palm, target_in_palm, target_in_object,
             self.target_pose.q, np.array([theta])])

    def get_robot_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        palm_pose = self.palm_link.get_pose()
        return np.concatenate([robot_qpos_vec, palm_pose.p, self.target_pose.p, self.target_pose.q])

    def get_reward(self, action):
        object_pose = self.manipulated_object.get_pose()
        return 0
        target_pose = self.target.get_pose()
        distance =np.linalg.norm(target_pose.p[:2] - object_pose.p[:2])
        if distance < 0.05:
            reward = 1
        else:
            reward = 0
        return reward
    
    def save_data(self, dir, step=False):
        obj_pos, obj_orn = self.manipulated_object.get_pose().p, self.manipulated_object.get_pose().q
        obj_state = np.concatenate([obj_pos, obj_orn])
        wall_pos, wall_orn = self.well.get_pose().p, self.well.get_pose().q
        wall_state = np.concatenate([wall_pos, wall_orn])
        # palm_pos, palm_orn = self.palm_link.get_pose().p, self.palm_link.get_pose().q
        # eef_state = np.concatenate([palm_pos, palm_orn])
        eef_state = np.array([np.concatenate((link.get_pose().p, link.get_pose().q)) for link in self.robot.get_links()])
        
        self.obj_state_list.append(obj_state)
        self.wall_state_list.append(wall_state)
        self.eef_list.append(eef_state)
        views = self.gui.render()
        if self.save_img:
            if not os.path.exists(f"{dir}/imgs"):
                os.makedirs(f"{dir}/imgs")
            cv2.imwrite(f"{dir}/imgs/img_{self.count}.png", views[0])
        self.count += 1
        if step:
            self.step_list.append(self.count)


    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        qpos = np.zeros(self.robot.dof)
        arm_qpos = self.robot_info.arm_init_qpos
        qpos[:self.arm_dof] = arm_qpos
        self.robot.set_qpos(qpos)
        self.robot.set_drive_target(qpos)
        init_pos = ARM_INIT + self.robot_info.root_offset
        init_pose = sapien.Pose(init_pos, transforms3d.euler.euler2quat(0, 0, 0))

        self.robot.set_pose(init_pose)
        self.reset_internal()
        self.object_episode_init_pose = self.manipulated_object.get_pose()
        random_quat = transforms3d.euler.euler2quat(*(self.np_random.randn(3) * self.object_pose_noise * 10))
        random_pos = self.np_random.randn(3) * self.object_pose_noise
        self.object_episode_init_pose = self.object_episode_init_pose * sapien.Pose(random_pos, random_quat)
        self.update_cached_state()
        self.eef_list = []
        self.obj_state_list = []
        self.wall_state_list = []
        self.step_list = []
        return self.get_observation()

    @cached_property
    def obs_dim(self):
        if not self.use_visual_obs:
            return len(self.get_oracle_state())
        else:
            return len(self.get_robot_state())

    def is_done(self):
        return False
        return self.manipulated_object.pose.p[2] - self.object_height < OBJECT_LIFT_LOWER_LIMIT

    @cached_property
    def horizon(self):
        return 250

    def set_init(self, init_states):
        init_pose = sapien.Pose.from_transformation_matrix(init_states[0])
        self.manipulated_object.set_pose(init_pose)
        init_target_pose = sapien.Pose.from_transformation_matrix(init_states[1])
        # self.target.set_pose(init_target_pose)

def main_env():
    env = ReorientateRLEnv(use_gui=True, robot_name="xarm6_with_gripper",
                        frame_skip=10, use_visual_obs=False, use_ray_tracing=False)
    base_env = env
    robot_dof = env.robot.dof
    env.seed(0)
    env.reset()
    viewer = Viewer(base_env.renderer)
    viewer.set_scene(base_env.scene)
    pi = np.pi
    viewer.set_camera_rpy(r=0, p=-pi/4, y=pi/4)
    viewer.set_camera_xyz(x=-0.5, y=0.5, z=0.5)
    base_env.viewer = viewer

    for link in env.robot.get_links():
        print(link.get_name(),link.get_pose())
    for joint in env.robot.get_active_joints():
        print(joint.get_name())

    half_pi = np.pi/2
    action = np.array([0, -0.5*half_pi, -0.5*half_pi, 2*half_pi, -0.5*half_pi, 2*half_pi, 0])
    viewer.toggle_pause(True)
    for i in range(500):
        action[6] = i*0.85/500
        obs, reward, done, _ = env.step(action)
        # print(env.robot.get_qpos())
        env.render()

    while not viewer.closed:
        env.render()


if __name__ == '__main__':
    main_env()
