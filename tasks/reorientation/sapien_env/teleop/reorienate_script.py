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
from sapien_env.rl_env.reorientate_rlenv import ReorientateRLEnv
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.gui.teleop_gui_trossen import GUIBase, DEFAULT_TABLE_TOP_CAMERAS,WRTHCH_USING_CAMERAS
from sapien_env.teleop.teleop_robot import TeleopRobot
from sapien_env.utils.robot_sample import RobotPcSampler
from sapien_env.rl_env.para import ARM_INIT
import copy
import open3d as o3d
import json
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

        new_way_point = False
        if self.trajectory[0]['t'] == self.step_count:
            new_way_point = True
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
        return cartisen_action, new_way_point
    
    def generate_trajectory(self, env: ReorientateRLEnv, ee_link_pose):
        cube_position = env.manipulated_object.get_pose().p 
        grasp_position = cube_position
        grasp_position[0] += np.random.uniform(-env.object_size[0]+0.01, env.object_size[0]-0.01) # += 0.01 # 
        grasp_position[2] -= np.random.uniform(0.03, env.object_size[2]+0.01) # -= 0.03 # 
        print(f"grasp_position:{grasp_position}")
        theta_grip = np.random.uniform(0, 45)
        gripper_pick_quat = Quaternion(ee_link_pose.q[:4]) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=theta_grip)
        ee_link_grasp_orn_mat = transforms3d.euler.quat2mat(gripper_pick_quat.elements)
        
        # ee_link_orn_deg_start = transforms3d.euler.quat2euler(Quaternion(ee_link_pose.q[:4]).elements)
        # ee_link_orn_deg = transforms3d.euler.quat2euler(gripper_pick_quat.elements)
        # degree_between_obj_and_gripper = ee_link_orn_deg[1] + 90
        # obj_orn_deg = transforms3d.euler.quat2euler(env.manipulated_object.get_pose().q)
        # print(f"ee_link_orn_deg start:{np.array(ee_link_orn_deg_start)/np.pi*180}")
        # print(f"ee_link_orn_deg:{np.array(ee_link_orn_deg)/np.pi*180}")
        # print(f"obj_orn_deg:{np.array(obj_orn_deg)/np.pi*180}")

        eef_grasp_position = grasp_position - np.dot(ee_link_grasp_orn_mat, [0, 0, 0.29])
        
        # from exp, the ee_link_pose is kind of like: [ 1.79999922e+02 -4.49999363e+01 -2.49666617e-04] which only rotates around y axis
        theta_reori = theta_grip + np.random.uniform(30, 35)
        gripper_reori_quat = Quaternion(ee_link_pose.q[:4]) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=theta_reori)
        ee_link_reori_orn_mat = transforms3d.euler.quat2mat(gripper_reori_quat.elements)
        target_grasp_position = env.well.get_pose().p + [-env.well_size[0],0,0.1] + [-0.20, 0, 0.0]
        eef_target_position = target_grasp_position - np.dot(ee_link_reori_orn_mat, [0, 0, 0.29])
        # ee_link_euler = transforms3d.euler.quat2euler(ee_link_pose.q,axes='sxyz')
        # gripper_pick_euler = transforms3d.euler.quat2euler(gripper_pick_quat.elements,axes='sxyz')
        # gripper_reori_euler = transforms3d.euler.quat2euler(gripper_reori_quat.elements,axes='sxyz')
        # print(f"ee_link_euler:{np.array(ee_link_euler)/np.pi*180}")
        # print(f"gripper_pick_euler:{np.array(gripper_pick_euler)/np.pi*180}")
        # print(f"gripper_reori_euler:{np.array(gripper_reori_euler)/np.pi*180}")

        open_gripper = 0.0
        # 14 is a magic number
        close_gripper = 0.85 - env.object_size[1]*14
        delta_x1 = np.random.uniform(0.05, 0.1)
        delta_x2 = np.random.uniform(max(delta_x1,0.15), 0.19)
        delta_z = np.random.uniform(-0.1, 0.1)
        self.trajectory = [
            {"t": 0, "xyz": ee_link_pose.p, "quat": ee_link_pose.q, "gripper": open_gripper}, #init    
            # {"t": 200, "xyz": ee_link_pose.p, "quat": ee_link_pose.q, "gripper": 0.021}, #init 
            {"t": 20, "xyz": ee_link_pose.p, "quat": ee_link_pose.q, "gripper": open_gripper}, # open gripper
            {"t": 50, "xyz": eef_grasp_position, "quat": gripper_pick_quat.elements, "gripper": open_gripper}, 
            {"t": 70, "xyz": eef_grasp_position, "quat": gripper_pick_quat.elements, "gripper": open_gripper}, # move to pick
            {"t": 80, "xyz": eef_grasp_position, "quat": gripper_pick_quat.elements, "gripper": close_gripper}, #gripper close
            {"t": 90, "xyz": eef_grasp_position, "quat": gripper_pick_quat.elements, "gripper": close_gripper}, # stop for a while
            {"t": 110, "xyz": eef_grasp_position+[0,0,0.1], "quat": gripper_pick_quat.elements, "gripper": close_gripper}, # move up
            {"t": 160, "xyz": eef_target_position, "quat": gripper_pick_quat.elements, "gripper": close_gripper},# move to target
            {"t": 170, "xyz": eef_target_position, "quat": gripper_pick_quat.elements, "gripper": close_gripper},# stop for a while
            # start reorientate
            {"t": 180, "xyz": eef_target_position, "quat": gripper_reori_quat.elements, "gripper": close_gripper},
            {"t": 200, "xyz": eef_target_position+[delta_x1,0,0], "quat": gripper_reori_quat.elements, "gripper": close_gripper},
            {"t": 300, "xyz": eef_target_position+[delta_x2,0,delta_z], "quat": gripper_reori_quat.elements, "gripper": close_gripper},
            # end reorientate
            {"t": 350, "xyz": eef_target_position + [-delta_x1,0,0.1], "quat": gripper_pick_quat.elements, "gripper": close_gripper}, # open gripper
            # {"t": 250, "xyz": eef_target_position+np.array([-0.2,0,0]), "quat": gripper_pick_quat.elements, "gripper": 0.07}, # go back to init
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


def reorientate_env(info):
    robot_name = 'xarm6_with_gripper'
    teleop = TeleopRobot(robot_name)
    max_timesteps = 350
    num_episodes = 10
    onscreen = True
    headless = True
    episode_idx = info["episode_idx"]
    # for episode_idx in range(num_episodes):

    # trossen_vx300s_tactile_thin, trossen_vx300s_new_gripper
    env = ReorientateRLEnv(use_gui=True, robot_name=robot_name,
                        frame_skip=20, use_visual_obs=False, use_ray_tracing=False,
                        save_img=True if not headless else False,)
    env.seed(episode_idx)
    env.reset()

    # Setup viewer and camera
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer, headless=headless)
    for cam_name, params in DEFAULT_TABLE_TOP_CAMERAS.items():
        gui.create_camera(**params) # this is for cv2.imshow
    if not headless:
        gui.viewer.set_camera_rpy(0, -0.7, 0.01) # this is for viewer (official sapien viewer)
        gui.viewer.set_camera_xyz(-0.4, 0, 0.45)
        gui.viewer.toggle_axes(True)
    scene = env.scene
    steps = 0
    
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
    env.gui = gui
    epi_dir = os.path.join('data/sapien_v2', f"episode_{episode_idx}")
    os.makedirs(epi_dir, exist_ok=True)
    # calculate camera intrinsics and extrinsics
    extrinsic_mat = np.zeros((4,4))
    extrinsic_mat[3,3] = 1
    extrinsic_mat[:3,:3]  = np.matmul(transforms3d.euler.quat2mat(gui.cam_mounts[0].pose.q), \
                                    gui.cams[0].get_extrinsic_matrix()[:3, :3].T,
                            )
    extrinsic_mat[:3,3] = -extrinsic_mat[:3,:3] @ gui.cam_mounts[0].pose.p
    np.savetxt(f"{epi_dir}/camera_intrinsics.txt", gui.cams[0].get_camera_matrix())
    np.savetxt(f"{epi_dir}/camera_extrinsics.txt", extrinsic_mat)
    with open(f"{epi_dir}/property_params.json", 'w') as f:
        json.dump(env.property_params, f)
    
    for i in range(max_timesteps):
        # xyz, orn, gripper
        cartisen_action, new_way_point = scripted_policy.single_trajectory(env,env.palm_link.get_pose()) # link6 is the same as xarm_gripper_base_link
        cartisen_action_in_rob = transform_action_from_world_to_robot(cartisen_action,env.robot.get_pose()) # xyz, euler_orn, gripper
        action[:arm_dof] = teleop.teleop_ik(env.robot.get_qpos()[:],cartisen_action_in_rob) # current pose to cartisen_action_in_rob
        action[arm_dof:] = cartisen_action[6]
        # action[arm_dof:] = i*0.85/max_timesteps
        obs, reward, done, _ = env.step(action[:7])
        env.save_data(epi_dir, new_way_point)
        # if onscreen:
        #     time.sleep(0.01)
        #     views = env.gui.render()
    if epi_dir is not None:
        np.save(os.path.join(epi_dir, 'steps.npy'), env.step_list)
        np.save(os.path.join(epi_dir, 'obj_state.npy'), env.obj_state_list)
        np.save(os.path.join(epi_dir, 'eef_states.npy'), env.eef_list)
        np.save(os.path.join(epi_dir, 'wall_states.npy'), env.wall_state_list)
    if not headless:
        env.gui.viewer.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import multiprocessing as mp
    num_episodes = 10000
    for episode_idx in range(num_episodes):
        print(f"episode_idx:{episode_idx}")
        info = {"episode_idx": episode_idx}
        reorientate_env(info)

    # episode_idx_list = [10010, 10011, 10012, 10013, 10014, 10015, 10016, 10017, 10018, 10019]
    # for episode_idx in episode_idx_list:
    #     print(f"episode_idx:{episode_idx}")
    #     info = {"episode_idx": episode_idx}
    #     reorientate_env(info)

# env.robot.get_qpos()[:6]: it is the joint angle of the robot, in the same frame as teleop.teleop_ik(env.robot.get_qpos()[:],cartisen_action_in_rob)
# cartisen_action_in_rob is the position of xarm_gripper_base_link  
# x: forward, y:left, z:up