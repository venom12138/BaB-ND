from tasks.reorientation.sapien_env.rl_env.reorientate_rlenv import ReorientateRLEnv
from tasks.reorientation.sapien_env.teleop.teleop_robot import TeleopRobot
import torch
import numpy as np
import matplotlib.pyplot as plt
import transforms3d
from pyquaternion import Quaternion
import sapien
class ReorientationSim(ReorientateRLEnv):
    def __init__(self, param_dict, init_poses=None, target_poses=None, pusher_pos=None):
        super().__init__(use_gui=False, robot_name='xarm6_with_gripper',
                        frame_skip=20, use_visual_obs=False, use_ray_tracing=False,
                        save_img=False,)
        # redundent definitions
        self.classes = None
        self.label_list = None
        state_dim = param_dict['state_dim']
        action_dim = param_dict['action_dim']
        self.dim_of_work_space = 2
        assert action_dim == 4 and state_dim == 8
        self.reset()
        obj_orientation= transforms3d.euler.euler2quat(0, 0, 0)
        obj_position = np.array([0.15, 0.0, 0.08])
        obj_pose = sapien.core.Pose(obj_position, obj_orientation)
        self.manipulated_object.set_pose(obj_pose)
        
        self.teleop = TeleopRobot('xarm6_with_gripper')
        self.grasp_object() # first grasp the object
        # action: [2,2] each dim is [-0.1, 0.1]
    
    def grasp_object(self, ):
        cube_position = self.manipulated_object.get_pose().p
        ee_link_pose = self.palm_link.get_pose()
        grasp_position = cube_position
        grasp_position[0] -= 0.02 # np.random.uniform(-self.object_size[0]+0.01, self.object_size[0]-0.01)
        # grasp_position[0] += np.random.uniform(-self.object_size[0]+0.01, self.object_size[0]-0.01) # += 0.01 # 
        # grasp_position[2] -= np.random.uniform(0.03, self.object_size[2]+0.01) # -= 0.03 # 
        grasp_position[2] -= 0.05 # np.random.uniform(0.03, self.object_size[2]+0.01)
        theta_grip = 10 # np.random.uniform(0, 45)
        gripper_pick_quat = Quaternion(ee_link_pose.q[:4]) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=theta_grip)
        ee_link_grasp_orn_mat = transforms3d.euler.quat2mat(gripper_pick_quat.elements)
        print(f"gropper_pick_quat: {transforms3d.euler.quat2euler(gripper_pick_quat.elements, axes='sxyz')}")
        
        eef_grasp_position = grasp_position - np.dot(ee_link_grasp_orn_mat, [0, 0, 0.29])

        open_gripper = 0.0
        # 14 is a magic number
        close_gripper = 0.85 - self.object_size[1]*14
        self.trajectory = [
            {"t": 0, "xyz": ee_link_pose.p, "quat": ee_link_pose.q, "gripper": open_gripper}, #init    
            # {"t": 200, "xyz": ee_link_pose.p, "quat": ee_link_pose.q, "gripper": 0.021}, #init 
            {"t": 20, "xyz": ee_link_pose.p, "quat": ee_link_pose.q, "gripper": open_gripper}, # open gripper
            {"t": 50, "xyz": eef_grasp_position, "quat": gripper_pick_quat.elements, "gripper": open_gripper}, 
            {"t": 70, "xyz": eef_grasp_position, "quat": gripper_pick_quat.elements, "gripper": open_gripper}, # move to pick
            {"t": 80, "xyz": eef_grasp_position, "quat": gripper_pick_quat.elements, "gripper": close_gripper}, #gripper close
            {"t": 90, "xyz": eef_grasp_position, "quat": gripper_pick_quat.elements, "gripper": close_gripper}, # stop for a while
            {"t": 110, "xyz": eef_grasp_position+[0,0,0.1], "quat": gripper_pick_quat.elements, "gripper": close_gripper}, # move up
        ]
        
        action = np.zeros(7)
        for t in range(110):
            if self.trajectory[0]['t'] == t:
                curr_waypoint = self.trajectory.pop(0)
            next_waypoint = self.trajectory[0]

            # interpolate between waypoints to obtain current pose and gripper command
            xyz, quat, gripper = self.interpolate(curr_waypoint, next_waypoint, t)

            cartisen_action_dim =6
            grip_dim = 1
            cartisen_action = np.zeros(cartisen_action_dim+grip_dim)
            eluer = transforms3d.euler.quat2euler(quat,axes='sxyz')
            cartisen_action[0:3] = xyz
            cartisen_action[3:6] = eluer
            cartisen_action[6] = gripper
            cartisen_action_in_rob = self.transform_action_from_world_to_robot(cartisen_action, self.robot.get_pose()) # xyz, euler_orn, gripper
            action[:self.arm_dof] = self.teleop.teleop_ik(self.robot.get_qpos()[:],cartisen_action_in_rob) # current pose to cartisen_action_in_rob
            action[self.arm_dof:] = cartisen_action[6]
            # action[arm_dof:] = i*0.85/max_timesteps
            obs, reward, done, _ = self.step(action[:7])
    
    # target pose is a relative pose
    # the obj is assumed to be horizontal lying on the table
    def get_target_pose(self, ):
        x = -0.01 # np.random.uniform(-self.object_size[0], self.object_size[0])
        y = 0 # np.random.uniform(-self.object_size[2], self.object_size[2])
        theta = 1.57 # 2.52 # np.random.uniform(0, 2*np.pi)
        return np.array([x, y, theta])
    
    def get_fixed_action_sequence(self,):
        raise NotImplementedError
    
    def update(self, u):
        # u: [delta_x, delta_y, delta_theta(rad)] finger tip's position
        ee_pos, ee_orn = self.palm_link.get_pose().p, self.palm_link.get_pose().q
        ee_orn_mat = transforms3d.euler.quat2mat(ee_orn)
        lst_grasp_position = ee_pos + np.dot(ee_orn_mat, [0, 0, 0.29])
        new_grasp_position = lst_grasp_position[0]+u[0], lst_grasp_position[1], \
                                lst_grasp_position[2]+u[1]
        delta_theta_deg = u[2] / np.pi * 180
        new_ee_orn = Quaternion(ee_orn[:4]) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=delta_theta_deg)
        new_ee_pos = new_grasp_position - np.dot(new_ee_orn.rotation_matrix, [0, 0, 0.29])
        cur_waypoint = {'t': 0, 'xyz': ee_pos, 'quat': Quaternion(ee_orn), 'gripper': 0.85 - self.object_size[1]*14}
        new_waypoint = {'t': 8, 'xyz': new_ee_pos, 'quat': new_ee_orn, 'gripper': 0.85 - self.object_size[1]*14}
        for t in range(8):
            xyz, quat, gripper = self.interpolate(cur_waypoint, new_waypoint, t)
            cartisen_action_dim =6
            grip_dim = 1
            cartisen_action = np.zeros(cartisen_action_dim+grip_dim)
            eluer = transforms3d.euler.quat2euler(quat, axes='sxyz')
            cartisen_action[0:3] = xyz
            cartisen_action[3:6] = eluer
            cartisen_action[6] = gripper
            cartisen_action_in_rob = self.transform_action_from_world_to_robot(cartisen_action, self.robot.get_pose()) # xyz, euler_orn, gripper
            action = np.zeros(7)
            action[:self.arm_dof] = self.teleop.teleop_ik(self.robot.get_qpos()[:], cartisen_action_in_rob) # current pose to cartisen_action_in_rob
            action[self.arm_dof:] = cartisen_action[6]
            # action[arm_dof:] = i*0.85/max_timesteps
            obs, reward, done, _ = self.step(action[:7])
            
    # coordinate is tranformed to x, y, z
    def get_current_state(self,):
        obj_pos, obj_orn = self.manipulated_object.get_pose().p, self.manipulated_object.get_pose().q
        obj_orn_mat = transforms3d.euler.quat2mat(obj_orn)
        
        obj_points = np.zeros((4,2)) # (x, z)
        obj_points[0] = obj_pos[[0,2]] + np.dot(obj_orn_mat, self.object_size*np.array([1,1,1]))[[0,2]]
        obj_points[1] = obj_pos[[0,2]] + np.dot(obj_orn_mat, self.object_size*np.array([-1,1,1]))[[0,2]]
        obj_points[2] = obj_pos[[0,2]] + np.dot(obj_orn_mat, self.object_size*np.array([-1,1,-1]))[[0,2]]
        obj_points[3] = obj_pos[[0,2]] + np.dot(obj_orn_mat, self.object_size*np.array([1,1,-1]))[[0,2]]
        
        eef_state = np.array([np.concatenate((link.get_pose().p, link.get_pose().q)) for link in self.robot.get_links()])[8]
        eef_pos, eef_orn = eef_state[:3], eef_state[3:]
        eef_orn_mat = transforms3d.euler.quat2mat(eef_orn)
        tool_points = np.zeros((2, 2)) # (x, z)
        tool_points[0] = (eef_pos + np.dot(eef_orn_mat, [0,0,0.29]))[[0,2]]
        tool_points[1] = (eef_pos + np.dot(eef_orn_mat, [0,0,0.25]))[[0,2]]
        obj_points = obj_points - np.array([[0.25, 0.]])
        tool_points = tool_points - np.array([[0.25, 0.]])
        return np.concatenate([obj_points, tool_points], axis=0) # 6,2 
    
    # coordinate is tranformed to x, y, z
    def get_pusher_start_position(self,):
        raise NotImplementedError
        # return np.mean(self.eef_states_list[0], axis=0)[[0,2,1]] * np.array([1,-1,1]) # the last state is reset
    
    # coordinate is tranformed to x, y, z
    def get_pusher_position(self,):
        # end_effector_position is different from eef_states_list[-2](this is finger_position+0.5)
        eef_state = np.array([np.concatenate((link.get_pose().p, link.get_pose().q)) for link in self.robot.get_links()])[8]
        eef_pos, eef_orn = eef_state[:3], eef_state[3:]
        eef_orn_mat = transforms3d.euler.quat2mat(eef_orn)
        tool_points = np.zeros((2, 2)) # (x, z)
        tool_points[0] = (eef_pos + np.dot(eef_orn_mat, [0,0,0.29]))[[0,2]]
        tool_points[1] = (eef_pos + np.dot(eef_orn_mat, [0,0,0.25]))[[0,2]]
        tool_points = tool_points - np.array([[0.25, 0.]])
        return tool_points # 2,2
    
    # this is the end effector position used for self.update
    def get_end_effector_position(self,):
        raise NotImplementedError
        # return np.mean(self.eef_states_list[-2], axis=0)[[0,2,1]] * np.array([1,-1,1]) + np.array([0,0,1.616-0.5]) # the last state is reset
    
    def transform_action_from_world_to_robot(self, action : np.ndarray, pose):
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
    