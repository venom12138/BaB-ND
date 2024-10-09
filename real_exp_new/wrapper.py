import numpy as np
import pygame
import cv2
import pymunk
import multiprocessing as mp
from threading import Thread
from queue import Empty
import torch
import os
import open3d as o3d
import pickle
import time

FLOAT_TYPE = np.float16
file_dir = os.path.dirname(os.path.abspath(__file__))

class RealWrapper:
    def __init__(self, serial_numbers, cad_path_list, object_color_list, keypoint_offsets_2d_list):
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
        workspace2world[:3, 3] = np.array([-0.05, 0.2, 0])
        self.world2base = world2base

        self.workspace2world = workspace2world
        self.world2workspace = np.linalg.inv(self.workspace2world)
        workspace2world_offset_xy = self.workspace2world[:2, 3]
        self.xyz_bounds = np.array([[0 + workspace2world_offset_xy[0], 0.5 + workspace2world_offset_xy[0]], 
                                    [0 + workspace2world_offset_xy[1], 0.5 + workspace2world_offset_xy[1]], 
                                    [0.01, 0.05]])
        
        self.workspace2base = world2base @ workspace2world
        self.base2workspace = np.linalg.inv(self.workspace2base)
        self.init_pose = [196.2, -1.6, 434, 179.2, 0, 0.3]
        x, y, z, roll, pitch, yaw = self.init_pose

        # self.xyz = np.array([x, y, z]) / 1000
        self.rpy = np.array([roll, pitch, yaw])
        #  180.0 * np.pi

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

        # mesh part
        mesh_pcd_list, rotation_matrices_list = [], []
        assert len(cad_path_list) == len(object_color_list)
        self.num_objects = len(cad_path_list)
        self.cad_path_list = cad_path_list
        self.object_color_list = np.array(object_color_list, dtype=FLOAT_TYPE)

        self.split_num = 36
        self.object_height = 0.01

        for cad_path in cad_path_list:
            mesh = o3d.io.read_triangle_mesh(cad_path)
            # mesh.rotate(flipping_matrix, center=(0, 0, 0))
            mesh.compute_vertex_normals()
            mesh_pcd = mesh.sample_points_uniformly(number_of_points=500)
            cad_points = np.asarray(mesh_pcd.points)
            # cad_points = cad_points[cad_points[:, 2] > 0.0095]
            cad_points[:, 2] = self.object_height
            mesh_pcd = o3d.geometry.PointCloud()
            mesh_pcd.points = o3d.utility.Vector3dVector(cad_points)
            rotation_matrices = [
                mesh.get_rotation_matrix_from_xyz((0, 0, np.pi * 2 / self.split_num * i)).astype(FLOAT_TYPE)
                for i in range(self.split_num)
            ]
            mesh_pcd_list.append(mesh_pcd)
            rotation_matrices_list.append(rotation_matrices)
        self.mesh_pcd_list = mesh_pcd_list
        self.rotation_matrices_list = rotation_matrices_list # a list of rotation matrices
        self.keypoint_offsets_2d_list = keypoint_offsets_2d_list
        self.num_keypoint_list = [keypoint_offsets_2d.shape[0] for keypoint_offsets_2d in keypoint_offsets_2d_list]
        # for i in range(self.num_objects):
        #     self.keypoint_offsets_2d_list[i] = self.keypoint_offsets_2d_list[i].astype(FLOAT_TYPE)

        self.state_dim = 2 * (sum(self.num_keypoint_list))
        self.action_dim = 2
        self.env_state = np.zeros(self.state_dim + self.action_dim, dtype=FLOAT_TYPE)
        self.alive = True

        # self.update_obj_state_count = 0
        # self.update_pusher_pos_count = 0
        self.update_obj_state_t = Thread(target=self.update_obj_state, name="update_obj_state")
        self.update_pusher_pos_t = Thread(target=self.update_pusher_pos, name="update_pusher_pos")


    def get_perception_param(self):
        return {"R_list": self.R_list, "T_list": self.T_list, "mesh_pcd_list": self.mesh_pcd_list,
                "rotation_matrices_list": self.rotation_matrices_list, "object_color_list": self.object_color_list,
                "split_num": self.split_num, "xyz_bounds": self.xyz_bounds, "num_objects": self.num_objects,
                "workspace2world": self.workspace2world
                }

    def reset(self, perception_q, pusher_pos_q, xarm_command_q, xarm_exe_lock):
        self.perception_q = perception_q
        self.pusher_pos_q = pusher_pos_q
        self.xarm_command_q = xarm_command_q
        self.xarm_exe_lock = xarm_exe_lock

    def start(self):
        self.update_obj_state_t.start()
        self.update_pusher_pos_t.start()

    def stop(self):
        self.alive = False
        self.update_obj_state_t.join()
        self.update_pusher_pos_t.join()

    def get_env_state(self):
        return self.env_state

    def get_pusher_position(self):
        return self.env_state[-self.action_dim:]

    def update(self, x, y, z=115, wait=True):
        next_position_in_workspace = np.array([x, y, z, 1000]) / 1000
        next_position_in_base = (self.workspace2base @ next_position_in_workspace)[:3]
        next_position_in_base[2] = max(0.118, next_position_in_base[2])
        xarm_cmd = np.concatenate([next_position_in_base*1000, self.rpy])

        while not self.xarm_command_q.empty():
            time.sleep(0.01)
            # print("xarm_command_q not empty, wait")
        with self.xarm_exe_lock:
            # print(f"put command: {action}-->{xarm_cmd[:3]}")
            self.xarm_command_q.put([xarm_cmd, wait])

    def update_obj_state(self):
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        
        # # Set camera view
        # ctr = vis.get_view_control()
        # ctr.set_lookat([0, 0, 0])
        # ctr.set_front([0, -1, 0])
        # ctr.set_up([0, 0, -1])

        while self.alive:
            if not self.perception_q.empty():
                try:
                    # vis.clear_geometries()
                    # draw_list = []
                    transformation_list = self.perception_q.get(block=False)
                    keypoints_in_workspace_list = []
                    for i in range(self.num_objects):
                        object2world = transformation_list[i]
                        object2workspace = self.world2workspace @ object2world
                        keypoint_offsets_2d = np.array(self.keypoint_offsets_2d_list[i])/1000
                        num_keypoints = keypoint_offsets_2d.shape[0]
                        keypoints_homo = np.concatenate(
                            [keypoint_offsets_2d, np.tile([[self.object_height, 1]], (num_keypoints, 1))], axis=1
                        )
                        keypoints_in_workspace = (object2workspace @ keypoints_homo.T).T[:, :2]
                        keypoints_in_workspace_list.append(keypoints_in_workspace)
                        # draw_list.append(mesh_pcd_list[i])
                        # draw_list.append(object_pcd_list[i])
                    self.env_state[:self.state_dim] = np.array(keypoints_in_workspace_list).flatten() * 1000
                    # self.update_obj_state_count += 1
                    # print("update_obj_state_count", self.update_obj_state_count)
                    # vis.add_geometry(coordinate)
                    # for draw in draw_list:
                    #     vis.add_geometry(draw)
                    # vis.poll_events()
                    # vis.update_renderer()
                except Empty:
                    pass
            time.sleep(0.01)
        
        # vis.close()
    
    def update_pusher_pos(self):
        while self.alive:
            if not self.pusher_pos_q.empty():
                try:
                    pusher2base = self.pusher_pos_q.get(block=False)
                    pusher2workspace = self.base2workspace @ pusher2base
                    local_pose = np.array([[0,0,0,1]]).T
                    raw_pose_in_workspace = pusher2workspace @ local_pose
                    self.env_state[-self.action_dim:] = raw_pose_in_workspace[:self.action_dim, 0] * 1000
                    # self.update_pusher_pos_count += 1
                    # print("update_pusher_pos_count", self.update_pusher_pos_count)
                except Empty:
                    pass
            time.sleep(0.01)