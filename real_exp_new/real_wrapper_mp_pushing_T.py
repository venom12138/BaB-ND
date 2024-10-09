import sys
import os
from xarm import version
from xarm.wrapper import XArmAPI
import ctypes
import matplotlib.pyplot as plt
import transforms3d
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
from real_utils import rpy_to_rotation_matrix
from common.kinematics_utils import KinHelper
# from arm_controller import ArmController

FLOAT_TYPE = np.float16
file_dir = os.path.dirname(os.path.abspath(__file__))

class RealWrapper:
    def __init__(
        self, param_dict: dict, serial_numbers, visible_areas, cad_path_list, object_color_list, target_state=None
    ):
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
        if param_dict.get("reset", False):
            self.set_robot_pose(np.concatenate([self.xyz, self.rpy]), steps=1000, wait=False, sleep_time=0.005)

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
            visible_areas=visible_areas,
        )
        self.realsense.set_exposure(exposure=100, gain=60) # 200
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
        workspace2world[:3, 3] = np.array([-0.05, 0.2, 0])
        self.world2base = world2base

        self.workspace2world = workspace2world
        self.workspace2base = world2base @ workspace2world
        # self.workspace2base = np.array([[-0.00463999, -0.97254255 , 0.23267886 , 0.60582383],
        #                                 [ 0.99970695, -0.01003963 ,-0.02202753 ,-0.17207565],
        #                                 [ 0.02375872,  0.23250847 , 0.97230414 ,-0.07352659],
        #                                 [ 0.      ,    0.    ,      0.    ,      1.        ]])
        # positions_in_workspace = np.array([[0, 0, 0, 1],
        #                                     [0.4, 0, 0, 1],
        #                                     [0.4, 0.4, 0, 1],
        #                                     [0, 0.4, 0, 1],
        #                                     [0.2, 0.2, 0, 1],
        #                                     [0.05, 0.05, 0, 1]])

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

        # mesh part
        mesh_pcd_list, rotation_matrices_list = [], []
        assert len(cad_path_list) == len(object_color_list)
        self.num_objects = len(cad_path_list)
        self.cad_path_list = cad_path_list
        self.object_color_list = np.array(object_color_list, dtype=FLOAT_TYPE)
        # target state in workspace coordinate, in mm, only for visualization
        self.target_state = target_state
        self.split_num = 120

        for cad_path in cad_path_list:
            mesh = o3d.io.read_triangle_mesh(cad_path)
            # mesh.rotate(flipping_matrix, center=(0, 0, 0))
            mesh.compute_vertex_normals()
            mesh_pcd = mesh.sample_points_uniformly(number_of_points=500)
            cad_points = np.asarray(mesh_pcd.points)
            # cad_points = cad_points[cad_points[:, 2] > 0.0095]
            cad_points[:, 2] = 0.01
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
        self.object_height = 0.01
        self.alive = False

        # self.pose_queue = Queue()
        # self.pose_thread = Thread(target=self.process_poses)
        # self.pose_thread.daemon = True
        
        # self.arm = ArmController(self.shm_manager)
        # initialize object pcd from last frame
        self.last_mesh_transform = None


    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.realsense.is_ready

    def update_waypoint(self, waypoint):
        self.curr_waypoint = waypoint

    def start_realsense(self, wait=True, exposure_time=5):
        self.realsense.start(wait=False, put_start_time=time.time() + exposure_time)
        if wait:
            self.start_wait()
        self.alive = True

    def stop_realsense(self, wait=False):
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()
        self.alive = False

    # def start_pose_thread(self):
    #     self.pose_thread.start()
    
    # def stop_pose_thread(self):
    #     self.pose_thread.join()

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

    def get_state(
        self,
        keypoint_offsets_2d_list,
        image_path=None,
        visualize=False,
    ):
        # keypoint_offsets_2d_list: in mm
        self.last_realsense_data = self.realsense.get(out=self.last_realsense_data)
        # change to hsv

        for i in self.camera_indices:
            # cv2.imwrite(f"rgb_image_{i}.png", self.last_realsense_data[i]["color"])
            # print(f"self.last_realsense_data[i]['color'].shape: {self.last_realsense_data[i]['color']}")
            self.last_realsense_data[i]["color"] = cv2.cvtColor(self.last_realsense_data[i]["color"], cv2.COLOR_BGR2RGB) # cv2.cvtColor(self.last_realsense_data[i]["color"], cv2.COLOR_RGB2HSV)
            # self.last_realsense_data[i]["color"] = cv2.cvtColor(self.last_realsense_data[i]["color"], cv2.COLOR_RGB2HSV)
            # self.last_realsense_data[i]["color"] = self.last_realsense_data[i]["color"]

        pusher_position = self.get_pusher_position()
        print(f"last_realsense_data: {[float(self.last_realsense_data[0]['timestamp']) for i in range(len(self.last_realsense_data))]}")
        color_list = [self.last_realsense_data[i]["color"] for i in self.camera_indices]
        points_list = [self.last_realsense_data[i]["points"] for i in self.camera_indices]
        align_transformation_list, aligned_mesh_pcd_list, aligned_object_pcd_list = self.process(
            color_list, points_list, visualize=visualize
        )
        keypoint_offsets_2d_list = np.array(keypoint_offsets_2d_list)/1000 # pixel to m
        keypoints_in_workspace_list = []
        for i in range(self.num_objects):
            aligned_mesh_pcd = aligned_mesh_pcd_list[i]
            aligned_pcd = aligned_object_pcd_list[i]
            align_transformation = align_transformation_list[i]
            keypoint_offsets_2d = keypoint_offsets_2d_list[i]
            num_keypoints = keypoint_offsets_2d.shape[0]
            keypoints_homo = np.concatenate(
                [keypoint_offsets_2d, np.tile([[self.object_height, 1]], (num_keypoints, 1))], axis=1
            )
            keypoints_in_workspace = (np.linalg.inv(self.workspace2world) @ align_transformation @ keypoints_homo.T).T[
                :, :2
            ] # 
            keypoints_in_workspace_list.append(keypoints_in_workspace)

        state = np.array(keypoints_in_workspace_list).flatten()
        state = np.concatenate([state, pusher_position[:2]])
        # return in mm
        return state * 1000 # 1mm is 1 pixel,

    def get_pusher_position(self):
        # position in workspace in mm
        raw_pose_in_base = np.array(self.get_robot_pose())
        raw_pose_in_workspace = np.linalg.inv(self.workspace2base) @ np.concatenate([raw_pose_in_base[:3] / 1000, [1]])
        return raw_pose_in_workspace[:3]

    def get_robot_pose(self):
        # code, raw_pose_in_base = self.arm.get_position()
        # use qpos to compute the pose
        cur_qpos = np.array(self.arm.get_servo_angle()[1][0:6]) / 180. * np.pi
        fk_trans_mat = self.kin_helper.compute_fk_sapien_links(cur_qpos, [self.kin_helper.sapien_eef_idx])[0]
        cur_position = np.zeros(6)
        cur_position[:3] = fk_trans_mat[:3,3]*1000
        cur_position[3:] = transforms3d.euler.mat2euler(fk_trans_mat[:3,:3],)
        cur_position[3:] = cur_position[3:] / np.pi * 180
        raw_pose_in_base = np.array(cur_position)
        return raw_pose_in_base
    # z= 175
    def init_pusher_pos(self, x, y, z=118):
        # x, y, z in mm (pixel in sim)
        self.update(x, y, z + 100, wait=True)
        self.update(x, y, z, wait=True)
        return

    # perform action given pusher position
    def update(self, x, y, z=118, wait=False):
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



    # def set_robot_pose(self, pose, steps=1000, wait=False, sleep_time=0.02):
    #     initial_qpos = np.array(self.xarm.get_servo_angle()[1][0:6]) / 180.0 * np.pi
    #     next_servo_angle = self.kin_helper.compute_ik_sapien(initial_qpos, pose)
    #     angles = np.linspace(initial_qpos, next_servo_angle, steps)
        
    #     # Queue the new pose steps
    #     for angle in angles:
    #         self.arm.set_servo_angle_j(angles=angle, is_radian=True)
    #         time.sleep(0.02)
    #     self.arm.clean_error()
    #     self.arm.clean_warn()

    #     return
    
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

    # def set_robot_pose(self, pose, steps=1000, wait=False, sleep_time=0.02):
    #     initial_qpos = np.array(self.arm.get_servo_angle()[1][0:6]) / 180.0 * np.pi
    #     next_servo_angle = self.kin_helper.compute_ik_sapien(initial_qpos, pose)
    #     angles = np.linspace(initial_qpos, next_servo_angle, steps)
        
    #     print(f"final angles: {angles[0]} {angles[-1]}")
    #     # self.arm_controller.clear()
    #     # Queue the new pose steps
    #     for angle in angles:
    #         self.arm.put((angle, sleep_time))
        
        # act all the actions before next step
        # if wait:
        #     while not self.arm.empty():
        #         time.sleep(0.1)
        # return

    
    
    # ANONYMOUS's raw code
    # def set_robot_pose(self, pose, steps=1000, wait=False, sleep_time=0.02):
    #     initial_qpos = np.array(self.arm.get_servo_angle()[1][0:6]) / 180.0 * np.pi
    #     next_servo_angle = self.kin_helper.compute_ik_sapien(initial_qpos, pose)
    #     angles = np.linspace(initial_qpos, next_servo_angle, steps)
        
    #     # Clear the queue if we need to prioritize this new pose
    #     try:
    #         while True:
    #             self.pose_queue.get_nowait()
    #     except Empty:
    #         pass
    
    #     with self.pose_queue.mutex:
    #         self.pose_queue.queue.clear()
    #     # Queue the new pose steps
    #     for angle in angles:
    #         self.pose_queue.put((angle, sleep_time))
        
    #     # act all the actions before next step
    #     if wait:
    #         while not self.pose_queue.empty():
    #             time.sleep(0.1)
    #     return

    # ANONYMOUS's raw code
    # def process_poses(self, pose_queue):
    #     while True: # self.alive:
    #         if self.alive:
    #             print("pose thread alive")
    #             try:
    #                 angles, sleep_time = pose_queue.get_nowait()
    #                 self.arm.set_servo_angle_j(angles=angles, is_radian=True)
    #                 time.sleep(sleep_time)
    #             except Empty:
    #                 continue  # Wait for new poses
    #         else:
    #             print("pose thread not alive")
    #             time.sleep(5)


    def reset_robot(self):
        self.arm.reset(wait=True)

    def process(self, color_list, points_list, visualize=False):
        verbose = False
        # visualize = True
        FLOAT_TYPE = np.float16
        n_cameras = len(color_list)
        if verbose or visualize:
            for i in range(n_cameras):
                cv2.imwrite(f"{file_dir}/test{i}.png", color_list[i])
        # process_start_time = time.time()
        object_point_dict = {i: [] for i in range(self.num_objects)}
        object_color_dict = {i: [] for i in range(self.num_objects)}

        valid_points_list = []
        valid_colors_list = []
        R_list = self.R_list
        T_list = self.T_list
        mesh_pcd_list = self.mesh_pcd_list
        rotation_matrices_list = self.rotation_matrices_list
        split_num = self.split_num
        aligned_mesh_pcd_list = []
        aligned_object_pcd_list = []
        align_transformation_list = []
        if visualize:
            visualize_list = []
        for i in range(n_cameras):
            colors = (color_list[i] / 255).astype(FLOAT_TYPE).reshape(-1, 3)
            points = (points_list[i]).astype(FLOAT_TYPE).reshape(-1, 3)
            valid_points = points
            valid_colors = colors
            # if verbose:
            #     print(f"pre mask time: {time.time() - process_start_time}")
            


            # visualize each camera point cloud
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(valid_points)
            # pcd.colors = o3d.utility.Vector3dVector(valid_colors)
            # o3d.visualization.draw_geometries([pcd])


            # post_start_time = time.time()
            valid_points = valid_points @ R_list[i] + T_list[i]
            # if verbose:
            #     print(f"transform time: {time.time() - post_start_time}")

            # TODO: update mask to fit workspace
            workspace2world_offset_xy = self.workspace2world[:2, 3] # translation between workspace and world
            mask = (
                (valid_points[:, 0] > 0 + workspace2world_offset_xy[0])
                & (valid_points[:, 0] < 0.5 + workspace2world_offset_xy[0])
                & (valid_points[:, 1] > 0 + workspace2world_offset_xy[1])
                & (valid_points[:, 1] < 0.5 + workspace2world_offset_xy[1])
                & (valid_points[:, 2] > 0.01)
                & (valid_points[:, 2] < 0.05)
            )
            valid_points = valid_points[mask]
            valid_colors = valid_colors[mask]
            valid_points_list.append(valid_points)
            valid_colors_list.append(valid_colors)

            # object_start_time = time.time()
            for j in range(self.num_objects):
                object_color = self.object_color_list[j]
                
                
                
                # visualize each camera point cloud and object point cloud
                # before color filtering
                # print(f"camera {i}")
                # visualize_list = []
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(valid_points)
                # pcd.colors = o3d.utility.Vector3dVector(valid_colors)
                # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
                # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
                # coordinate.transform(self.workspace2world)
                # visualize_list.append(pcd)
                # visualize_list.append(coordinate)
                # o3d.visualization.draw_geometries(visualize_list)


                color_mask = np.linalg.norm(valid_colors - object_color, axis=1, ord=2) < 0.35 # 0.5
                # import pdb; pdb.set_trace()
                object_points = valid_points[color_mask]

                object_colors = valid_colors[color_mask]
                
                
                
                # # visualize each camera point cloud and object point cloud
                # print(f"camera {i}")
                # visualize_list = []
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(object_points)
                # pcd.colors = o3d.utility.Vector3dVector(object_colors)
                # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
                # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
                # coordinate.transform(self.workspace2world)
                # visualize_list.append(pcd)
                # visualize_list.append(coordinate)
                # o3d.visualization.draw_geometries(visualize_list)



                object_point_dict[j].append(object_points)
                object_color_dict[j].append(object_colors)

        valid_points = np.concatenate(valid_points_list, axis=0)
        valid_colors = np.concatenate(valid_colors_list, axis=0)
        
        object_point_dict = {i: np.concatenate(object_point_dict[i], axis=0) for i in range(self.num_objects)}
        object_color_dict = {i: np.concatenate(object_color_dict[i], axis=0) for i in range(self.num_objects)}
        assert len(object_points) > 0



        # print('visualize the combined pointcloud')
        # visualize_list = []
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(object_point_dict[i])
        # pcd.colors = o3d.utility.Vector3dVector(object_color_dict)
        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        # coordinate.transform(self.workspace2world)
        # visualize_list.append(pcd)
        # visualize_list.append(coordinate)
        # o3d.visualization.draw_geometries(visualize_list)



        for j in range(self.num_objects):
            if verbose:
                print(f"process object {j}")
            object_points = object_point_dict[j]
            object_colors = object_color_dict[j]
            mesh_pcd = mesh_pcd_list[j]
            rotation_matrices = rotation_matrices_list[j]
            object_points[:, 2] = np.max(object_points[:, 2])
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(object_points)
            object_pcd.colors = o3d.utility.Vector3dVector(object_colors)
            


            # for debug outliers
            # def remove_outliers_and_visualize(nb, std_ratio):
            #     cl, ind = object_pcd.remove_statistical_outlier(nb_neighbors=nb,
            #                                             std_ratio=std_ratio)
            #     object_pcd_for_visualize = object_pcd.select_by_index(ind)
            #     coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            #     coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            #     coordinate.transform(self.workspace2world)
            #     o3d.visualization.draw_geometries([coordinate, object_pcd_for_visualize])

            # while True:
            #     import pdb; pdb.set_trace()
            #     # remove_outliers_and_visualize()
            #     input_key = input("Press x to break...")
            #     if input_key == 'x':
            #         break
            
            
            
            # filter out the outliers
            cl, ind = object_pcd.remove_statistical_outlier(nb_neighbors=100,
                                                    std_ratio=0.8)
            object_pcd = object_pcd.select_by_index(ind)          
            
            mesh_center = mesh_pcd.get_center().astype(FLOAT_TYPE)
            object_center = object_pcd.get_center().astype(FLOAT_TYPE)
            
            mesh2world = np.eye(4, dtype=FLOAT_TYPE)
            mesh2world[:3, 3] = object_center - mesh_center
            center_mesh = copy.deepcopy(mesh_pcd) # translate mesh to object pcd position
            center_mesh.transform(mesh2world)
            
            # if self.last_mesh_transform is None:
            min_dist = np.inf
            best_result = None
            for i in range(split_num):
                copy_mesh = copy.deepcopy(center_mesh)
                rotate_matrix = rotation_matrices[i]
                rotate_center = copy_mesh.get_center().astype(FLOAT_TYPE)
                local_transformation = np.eye(4, dtype=FLOAT_TYPE)
                # rotate around the obj center
                # R(x-c) + c
                local_transformation[:3, :3] = rotate_matrix
                local_transformation[:3, 3] = rotate_center - np.dot(rotate_matrix, rotate_center)
                copy_mesh.transform(local_transformation)
                dists = copy_mesh.compute_point_cloud_distance(object_pcd)
                if np.mean(dists) < min_dist:
                    min_dist = np.mean(dists)
                    best_result = local_transformation
            # if verbose:
            #     print(f"rotation time: {time.time() - rotation_time}")
            mesh2world = np.dot(best_result, mesh2world)
            center_mesh.transform(best_result)

            # else:
            #     mesh2world = np.dot(self.last_mesh_transform, mesh2world)
            #     center_mesh.transform(self.last_mesh_transform)
            #     # local transformation around -30 deg to 30 deg
            #     local_deg_range = 45
            #     local_rotation_matrices = [
            #         o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.pi * 2 / 360 * i)).astype(FLOAT_TYPE)
            #         for i in range(-local_deg_range, local_deg_range)
            #     ]
            #     min_dist = np.inf
            #     best_result = None
            #     for i in range(2*local_deg_range):
            #         copy_mesh = copy.deepcopy(center_mesh)
            #         rotate_matrix = local_rotation_matrices[i]
            #         rotate_center = copy_mesh.get_center().astype(FLOAT_TYPE)
            #         local_transformation = np.eye(4, dtype=FLOAT_TYPE)
            #         # rotate around the obj center
            #         # R(x-c) + c
            #         local_transformation[:3, :3] = rotate_matrix
            #         local_transformation[:3, 3] = rotate_center - np.dot(rotate_matrix, rotate_center)
            #         copy_mesh.transform(local_transformation)
            #         dists = copy_mesh.compute_point_cloud_distance(object_pcd)
            #         if np.mean(dists) < min_dist:
            #             min_dist = np.mean(dists)
            #             best_result = local_transformation
            #     mesh2world = np.dot(best_result, mesh2world)
            #     center_mesh.transform(best_result)
            #     self.last_mesh_transform = np.dot(best_result, self.last_mesh_transform)
            
            # if verbose:
            #     print(f"distance before icp: {min_dist}")
            # icp_start_time = time.time()
            reg_p2p = o3d.pipelines.registration.registration_icp(
                center_mesh,
                object_pcd,
                0.005,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
            )
            mesh2world = np.dot(reg_p2p.transformation.astype(FLOAT_TYPE), mesh2world)
            center_mesh.transform(reg_p2p.transformation)
            # if self.last_mesh_transform is None:
            #     self.last_mesh_transform = best_result # np.dot(reg_p2p.transformation, best_result)
            # else:
            #     self.last_mesh_transform = self.last_mesh_transform # np.dot(reg_p2p.transformation, self.last_mesh_transform)
            
            
            if visualize:
                center_mesh.paint_uniform_color([1, 0, 0])
                visualize_list.append(center_mesh)
                visualize_list.append(object_pcd)
            align_transformation_list.append(mesh2world)
            aligned_mesh_pcd_list.append(center_mesh)
            aligned_object_pcd_list.append(object_pcd)
        if visualize:
            # import pdb; pdb.set_trace()
            raw_pusher_pos_in_base = np.array(self.get_robot_pose())
            pusher_pos = np.linalg.inv(self.world2base) @ np.concatenate([raw_pusher_pos_in_base[:3] / 1000, [1]])

            pusher_z_points = np.linspace(0, 0.1, 1000)
            pusher_points = np.array([[pusher_pos[0], pusher_pos[1]]])
            pusher_points = np.tile(pusher_points, (1000, 1))
            pusher_points = np.concatenate([pusher_points, pusher_z_points.reshape(-1, 1)], axis=1)
            pusher_colors = np.tile(np.array([[0,0,1]]), (1000, 1))
            pusher_pcd = o3d.geometry.PointCloud()
            pusher_pcd.points = o3d.utility.Vector3dVector(pusher_points)
            pusher_pcd.colors = o3d.utility.Vector3dVector(pusher_colors)
            visualize_list.append(pusher_pcd)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(valid_points) # whole workspace
            pcd.colors = o3d.utility.Vector3dVector(valid_colors)
            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            coordinate.transform(self.workspace2world)
            # visualize_list.append(pcd)
            visualize_list.append(coordinate)
            o3d.visualization.draw_geometries(visualize_list)

        return align_transformation_list, aligned_mesh_pcd_list, aligned_object_pcd_list


def get_visible_areas(serial_numbers):
    visible_areas = []
    for serial_number in serial_numbers:
        if serial_number == "311322303615":
            visible_area = [[0.35, 0.75], [0.15, 0.75]]
        elif serial_number == "215122252880":
            visible_area = [[0.25, 0.65], [0.25, 0.8]]
        elif serial_number == "151422251501":
            visible_area = [[0.35, 0.75], [0.15, 0.7]]
        elif serial_number == "246322303954":
            visible_area = [[0.40, 0.75], [0.2, 0.75]]
        else:
            visible_area = None
        visible_areas.append(visible_area)
    # for serial_number in serial_numbers:
    #     if serial_number == "246322301893":
    #         visible_area = [[0.35, 0.75], [0.15, 0.75]]
    #     elif serial_number == "246322303954":
    #         visible_area = [[0.25, 0.65], [0.25, 0.8]]
    #     elif serial_number == "311322303615":
    #         visible_area = [[0.35, 0.75], [0.15, 0.7]]
    #     elif serial_number == "311322300308":
    #         visible_area = [[0.40, 0.75], [0.2, 0.75]]
    #     else:
    #         visible_area = None
    #     visible_areas.append(visible_area)
    # import pdb; pdb.set_trace()
    visible_areas = np.array(visible_areas)
    assert len(serial_numbers) == visible_areas.shape[0]
    return visible_areas


def main():
    # serial_numbers = ["246322301893", "246322303954", "311322303615", "311322300308"]
    # ["311322303615", "215122252880", "151422251501", "246322303954"]
    serial_numbers = ["311322303615", "215122252880", "151422251501", "213622251153"]
    # visible_areas = get_visible_areas(serial_numbers)
    visible_areas = None
    param_dict = {
        "capture_fps": 15,
        "record_fps": 15,
        "record_time": 10,
        "reset": False,
    }
    cad_path_list = [
        f"{file_dir}/object/t.obj",
    ]
    object_color_list = [np.array([0.80, 0.35, 0.15], dtype=FLOAT_TYPE)]
    real_wrapper = RealWrapper(param_dict, serial_numbers, visible_areas, cad_path_list, object_color_list, target_state=None)
    real_wrapper.start()
    time.sleep(1)
    while True:
        try:
            # for small T
            # keypoint_offsets_2d_list=np.array([[[  0.,  -5.],
            #                                     [ 30.,  -5.],
            #                                     [ 60.,  -5.],
            #                                     [ 30., -70.]]])

            # for big T
            keypoint_offsets_2d_list=np.array([[[  0.,  -15.],
                                                [ 60.,  -15.],
                                                [ 120.,  -15.],
                                                [ 60., -120.]]])

            pusher_pos = real_wrapper.get_pusher_position()
            print(f"pusher_pos:{pusher_pos}\n")
            # for big T
            state = real_wrapper.get_state(
                keypoint_offsets_2d_list=keypoint_offsets_2d_list,
                visualize=False,
            )
            print(f"state:{state}\n")
            
            # plt.figure()
            # plt.xlim(0, 500)
            # plt.ylim(0, 500)
            # plt.axis('equal')
            # plt.scatter(state[0::2], state[1::2], color='g')
            # plt.scatter(state[-2], state[-1], color='r')
            # plt.show()
        except:
            pass
        # input("press enter to continue")
    # start_time = time.time()
    # test_steps = 10
    # pusher_pos = real_wrapper.get_pusher_position() *1000
    # for i in range(test_steps):
    #     # 0.01s
    #     real_wrapper.update(pusher_pos[0]-15*(i+1), pusher_pos[1], pusher_pos[2])
    #     # if i % 3 == 2:
    #     #     time.sleep(0.1)
    #     time.sleep(0.1)
    #     # if i == test_steps//2:
    #     #     time.sleep(0.1)
    # print(f"total time: {time.time() - start_time}")

    real_wrapper.stop()


if __name__ == "__main__":
    main()
