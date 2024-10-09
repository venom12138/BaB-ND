import sys
import os
from xarm import version
from xarm.wrapper import XArmAPI
import ctypes
# ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(ROOT_DIR)
# os.chdir(ROOT_DIR)
file_dir = os.path.dirname(os.path.abspath(__file__))
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

class ImageProcessor(mp.Process):

    def __init__(self, serial_numbers, realsense, capture_fps, record_fps, record_time, record_flag):
        super().__init__()
        # self.index = index
        self.capture_fps = capture_fps
        self.record_fps = record_fps
        self.record_time = record_time

        self.realsense = realsense
        self.record_flag = record_flag
        # self.robot_obs = robot_obs

        self.lock = Lock()

        mesh_pcd, R_list, T_list, rotation_matrices, split_num = get_params(serial_numbers)
        self.mesh_pcd = mesh_pcd
        self.R_list = R_list
        self.T_list = T_list
        self.rotation_matrices = rotation_matrices
        self.split_num = split_num
        self.indices = [i for i in range(len(serial_numbers))]

    def run(self):
        # limit threads
        threadpool_limits(1)
        cv2.setNumThreads(1)

        # i = self.index
        capture_fps = self.capture_fps
        record_fps = self.record_fps
        record_time = self.record_time

        realsense = self.realsense
        # robot_obs = self.robot_obs

        out = None
        next_step_idx = 0

        # f = open(f'recording/timestamps.txt', 'a')
        
        while self.alive:
            try:
                out = realsense.get(out=out)
                if out[0]['step_idx'] == next_step_idx * (capture_fps // record_fps):
                    if next_step_idx == 0:
                        start_time = time.time()
                    print(f'step {out[0]["step_idx"]} timestamp {out[0]["timestamp"]}')
                    with self.lock:
                        color_list = [out[i]['color'] for i in self.indices]
                        points_list = [out[i]['points'] for i in self.indices]
                        process_start_time = time.time()
                        process(color_list, points_list, self.mesh_pcd, self.R_list, self.T_list, self.rotation_matrices, self.split_num)
                        print(f"process time: {time.time() - process_start_time}")

                    # with self.lock:
                    #     cv2.imwrite(f'recording/camera_{i}/{next_step_idx:06}.jpg', out[i]['color'])
                    #     cv2.imwrite(f'recording/camera_{i}/{next_step_idx:06}_depth.png', out[i]['depth'])
                    #     f.write(f'{timestamp}\n')
                    #     f.flush()

                    next_step_idx += 1
                    self.record_flag.value = True

                if next_step_idx >= record_time * record_fps:
                    # f.close()
                    self.alive = False
                    self.record_flag.value = False

            except Exception as e:
                print(f"Error in camera: {e}")
                # f.close()
                self.alive = False
                self.record_flag.value = False
        finish_time = time.time()
        print(f"total time: {finish_time - start_time}")
        print(f"fps: {next_step_idx / (finish_time - start_time)}")
            

    def start(self):
        self.alive = True
        super().start()

    def stop(self):
        self.alive = False
    
    def join(self):
        super().join()

def process(color_list, points_list, mesh_pcd, R_list, T_list, rotation_matrices, split_num):
    verbose = False
    FLOAT_TYPE = np.float16
    n_cameras = len(color_list)
    if verbose:
        for i in range(n_cameras):
            cv2.imwrite(f'test{i}.png', color_list[i])
    process_start_time = time.time()
    object_points_list = []
    object_colors_list = []
    valid_points_list = []
    valid_colors_list = []
    for i in range(n_cameras):
        colors = (color_list[i]/255).astype(FLOAT_TYPE).reshape(-1,3)
        points = (points_list[i]).astype(FLOAT_TYPE).reshape(-1, 3)
        valid_points = points
        valid_colors = colors
        # if verbose:
        #     print(f"pre mask time: {time.time() - process_start_time}")
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(valid_points)
        # pcd.colors = o3d.utility.Vector3dVector(valid_colors)
        # o3d.visualization.draw_geometries([pcd])
        
        # post_start_time = time.time()
        valid_points = valid_points @ R_list[i] + T_list[i]
        # if verbose:
        #     print(f"transform time: {time.time() - post_start_time}")
        workspace2world_offset = [0.2, -0.05]
        
        mask = (
            (valid_points[:, 0] > 0 + workspace2world_offset[0]) & 
            (valid_points[:, 0] < 0.4 + workspace2world_offset[0]) & 
            (valid_points[:, 1] > 0 + workspace2world_offset[1]) & 
            (valid_points[:, 1] < 0.4 + workspace2world_offset[1]) &
            (valid_points[:, 2] > -0.02) & 
            (valid_points[:, 2] < 0.0)
        )
        valid_points = valid_points[mask]
        valid_colors = valid_colors[mask]
        valid_points_list.append(valid_points)
        valid_colors_list.append(valid_colors)
        
        # object_start_time = time.time()
        block_color = np.array([0, 0, 0], dtype=FLOAT_TYPE)
        color_mask = np.linalg.norm(valid_colors - block_color, axis=1, ord=2) < 0.3
        object_points = valid_points[color_mask]

        object_colors = valid_colors[color_mask]
        object_points_list.append(object_points)
        object_colors_list.append(object_colors)

    valid_points = np.concatenate(valid_points_list, axis=0)
    valid_colors = np.concatenate(valid_colors_list, axis=0)
    object_points = np.concatenate(object_points_list, axis=0)
    object_colors = np.concatenate(object_colors_list, axis=0)
    assert len(object_points) > 0

    object_points[:, 2] = np.min(object_points[:, 2])
    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(object_points)
    object_pcd.colors = o3d.utility.Vector3dVector(object_colors)
    mesh_center = mesh_pcd.get_center().astype(FLOAT_TYPE)
    object_center = object_pcd.get_center().astype(FLOAT_TYPE)
    # if verbose:
    #     print(f"object time: {time.time() - object_start_time}")
    
    # rotation_time = time.time()
    total_transformation = np.eye(4, dtype=FLOAT_TYPE)
    total_transformation[:3, 3] = object_center - mesh_center
    center_mesh = copy.deepcopy(mesh_pcd)
    center_mesh.transform(total_transformation)
    min_dist = np.inf
    best_result = None
    for i in range(split_num):
        copy_mesh = copy.deepcopy(center_mesh)
        rotate_matrix = rotation_matrices[i]
        rotate_center = copy_mesh.get_center().astype(FLOAT_TYPE)
        local_transformation = np.eye(4, dtype=FLOAT_TYPE)
        local_transformation[:3, :3] = rotate_matrix
        local_transformation[:3, 3] = rotate_center - np.dot(rotate_matrix, rotate_center)
        copy_mesh.transform(local_transformation)
        dists = copy_mesh.compute_point_cloud_distance(object_pcd)
        if np.mean(dists) < min_dist:
            min_dist = np.mean(dists)
            best_result = local_transformation
    # if verbose:
    #     print(f"rotation time: {time.time() - rotation_time}")
    total_transformation = np.dot(best_result, total_transformation)
    mesh_pcd.transform(total_transformation)
    if verbose:
        print(min_dist)
    
    # icp_start_time = time.time()
    reg_p2p = o3d.pipelines.registration.registration_icp(
        mesh_pcd,
        object_pcd,
        0.005,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
    )
    total_transformation = np.dot(reg_p2p.transformation.astype(FLOAT_TYPE), total_transformation)
    mesh_pcd.transform(reg_p2p.transformation)
    if verbose:
        print(np.mean(mesh_pcd.compute_point_cloud_distance(object_pcd)))
        # print(f"icp time: {time.time() - icp_start_time}")

        # print(f"total time: {time.time() - process_start_time}")
    print(f"total time: {time.time() - process_start_time}")
    # # visualize
    mesh_pcd.paint_uniform_color([1, 0, 0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.colors = o3d.utility.Vector3dVector(valid_colors)
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    o3d.visualization.draw_geometries([object_pcd, mesh_pcd, coordinate])

    return 

def get_params(serial_numbers = ['246322301893']):
    FLOAT_TYPE = np.float16
    cad_path = f'{file_dir}/object/t.obj'
    mesh = o3d.io.read_triangle_mesh(cad_path)
    mesh.compute_vertex_normals()
    mesh_pcd = mesh.sample_points_uniformly(number_of_points=500)
    cad_points = np.asarray(mesh_pcd.points)
    # cad_points = cad_points[cad_points[:, 2] > 0.0095]
    cad_points[:, 2] = 0.01
    mesh_pcd = o3d.geometry.PointCloud()
    mesh_pcd.points = o3d.utility.Vector3dVector(cad_points)
    split_num = 120
    rotation_matrices = [mesh.get_rotation_matrix_from_xyz((0, 0, np.pi * 2 / split_num * i)).astype(FLOAT_TYPE) for i in range(split_num)]

    calibrate_result_dir = f'{file_dir}/calibration_result'
    with open(f'{calibrate_result_dir}/rvecs.pkl', 'rb') as f:
        rvecs = pickle.load(f)
    with open(f'{calibrate_result_dir}/tvecs.pkl', 'rb') as f:
        tvecs = pickle.load(f)
    with open(f'{calibrate_result_dir}/intrinsics.pkl', 'rb') as f:
        intrinsics = pickle.load(f)
    R_list, T_list = [], []
    for i in range(len(serial_numbers)):
        serial_number = serial_numbers[i]
        R_cam2board = rvecs[serial_number] # (3, 1)
        # convert to rotation matrix
        R_cam2board = cv2.Rodrigues(R_cam2board)[0]
        t_cam2board = tvecs[serial_number] # (3, 1)
        intr = intrinsics[serial_number]
        cam2base = np.eye(4)
        cam2base[:3, :3] = R_cam2board
        cam2base[:3, 3] = t_cam2board.flatten()
        cam2base = np.linalg.inv(cam2base)
        R_list.append(cam2base[:3, :3].T.astype(FLOAT_TYPE))
        T_list.append(cam2base[:3, 3].astype(FLOAT_TYPE))
    return mesh_pcd, R_list, T_list, rotation_matrices, split_num

def main():
    # # serial_numbers = ['246322301893', '311322303615']
    # serial_numbers = ['246322301893', '246322303954', '311322303615', '311322300308']
    # # visible_area = np.array([[[0.35, 0.85], [0.1, 0.6]]])
    # visible_areas = []
    # for serial_number in serial_numbers:
    #     if serial_number == '246322301893':
    #         visible_area = [[0.35, 0.75], [0.15, 0.75]]
    #     elif serial_number == '246322303954':
    #         visible_area = [[0.25, 0.65], [0.25, 0.8]]
    #     elif serial_number == '311322303615':
    #         visible_area = [[0.35, 0.75], [0.15, 0.7]]
    #     elif serial_number == '311322300308':
    #         visible_area = [[0.40, 0.75], [0.2, 0.75]]
    #     else:
    #         visible_area = None
    #     visible_areas.append(visible_area)

    # serial_numbers = ['246322301893', '246322303954', '311322303615', '311322300308']
    serial_numbers = ['246322301893', '246322303954', '311322303615', '311322300308']
    # visible_area = np.array([[[0.35, 0.85], [0.1, 0.6]]])
    visible_areas = []
    for serial_number in serial_numbers:
        if serial_number == '246322301893':
            visible_area = [[0.35, 0.75], [0.15, 0.75]]
        elif serial_number == '246322303954':
            visible_area = [[0.25, 0.65], [0.25, 0.8]]
        elif serial_number == '311322303615':
            visible_area = [[0.35, 0.75], [0.15, 0.7]]
        elif serial_number == '311322300308':
            visible_area = [[0.40, 0.75], [0.2, 0.75]]
        else:
            visible_area = None
        visible_areas.append(visible_area)
    visible_areas = np.array(visible_areas)
    assert len(serial_numbers) == visible_areas.shape[0]
    record_fps = 15
    record_time = 5
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    camera = MultiRealsense(
                shm_manager=shm_manager,
                serial_numbers=serial_numbers,
                resolution=(640, 360),
                # resolution=(1280, 720),
                capture_fps=record_fps,
                enable_depth=True,
                enable_pcd=True,
                verbose=False,
                visible_areas=visible_areas
                # visible_area=None
            )
    try:
        camera.start()
        camera.set_exposure(exposure=200, gain=60)
        exposure_time = 5
        rec_start_time = time.time() + exposure_time
        camera.restart_put(start_time=rec_start_time)
        record_time = 5


        manager = Manager()
        record_flag = manager.Value(ctypes.c_bool, False)
        process = ImageProcessor(serial_numbers, camera, record_fps, record_fps, record_time, record_flag)
        process.start()

        process.join()
    finally:
        camera.stop()
    
    pass

if __name__ == "__main__":
    main()

