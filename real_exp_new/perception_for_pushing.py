import cv2
from threadpoolctl import threadpool_limits
import os
import numpy as np
import copy
import multiprocessing as mp
import time
from threading import Lock
import open3d as o3d
file_dir = os.path.dirname(os.path.abspath(__file__))
class Perception(mp.Process):
    name = "Perception"
    def __init__(self, 
                 serial_numbers, 
                 realsense, 
                 param_dict,
                 process_func,
                 ):
        super().__init__()
        # self.index = index
        self.capture_fps = param_dict["capture_fps"]
        self.record_fps = param_dict["record_fps"]
        self.record_time = param_dict["record_time"]

        self.realsense = realsense
        # self.record_flag = record_flag
        # self.robot_obs = robot_obs

        self.lock = Lock()
        self.perception_q = mp.Queue(maxsize=1)

        self.process_func = process_func
        self.indices = [i for i in range(len(serial_numbers))]
        self.alive = mp.Value('b', False)

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

        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        # vis.add_geometry(coordinate)
        
        # # Set camera view
        # ctr = vis.get_view_control()
        # ctr.set_lookat([0, 0, 0])
        # ctr.set_front([0, -1, 0])
        # ctr.set_up([0, 0, -1])

        while self.alive.value:
            try: 
                out = realsense.get(out=out)
                if next_step_idx == 0:
                    start_time = time.time()
                cut_time = time.time()
                with self.lock:
                    color_list = [cv2.cvtColor(out[i]['color'], cv2.COLOR_BGR2RGB) for i in self.indices]
                    points_list = [out[i]['points'] for i in self.indices]
                    process_start_time = time.time()
                    process_out = self.process_func(
                        color_list, 
                        points_list, 
                    )
                    # print(f"process time: {time.time() - process_start_time}")
                    # print(f"T: {process_out['transformation']}")
                    # print(f"R: {process_out['transformation'][:3, :3]}")
                    if not self.perception_q.full():
                        # print(f"put block: {time.time()}")
                        self.perception_q.put(process_out["transformation_list"])
                        # self.perception_q.put(process_out)
                        # else:
                        #     print("!! perception_q is full")

                    # vis.clear_geometries()
                    # # vis.add_geometry(coordinate)
                    # for i in range(len(process_out["mesh_pcd_list"])):
                    #     vis.add_geometry(process_out["mesh_pcd_list"][i])
                    #     vis.add_geometry(process_out["object_pcd_list"][i])
                    # vis.poll_events()
                    # vis.update_renderer()

                    next_step_idx += 1


                if next_step_idx >= record_time * record_fps:
                    self.alive.value = False

            except Exception as e:
                print(f"Error in camera: {e}")
                self.alive.value = False

        finish_time = time.time()
        print(f"total time: {finish_time - start_time}")
        print(f"fps: {next_step_idx / (finish_time - start_time)}")


    def start(self):
        self.alive.value = True
        super().start()

    def stop(self):
        self.alive.value = False
        self.join()

    def join(self):
        super().join()
    

def process_for_T(color_list, points_list, param_dict,
                  visualize=False):
    # start_time = time.time()
    verbose = False
    # visualize = True
    FLOAT_TYPE = np.float16
    n_cameras = len(color_list)
    if verbose:
        for i in range(n_cameras):
            cv2.imwrite(f"{file_dir}/test{i}.png", color_list[i])
    # process_start_time = time.time()

    mesh_pcd_list, rotation_matrices_list, object_color_list, split_num, xyz_bounds, num_objects, workspace2world, R_list, T_list = param_dict["mesh_pcd_list"], param_dict["rotation_matrices_list"], param_dict["object_color_list"], param_dict["split_num"], param_dict["xyz_bounds"], param_dict["num_objects"], param_dict["workspace2world"], param_dict["R_list"], param_dict["T_list"]
    object_point_dict = {i: [] for i in range(num_objects)}
    object_color_dict = {i: [] for i in range(num_objects)}

    aligned_mesh_pcd_list = []
    aligned_object_pcd_list = []
    align_transformation_list = []
    if visualize:
        visualize_list = []
        vis_valid_points_list = []
        vis_valid_colors_list = []
    
    def get_mask(points, bounds):
        return (
            (points[:, 0] > bounds[0,0])
            & (points[:, 0] < bounds[0,1])
            & (points[:, 1] > bounds[1,0])
            & (points[:, 1] < bounds[1,1])
            & (points[:, 2] > bounds[2,0])
            & (points[:, 2] < bounds[2,1])
        )

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
        if visualize:
            mask = get_mask(valid_points, xyz_bounds)
            vis_valid_points = valid_points[mask]
            vis_valid_colors = valid_colors[mask]
            vis_valid_points_list.append(valid_points)
            vis_valid_colors_list.append(valid_colors)

        # object_start_time = time.time()
        for j in range(num_objects):
            object_color = object_color_list[j]
            
            
            
            # visualize each camera point cloud and object point cloud
            # before color filtering
            # print(f"camera {i}")
            # visualize_list = []
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(valid_points)
            # pcd.colors = o3d.utility.Vector3dVector(valid_colors)
            # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            # coordinate.transform(workspace2world)
            # visualize_list.append(pcd)
            # visualize_list.append(coordinate)
            # o3d.visualization.draw_geometries(visualize_list)


            color_mask = np.linalg.norm(valid_colors - object_color, axis=1, ord=2) < 0.35 # 0.5
            # import pdb; pdb.set_trace()
            object_points = valid_points[color_mask]
            object_colors = valid_colors[color_mask]
            mask = get_mask(object_points, xyz_bounds)
            object_points = object_points[mask]
            object_colors = object_colors[mask]
            
            
            
            # # visualize each camera point cloud and object point cloud
            # print(f"camera {i}")
            # visualize_list = []
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(object_points)
            # pcd.colors = o3d.utility.Vector3dVector(object_colors)
            # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            # coordinate.transform(workspace2world)
            # visualize_list.append(pcd)
            # visualize_list.append(coordinate)
            # o3d.visualization.draw_geometries(visualize_list)



            object_point_dict[j].append(object_points)
            object_color_dict[j].append(object_colors)

    object_point_dict = {i: np.concatenate(object_point_dict[i], axis=0) for i in range(num_objects)}
    object_color_dict = {i: np.concatenate(object_color_dict[i], axis=0) for i in range(num_objects)}
    # assert len(object_points) > 0



    # print('visualize the combined pointcloud')
    # visualize_list = []
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(object_point_dict[i])
    # pcd.colors = o3d.utility.Vector3dVector(object_color_dict)
    # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    # coordinate.transform(workspace2world)
    # visualize_list.append(pcd)
    # visualize_list.append(coordinate)
    # o3d.visualization.draw_geometries(visualize_list)



    for j in range(num_objects):
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
        #     coordinate.transform(workspace2world)
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
        
        # if last_mesh_transform is None:
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

        # if verbose:
        #     print(f"distance before icp: {min_dist}")
        # icp_start_time = time.time()
        reg_p2p = o3d.pipelines.registration.registration_icp(
            center_mesh,
            object_pcd,
            0.005,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
        )
        mesh2world = np.dot(reg_p2p.transformation.astype(FLOAT_TYPE), mesh2world)
        center_mesh.transform(reg_p2p.transformation)

        if visualize:
            center_mesh.paint_uniform_color([1, 0, 0])
            visualize_list.append(center_mesh)
            visualize_list.append(object_pcd)
        align_transformation_list.append(mesh2world)
        aligned_mesh_pcd_list.append(center_mesh)
        aligned_object_pcd_list.append(object_pcd)
    if visualize:
        vis_valid_points = np.concatenate(vis_valid_points_list, axis=0)
        vis_valid_colors = np.concatenate(vis_valid_colors_list, axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vis_valid_points) # whole workspace
        pcd.colors = o3d.utility.Vector3dVector(vis_valid_colors)
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        coordinate.transform(workspace2world)
        # visualize_list.append(pcd)
        visualize_list.append(coordinate)
        o3d.visualization.draw_geometries(visualize_list)

    # return align_transformation_list, aligned_mesh_pcd_list, aligned_object_pcd_list
    # print(f"total time: {time.time() - start_time}")
    return {"transformation_list": align_transformation_list,
            "mesh_pcd_list": aligned_mesh_pcd_list,
            "object_pcd_list": aligned_object_pcd_list,
    }


def process_for_L(color_list, points_list, param_dict,
                  visualize=False):
    verbose = False
    # visualize = True
    FLOAT_TYPE = np.float16
    n_cameras = len(color_list)
    if verbose:
        for i in range(n_cameras):
            cv2.imwrite(f"{file_dir}/test{i}.png", color_list[i])
    # process_start_time = time.time()

    mesh_pcd_list, rotation_matrices_list, object_color_list, split_num, xyz_bounds, num_objects, workspace2world, R_list, T_list = param_dict["mesh_pcd_list"], param_dict["rotation_matrices_list"], param_dict["object_color_list"], param_dict["split_num"], param_dict["xyz_bounds"], param_dict["num_objects"], param_dict["workspace2world"], param_dict["R_list"], param_dict["T_list"]
    xyz_bounds[2,0] = 0.008
    xyz_bounds[2,1] = 0.05
    object_point_dict = {i: [] for i in range(num_objects)}
    object_color_dict = {i: [] for i in range(num_objects)}

    aligned_mesh_pcd_list = []
    aligned_object_pcd_list = []
    align_transformation_list = []
    if visualize:
        visualize_list = []
        vis_valid_points_list = []
        vis_valid_colors_list = []
    
    def get_mask(points, bounds):
        return (
            (points[:, 0] > bounds[0,0])
            & (points[:, 0] < bounds[0,1])
            & (points[:, 1] > bounds[1,0])
            & (points[:, 1] < bounds[1,1])
            & (points[:, 2] > bounds[2,0])
            & (points[:, 2] < bounds[2,1])
        )

    for i in range(n_cameras):
        colors = (color_list[i] / 255).astype(FLOAT_TYPE).reshape(-1, 3)
        points = (points_list[i]).astype(FLOAT_TYPE).reshape(-1, 3)
        valid_points = points
        valid_colors = colors
        # if verbose:
        #     print(f"pre mask time: {time.time() - process_start_time}")
        
        # post_start_time = time.time()
        valid_points = valid_points @ R_list[i] + T_list[i]
        # if verbose:
        #     print(f"transform time: {time.time() - post_start_time}")

        # TODO: update mask to fit workspace
        if visualize:
            mask = get_mask(valid_points, xyz_bounds)
            vis_valid_points = valid_points[mask]
            vis_valid_colors = valid_colors[mask]
            vis_valid_points_list.append(valid_points)
            vis_valid_colors_list.append(valid_colors)

        # object_start_time = time.time()
        for j in range(num_objects):
            object_color = object_color_list[j]
            
            if j == 0:
                color_mask = np.linalg.norm(valid_colors - object_color, axis=1, ord=1) < 0.5 # 0.5
            else:
                color_mask = np.linalg.norm(valid_colors - object_color, axis=1, ord=1) < 0.35
            # import pdb; pdb.set_trace()
            object_points = valid_points[color_mask]
            object_colors = valid_colors[color_mask]
            mask = get_mask(object_points, xyz_bounds)
            object_points = object_points[mask]
            object_colors = object_colors[mask]
            
            min_rgb = np.min(object_colors, axis=1)
            max_rgb = np.max(object_colors, axis=1)
            shadow_mask = ((max_rgb - min_rgb) < 0.1)
            object_points = object_points[~shadow_mask]
            object_colors = object_colors[~shadow_mask]

            object_point_dict[j].append(object_points)
            object_color_dict[j].append(object_colors)

    object_point_dict = {i: np.concatenate(object_point_dict[i], axis=0) for i in range(num_objects)}
    object_color_dict = {i: np.concatenate(object_color_dict[i], axis=0) for i in range(num_objects)}
    # assert len(object_points) > 0

    for j in range(num_objects):
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
        
        # if last_mesh_transform is None:
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

        if visualize:
            center_mesh.paint_uniform_color([1, 0, 0])
            visualize_list.append(center_mesh)
            visualize_list.append(object_pcd)
        align_transformation_list.append(mesh2world)
        aligned_mesh_pcd_list.append(center_mesh)
        aligned_object_pcd_list.append(object_pcd)
    if visualize:
        vis_valid_points = np.concatenate(vis_valid_points_list, axis=0)
        vis_valid_colors = np.concatenate(vis_valid_colors_list, axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vis_valid_points) # whole workspace
        pcd.colors = o3d.utility.Vector3dVector(vis_valid_colors)
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        coordinate.transform(workspace2world)
        # visualize_list.append(pcd)
        visualize_list.append(coordinate)
        o3d.visualization.draw_geometries(visualize_list)

    # return align_transformation_list, aligned_mesh_pcd_list, aligned_object_pcd_list

    return {"transformation_list": align_transformation_list,
            "mesh_pcd_list": aligned_mesh_pcd_list,
            "object_pcd_list": aligned_object_pcd_list,
    }
