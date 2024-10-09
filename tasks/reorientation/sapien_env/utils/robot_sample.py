import sapien.core as sapien
import transforms3d
from pathlib import Path
from urdfpy import URDF
import numpy as np
import copy

class RobotPcSampler:
    def __init__(self, robot_name="trossen_vx300s_tactile_thin"):
        self.engine = sapien.Engine()
        self.scene = self.engine.create_scene()
        loader = self.scene.create_urdf_loader()
        current_dir = Path(__file__).parent
        package_dir = (current_dir.parent / "assets").resolve()
        if "trossen" in robot_name:
            urdf_path = f"{package_dir}/robot/trossen_description/vx300s.urdf"
        elif "panda" in robot_name:
            urdf_path = f"{package_dir}/robot/panda/panda.urdf"
        elif "xarm" in robot_name:
            urdf_path = f"{package_dir}/robot/xarm6_description/xarm6_with_gripper.urdf"
        self.sapien_robot = loader.load(urdf_path)
        self.urdf_robot = URDF.load(urdf_path)
        self.robot_model = self.sapien_robot.create_pinocchio_model()
        self.robot_name = robot_name

        # load meshes and offsets from urdf_robot
        self.meshes = {}
        self.scales = {}
        self.offsets = {}
        for link in self.urdf_robot.links:
            if len(link.collisions) > 0:
                collision = link.collisions[0]
                if collision.geometry.mesh != None:
                    if len(collision.geometry.mesh.meshes) > 0:
                        mesh = collision.geometry.mesh.meshes[0]
                        self.meshes[link.name] = mesh.as_open3d
                        self.scales[link.name] = collision.geometry.mesh.scale[0] if collision.geometry.mesh.scale is not None else 1.0
            self.offsets[link.name] = collision.origin
            
        self.pcd_dict = {}

    def mesh_poses_to_pc(self, poses, meshes, offsets, num_pts, scales, pcd_name=None):

        try:
            assert poses.shape[0] == len(meshes)
            assert poses.shape[0] == len(offsets)
            assert poses.shape[0] == len(num_pts)
            assert poses.shape[0] == len(scales)
        except:
            raise RuntimeError('poses and meshes must have the same length')

        N = poses.shape[0]
        all_pc = []
        for index in range(N):
            mat = poses[index]
            if pcd_name is None or pcd_name not in self.pcd_dict or len(self.pcd_dict[pcd_name]) <= index:
                mesh = copy.deepcopy(meshes[index])  # .copy()
                mesh.scale(scales[index], center=np.array([0, 0, 0]))
                sampled_cloud= mesh.sample_points_poisson_disk(number_of_points=num_pts[index])
                cloud_points = np.asarray(sampled_cloud.points)
                if pcd_name not in self.pcd_dict:
                    self.pcd_dict[pcd_name] = []
                self.pcd_dict[pcd_name].append(cloud_points)
            else:
                cloud_points = self.pcd_dict[pcd_name][index]
            
            tf_obj_to_link = offsets[index]
            mat = mat @ tf_obj_to_link
            transformed_points = cloud_points @ mat[:3, :3].T + mat[:3, 3]
            all_pc.append(transformed_points)
        all_pc =np.concatenate(all_pc,axis=0)
        return all_pc


    def compute_mesh_poses(self, qpos, link_names = None):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        if link_names is None:
            link_names = self.meshes.keys()
        link_idx_ls = []
        for link_name in link_names:
            for link_idx, link in enumerate(self.sapien_robot.get_links()):
                if link.name == link_name:
                    link_idx_ls.append(link_idx)
                    break
        link_pose_ls = np.stack([self.robot_model.get_link_pose(link_idx).to_transformation_matrix() for link_idx in link_idx_ls])
        # meshes_ls = [self.meshes[link_name] for link_name in link_names]
        offsets_ls = [self.offsets[link_name] for link_name in link_names]
        scales_ls = [self.scales[link_name] for link_name in link_names]
        poses= self.get_mesh_poses(poses=link_pose_ls,offsets=offsets_ls, scales=scales_ls)
        return poses
    
    def get_mesh_poses(self, poses, offsets,scales):

        try:
            assert poses.shape[0] == len(offsets)
        except:
            raise RuntimeError('poses and meshes must have the same length')

        N = poses.shape[0]
        all_mats= []
        for index in range(N):
            mat = poses[index]
            tf_obj_to_link = offsets[index]
            # print('offsets_ls[index]',offsets[index])
            mat = mat @ tf_obj_to_link
            all_mats.append(mat)
        return np.stack(all_mats)
        


    # compute robot pcd given qpos
    def compute_robot_pcd(self, qpos, link_names = None, num_pts = None, pcd_name=None):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        if link_names is None:
            link_names = self.meshes.keys()
        if num_pts is None:
            num_pts = [32] * len(link_names)
        link_idx_ls = []
        for link_name in link_names:
            for link_idx, link in enumerate(self.sapien_robot.get_links()):
                if link.name == link_name:
                    link_idx_ls.append(link_idx)
                    break
        link_pose_ls = np.stack([self.robot_model.get_link_pose(link_idx).to_transformation_matrix() for link_idx in link_idx_ls])
        meshes_ls = [self.meshes[link_name] for link_name in link_names]
        offsets_ls = [self.offsets[link_name] for link_name in link_names]
        scales_ls = [self.scales[link_name] for link_name in link_names]
        pcd = self.mesh_poses_to_pc(poses=link_pose_ls, meshes=meshes_ls, offsets=offsets_ls, num_pts=num_pts, scales=scales_ls, pcd_name=pcd_name)
        return pcd
    
    def compute_sensor_pc(self, qpos, sensor_names, pcd_name=None):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        sensor_idx_ls = []
        for link_name in sensor_names:
            for link_idx, link in enumerate(self.sapien_robot.get_links()):
                if link.name == link_name:
                    sensor_idx_ls.append(link_idx)
                    break
        sensor_pose = np.stack([self.robot_model.get_link_pose(link_idx).to_transformation_matrix() for link_idx in sensor_idx_ls])
        offsets_ls = [self.offsets[link_name] for link_name in sensor_names]
        all_sensor_pc = []
        N = sensor_pose.shape[0]
        for index in range(N):
            sensor_pc = sensor_pose[index] @ offsets_ls[index]
            sensor_pc = sensor_pc[:3, 3]
            all_sensor_pc.append(sensor_pc)
        all_sensor_pc = np.stack(all_sensor_pc)
        return all_sensor_pc
    
def test_sample_robot():
    import open3d as o3d
    
    robot_name = 'trossen_vx300s_tactile_thin'
    finger_names = [
                    # 'vx300s/base_link',
                    # 'vx300s/shoulder_link',
                    # 'vx300s/upper_arm_link',
                    # 'vx300s/upper_forearm_link',
                    # 'vx300s/lower_forearm_link',
                    # 'vx300s/wrist_link',
                    # 'vx300s/gripper_link',
                    'vx300s/gripper_bar_link',
                    'vx300s/left_finger_link',
                    'vx300s/right_finger_link',]
    finger_contact_link_name = [
        "vx300s/left_tactile_map_link1", "vx300s/left_tactile_map_link2","vx300s/left_tactile_map_link3","vx300s/left_tactile_map_link4",
        "vx300s/left_tactile_map_link5", "vx300s/left_tactile_map_link6","vx300s/left_tactile_map_link7","vx300s/left_tactile_map_link8",
        "vx300s/left_tactile_map_link9", "vx300s/left_tactile_map_link10","vx300s/left_tactile_map_link11","vx300s/left_tactile_map_link12",
        "vx300s/left_tactile_map_link13", "vx300s/left_tactile_map_link14","vx300s/left_tactile_map_link15","vx300s/left_tactile_map_link16",
        "vx300s/right_tactile_map_link1","vx300s/right_tactile_map_link2","vx300s/right_tactile_map_link3","vx300s/right_tactile_map_link4",
        "vx300s/right_tactile_map_link5","vx300s/right_tactile_map_link6","vx300s/right_tactile_map_link7","vx300s/right_tactile_map_link8",
        "vx300s/right_tactile_map_link9","vx300s/right_tactile_map_link10","vx300s/right_tactile_map_link11","vx300s/right_tactile_map_link12",
        "vx300s/right_tactile_map_link13","vx300s/right_tactile_map_link14","vx300s/right_tactile_map_link15","vx300s/right_tactile_map_link16",
    ]
    # num_pts = [1000, 1000]
    init_qpos = np.array([0,0,0,0,0,0,0.021,-0.021])
    end_qpos = np.array([0, -0.96, 1.16, 0, -0.3, 0, 0.057, -0.057])
    sample_robot = RobotPcSampler(robot_name=robot_name)
    img_obs = o3d.geometry.PointCloud()
    contact_obs = o3d.geometry.PointCloud()
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()


    for i in range(100):
        curr_qpos = init_qpos + (end_qpos - init_qpos) * i / 100
        img_pc = sample_robot.compute_robot_pcd(curr_qpos, link_names=finger_names, )
        contact_pc = sample_robot.compute_sensor_pc(curr_qpos, sensor_names=finger_contact_link_name)
        if i == 0:
            img_obs.points = o3d.utility.Vector3dVector(img_pc)
            contact_obs.points = o3d.utility.Vector3dVector(contact_pc)
            img_obs.paint_uniform_color([1, 0, 0])
            contact_obs.paint_uniform_color([0, 1, 0])
            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
            # visualizer.add_geometry(camera_obs)
            visualizer.add_geometry(coordinate)
            visualizer.add_geometry(img_obs)
            visualizer.add_geometry(contact_obs)
        img_obs.points = o3d.utility.Vector3dVector(img_pc)
        contact_obs.points = o3d.utility.Vector3dVector(contact_pc)
        visualizer.update_geometry(img_obs)
        visualizer.update_geometry(contact_obs)
        visualizer.poll_events()
        visualizer.update_renderer()


if __name__ == "__main__":
    test_sample_robot()