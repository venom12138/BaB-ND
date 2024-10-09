from typing import List, Dict, Optional, Callable, Union

import cv2
import numpy as np
import sapien.core as sapien
import torch.utils.dlpack
from sapien.core import Pose
from sapien.core.pysapien import renderer as R
from sapien.utils import Viewer
from sapien_env.utils.render_scene_utils import add_mesh_to_renderer
import transforms3d
# DEFAULT_TABLE_TOP_CAMERAS = {
#     "left": dict(position=np.array([0, 1, 0.6]), look_at_dir=np.array([0, -1, -0.6]), right_dir=np.array([-1, 0, 0]),
#                  name="left_view", ),
#     "bird": dict(position=np.array([0, 0, 1.0]), look_at_dir=np.array([0, 0, -1]), right_dir=np.array([0, -1, 0]),
#                  name="bird_view", ),
# }

DEFAULT_TABLE_TOP_CAMERAS = {
    "left": dict(position=np.array([0, 0.5, 0.45]), look_at_dir=np.array([0, -0.5, -0.35]), right_dir=np.array([-1, 0, 0]),
                 name="left_view", ),
    "bird": dict(position=np.array([0, 0, 0.8]), look_at_dir=np.array([0, 0, -1]), right_dir=np.array([0, -1, 0]),
                 name="bird_view", ),
}

WRTHCH_USING_CAMERAS = {
    # "left": dict(position=np.array([0.05, 0.5, 0.5]), look_at_dir=np.array([0.05, -0.5, -0.5]), right_dir=np.array([-1, 0, 0]),
    #              name="left_view", ),
    "bird": dict(position=np.array([0.05, 0, 0.5]), look_at_dir=np.array([0.05, 0, -1]), right_dir=np.array([0, -1, 0]),
                 name="bird_view", ),
    "right": dict(position=np.array([0.05, -0.5, 0.5]), look_at_dir=np.array([0.05, 0.5, -0.5]), right_dir=np.array([1, 0, 0]),
                 name="right_view", ),
}
YX_TABLE_TOP_CAMERAS = {
    # "front": dict(position=np.array([0, 0.5, 0.6]), look_at_dir=np.array([0, -0.5, -0.6]), right_dir=np.array([-1, 0, 0]),
    #               name="front_view", ),
    "left_bottom": dict(position=np.array([0.5, 0.5, 0.6]), look_at_dir=np.array([-0.5, -0.5, -0.6]), right_dir=np.array([-1, 1, 0]),
                        name="left_bottom_view", ),
    "right_bottom": dict(position=np.array([-0.5, 0.5, 0.6]), look_at_dir=np.array([0.5, -0.5, -0.6]), right_dir=np.array([-1, -1, 0]),
                         name="right_bottom_view", ),
    "left_top": dict(position=np.array([0.5, -0.5, 0.6]), look_at_dir=np.array([-0.5, 0.5, -0.6]), right_dir=np.array([1, 1, 0]),
                     name="left_top_view", ),
    "right_top": dict(position=np.array([-0.5, -0.5, 0.6]), look_at_dir=np.array([0.5, 0.5, -0.6]), right_dir=np.array([1, -1, 0]),
                      name="right_top_view", ),
    "front": dict(position=np.array([0, 0.6, 0.2]), look_at_dir=np.array([0, -0.6, -0.2]), right_dir=np.array([-1, 0, 0]),
                  name="front_view", ),
    "right": dict(position=np.array([-0.6, 0, 0.2]), look_at_dir=np.array([0.6, 0, -0.2]), right_dir=np.array([0, -1, 0]),
                  name="right_view", ),
    # "front": dict(position=np.array([0, 0.6, 0.8]), look_at_dir=np.array([0, -0.6, -0.8]), right_dir=np.array([-1, 0, 0]),
    #               name="front_view", ),
    # "right": dict(position=np.array([-0.6, 0, 0.6]), look_at_dir=np.array([0.6, 0, -0.6]), right_dir=np.array([0, -1, 0]),
    #               name="right_view", ),
}

def depth_to_vis_depth(depth):
    # :param depth: depth image in np.uint16
    # :return: visualized depth image in np.uint8
    depth = depth.astype(np.float32) / 1000
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    depth_norm = np.clip(depth_norm * 255, 0, 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_VIRIDIS)
    return depth_vis

class GUIBase:
    def __init__(self, scene: sapien.Scene, renderer: Union[sapien.VulkanRenderer, sapien.KuafuRenderer],
                 resolution=(640, 480), window_scale=0.5, headless=False):
        use_ray_tracing = isinstance(renderer, sapien.KuafuRenderer)
        self.scene = scene
        self.renderer = renderer
        self.headless = headless
        self.cams: List[sapien.CameraEntity] = []
        self.cam_mounts: List[sapien.ActorBase] = []

        # Context
        self.use_ray_tracing = use_ray_tracing
        if not use_ray_tracing:
            self.context: R.Context = renderer._internal_context
            self.render_scene: R.Scene = scene.get_renderer_scene()._internal_scene
            self.nodes: List[R.Node] = []
        self.sphere_nodes: Dict[str, List[R.Node]] = {}
        self.sphere_model: Dict[str, R.Model] = {}

        # Viewer
        if not use_ray_tracing and not headless:
            self.viewer = Viewer(renderer)
            self.viewer.set_scene(scene)
            self.viewer.toggle_axes(False)
            self.viewer.toggle_camera_lines(False)
            self.viewer.set_camera_xyz(-0.3, 0, 0.5)
            self.viewer.set_camera_rpy(0, -1.4, 0)
        self.resolution = resolution
        self.window_scale = window_scale

        # Key down action map
        self.keydown_map: Dict[str, Callable] = {}

        # Common material
        self.viz_mat_hand = self.renderer.create_material()
        self.viz_mat_hand.set_base_color(np.array([0.96, 0.75, 0.69, 1]))
        self.viz_mat_hand.set_specular(0)
        self.viz_mat_hand.set_metallic(0.8)
        self.viz_mat_hand.set_roughness(0)

        # self.idx = 0
        
    def create_camera(self, position, look_at_dir, right_dir, name):
        builder = self.scene.create_actor_builder()
        builder.set_mass_and_inertia(1e-2, Pose(np.zeros(3)), np.ones(3) * 1e-4)
        mount = builder.build_static(name=f"{name}_mount")
        cam = self.scene.add_mounted_camera(name, mount, Pose(), width=self.resolution[0], height=self.resolution[1],
                                            fovy=0.9, fovx=0.9, near=0.1, far=10)

        # Construct camera pose
        look_at_dir = look_at_dir / np.linalg.norm(look_at_dir)
        right_dir = right_dir - np.sum(right_dir * look_at_dir).astype(np.float64) * look_at_dir
        right_dir = right_dir / np.linalg.norm(right_dir)
        up_dir = np.cross(look_at_dir, -right_dir)
        rot_mat_homo = np.stack([look_at_dir, -right_dir, up_dir, position], axis=1)
        pose_mat = np.concatenate([rot_mat_homo, np.array([[0, 0, 0, 1]])])

        # Add camera to the scene
        mount.set_pose(Pose.from_transformation_matrix(pose_mat))
        self.cams.append(cam)
        self.cam_mounts.append(mount)

    # def setup_visual_obs_config(self):
    #     for name, camera_cfg in config.items():
    #         if name not in self.cameras.keys():
    #             raise ValueError(
    #                 f"Camera {name} not created. Existing {len(self.cameras)} cameras: {self.cameras.keys()}")
    #         self.camera_infos[name] = {}
    #         banned_modality_set = {"point_cloud", "depth"}
    #         if len(banned_modality_set.intersection(set(camera_cfg.keys()))) == len(banned_modality_set):
    #             raise RuntimeError(f"Request both point_cloud and depth for same camera is not allowed. "
    #                                f"Point cloud contains all information required by the depth.")

    #         for modality, cfg in camera_cfg.items():
    #             if modality == "point_cloud":
    #                 if "process_fn" not in cfg or "num_points" not in cfg:
    #                     raise RuntimeError(f"Missing process_fn or num_points in camera {name} point_cloud config.")

    #             self.camera_infos[name][modality] = cfg


    @property
    def closed(self):
        return self.viewer.closed

    def _fetch_all_views(self, use_bgr=False, render_depth=False) -> List[np.ndarray]:
        views = []
        if render_depth:
            depths = []
        for cam in self.cams:
            cam.take_picture()
            
            # tensor -> np
            # rgb_tensor = torch.clamp(torch.utils.dlpack.from_dlpack(cam.get_dl_tensor("Color"))[..., :3], 0, 1) * 255
            # if use_bgr:
            #     rgb_tensor = torch.flip(rgb_tensor, [-1])
            # rgb = rgb_tensor.type(torch.uint8).cpu().numpy()
            
            # direct np
            rgb = (np.clip(cam.get_float_texture("Color")[..., :3],0,1) * 255).astype(np.uint8)
            if use_bgr:
                rgb = rgb[..., ::-1]
                
            if render_depth:
                # # tensor -> np
                # depth_tensor = torch.utils.dlpack.from_dlpack(cam.get_dl_tensor("Position"))[..., 2] * -1000
                # depth = depth_tensor.cpu().numpy().astype(np.uint16)
                # depths.append(depth)
                depth = (-cam.get_float_texture("Position")[..., 2] * 1000).astype(np.uint16)
                depths.append(depth)
            views.append(rgb)
        if render_depth:
            return views, depths
        return views

    def take_single_view(self, camera_name: str, use_bgr=False, render_depth=False) -> np.ndarray:
        import cupy
        for cam in self.cams:
            if cam.get_name() == camera_name:
                cam.take_picture()
                dlpack = cam.get_dl_tensor("Color")
                rgb = np.clip(cupy.asnumpy(cupy.from_dlpack(dlpack))[..., :3], 0, 1) * 255
                rgb = rgb.astype(np.uint8)
                if use_bgr:
                    rgb = np.flip(rgb, [-1])
                if render_depth:
                    dlpack = cam.get_dl_tensor("Position")
                    depth = (-cupy.asnumpy(cupy.from_dlpack(dlpack))[..., 2] * 1000).astype(np.uint16)
                    return rgb, depth
                return rgb
        raise RuntimeError(f"Camera name not found: {camera_name}")
        
    def render(self,horizontal=True,depth=False):
        self.scene.update_render()
        if not self.headless:
            self.viewer.render()
            if not self.viewer.closed:
                for key, action in self.keydown_map.items():
                    if self.viewer.window.key_down(key):
                        action()
        if depth:
            views, depths = self._fetch_all_views(use_bgr=True,render_depth=depth)
        else:
            views = self._fetch_all_views(use_bgr=True,render_depth=depth)
        
        if not self.headless:
            if horizontal:
                pad = np.ones([views[0].shape[0], 200, 3], dtype=np.uint8) * 255
            else:
                pad = np.ones([200, views[0].shape[1], 3], dtype=np.uint8) * 255

            final_views = [views[0]]
            if depth:
                final_depths = [depth_to_vis_depth(depths[0])]
            for i in range(1, len(views)):
                final_views.append(pad)
                final_views.append(views[i])
                if depth:
                    final_depths.append(pad)
                    final_depths.append(depth_to_vis_depth(depths[i]))
            axis = 1 if horizontal else 0
            final_views = np.concatenate(final_views, axis=axis)
            target_shape = final_views.shape
            target_shape = (int(target_shape[1] * self.window_scale), int(target_shape[0] * self.window_scale))
            final_views = cv2.resize(final_views, target_shape)
            # self.idx += 1
            # if self.idx % 10 == 0:
            #     cv2.imwrite(f"data/monitor_{self.idx}.png", views[0])
            cv2.imshow("Monitor", final_views)
            if depth:
                final_depths = np.concatenate(final_depths, axis=axis)
                final_depths = cv2.resize(final_depths, target_shape)
                cv2.imshow("Depth", final_depths)
            cv2.waitKey(1)
            
        if depth:
            return views, depths
        else:
            return views

    def update_mesh(self, v, f, viz_mat: R.Material, pose: Pose, use_shadow=True, clear_context=False):
        if clear_context:
            for i in range(len(self.nodes)):
                node = self.nodes.pop()
                self.render_scene.remove_node(node)
        node = add_mesh_to_renderer(self.scene, self.renderer, v, f, viz_mat)
        node.set_position(pose.p)
        node.set_rotation(pose.q)
        if use_shadow:
            node.shading_mode = 0
            node.cast_shadow = True
        self.nodes.append(node)

    def register_keydown_action(self, key, action: Callable):
        if key in self.keydown_map:
            raise RuntimeError(f"Key {key} has already been registered")
        self.keydown_map[key] = action

    def add_sphere_visual(self, label, pos: np.ndarray, rgba: np.ndarray = np.array([1, 0, 0, 1]),
                          radius: float = 0.01):
        if label not in self.sphere_model:
            mesh = self.context.create_uvsphere_mesh()
            material = self.context.create_material(emission=np.zeros(4), base_color=rgba, specular=0.4, metallic=0,
                                                    roughness=0.1)
            self.sphere_model[label] = self.context.create_model([mesh], [material])
            self.sphere_nodes[label] = []
        model = self.sphere_model[label]
        node = self.render_scene.add_object(model, parent=None)
        node.set_scale(np.ones(3) * radius)
        self.sphere_nodes[label].append(node)
        node.set_position(pos)
        return node
    def close(self):
        for camera in self.cams:
            self.scene.remove_camera(camera)
        self.cams = []
        self.scene = None

