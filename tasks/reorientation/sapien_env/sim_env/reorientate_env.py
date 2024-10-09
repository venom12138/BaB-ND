import numpy as np
import sapien.core as sapien
import transforms3d
import sys
import os
sys.path.append(os.getcwd())
from tasks.reorientation.sapien_env.sim_env.base import BaseSimulationEnv
from tasks.reorientation.sapien_env.utils.render_scene_utils import set_entity_color
from tasks.reorientation.sapien_env.utils.ycb_object_utils import load_ycb_object, YCB_SIZE, YCB_ORIENTATION
from sapien.utils import Viewer
from tasks.reorientation.sapien_env.sim_env.constructor import add_default_scene_light

class ReorientateEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5,  randomness_scale=1, friction=1, use_visual_obs=False, use_ray_tracing =False,**renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, use_visual_obs=use_visual_obs,use_ray_tracing=use_ray_tracing, **renderer_kwargs)


        self.target_pose = sapien.Pose()

        # Dynamics info
        self.randomness_scale = randomness_scale
        self.friction = friction

        # Construct scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(0.005)

        # Dummy camera creation to initial geometry object
        if self.renderer and not self.no_rgb:
            cam = self.scene.add_camera("init_not_used", width=10, height=10, fovy=1, near=0.1, far=1)
            self.scene.remove_camera(cam)

        # Load table
        self.tables = self.create_table(table_height=0.6, table_half_size=[0.65, 0.65, 0.025])
        # sizes of half sizes in fact
        well_size = 0.2
        self.well_size = [well_size/4, well_size, well_size]
        self.well = self.load_cube(self.well_size, is_static=True)
        object_size = 0.03
        self.object_size = [2*object_size, 0.5*object_size, object_size]
        self.manipulated_object = self.load_cube(self.object_size)
        # self.target= self.load_target()

    def load_cube(self, cube_size, is_static=False):
        cube_physics = self.scene.create_physical_material(5 * self.friction, 0.5 * self.friction, 0.01)
        builder = self.scene.create_articulation_builder()
        cube = builder.create_link_builder()
        cube.set_name("cube")
        cube.add_box_collision(pose=sapien.Pose([0, 0, 0]), half_size=cube_size, density=100000 if is_static else 1,material=cube_physics)
        if self.use_gui:
            cube.add_box_visual(pose=sapien.Pose([0, 0, 0]), half_size=cube_size, color=[0.5, 0.5, 0.5, 1] if is_static else [1, 1, 0, 1])
        cube = builder.build(fix_root_link=False)
        return cube

    # def load_target(self):
    #     target_size = [0.06,0.06,0.01]
    #     target_physics = self.scene.create_physical_material(1 * self.friction, 0.5 * self.friction, 0.01)
    #     builder = self.scene.create_articulation_builder()
    #     target = builder.create_link_builder()
    #     target.set_name("target")
    #     target.add_box_collision(pose=sapien.Pose([0, 0, 0]), half_size=target_size, density=1,material=target_physics)
    #     if self.use_gui:
    #         target.add_box_visual(pose=sapien.Pose([0, 0, 0]), half_size=target_size,color=[1, 1, 0, 1])
    #     target = builder.build(fix_root_link=True)
    #     return target
    
    # def load_workspace(self):
    #     block_size = [0.2,0.2,0.05]
    #     block_physics = self.scene.create_physical_material(1 * self.friction, 0.5 * self.friction, 0.01)
    #     builder = self.scene.create_articulation_builder()
    #     block = builder.create_link_builder()
    #     block.set_name("workspace")
    #     block.add_box_collision(pose=sapien.Pose([0, 0, 0]), half_size=block_size, density=1,material=block_physics)
    #     if self.use_gui:
    #         block.add_box_visual(pose=sapien.Pose([0, 0, 0]), half_size=block_size)
    #     block = builder.build(fix_root_link=True)
    #     return block
        
    def generate_random_object_pose(self, randomness_scale):
        pos0 = self.np_random.uniform(low=0.05, high=0.1, size=1) * randomness_scale
        # pos1 = self.np_random.uniform(low=0, high=0.05, size=1) * randomness_scale
        pos1 = self.np_random.uniform(low=-0.03, high=0.03, size=1) * randomness_scale
        orientation= transforms3d.euler.euler2quat(0, 0, 0)
        position = np.array([pos0[0],pos1[0], 0.08])
        pose = sapien.Pose(position, orientation)
        return pose

    # def generate_random_target_pose(self, randomness_scale):
    #     pos0 = self.np_random.uniform(low=0.1, high=0.2, size=1) * randomness_scale
    #     # pos1 = self.np_random.uniform(low=0, high=0.05, size=1) * randomness_scale
    #     pos1 = self.np_random.uniform(low=-0.03, high=0.03, size=1) * randomness_scale
    #     orientation= transforms3d.euler.euler2quat(0, 0, 0)
    #     position = np.array([pos0[0],pos1[0], 0.01])
    #     pose = sapien.Pose(position, orientation)
    #     return pose

    def reset_env(self):
        self.well.set_pose(sapien.Pose([0.3, 0, self.well_size[2]*1]))
        pose = self.generate_random_object_pose(self.randomness_scale)
        self.manipulated_object.set_pose(pose)
        orientation= transforms3d.euler.euler2quat(0, 0, 0)
        # target_pose = self.generate_random_target_pose(self.randomness_scale)
        # self.target.set_pose(target_pose)


def env_test():

    env = ReorientateEnv(use_gui=True, use_ray_tracing=False)
    env.reset_env()
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    pi = np.pi
    viewer.set_camera_rpy(r=0, p=-pi/4, y=pi/4)
    viewer.set_camera_xyz(x=-0.5, y=0.5, z=0.5)
    env.viewer = viewer
    viewer.toggle_pause(True)

    while not viewer.closed:
        env.reset_env()
        # env.simple_step()
        env.render()


if __name__ == '__main__':
    env_test()
