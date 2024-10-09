import sys, os
import pymunk
from pymunk import Vec2d
import math
import numpy as np
import random
import time

sys.path.append(os.getcwd())
from base_sim import Base_Sim
from others.helper import *

""" 
main class for box pushing task
"""


class BoxSim(Base_Sim):
    def __init__(self, param_dict, init_poses=None, target_poses=None, pusher_pos=None):
        super().__init__(param_dict)
        if target_poses is None:
            self.target_positions = None
            self.target_angles = None
        else:
            self.target_positions = [target_pose[:2] for target_pose in target_poses]
            self.target_angles = [target_pose[2] for target_pose in target_poses]  # in radians
        if init_poses is not None:
            self.obj_num = len(init_poses)
        elif "obj_num" in param_dict:
            self.obj_num = param_dict["obj_num"]
        else:
            self.obj_num = 1
        self.param_dict = param_dict
        self.box_size = param_dict["box_size"]
        self.create_world(init_poses, pusher_pos)

    def create_object(self, id, pose=None):
        box_size = self.box_size
        mass = self.obj_mass
        moment = pymunk.moment_for_box(mass, (box_size, box_size))
        body = pymunk.Body(mass, moment)
        if pose is None:
            body.angle = random.random() * math.pi * 2
            body.position = Vec2d(random.randint(200, 300), random.randint(200, 300))
        else:
            body.angle = pose[2]
            body.position = Vec2d(pose[0], pose[1])
        shape = pymunk.Poly.create_box(body, (box_size, box_size))
        shape.color = self.object_colors[id % len(self.object_colors)]
        shape.elasticity = self.elasticity
        shape.friction = self.friction

        return body, [shape]

    def get_object_keypoints(self, index, target=False):
        return get_keypoints_from_pose(self.get_object_pose(index, target), self.param_dict, self.include_com)

    def get_object_vertices(self, index, target=False):
        return transform_polys_wrt_pose_2d(self.get_box_vertices(), self.get_object_pose(index, target))

    def gen_vertices_from_pose(self, pose):
        return transform_polys_wrt_pose_2d(self.get_box_vertices(), pose)

    def get_current_state(self):
        obj_keypoints = self.get_all_object_keypoints()[0]
        state = np.concatenate(
            (obj_keypoints, np.array([self.pusher_body.position]), np.array([self.velocity])), axis=0
        )
        return state.flatten()

    def get_box_vertices(self, flatten=False):
        box_size = self.box_size
        box_vertices = get_rect_vertices(box_size, box_size)
        if flatten:
            return box_vertices.flatten()
        return [box_vertices.tolist()]


""" 
some helper functions
"""


def generate_init_target_states(init_poses, target_poses, param_dict, include_com=False):
    init_states = get_keypoints_from_pose(init_poses[0], param_dict, include_com)
    target_states = get_keypoints_from_pose(target_poses[0], param_dict, include_com)
    return init_states.flatten(), target_states.flatten()


def get_keypoints_from_pose(pose, param_dict, include_com=False):
    box_size = param_dict["box_size"]
    pos = Vec2d(pose[0], pose[1])
    angle = pose[2]
    half_size = box_size / 2
    offsets = [
        Vec2d(-half_size, -half_size),  # Bottom Left
        Vec2d(half_size, -half_size),  # Bottom Right
        Vec2d(half_size, half_size),  # Top Right
        Vec2d(-half_size, half_size),  # Top Left
    ]
    if include_com:
        offsets.append(Vec2d(0, 0))
    # Calculate the global position of each keypoint
    keypoints = []
    for offset in offsets:
        rotated_offset = offset.rotated(angle)
        keypoints.append(pos + rotated_offset)

    return np.array(keypoints)


def get_pose_from_keypoints(keypoints, param_dict):
    box_size = param_dict["box_size"]
    half_size = box_size / 2
    model_points = np.array(
        [
            [-half_size, -half_size],
            [half_size, -half_size],
            [half_size, half_size],
            [-half_size, half_size],
        ]
    )

    return keypoints_to_pose_2d_SVD(model_points, keypoints)


def get_offests_w_origin(param_dict):
    box_size = param_dict["box_size"]
    # hardcode as the top left corner of the box, deponds on the .obj file
    center_w_origin = np.array([box_size / 2, -box_size / 2])
    offest_w_center = get_keypoints_from_pose([0, 0, 0], param_dict)
    return offest_w_center + center_w_origin


if __name__ == "__main__":
    param_dict = {
        "box_size": 60,
        "pusher_size": 10,
        "save_img": True,
        "enable_vis": True,
        "img_state": True,
        "img_size": 200,
    }

    # init_poses = [[[250,250,math.radians(45)], [150,150,math.radians(-45)]]]
    init_poses = [[250, 250, math.radians(0)]]
    target_poses = [[350, 350, math.radians(60)]]
    sim = BoxSim(
        param_dict=param_dict,
        init_poses=init_poses,
        # target_poses=target_poses,
    )
    # [[250,250,math.radians(45)], [150,150,math.radians(-45)]]
    sim.render()
    img = sim.update(sim.get_pusher_position())
    print(sim.get_all_object_positions())
    print(sim.get_all_object_keypoints())
    print(sim.get_current_state())
    print(sim.get_all_object_keypoints(target=False))
    for i in range(1):
        sim.render()
        time.sleep(0.5)
    sim.save_gif("box_pushing.gif")
    sim.close()
