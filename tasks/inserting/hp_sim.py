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
main class for hole-peg insertion task
"""


class HP_Sim(Base_Sim):
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
            self.obj_num = 2
        self.param_dict = param_dict
        self.unit_size = param_dict["unit_size"]
        self.hp_ratio = param_dict["hp_ratio"]
        self.fix_hole = param_dict["fix_hole"]
        self.create_world(init_poses, pusher_pos)

    def create_object(self, id, pose=None):
        """
        Create a hole or peg by defining its shapes, mass, etc., with the leg above the foot and both aligned to the left.
        """
        color = self.object_colors[id % len(self.object_colors)]
        if id % 2 == 0:
            return self.create_hole_shape(color, pose)
        else:
            return self.create_peg_shape(color, pose)

    def create_hole_shape(self, color, pose=None):
        """
        Create a hole by defining its shape, mass, etc.
        """
        unit_size = self.unit_size
        base_vertices, side1_vertices, side2_vertices = self.get_hole_comp_vertices()
        if self.fix_hole:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
        else:
            mass = self.obj_mass
            base_square = 3
            side_square = 1
            total_square = base_square + 2 * side_square
            base_moment = pymunk.moment_for_poly(mass * base_square / total_square, base_vertices)
            side1_moment = pymunk.moment_for_poly(mass * side_square / total_square, side1_vertices)
            side2_moment = pymunk.moment_for_poly(mass * side_square / total_square, side2_vertices)
            moment = base_moment + side1_moment + side2_moment
            body = pymunk.Body(mass, moment)
        if pose is None:
            body.angle = random.random() * math.pi * 2
            body.position = Vec2d(random.randint(200, 300), random.randint(200, 300))
        else:
            body.angle = pose[2]
            body.position = Vec2d(pose[0], pose[1])

        base = pymunk.Poly(body, base_vertices)
        base.color = color
        base.elasticity = self.elasticity
        base.friction = self.friction
        side_1 = pymunk.Poly(body, side1_vertices)
        side_1.color = color
        side_1.elasticity = self.elasticity
        side_1.friction = self.friction
        side_2 = pymunk.Poly(body, side2_vertices)
        side_2.color = color
        side_2.elasticity = self.elasticity
        side_2.friction = self.friction
        return body, [base, side_1, side_2]

    def create_peg_shape(self, color, pose=None):
        """
        Create a peg by defining its shape, mass, etc.
        """
        unit_size = self.unit_size
        hp_ratio = self.hp_ratio
        mass = self.obj_mass  # Define the mass of the L shape

        base_vertices, peg_vertices = self.get_peg_comp_vertices()
        base_square = 3
        peg_square = 1 * hp_ratio
        total_square = base_square + peg_square
        base_moment = pymunk.moment_for_poly(mass * base_square / total_square, base_vertices)
        peg_moment = pymunk.moment_for_poly(mass * peg_square / total_square, peg_vertices)
        moment = base_moment + peg_moment
        body = pymunk.Body(mass, moment)
        if pose is None:
            body.angle = random.random() * math.pi * 2
            body.position = Vec2d(random.randint(200, 300), random.randint(200, 300))
        else:
            body.angle = pose[2]
            body.position = Vec2d(pose[0], pose[1])

        base = pymunk.Poly(body, base_vertices)
        base.color = color
        base.elasticity = self.elasticity
        base.friction = self.friction
        half_peg_width = hp_ratio * unit_size * 0.5
        peg = pymunk.Poly(body, peg_vertices)
        peg.color = color
        peg.elasticity = self.elasticity
        peg.friction = self.friction

        return body, [base, peg]

    def get_object_keypoints(self, index, target=False):
        if index % 2 == 0:
            return get_keypoints_from_pose_hole(self.get_object_pose(index, target), self.param_dict, self.include_com)
        else:
            return get_keypoints_from_pose_peg(self.get_object_pose(index, target), self.param_dict, self.include_com)

    def get_object_vertices(self, index, target=False):
        if index % 2 == 0:
            return transform_polys_wrt_pose_2d(self.get_hole_comp_vertices(), self.get_object_pose(index, target))
        else:
            return transform_polys_wrt_pose_2d(self.get_peg_comp_vertices(), self.get_object_pose(index, target))

    def gen_vertices_from_pose(self, pose, obj_type):
        if obj_type == "hole":
            return transform_polys_wrt_pose_2d(self.get_hole_comp_vertices(), pose)
        else:
            return transform_polys_wrt_pose_2d(self.get_peg_comp_vertices(), pose)

    def get_object_pose(self, index, target=False):
        if target:
            pos = self.target_positions[index]
            angle = self.target_angles[index]
        else:
            body: pymunk.Body = self.obj_list[index][0]
            pos = body.position
            angle = body.angle
        return [pos.x, pos.y, angle]

    def get_current_state(self):
        obj_keypoints = self.get_all_object_keypoints(peg_only=False)[:2]
        state = np.concatenate(
            (obj_keypoints[0], obj_keypoints[1], np.array([self.pusher_body.position]), np.array([self.velocity])),
            axis=0,
        )
        return state.flatten()

    def get_hole_comp_vertices(self, flatten=False):
        """
        Get the vertices of the hole component of a hole in local coordinates.
        """
        unit_size = self.unit_size
        center_base, center_side1, center_side2, com = get_hole_comp_center(unit_size)
        center_base = np.array(center_base) - np.array(com)
        center_side1 = np.array(center_side1) - np.array(com)
        center_side2 = np.array(center_side2) - np.array(com)
        base_vertices = get_rect_vertices(3 * unit_size, unit_size)
        base_vertices = base_vertices + center_base
        if flatten:
            base_vertices = base_vertices.flatten()
        side1_vertices = get_rect_vertices(unit_size, unit_size)
        side1_vertices = side1_vertices + center_side1
        if flatten:
            side1_vertices = side1_vertices.flatten()
        side2_vertices = get_rect_vertices(unit_size, unit_size)
        side2_vertices = side2_vertices + center_side2
        if flatten:
            side2_vertices = side2_vertices.flatten()
        return base_vertices.tolist(), side1_vertices.tolist(), side2_vertices.tolist()

    def get_peg_comp_vertices(self, flatten=False):
        """
        Get the vertices of the peg component of a peg in local coordinates.
        """
        unit_size = self.unit_size
        hp_ratio = self.hp_ratio
        center_base, center_peg, com = get_peg_comp_center(unit_size, hp_ratio)
        center_base = np.array(center_base) - np.array(com)
        center_peg = np.array(center_peg) - np.array(com)
        base_vertices = get_rect_vertices(3 * unit_size, unit_size)
        base_vertices = base_vertices + center_base
        if flatten:
            base_vertices = base_vertices.flatten()
        peg_vertices = get_rect_vertices(unit_size * hp_ratio, unit_size)
        peg_vertices = peg_vertices + center_peg
        if flatten:
            peg_vertices = peg_vertices.flatten()
        return base_vertices.tolist(), peg_vertices.tolist()


""" 
some helper functions
"""


def parse_merging_pose(merging_pose, param_dict, distance=0, pose_mode=0, noisy=False):
    # assume the hole object is at the origin
    # if noisy, the peg object will have a random offset
    unit_size, hp_ratio = param_dict["unit_size"], param_dict["hp_ratio"]
    merging_pose = np.array(merging_pose)
    merging_angle = merging_pose[2]
    noise_bound = 20
    _, _, _, [x_m, y_m] = get_hole_comp_center(unit_size)
    _, _, [x_g_peg, y_g_peg] = get_peg_comp_center(unit_size, hp_ratio)
    delta_y = 3 * unit_size - y_g_peg - y_m
    if pose_mode == 0:
        shift_size = delta_y + distance
        offset = np.array(
            [
                [0, 0, 0],
                [-shift_size * math.sin(merging_angle), shift_size * math.cos(merging_angle), math.pi],
            ]
        )
        if noisy:
            offset[1] += np.array(
                [
                    rand_float(-noise_bound, noise_bound),
                    rand_float(-noise_bound, noise_bound),
                    rand_float(-math.pi, math.pi),
                ]
            )
    elif pose_mode == 1:
        shift_size = 2 * delta_y + distance
        offset = np.array(
            [
                [0, 0, 0],
                [-shift_size * math.sin(merging_angle), shift_size * math.cos(merging_angle), 0],
            ]
        )
        if noisy:
            offset[1] += np.array(
                [
                    rand_float(-noise_bound, noise_bound),
                    rand_float(-noise_bound, noise_bound),
                    rand_float(-math.pi, math.pi),
                ]
            )
    else:
        shift_size = 2 * delta_y + distance
        offset = np.array(
            [
                [0, 0, 0],
                [shift_size * rand_float(-1, 1), shift_size * rand_float(-1, 1), rand_float(0, math.pi * 2)],
            ]
        )
    return (merging_pose + offset).tolist()


def generate_init_target_states(init_poses, target_poses, param_dict, include_com=False):
    init_states = get_keypoints_from_pose_hole(init_poses[0], param_dict, include_com)
    init_states = np.concatenate((init_states, get_keypoints_from_pose_peg(init_poses[1], param_dict, include_com)), axis=0)
    target_states = get_keypoints_from_pose_hole(target_poses[0], param_dict, include_com)
    target_states = np.concatenate((target_states, get_keypoints_from_pose_peg(target_poses[1], param_dict, include_com)), axis=0)
    return init_states.flatten(), target_states.flatten()


def get_hole_comp_center(unit_size):
    x_b, y_b = unit_size * 1.5, unit_size / 2
    x_s1, y_s1 = unit_size / 2, unit_size * 1.5
    x_s2, y_s2 = unit_size * 2.5, unit_size * 1.5
    m_b, m_s1, m_s2 = 3, 1, 1
    x_m, y_m = calculate_com([x_b, x_s1, x_s2], [y_b, y_s1, y_s2], [m_b, m_s1, m_s2])
    return [x_b, y_b], [x_s1, y_s1], [x_s2, y_s2], [x_m, y_m]


def get_keypoints_from_pose_hole(pose, param_dict, include_com):
    unit_size = param_dict["unit_size"]
    pos = Vec2d(pose[0], pose[1])
    angle = pose[2]
    [x_b, y_b], [x_s1, y_s1], [x_s2, y_s2], [x_m, y_m] = get_hole_comp_center(unit_size)
    com = Vec2d(x_m, y_m)

    # Define offsets for the L shape key points relative to the L shape's center
    offsets = [
        Vec2d(x_s1, y_s1),  # Center of left unit block of the side
        Vec2d(x_s1, y_b),  # Center of left unit block of the base
        Vec2d(x_b, y_b),  # Center of right unit block of the base
        Vec2d(x_s2, y_b),  # Center of right unit block of the base
        Vec2d(x_s2, y_s2),  # Center of right unit block of the side
    ]
    if include_com:
        offsets.append(Vec2d(0, 0))
    # Calculate the global position of each keypoint
    keypoints = []
    for offset in offsets:
        offset = offset - com
        rotated_offset = offset.rotated(angle)
        keypoints.append(pos + rotated_offset)

    return np.array(keypoints)


def get_pose_from_keypoints_hole(keypoints, unit_size):
    [x_b, y_b], [x_s1, y_s1], [x_s2, y_s2], [x_m, y_m] = get_hole_comp_center(unit_size)
    model_points = np.array(
        [
            [x_s1, y_s1],
            [x_s1, y_b],
            [x_b, y_b],
            [x_s2, y_b],
            [x_s2, y_s2],
        ]
    ) - np.array([x_m, y_m])

    return keypoints_to_pose_2d_SVD(model_points, keypoints)


def get_peg_comp_center(unit_size, hp_ratio):
    x_b, y_b = unit_size * 1.5, unit_size / 2
    x_p, y_p = unit_size * 1.5, unit_size * 1.5
    m_b, m_p = 3, 1 * hp_ratio
    x_m, y_m = calculate_com([x_b, x_p], [y_b, y_p], [m_b, m_p])
    return [x_b, y_b], [x_p, y_p], [x_m, y_m]


def get_keypoints_from_pose_peg(pose, param_dict, include_com):
    unit_size, hp_ratio = param_dict["unit_size"], param_dict["hp_ratio"]
    pos = Vec2d(pose[0], pose[1])
    angle = pose[2]
    half_unit_size = unit_size / 2
    [x_b, y_b], [x_p, y_p], [x_m, y_m] = get_peg_comp_center(unit_size, hp_ratio)
    com = Vec2d(x_m, y_m)

    # Define offsets for the peg key points relative to botter left corner of the L shape
    offsets = [
        Vec2d(half_unit_size, y_b),  # Center of left unit block of the base
        Vec2d(x_b, y_b),  # Center of right unit block of the base
        Vec2d(2.5 * half_unit_size, y_b),  # Center of right unit block of the base
        Vec2d(x_p, y_p),  # Center of unit block of the peg
    ]
    if include_com:
        offsets.append(Vec2d(0, 0))
    # Calculate the global position of each keypoint
    keypoints = []
    for offset in offsets:
        offset = offset - com
        rotated_offset = offset.rotated(angle)
        keypoints.append(pos + rotated_offset)

    return np.array(keypoints)


def get_pose_from_keypoints_peg(keypoints, unit_size, hp_ratio):
    half_unit_size = unit_size / 2
    [x_b, y_b], [x_p, y_p], [x_m, y_m] = get_peg_comp_center(unit_size, hp_ratio)
    model_points = np.array(
        [
            [half_unit_size, y_b],
            [x_b, y_b],
            [2.5 * half_unit_size, y_b],
            [x_p, y_p],
        ]
    ) - np.array([x_m, y_m])

    return keypoints_to_pose_2d_SVD(model_points, keypoints)


def get_keypoints_from_pose(pose, param_dict, obj_type):
    assert obj_type in ["hole", "peg"]
    unit_size, hp_ratio = param_dict["unit_size"], param_dict["hp_ratio"]
    if obj_type == "hole":
        return get_keypoints_from_pose_hole(pose, param_dict)
    else:
        return get_keypoints_from_pose_peg(pose, param_dict)


def get_pose_from_keypoints(keypoints, param_dict):
    unit_size, hp_ratio = param_dict["unit_size"], param_dict["hp_ratio"]
    shape = keypoints.shape
    # w/ or w/o batch
    assert len(shape) == 2 or len(shape) == 3
    assert shape[-2] == 5 or shape[-2] == 4
    if shape[-2] == 5:
        return get_pose_from_keypoints_hole(keypoints, unit_size)
    else:
        return get_pose_from_keypoints_peg(keypoints, unit_size, hp_ratio)


def get_offests_w_origin(param_dict):
    unit_size, hp_ratio = param_dict["unit_size"], param_dict["hp_ratio"]
    # hardcode as the top left corner of the left side of hole, deponds on the .obj file
    _, _, _, [x_g_h, y_g_h] = get_hole_comp_center(unit_size)
    com_w_origin_h = np.array([x_g_h, y_g_h - 2 * unit_size])
    offest_w_com_h = get_keypoints_from_pose_hole([0, 0, 0], param_dict)
    # hardcode as the top left corner of the base of peg, deponds on the .obj file
    _, _, [x_g_p, y_g_p] = get_peg_comp_center(unit_size, hp_ratio)
    com_w_origin_p = np.array([x_g_p, y_g_p - 2 * unit_size])
    offest_w_com_p = get_keypoints_from_pose_peg([0, 0, 0], param_dict)
    return [offest_w_com_h + com_w_origin_h, offest_w_com_p + com_w_origin_p]


if __name__ == "__main__":
    unit_size = 20
    hp_ratio = 0.8
    param_dict = {
        "save_img": True,
        "enable_vis": True,
        "unit_size": unit_size,
        "hp_ratio": hp_ratio,
        "fix_hole": True,
        "pusher_size": 5,
    }
    # init_poses = [[[250,250,math.radians(45)], [150,150,math.radians(-45)]]]
    init_poses = [parse_merging_pose([250, 250, math.radians(45)], param_dict, distance=50, pose_mode=1)]
    merging_pose = [350, 350, math.radians(135)]
    target_poses = [parse_merging_pose(merging_pose, param_dict, distance=0, pose_mode=0)]

    sim = HP_Sim(
        param_dict=param_dict,
        init_poses=init_poses,
        target_poses=target_poses,
    )
    # [[250,250,math.radians(45)], [150,150,math.radians(-45)]]
    sim.render()
    print(sim.get_all_object_positions())
    print(sim.get_all_object_keypoints())
    print(sim.get_current_state())
    print(sim.get_all_object_keypoints(target=True))
    for i in range(5):
        sim.render()
        time.sleep(0.5)
