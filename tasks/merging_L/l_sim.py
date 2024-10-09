import sys, os
import pymunk
from pymunk import Vec2d
import math
import numpy as np
import random
import time

import pymunk
from pymunk import Vec2d
import math
import numpy as np
import random
import sys, os

sys.path.append(os.getcwd())
from base_sim import Base_Sim
from others.helper import *

""" 
main class for the L-shaped block merging task
"""


class L_Sim(Base_Sim):
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
        unit_size, leg_shape, foot_shape, pusher_size = (
            param_dict["unit_size"],
            param_dict["leg_shape"],
            param_dict["foot_shape"],
            param_dict["pusher_size"],
        )
        self.unit_size = unit_size
        self.leg_size = (leg_shape[0] * unit_size, leg_shape[1] * unit_size)
        self.foot_size = (foot_shape[0] * unit_size, foot_shape[1] * unit_size)
        self.leg_shape = leg_shape
        self.foot_shape = foot_shape
        self.create_world(init_poses, pusher_pos)

    def create_object(self, id, pose=None):
        """
        Create a single L-shaped piece by defining its shapes, mass, etc., with the leg above the foot and both aligned to the left.
        """
        color = self.object_colors[id % len(self.object_colors)]
        foot_size, leg_size = self.foot_size, self.leg_size
        mass = self.obj_mass  # Define the mass of the L shape

        # Get vertices for the leg and foot
        leg_vertices, foot_vertices = self.get_l_comp_vertices()

        # Assuming equal mass distribution between the leg and the foot
        foot_square = foot_size[0] * foot_size[1]
        leg_square = leg_size[0] * leg_size[1]
        total_square = foot_square + leg_square
        leg_moment = pymunk.moment_for_poly(mass * leg_square / total_square, leg_vertices)
        foot_moment = pymunk.moment_for_poly(mass * foot_square / total_square, foot_vertices)

        # Total moment for the L-shape
        moment = leg_moment + foot_moment

        body = pymunk.Body(mass, moment)
        if pose is None:
            body.angle = random.random() * math.pi * 2
            body.position = Vec2d(random.randint(200, 300), random.randint(200, 300))
        else:
            body.angle = pose[2]
            body.position = Vec2d(pose[0], pose[1])

        leg = pymunk.Poly(body, leg_vertices)
        leg.color = color
        leg.elasticity = self.elasticity
        leg.friction = self.friction

        foot = pymunk.Poly(body, foot_vertices)
        foot.color = color
        foot.elasticity = self.elasticity
        foot.friction = self.friction
        return body, [leg, foot]

    def get_object_keypoints(self, index, target=False):
        return get_keypoints_from_pose(self.get_object_pose(index, target), self.param_dict, self.include_com)

    def get_object_vertices(self, index, target=False):
        return transform_polys_wrt_pose_2d(self.get_l_comp_vertices(), self.get_object_pose(index, target))

    def gen_vertices_from_pose(self, pose):
        return transform_polys_wrt_pose_2d(self.get_l_comp_vertices(), pose)

    def get_current_state(self):
        obj_keypoints = self.get_all_object_keypoints()[:2]
        state = np.concatenate(
            (obj_keypoints[0], obj_keypoints[1], np.array([self.pusher_body.position]), np.array([self.velocity])),
            axis=0,
        )
        return state.flatten()

    def get_l_comp_vertices(self, flatten=False):
        """
        Get the vertices of the leg and foot of the L shape.
        """
        leg_size, foot_size = self.leg_size, self.foot_size
        center_leg, center_foot, com = get_l_comp_centers(leg_size, foot_size)
        center_leg = np.array(center_leg) - np.array(com)
        center_foot = np.array(center_foot) - np.array(com)
        leg_vertices = get_rect_vertices(*leg_size)
        leg_vertices = leg_vertices + center_leg
        if flatten:
            leg_vertices = leg_vertices.flatten()
        foot_vertices = get_rect_vertices(*foot_size)
        foot_vertices = foot_vertices + center_foot
        if flatten:
            foot_vertices = foot_vertices.flatten()
        return leg_vertices.tolist(), foot_vertices.tolist()


""" 
some helper functions
"""


def parse_merging_pose(merging_pose, param_dict, distance=0, pose_mode=0, noisy=False):
    unit_size, leg_shape, foot_shape = (
        param_dict["unit_size"],
        param_dict["leg_shape"],
        param_dict["foot_shape"],
    )
    assert leg_shape[1] == foot_shape[1]
    merging_pose = np.array(merging_pose)
    merging_angle = merging_pose[2]
    leg_size = np.array(leg_shape) * unit_size
    foot_size = np.array(foot_shape) * unit_size
    w_l, h_l = leg_size
    w_f, h_f = foot_size
    noise = 20
    _, _, [x_m, y_m] = get_l_comp_centers(leg_size, foot_size)
    if pose_mode == 0 or pose_mode == 1:
        x_c, y_c = (w_l + w_f) / 2, h_f
        delta_x = x_c - x_m + distance
        delta_y = y_c - y_m
        offset = np.array(
            [
                [
                    -delta_x * math.cos(merging_angle) + delta_y * math.sin(merging_angle),
                    -delta_x * math.sin(merging_angle) - delta_y * math.cos(merging_angle),
                    0,
                ],
                [
                    delta_x * math.cos(merging_angle) - delta_y * math.sin(merging_angle),
                    delta_x * math.sin(merging_angle) + delta_y * math.cos(merging_angle),
                    math.pi,
                ],
            ]
        )
        if noisy:
            offset += np.array(
                [
                    [rand_float(-noise, noise), rand_float(-noise, noise), rand_float(0, math.pi * 2)],
                    [rand_float(-noise, noise), rand_float(-noise, noise), rand_float(-math.pi, math.pi)],
                ]
            )
            offset += np.array(
                [
                    [rand_float(-noise, noise), rand_float(-noise, noise), rand_float(0, math.pi * 2)],
                    [rand_float(-noise, noise), rand_float(-noise, noise), rand_float(-math.pi, math.pi)],
                ]
            )
        if pose_mode == 1:
            offset = offset[::-1]
    elif pose_mode == 2 or pose_mode == 3:
        x_c, y_c = (w_l + w_f) / 2, h_f
        delta_x = x_c - x_m + distance
        delta_y = y_c - y_m
        offset = np.array(
            [
                [
                    -delta_x * math.cos(merging_angle) + delta_y * math.sin(merging_angle),
                    -delta_x * math.sin(merging_angle) - delta_y * math.cos(merging_angle),
                    0,
                ],
                [
                    delta_x * math.cos(merging_angle) - delta_y * math.sin(merging_angle),
                    delta_x * math.sin(merging_angle) + delta_y * math.cos(merging_angle),
                    0.5 * math.pi,
                ],
            ]
        )
        if pose_mode == 3:
            offset = offset[::-1]
    else:
        half_height = (h_l + h_f + distance) / 2
        offset = np.array(
            [
                [
                    -half_height * math.sin(merging_angle) * rand_float(0, 1),
                    half_height * math.cos(merging_angle) * rand_float(0, 1),
                    rand_float(0, 2 * math.pi),
                ],
                [
                    half_height * math.sin(merging_angle) * rand_float(0, 1),
                    -half_height * math.cos(merging_angle) * rand_float(0, 1),
                    rand_float(0, 2 * math.pi),
                ],
            ]
        )
        # # # random exachange the order of the two L shapes
        # if random.randint(0, 1) == 0:
        #     offset = offset[::-1]
    return (merging_pose + offset).tolist()


def generate_init_target_states(init_poses, target_poses, param_dict, include_com=False):
    # init_poses, target_poses: list of (x,y,angle)
    init_states = get_keypoints_from_pose(init_poses[0], param_dict)
    init_states = np.concatenate((init_states, get_keypoints_from_pose(init_poses[1], param_dict, include_com)), axis=0)
    target_states = get_keypoints_from_pose(target_poses[0], param_dict)
    target_states = np.concatenate((target_states, get_keypoints_from_pose(target_poses[1], param_dict, include_com)), axis=0)
    return init_states.flatten(), target_states.flatten()


def get_l_comp_centers(leg_size, foot_size):
    """
    Get the center of the leg, foot and the L shape.
    """
    w_l, h_l = leg_size
    w_f, h_f = foot_size
    x_l, y_l = w_l / 2, h_f + h_l / 2
    x_f, y_f = w_f / 2, h_f / 2
    m_l, m_f = w_l * h_l, w_f * h_f
    x_m, y_m = calculate_com([x_l, x_f], [y_l, y_f], [m_l, m_f])
    return [x_l, y_l], [x_f, y_f], [x_m, y_m]


def get_keypoints_from_pose(pose, param_dict, include_com=False):
    unit_size, leg_shape, foot_shape = (
        param_dict["unit_size"],
        param_dict["leg_shape"],
        param_dict["foot_shape"],
    )
    pos = Vec2d(pose[0], pose[1])
    angle = pose[2]
    half_unit_size = unit_size / 2
    leg_size = np.array(leg_shape) * unit_size
    foot_size = np.array(foot_shape) * unit_size
    w_l, h_l = leg_size
    w_f, h_f = foot_size
    [x_l, y_l], [x_f, y_f], [x_m, y_m] = get_l_comp_centers(leg_size, foot_size)
    com = Vec2d(x_m, y_m)

    # Define offsets for the L shape key points relative to bottom left corner of the L shape
    offsets = [
        # Vec2d(w_l - half_unit_size, y_l),  # Center of right unit block of the leg
        Vec2d(half_unit_size, y_l+half_unit_size),  # top Center of left unit block of the leg
        Vec2d(half_unit_size, y_f),  # Center of left unit block of the foot
        # Vec2d(w_f/2, y_f), 
        Vec2d(w_f, y_f),  # right Center of right unit block of the foot
    ]
    # offsets = [
    #     # Vec2d(w_l - half_unit_size, y_l),  # Center of right unit block of the leg
    #     Vec2d(half_unit_size, y_l),  # top Center of left unit block of the leg
    #     Vec2d(half_unit_size, y_f),  # Center of left unit block of the foot
    #     # Vec2d(w_f/2, y_f), 
    #     Vec2d(w_f-half_unit_size, y_f),  # right Center of right unit block of the foot
    # ]
    if include_com:
        offsets.append(Vec2d(0, 0))
    # Calculate the global position of each keypoint
    keypoints = []
    for offset in offsets:
        offset = offset - com
        rotated_offset = offset.rotated(angle)
        keypoints.append(pos + rotated_offset)
    return np.array(keypoints)


def get_pose_from_keypoints(keypoints, param_dict):
    """
    Get the position and angle of COM of the L shape from absolute positions of its keypoints.
    it is a reverse function of get_keypoints_from_pose
    """
    unit_size, leg_shape, foot_shape = (
        param_dict["unit_size"],
        param_dict["leg_shape"],
        param_dict["foot_shape"],
    )
    half_unit_size = unit_size / 2
    leg_size = np.array(leg_shape) * unit_size
    foot_size = np.array(foot_shape) * unit_size
    w_l, h_l = leg_size
    w_f, h_f = foot_size
    [x_l, y_l], [x_f, y_f], [x_m, y_m] = get_l_comp_centers(leg_size, foot_size)

    # Define model points for the L shape key points relative to COM
    model_points = np.array(
        [
            # [w_l - half_unit_size, y_l],  # Center of right unit block of the leg
            [half_unit_size, y_l],  # Center of left unit block of the leg
            [half_unit_size, y_f],  # Center of left unit block of the foot
            # [w_f/2, y_f],  
            [w_f - half_unit_size, y_f],  # Center of right unit block of the foot
        ]
    ) - np.array([x_m, y_m])

    return keypoints_to_pose_2d_SVD(model_points, keypoints)


def get_offests_w_origin(param_dict):
    unit_size, leg_shape, foot_shape = (
        param_dict["unit_size"],
        param_dict["leg_shape"],
        param_dict["foot_shape"],
    )
    # hardcode as the top left corner of the leg, deponds on the .obj file
    _, _, [x_g, y_g] = get_l_comp_centers(leg_shape, foot_shape)
    com_w_origin = np.array([x_g, y_g - (leg_shape[1] + foot_shape[1])]) * unit_size
    offest_w_com = get_keypoints_from_pose([0, 0, 0], param_dict)
    return offest_w_com + com_w_origin


if __name__ == "__main__":
    unit_size = 20
    leg_shape = (1, 1)
    foot_shape = (3, 1)
    pusher_size = 5
    param_dict = {
        "unit_size": unit_size,
        "leg_shape": leg_shape,
        "foot_shape": foot_shape,
        "pusher_size": pusher_size,
        "save_img": False,
        "enable_vis": True,
    }
    seed = 0
    keypoints = get_keypoints_from_pose([250, 250, math.radians(45)], param_dict)
    print(keypoints)
    pose = get_pose_from_keypoints(keypoints, param_dict)
    print(pose)
    # num_test = 1
    # random.seed(seed)
    # init_pusher_pos_list = [np.array([random.randint(100, 150), random.randint(100, 150)]) for i in range(num_test)]
    # random.seed(seed)
    # init_pose_list = [
    #     np.array([random.randint(170, 200), random.randint(170, 200), math.radians(random.randint(90, 180) * 0)])
    #     for i in range(num_test)
    # ]
    # random.seed(seed)
    # target_pose_list = [
    #     np.array([random.randint(200, 230), random.randint(200, 230), math.radians(random.randint(0, 90) * 1)])
    #     for i in range(num_test)
    # ]
    # init_pusher_pos = init_pusher_pos_list[0]
    # init_poses = parse_merging_pose(init_pose_list[0], param_dict, 10, 0)
    # init_poses[1][2] -= math.pi * 0.5
    # target_poses = parse_merging_pose(target_pose_list[0], param_dict, 0, 0)
    # sim = L_Sim(
    #     param_dict=param_dict,
    #     init_poses=init_poses,
    #     target_poses=target_poses,
    #     pusher_pos=init_pusher_pos,
    # )
    # # [[250,250,math.radians(45)], [150,150,math.radians(-45)]]
    # sim.render()
    # print(sim.get_all_object_positions())
    # print(sim.get_all_object_keypoints())
    # print(sim.get_current_state())
    # print(sim.get_all_object_keypoints(target=True))
    # while True:
    #     sim.render()
    #     input("press enter to exit")
    #     exit()
