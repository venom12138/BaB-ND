import torch
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
import random
import math
import re
import cv2
from PIL import Image
from io import TextIOWrapper
import os, psutil
from scipy.stats import truncnorm

"""
helper functions for parsing object geometry
"""


def format_time_tag(time_slot):
    # time_slot: list of integers representing the time slot
    print(f"time_slot: {time_slot}")
    if len(time_slot) == 3:
        time_tag = "_{}_{}_{}".format(*time_slot)
    elif len(time_slot) == 4:
        time_tag = "_{}_{}_{}_{}".format(*time_slot)
    else:
        time_tag = ""
    return time_tag


def save_gif(images, gif_name):
    pil_images = [Image.fromarray((cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))) for img in images]
    pil_images[0].save(
        gif_name,
        save_all=True,
        append_images=pil_images[1:],
        duration=200,  # Duration of each frame in milliseconds
        loop=0,  # Loop forever
    )
    print(f"Save gif to {gif_name}")


def transform_polys_wrt_pose_2d(poly_list, pose):
    # poly_list: list of 2D polygons, each represented by a list of vertices
    x, y, angle = pose
    translation_vector = np.array([x, y])
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed_poly_list = []
    for vertices in poly_list:
        transformed_vertices = np.dot(vertices, rotation_matrix.T) + translation_vector
        transformed_poly_list.append(transformed_vertices)

    return transformed_poly_list


def rotate_state(state, seed=None):
    """
    Rotate the state by some degrees.

    Parameters:
    - state: A array of batch of 2D coordinates, shape: (B, ..., n*2)

    Returns:
    - state: A array of batch of 2D coordinates rotated by some degrees, shape: (B, ..., n*2)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.tensor(state, dtype=torch.float32, device=device)
    shape = state.shape
    B = state.shape[0]
    state = state.view(B, -1, 2)
    if seed is not None:
        reset_seed(seed)
    angles_rad = torch.rand(B, device=device) * 2 * torch.pi
    if seed is not None:
        reset_seed(seed)
    cos_vals = torch.cos(angles_rad).unsqueeze(1).unsqueeze(2)
    sin_vals = torch.sin(angles_rad).unsqueeze(1).unsqueeze(2)
    rotation_matrices = torch.stack([cos_vals, -sin_vals, sin_vals, cos_vals], dim=-1).reshape(B, 2, 2)

    rotated_state = torch.einsum("bij,bjk->bik", state, rotation_matrices)
    return rotated_state.view(*shape).detach().cpu().numpy()


def keypoints_to_pose_2d_SVD(kp1, kp2):
    """
    Calculate the 2D pose transformation from keypoints using SVD supporting batch processing with torch tensors.

    Parameters:
    - kp1: 2D keypoints in the original frame. Shape: (n, 2) array
    - kp2: 2D keypoints in the target frame. Shape: (batch_size, n, 2) tensor or (n, 2) array

    Returns:
    - pose: 2D pose transformations for each batch, Shape: (batch_size, 3) or (3,)
            each containing 2D position (x, y) and rotation angle in radians.
    """
    # Convert inputs to torch.Tensor if they are numpy.ndarray
    original_is_ndarray = False
    if isinstance(kp2, np.ndarray):
        original_is_ndarray = True
        kp1 = torch.tensor(kp1, dtype=torch.float32)
        kp2 = torch.tensor(kp2, dtype=torch.float32)
    elif isinstance(kp2, torch.Tensor):
        kp1 = torch.tensor(kp1, dtype=torch.float32, device=kp2.device)
    kp1 = kp1.unsqueeze(0)
    original_in_batch = True
    if kp2.dim() == 2:
        original_in_batch = False
        kp2 = kp2.unsqueeze(0)

    # Center the keypoints
    kp1_center = kp1.mean(dim=1, keepdim=True)
    kp2_center = kp2.mean(dim=1, keepdim=True)
    kp1_centered = kp1 - kp1_center
    kp2_centered = kp2 - kp2_center

    # Compute the covariance matrix for each batch
    H = torch.matmul(kp1_centered.transpose(-2, -1), kp2_centered)

    # Perform SVD
    U, S, Vt = torch.linalg.svd(H, full_matrices=True)

    # Compute the rotation matrix
    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))
    # Ensure the determinant of the rotation matrix is 1 (correcting for reflection if necessary)
    det_R = torch.linalg.det(R)
    reflection_correction = torch.diag_embed(torch.ones(R.shape[:-1], device=R.device))
    reflection_correction[:, -1, -1] = torch.sign(det_R)
    R = torch.matmul(Vt.transpose(-2, -1), reflection_correction.matmul(U.transpose(-2, -1)))

    # Compute the translation vector
    t = (kp2_center - torch.matmul(R, kp1_center.transpose(-2, -1)).transpose(-2, -1)).squeeze(1)

    # Compute the rotation angle in radians
    theta = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    # Concatenate translation and rotation to form the pose
    pose = torch.cat((t, theta.unsqueeze(-1)), dim=1)

    # Convert back to numpy.ndarray if the original input was ndarray
    if not original_in_batch:
        pose = pose.squeeze(0)
    if original_is_ndarray:
        pose = pose.numpy()

    return pose


def in_range(x, lo, hi):
    return x >= lo and x <= hi


def get_rect_vertices(w, h):
    w /= 2
    h /= 2
    return np.array([[-w, -h], [w, -h], [w, h], [-w, h]])


def calculate_com(x_i, y_i, m_i):
    """
    Calculate the center of mass (CoM) for a composite object based on
    the masses (or areas) and coordinates of individual parts.

    Parameters:
    - x_i: List or array of x-coordinates of the centers of mass of the components.
    - y_i: List or array of y-coordinates of the centers of mass of the components.
    - m_i: List or array of masses (or areas) of the components.

    Returns:
    - (C_x, C_y): A tuple representing the x and y coordinates of the composite CoG.
    """
    total_mass = sum(m_i)
    C_x = sum(m * x for m, x in zip(m_i, x_i)) / total_mass
    C_y = sum(m * y for m, y in zip(m_i, y_i)) / total_mass
    return (C_x, C_y)


"""
helper functions for experiments
"""


def get_model_file_list(model_dir, model_type):
    model_file_list = []
    for filename in os.listdir(model_dir):
        if filename.endswith(".pth") and "state_dict" not in filename and "pruned" not in filename:
            model_file_list.append(filename)
    if model_type == "best":
        model_file_list = [x for x in model_file_list if "best" in x]
    elif model_type == "before":
        model_file_list = [x for x in model_file_list if "best" not in x]
    else:
        raise NotImplementedError
    model_file_list.sort()
    return model_file_list


def reset_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_task_spec_dict(config):
    data_config = config["data"]
    task_name = config["task_name"]
    if "merging_L" in task_name:
        assert data_config["obj_num"] == 2
        unit_size, leg_shape, foot_shape, pusher_size = (
            data_config["unit_size"],
            data_config["leg_shape"],
            data_config["foot_shape"],
            data_config["pusher_size"],
        )
        task_spec_dict = {
            "unit_size": unit_size,
            "leg_shape": leg_shape,
            "foot_shape": foot_shape,
            "pusher_size": pusher_size,
        }
    elif "pushing_T_latent" in task_name:
        assert data_config["obj_num"] == 1
        stem_size, bar_size, pusher_size = data_config["stem_size"], data_config["bar_size"], data_config["pusher_size"]
        task_spec_dict = {"stem_size": stem_size, "bar_size": bar_size, "pusher_size": pusher_size}
    elif "pushing_T" in task_name:
        assert data_config["obj_num"] == 1
        stem_size, bar_size, pusher_size = data_config["stem_size"], data_config["bar_size"], data_config["pusher_size"]
        task_spec_dict = {"stem_size": stem_size, "bar_size": bar_size, "pusher_size": pusher_size}
    elif "box_pushing" in task_name:
        assert data_config["obj_num"] == 1
        box_size, pusher_size = data_config["box_size"], data_config["pusher_size"]
        task_spec_dict = {"box_size": box_size, "pusher_size": pusher_size}
    elif "inserting" in task_name:
        assert data_config["obj_num"] == 2
        unit_size, hp_ratio, fix_hole, pusher_size = (
            data_config["unit_size"],
            data_config["hp_ratio"],
            data_config["fix_hole"],
            data_config["pusher_size"],
        )
        task_spec_dict = {
            "unit_size": unit_size,
            "hp_ratio": hp_ratio,
            "fix_hole": fix_hole,
            "pusher_size": pusher_size,
        }
    elif task_name == "obj_pile":
        obj_size, pusher_size = data_config["obj_size"], data_config["pusher_size"]
        task_spec_dict = {"obj_size": obj_size, "pusher_size": pusher_size, \
            "obj_num": (data_config["state_dim"])//2, "env_type": data_config["env_type"], \
            'classes': data_config['classes'], "push_single": data_config["push_single"]}
    elif task_name == "pushing_rope":
        task_spec_dict = {"obj": data_config["obj"], "wkspc_w": data_config["wkspc_w"], "headless": data_config["headless"], \
            "robot_type": data_config["robot_type"], "cont_motion": data_config["cont_motion"], "camera_view": data_config["camera_view"], \
            "gripper": data_config["gripper"], "grasp": data_config["grasp"], \
            "physics": data_config["physics"], "fps": data_config["fps"], "fps_number": data_config["fps_number"], \
            "max_nobj": data_config["max_nobj"], "state_dim": data_config["state_dim"], "action_dim": data_config["action_dim"], \
            "obj_size": 0.5, \
            } # random set obj size as 0.5
    elif task_name == "rope_3d":
        task_spec_dict = {"obj": data_config["obj"], "wkspc_w": data_config["wkspc_w"], "headless": data_config["headless"], \
            "robot_type": data_config["robot_type"], "cont_motion": data_config["cont_motion"], "camera_view": data_config["camera_view"], \
            "gripper": data_config["gripper"], "grasp": data_config["grasp"], \
            "physics": data_config["physics"], "fps": data_config["fps"], "fps_number": data_config["fps_number"], \
            "max_nobj": data_config["max_nobj"], "state_dim": data_config["state_dim"], "action_dim": data_config["action_dim"], \
            "obj_size": 0.5, "fix_rope_end": data_config["fix_rope_end"], "seed": config["seed"], \
            } # random set obj size as 0.5
    elif task_name == "reorientation":
        task_spec_dict = {"state_dim": data_config["state_dim"], \
            "action_dim": data_config["action_dim"], \
            "pusher_lo": data_config["pusher_lo"], \
            "pusher_hi": data_config["pusher_hi"], \
            "action_bound": data_config["action_bound"]} # random set obj size as 0.5
    else:
        raise NotImplementedError
    if task_name in ["merging_L", "pushing_T_latent", "pushing_T", "box_pushing", "inserting"]:
        task_spec_dict["obj_num"] = data_config["obj_num"]
        task_spec_dict["include_com"] = config["train"]["include_com"]
    task_spec_dict["task_name"] = task_name
    task_spec_dict["state_dim"] = data_config["state_dim"]
    task_spec_dict["save_img"] = data_config["gif"]
    task_spec_dict["enable_vis"] = data_config["visualizing"]
    task_spec_dict["img_state"] = False
    task_spec_dict["img_size"] = 0
    task_spec_dict["window_size"] = data_config.get("window_size", None)

    if "latent" in task_name:
        latent_config = config["latent"]
        task_spec_dict["img_state"] = latent_config["enable"]
        task_spec_dict["img_size"] = latent_config["img_size"]

    task_spec_dict["task_name"] = task_name
    return task_spec_dict


def _gen_pose_list(num_test, seed, x_bound, y_bound, theta_bound=None, theta_factor=1):
    shift = 0
    num_test+=shift
    random.seed(seed)
    if theta_bound is None:
        return [np.array([random.randint(*x_bound), random.randint(*y_bound)]) for i in range(num_test)][shift:]
    return [
        np.array(
            [
                random.randint(*x_bound),
                random.randint(*y_bound),
                math.radians(random.randint(*theta_bound) * theta_factor),
            ]
        )
        for i in range(num_test)
    ][shift:]

def _gen_pose_list_for_obj_pile(num_test, num_objs, seed, x_bound, y_bound, theta_bound=None, theta_factor=1):
    random.seed(seed)
    
    return [[np.array([random.randint(*x_bound), random.randint(*y_bound)]) for i in range(num_objs)] for _ in range(num_test)]


def generate_test_cases(seed, num_test, task_spec_dict, test_id, cost_mode=None):
    task_name = task_spec_dict["task_name"]
    test_id = int(test_id)
    # only used for latent task
    target_pusher_pos_list = None
    if "merging_L" == task_name:
        if test_id == 0:
            # init_pusher_pos_list = _gen_pose_list(num_test, seed, (100, 100), (100, 100), None)
            init_pusher_pos_list = _gen_pose_list(num_test, seed, (100, 150), (100, 150), None)
            init_pose_list = _gen_pose_list(num_test, seed, (180, 210), (170, 200), (90, 180), 0)
            target_pusher_pos_list = _gen_pose_list(num_test, seed, (170, 180), (170, 180), None)
            # target_pose_list = [np.array([250, 250, 0])]
            target_pose_list = _gen_pose_list(num_test, seed, (200, 220), (220, 250), (90, 180), 0)
        elif test_id == 1:
            init_pusher_pos_list = _gen_pose_list(num_test, seed, (300, 350), (120, 150), None)
            init_pose_list = _gen_pose_list(num_test, seed, (250, 280), (170, 200), (-45, 45))
            target_pusher_pos_list = _gen_pose_list(num_test, seed, (250, 260), (190, 200), None)
            target_pose_list = _gen_pose_list(num_test, seed, (200, 230), (220, 250), (-60, 60))
        elif test_id == 3:
            init_pusher_pos_list = _gen_pose_list(num_test, seed, (220, 220), (100, 100), None)
            # init_pose_list_1 =_gen_pose_list(num_test, seed, (180, 180), (250, 250), (135, 135), 1)
            init_pose_list_1 =_gen_pose_list(num_test, seed, (160, 160), (180, 180), (-90, -90), 1)
            init_pose_list_2 = _gen_pose_list(num_test, seed, (220, 220), (150, 150), (45, 45), 1)
            init_pose_list = [np.concatenate((init_pose_list_1[i], init_pose_list_2[i])) for i in range(num_test)]
            target_pusher_pos_list = _gen_pose_list(num_test, seed, (170, 180), (170, 180), None)
            # target_pose_list = [np.array([250, 250, 0])]
            target_pose_list = _gen_pose_list(num_test, seed, (180, 220), (200, 220), (-45, 45), 1)
        elif test_id == 4:
            init_pusher_pos_list = _gen_pose_list(num_test, seed, (230, 230), (140, 140), None)
            init_pose_list_1 =_gen_pose_list(num_test, seed, (180, 180), (130, 130), (-30, -30), 1)
            init_pose_list_2 = _gen_pose_list(num_test, seed, (300, 300), (230, 230), (135, 135), 1)
            init_pose_list = [np.concatenate((init_pose_list_1[i], init_pose_list_2[i])) for i in range(num_test)]
            target_pusher_pos_list = _gen_pose_list(num_test, seed, (170, 180), (170, 180), None)
            # target_pose_list = [np.array([250, 250, 0])]
            target_pose_list = _gen_pose_list(num_test, seed, (220, 250), (220, 250), (-45, -45), 0)
        elif test_id == 5:
            init_pusher_pos_list = _gen_pose_list(num_test, seed, (240, 240), (150, 150), None)
            init_pose_list_1 =_gen_pose_list(num_test, seed, (190, 190), (140, 140), (-30, -30), 1)
            init_pose_list_2 = _gen_pose_list(num_test, seed, (300, 300), (220, 220), (150, 150), 1)
            init_pose_list = [np.concatenate((init_pose_list_1[i], init_pose_list_2[i])) for i in range(num_test)]
            target_pusher_pos_list = _gen_pose_list(num_test, seed, (170, 180), (170, 180), None)
            # target_pose_list = [np.array([250, 250, 0])]
            target_pose_list = _gen_pose_list(num_test, seed, (220, 250), (220, 250), (-45, -45), 0)
        elif test_id == -2:
            init_pusher_pos_list = [np.array([50,50])]# _gen_pose_list(num_test, seed, (250, 250), (50, 75), None)
            init_pose_list = _gen_pose_list(num_test, seed, (225, 275), (125, 150), (-90, -90))
            # target_pusher_pos_list = _gen_pose_list(num_test, seed, (200, 210), (250, 260), None)
            target_pose_list = [np.array([200, 200, 0])]# _gen_pose_list(num_test, seed, (270, 300), (400, 400), (-90, -90))
        # elif test_id == 2:
        #     init_pusher_pos_list = _gen_pose_list(num_test, seed, (150, 200), (100, 120), None)
        #     init_pose_list = _gen_pose_list(num_test, seed, (170, 200), (170, 200), (-45, 45))
        #     target_pose_list = _gen_pose_list(num_test, seed, (300, 330), (320, 350), (-60, 60))
        # elif test_id == 3:
        #     init_pusher_pos_list = _gen_pose_list(num_test, seed, (250, 300), (120, 150), None)
        #     init_pose_list = _gen_pose_list(num_test, seed, (170, 200), (170, 200), (-45, 45))
        #     target_pose_list = _gen_pose_list(num_test, seed, (300, 330), (320, 350), (-60, 60))
        else:
            raise NotImplementedError
    elif "pushing_T" == task_name:
        if test_id == 0:
            init_pusher_pos_list = [np.array([385.0, 87.0, 0])]
            init_pose_list = [np.array([315.0, 80.0, -math.pi*0.5])]
            target_pusher_pos_list = [np.array([0.0, 0,0, 0,0])]
            target_pose_list = [np.array([120.0, 410.0, -math.pi*0.95])]
        elif test_id == 1:
            init_pusher_pos_list = [np.array([85.0, 421.0, 0])]
            init_pose_list = [np.array([162.0, 400.0, math.pi*0.25])]
            target_pusher_pos_list = [np.array([0.0, 0,0, 0,0])]
            target_pose_list = [np.array([120.0, 120.0, -math.pi*0.25])]
        elif test_id == 2:
            # init_pusher_pos_list = [np.array([387.0, 66.0, 0])]
            # init_pose_list = [np.array([315.0, 95.0, 1.23])]
            # target_pusher_pos_list = [np.array([0.0, 0,0, 0,0])]
            # target_pose_list = [np.array([100.0, 380.0, -2.356194490192345])]
            init_pusher_pos_list = [np.array([150.0, 50.0, 0])]
            init_pose_list = [np.array([150.0, 110.0, math.pi])]
            target_pusher_pos_list = [np.array([0.0, 0,0, 0,0])]
            target_pose_list = [np.array([200.0, 420.0, math.pi*1.5])]
        else:
            raise NotImplementedError(f"{test_id} not implemented")
    elif "pushing_T_latent" == task_name:
        init_pusher_pos_list = _gen_pose_list(num_test, seed, (180, 200), (170, 190), None)
        init_pose_list = _gen_pose_list(num_test, seed, (240, 250), (130, 150), (30, 60))
        target_pusher_pos_list = _gen_pose_list(num_test, seed, (200, 210), (250, 260), None)
        target_pose_list = _gen_pose_list(num_test, seed, (230, 250), (280, 300), (90, 120))
    elif "box_pushing" == task_name:
        init_pusher_pos_list = _gen_pose_list(num_test, seed, (220, 280), (100, 110), None)
        init_pose_list = _gen_pose_list(num_test, seed, (230, 270), (160, 180), (30, 60))
        target_pusher_pos_list = _gen_pose_list(num_test, seed, (170, 180), (220, 230), None)
        target_pose_list = _gen_pose_list(num_test, seed, (200, 300), (250, 300), (90, 120))
    elif "inserting" == task_name:
        init_pusher_pos_list = _gen_pose_list(num_test, seed, (100, 150), (100, 150), None)
        init_pose_list = _gen_pose_list(num_test, seed, (170, 200), (170, 200), (90, 180), 0)
        if task_spec_dict["fix_hole"]:
            target_pusher_pos_list = _gen_pose_list(num_test, seed, (140, 150), (140, 150), None)
            target_pose_list = init_pose_list.copy()
        else:
            target_pusher_pos_list = _gen_pose_list(num_test, seed, (170, 180), (170, 180), None)
            target_pose_list = _gen_pose_list(num_test, seed, (200, 230), (200, 230), (90, 180), 0)
    elif "box_pushing_latent" == task_name:
        init_pusher_pos_list = _gen_pose_list(num_test, seed, (220, 220), (100, 100), None)
        init_pose_list = _gen_pose_list(num_test, seed, (270, 270), (160, 160), (30, 60))
        target_pusher_pos_list = _gen_pose_list(num_test, seed, (250, 250), (250, 250), None)
        target_pose_list = _gen_pose_list(num_test, seed, (280, 300), (280, 300), (90, 120))
    elif task_name == "obj_pile":
        # corners = [[100,100], [200,200], [100, 300], [300, 100]]
        corners = [[300,300], [100,100], [100, 300], [300, 100]]
        # corners = [[125,125], [275,275], [100, 300], [300, 100]]
        num_objs = task_spec_dict["obj_num"]
        class_num = task_spec_dict['classes']
        # pusher_pos_range = [(120, 170), (170, 220), (270, 320), (220,270)]
        # init_pusher_pos_list =  _gen_pose_list(num_test, seed, \
        #     pusher_pos_range[np.random.randint(0, len(pusher_pos_range))], \
        #     pusher_pos_range[np.random.randint(0, len(pusher_pos_range))], None)
        init_pusher_pos_list = [np.array([10, 10]) for i in range(num_test)]
        # init_pose_list = _gen_pose_list_for_obj_pile(num_test, num_objs, seed, (150, 250), (150, 250), None)
        init_pose_list = _gen_pose_list_for_obj_pile(num_test, num_objs, seed, (125, 275), (125, 275), None)
        for i in range(num_test):
            init_pose_list[i][0] = np.array([200, 200])
        if cost_mode == 'target':
            target_pose_list = [[np.array(corners[i]) for i in range(class_num)] for _ in range(num_test)]
        else:
            target_pose_list = None
    elif task_name == "pushing_rope":
        corners = [[200,200], [600,600], [200, 600], [600, 200]]
        init_pusher_pos_list = [np.array([10, 10]) for i in range(num_test)]
        init_pose_list = [None for i in range(num_test)]
        target_pose_list = None
    elif task_name == "rope_3d":
        # corners = [[200,200], [600,600], [200, 600], [600, 200]]
        init_pusher_pos_list = [np.array([10, 10]) for i in range(num_test)]
        init_pose_list = [None for i in range(num_test)]
        target_pose_list = None
        max_test_cases = 20
        if test_id == -1:
            task_spec_dict["target_offest"] = [random.uniform(0, 0), random.uniform(0.05, 0.05)]
            task_spec_dict["obs_gap"] = random.uniform(0.15, 0.15)
            task_spec_dict["init_y_angle"] = random.uniform(50, 50)
        elif test_id >= 0 and test_id < max_test_cases:
            offset_target_y = np.linspace(0.1, 0.2, max_test_cases+1)
            offset_y_angle = np.linspace(53, 57, max_test_cases+1)
            task_spec_dict["target_offest"] = [0, offset_target_y[test_id]]
            task_spec_dict["obs_gap"] = 0.1
            task_spec_dict["init_y_angle"] = offset_y_angle[test_id]
        else:
            raise NotImplementedError(f"{test_id} not implemented")
    elif task_name == "reorientation":
        init_pusher_pos_list = [np.array([10, 10]) for i in range(num_test)]
        init_pose_list = [None for i in range(num_test)]
        target_pose_list = None
    else:
        raise NotImplementedError
    random.seed(seed)
    # target_pusher_pos_list is only used for latent task
    if task_name == "obj_pile":
        return init_pose_list, target_pose_list, init_pusher_pos_list, None
    return init_pose_list, target_pose_list, init_pusher_pos_list, target_pusher_pos_list

def convert_task_to_obj(task_name):
    if task_name == "merging_L":
        return "L"
    elif task_name == "pushing_T":
        return "T"
    elif task_name == "box_pushing":
        return "box"
    elif task_name == "inserting":
        return "HP"
    elif task_name == "obj_pile":
        return "obj_pile"
    else:
        raise NotImplementedError


"""
helper functions for generating data
"""


def get_truncated_normal(mean=0, sd=1, low=-10, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def gen_act(delta, scale=1, bound=30):
    if delta > 0:
        act = random.random() * bound * (scale + 1) - bound
        return act if act <= 0 or delta > bound else act / scale
    else:
        act = -random.random() * bound * (scale + 1) + bound
        return act if act >= 0 or delta < -bound else act / scale

def gen_start_pusher_pos(centers, spread_range):
    # centers: [B, 2]
    # spread_range: scalar
    mu = np.random.choice(centers, 1)
    pos = np.random.normal(loc=mu, scale=spread_range/2)
    return pos

def gen_act_pile(delta, scale=1, bound=30):
    if delta > 0:
        norm_mu = bound*(scale+1)/2 - 0.2 * bound
        norm_scale = bound / 3
        act = np.random.normal(loc=norm_mu, scale=norm_scale) # random.randint(-bound, int(bound * scale))
        return act if act <= 0 or delta > bound else act / scale
    else:
        norm_mu = -bound*(scale+1)/2 + 0.2 * bound
        norm_scale = bound / 3
        act = np.random.normal(loc=norm_mu, scale=norm_scale) # random.randint(int(-bound * scale), bound)
        return act if act >= 0 or delta < -bound else act / scale


"""
helper functions for running and parsing CROWN
"""


def print_prop(num, f: TextIOWrapper, act_seq_dim, action_lb, action_ub, big_M=1000):
    # action_lb, action_ub: scalar or torch.Tensor
    if not isinstance(action_lb, torch.Tensor):
        action_lb = torch.tensor(action_lb)
    if not isinstance(action_ub, torch.Tensor):
        action_ub = torch.tensor(action_ub)
    # if scalar, every dim of action has the same bounds, action_dim doesn't matter
    action_dim = len(action_lb)
    assert act_seq_dim % action_dim == 0 and action_dim == len(action_ub)
    f.write(f"; Property for step {num}\n\n")

    # declare constants
    for i in range(act_seq_dim):
        f.write(f"(declare-const X_{i} Real)\n")

    f.write("\n")
    for i in range(1):
        f.write(f"(declare-const Y_{i} Real)\n\n")

    # constraints
    f.write(f"; Input constraints:\n")
    for i in range(act_seq_dim):
        f.write(f"(assert (<= X_{i} {float(action_ub[i%action_dim])}))\n")
        f.write(f"(assert (>= X_{i} {float(action_lb[i%action_dim])}))\n\n")

    f.write(f"; Output constraints: (Y <= big_M: always sat, solve best solution)\n")
    f.write(f"(assert (<= Y_0 {big_M}))\n")


def parse_result(output_filename, verbose=False, output_lb=False):
    with open(output_filename, "r") as file:
        text = file.read()

    # For final results
    final_pattern = r"Final solution found in \d+:\s+Adv example: (\[.*?\])\s+Adv output: (\[.*?\])\s+best upper bound: ([+-]?\d+\.\d+)\s+Current \(lb-rhs\): ([+-]?\d+\.\d+)"
    final_match = re.search(final_pattern, text)

    # Initialize lists for intermediate results
    intermediate_feasible_sols = []
    intermediate_best_outputs = []

    if verbose:
        # For intermediate results (new solutions)
        iter_pattern = r"New solution found in \d+:\s+Adv example: (\[.*?\])\s+Adv output: (\[.*?\])"
        iter_matches = re.finditer(iter_pattern, text)

        for inter_match in iter_matches:
            intermediate_feasible_sols.append(np.array(eval(inter_match.group(1))))
            intermediate_best_outputs.append(np.array(eval(inter_match.group(2))))

    final_lb = None
    if final_match:
        feasible_sol = np.array(eval(final_match.group(1)))
        best_output = np.array(eval(final_match.group(2)))
        best_upper_bound = float(final_match.group(3))
        current_lb_rhs = float(final_match.group(4))
        if output_lb:
            final_lb = current_lb_rhs + best_upper_bound
    else:
        feasible_sol = None
        best_output = None

    return feasible_sol, best_output, np.array(intermediate_feasible_sols), np.array(intermediate_best_outputs), final_lb


"""
miscellaneous helper functions
"""


def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2
    print(f"Memory Usage: {mem} MB")
    return mem


def env_state_to_input(state, device):
    # :return (1, state_dim)
    return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)


"""
functions for visualization of MPPI reward curves. Now not used.
"""


def vis_res(res):
    eval_outputs = res["eval_outputs"]
    rew = [out["rewards"].mean().item() for out in eval_outputs]
    plt.plot(rew)
    plt.show()


def vis_all_res(all_res, image_name, save=False):
    """
    Visualize the mean reward across multiple timesteps and update iterations.

    Parameters:
    - all_res: List of 'res' dictionaries obtained from each timestep.
    """
    # Initialize an empty list to hold all reward curves
    all_rew = []

    for res in all_res:
        eval_outputs = res["eval_outputs"]
        # Compute the mean reward for each update iteration in the current timestep
        rew = [out["rewards"].mean().item() for out in eval_outputs]
        all_rew.append(rew)

    # Plotting
    n_timesteps = len(all_rew)
    colors = cm.viridis(np.linspace(0, 1, n_timesteps))
    for i, (rew, color) in enumerate(zip(all_rew, colors)):
        plt.plot(rew, color=color)

    # Adding colorbar
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=1, vmax=n_timesteps))
    sm.set_array([])
    plt.colorbar(sm, ticks=np.linspace(0, n_timesteps, 11), label="Timestep")

    plt.xlabel("Update Iteration")
    plt.ylabel("Mean Reward")
    plt.title("Mean Reward across All Timesteps")
    if save:
        plt.savefig(image_name, bbox_inches="tight")
    plt.show()


def vis_best_eval_output(all_res, image_name, save=False):
    """
    Visualize the best evaluation output (best reward) for each timestep.

    Parameters:
    - all_res: List of 'res' dictionaries obtained from each timestep.
    """
    # Initialize an empty list to hold the best evaluation output for each timestep
    best_eval_outputs = []

    for res in all_res:
        best_eval_output = res["best_eval_output"]
        best_eval_outputs.append(best_eval_output["rewards"].item())
        # best_eval_output = res.get('best_eval_output', None)
        # if best_eval_output is not None:
        #     best_eval_outputs.append(best_eval_output[0].item())  # Assuming best_eval_output is a 1-element tensor

    # Plotting
    plt.plot(best_eval_outputs, marker="o", linestyle="-")

    plt.xlabel("Timestep")
    plt.ylabel("Best Evaluation Output (Best Reward)")
    plt.title("Best Evaluation Output across All Timesteps")
    if save:
        plt.savefig(image_name, bbox_inches="tight")
    plt.show()

from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import imageio
from copy import deepcopy
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

COLORS = ['orange', 'b', 'g', 'r', 'm', 'y', 'k']

def plot_piles_only(ax, state, config, labels, colors):
    # print(f"-----------------step:{step}-----------------")
    plot_circles(ax, state, config['data']['obj_size'], labels, colors=colors)
    
    return ax

def plot_obj_pile_single_img(state, pusher_poses, action_seq, \
    config, save_path, labels=None, filename=None):
    steps, state_dim = state.shape
    lo = -100
    hi = 500
    if labels is None:
        labels = [0 for _ in range(state_dim//2)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    frames = []
    
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax1.set_title('gt')
    ax2.set_title('pred')
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    # print(f"-----------------step:{step}-----------------")
    plot_circles(ax1, state[0], config['data']['obj_size'], labels)
    plot_pusher(ax1, pusher_poses[0], config['data']['pusher_size'], action_seq[0])

    plot_circles(ax2, state[1], config['data']['obj_size'], labels)
    plot_pusher(ax2, pusher_poses[0]+action_seq[0], config['data']['pusher_size'], action_seq[0])
    
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    image = data.reshape((height, width, 4))
    
    frames.append(deepcopy(image)) 
    plt.cla()
    ax1.clear()
    ax2.clear()
    for frame in frames:
        plt.imsave(f"{save_path}/{filename}.png", frame)
    fig.clear()
    plt.close()

def plot_task(task_name, **kwargs):
    if task_name == "obj_pile":
        plot_obj_pile(**kwargs)
    elif task_name == "pushing_rope":
        plot_pushing_rope(**kwargs)
    elif task_name == "rope_3d":
        plot_rope_3d(**kwargs)
    elif task_name == "reorientation":
        plot_reorientation(**kwargs)
    else:
        raise NotImplementedError

def plot_reorientation(state_gt, state_pred, pusher_poses, action_seq, \
    config, start_idx, save_path, labels=None, filename=None, **kwargs):
    dim_of_work_space = kwargs["dim_of_work_space"]
    B, steps, state_dim = state_gt.shape
    if dim_of_work_space == 3:
        # turn to 2D when plotting
        state_gt = state_gt.reshape(B, steps, state_dim//dim_of_work_space, dim_of_work_space)[:,:,:,[0,2]].reshape(B, steps, -1)
        state_pred = state_pred.reshape(B, steps, state_dim//dim_of_work_space, dim_of_work_space)[:,:,:,[0,2]].reshape(B, steps, -1)
        pusher_poses = pusher_poses[:,:,[0,2]]
        action_seq = action_seq[:,:,[0,2]]
    lo = -0.4
    hi = 0.4
    for b in range(B):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        frames = []
        for step in range(steps):
            ax1.set_xlim(lo, hi)
            ax1.set_ylim(lo, hi)
            ax2.set_xlim(lo, hi)
            ax2.set_ylim(lo, hi)
            ax1.set_title('gt')
            ax2.set_title('pred')
            ax1.set_aspect('equal')
            ax2.set_aspect('equal')
            # print(f"-----------------step:{step}-----------------")
            # import pdb; pdb.set_trace()
            plot_box_2D(ax1, state_gt[b, step])
            plot_box_2D(ax2, state_pred[b, step])
            plot_gripper_2D(ax1, pusher_poses[b, step])
            plot_gripper_2D(ax2, pusher_poses[b, step])
            plot_target_gripper(ax1, kwargs["target_state"][0], state_gt[b, step], gripper_points_interval=0.04)
            plot_target_gripper(ax2, kwargs["target_state"][0], state_pred[b, step], gripper_points_interval=0.04)
            plot_wall(ax1)
            plot_wall(ax2)
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            width, height = fig.canvas.get_width_height()
            image = data.reshape((height, width, 4))
            
            frames.append(deepcopy(image)) 
            plt.cla()
            ax1.clear()
            ax2.clear()
        fps = 2
        if filename is not None:
            with imageio.get_writer(f'{save_path}/{filename}.mp4', fps=fps) as video_writer:
                for image in frames:
                    video_writer.append_data(image)
            imageio.mimsave(f'{save_path}/{filename}.gif', frames, 'GIF', fps=fps)
            print(f"{save_path}/{filename}.mp4 saved")
        else:
            with imageio.get_writer(f'{save_path}/reorientation_{start_idx+b}.mp4', fps=fps) as video_writer:
                for image in frames:
                    video_writer.append_data(image)
            imageio.mimsave(f'{save_path}/reorientation_{start_idx+b}.gif', frames, 'GIF', fps=fps)
            print(f"{save_path}/reorientation_{start_idx+b}.mp4 saved")
        fig.clear()
        plt.close()

# target_state: [x,y,theta]
def plot_target_gripper(ax, target_state, box_points, gripper_points_interval=0.04):
    original_box = torch.tensor([[0.06, 0.03],
                            [-0.06, 0.03],
                            [-0.06, -0.03],
                            [0.06, 0.03]])
    from util.utils import rigid_transform_2D
    original_box = original_box.unsqueeze(0) # 1, N, 2
    box_points = box_points.reshape(-1, 2)
    box_points = torch.tensor(box_points).unsqueeze(0) # 1, N, 2
    R, _ = rigid_transform_2D(original_box, box_points)
    # import pdb; pdb.set_trace()
    finger_pos_1 = target_state[:2]
    finger_pos_2 = target_state[:2] + np.array([np.cos(target_state[2]), np.sin(target_state[2])]) * gripper_points_interval
    finger_pos = np.concatenate([finger_pos_1[np.newaxis,:], finger_pos_2[np.newaxis,:]], axis=0)[np.newaxis,:,:] # 1, 2, 2
    finger_pos = torch.tensor(finger_pos, dtype=torch.float32)
    finger_pos = torch.bmm(finger_pos, R.transpose(1, 2)) # .transpose(1, 2)
    finger_pos = finger_pos.squeeze(0).numpy()
    finger_pos += np.mean(box_points[0].numpy(), axis=0)
    ax.plot(finger_pos[:, 0], finger_pos[:, 1], 'g-')

def plot_wall(ax, wall_position=0.):
    ax.plot([wall_position, wall_position], [-1, 1], 'k-')

def plot_box_2D(ax, corner_points):
    corner_points = corner_points.reshape(-1, 2) # N, 2
    x_points = corner_points[:, 0]
    y_points = corner_points[:, 1]
    x_points = np.concatenate([x_points, [x_points[0]]])
    y_points = np.concatenate([y_points, [y_points[0]]])
    
    ax.plot(x_points, y_points, 'b-')
    ax.fill(x_points, y_points, 'b', alpha=0.3)

def plot_gripper_2D(ax, gripper_points):
    gripper_points = gripper_points.reshape(-1, 2) # N, 2
    ax.scatter(gripper_points[0, 0], gripper_points[0, 1], c='orange', s=10)
    ax.scatter(gripper_points[1, 0], gripper_points[1, 1], c='r', s=10)
    ax.plot(gripper_points[:, 0], gripper_points[:, 1], c='Darkred')

def plot_pushing_rope(state_gt, state_pred, pusher_poses, action_seq, \
    config, start_idx, save_path, labels=None, filename=None, **kwargs):
    dim_of_work_space = kwargs["dim_of_work_space"]
    target_state = None
    if 'target_state' in kwargs.keys():
        target_state = kwargs["target_state"]
    B, steps, state_dim = state_gt.shape
    if dim_of_work_space == 3:
        # turn to 2D when plotting
        state_gt = state_gt.reshape(B, steps, state_dim//dim_of_work_space, dim_of_work_space)[:,:,:,[0,2]].reshape(B, steps, -1)
        state_pred = state_pred.reshape(B, steps, state_dim//dim_of_work_space, dim_of_work_space)[:,:,:,[0,2]].reshape(B, steps, -1)
        pusher_poses = pusher_poses[:,:,[0,2]]
        action_seq = action_seq[:,:,[0,2]]
    lo = -4
    hi = 4
    for b in range(B):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        frames = []
        for step in range(steps):
            if step == 0:
                substeps = [1]
            elif step == steps-1:
                substeps = [0]
            else:
                substeps = [0,1]
            # substeps = [1]
            
            for substep in substeps:
                ax1.set_xlim(lo, hi)
                ax1.set_ylim(lo, hi)
                ax2.set_xlim(lo, hi)
                ax2.set_ylim(lo, hi)
                ax1.set_title('gt')
                ax2.set_title('pred')
                ax1.set_aspect('equal')
                ax2.set_aspect('equal')
                # print(f"-----------------step:{step}-----------------")
                plot_rope(ax1, state_gt[b, step], 'r-')
                plot_rope(ax2, state_pred[b, step], 'r-')
                if target_state is not None:
                    plot_rope(ax1, target_state[b], 'b:')
                    plot_rope(ax2, target_state[b], 'b:')
                if substep == 1:
                    plot_cylinder(ax1, pusher_poses[b, step], 0.2)
                    plot_cylinder(ax2, pusher_poses[b, step], 0.2)
                elif substep == 0:
                    plot_cylinder(ax1, pusher_poses[b, step-1]+action_seq[b, step-1], 0.2)
                    plot_cylinder(ax2, pusher_poses[b, step-1]+action_seq[b, step-1], 0.2)
                
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                width, height = fig.canvas.get_width_height()
                image = data.reshape((height, width, 4))
                
                frames.append(deepcopy(image)) 
                plt.cla()
                ax1.clear()
                ax2.clear()
        fps = 3
        if filename is not None:
            imageio.mimsave(f'{save_path}/{filename}.gif', frames, 'GIF', fps=fps)
            print(f"{save_path}/{filename}.gif saved")
        else:
            imageio.mimsave(f'{save_path}/pushing_rope_{start_idx+b}.gif', frames, 'GIF', fps=fps)
            print(f"{save_path}/pushing_rope_{start_idx+b}.gif saved")
        fig.clear()
        plt.close()

def plot_rope(ax, states, style='r-'):
    states = states.reshape(-1, 2) # N, 2
    # distances = np.sqrt(((states[:, np.newaxis, :] - states[np.newaxis, :, :]) ** 2).sum(axis=2))
    # np.fill_diagonal(distances, np.inf)
    # nearest_points = np.argmin(distances, axis=1)

    for i, point in enumerate(states):
        particle = Circle((point[0], point[1]), 0.1, color=style[0])
        particle.set_zorder(10)
        ax.add_patch(particle)
        # ax.plot(point[0], point[1], color=style[0], markersize=3) # points
        # nearest_point = states[nearest_points[i]]
        # ax.plot([point[0], nearest_point[0]], [point[1], nearest_point[1]], color=style[0], linestyle=style[1]) # lines

def plot_cylinder(ax, pusher_pos, pusher_radius):
    # left bottom corner coordinate 
    circle = Circle((pusher_pos[0], pusher_pos[1]), pusher_radius, color='g')
    circle.set_zorder(10)
    ax.add_patch(circle)
    
    return ax

def plot_rope_3d(state_gt, state_pred, pusher_poses, action_seq, \
    config, start_idx, save_path, labels=None, filename=None, **kwargs):
    dim_of_work_space = kwargs["dim_of_work_space"]
    target_state = None
    forbidden_area = None
    if 'target_state' in kwargs.keys():
        target_state = kwargs["target_state"]
    if "forbidden_area" in kwargs.keys():
        forbidden_area = kwargs["forbidden_area"]
    B, steps, state_dim = state_gt.shape
    
    state_gt = state_gt.reshape(B, steps, state_dim//dim_of_work_space, dim_of_work_space)
    state_pred = state_pred.reshape(B, steps, state_dim//dim_of_work_space, dim_of_work_space)
    pusher_poses = pusher_poses
    action_seq = action_seq
    
    lo = -2
    hi = 2
    for b in range(B):
        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        frames = []
        for step in range(steps):            
            ax1.set_xlim(lo, hi)
            ax1.set_ylim(lo, hi)
            ax1.set_zlim(0.5, hi)
            ax2.set_xlim(lo, hi)
            ax2.set_ylim(lo, hi)
            ax2.set_zlim(0.5, hi)
            ax1.set_title('gt')
            ax2.set_title('pred')
            ax1.set_aspect('equal')
            ax2.set_aspect('equal')
            # print(f"-----------------step:{step}-----------------")
            # import pdb; pdb.set_trace()
            
            # ax1.plot((state_gt[b, step]+pusher_poses[b, step])[:,0], \
            #         (state_gt[b, step]+pusher_poses[b, step])[:,1], \
            #         (state_gt[b, step]+pusher_poses[b, step])[:,2], 'r-')
            # ax2.plot((state_pred[b, step]+pusher_poses[b, step])[:,0], \
            #         (state_pred[b, step]+pusher_poses[b, step])[:,1], \
            #         (state_pred[b, step]+pusher_poses[b, step])[:,2], 'r-')
            # import pdb; pdb.set_trace()
            ax1.scatter(state_gt[b, step][:,0], \
                    state_gt[b, step][:,1], \
                    state_gt[b, step][:,2], 'b-')
            ax2.scatter(state_pred[b, step][:,0], \
                    state_pred[b, step][:,1], \
                    state_pred[b, step][:,2], 'b-')
            
            if forbidden_area is not None:
                plot_box(ax1, forbidden_area[0], forbidden_area[1])
                plot_box(ax1, forbidden_area[2], forbidden_area[3])
                
                plot_box(ax2, forbidden_area[0], forbidden_area[1])
                plot_box(ax2, forbidden_area[2], forbidden_area[3])
            
            # if target_state is not None:
            #     plot_rope(ax1, target_state[b], 'b:')
            #     plot_rope(ax2, target_state[b], 'b:')

            ax1.scatter(pusher_poses[b, step][0], pusher_poses[b, step][1], pusher_poses[b, step][2]-0.5, c='g', marker='o')
            ax2.scatter(pusher_poses[b, step][0], pusher_poses[b, step][1], pusher_poses[b, step][2]-0.5, c='g', marker='o')
            # plot_cylinder(ax1, pusher_poses[b, step], 0.2)
            # plot_cylinder(ax2, pusher_poses[b, step], 0.2)
            
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            width, height = fig.canvas.get_width_height()
            image = data.reshape((height, width, 4))
            
            frames.append(deepcopy(image)) 
            plt.cla()
            ax1.clear()
            ax2.clear()
        fps = 3
        if filename is not None:
            imageio.mimsave(f'{save_path}/{filename}.gif', frames, 'GIF', fps=fps)
            print(f"{save_path}/{filename}.gif saved")
        else:
            imageio.mimsave(f'{save_path}/rope_3d_{start_idx+b}.gif', frames, 'GIF', fps=fps)
            print(f"{save_path}/rope_3d_{start_idx+b}.gif saved")
        fig.clear()
        plt.cla()
        # plt.close()

def plot_box(ax, bottom_left, top_right):
    x1, y1, z1 = bottom_left
    x2, y2, z2 = top_right
    # Define the vertices of the box
    vertices = [
        [x1, y1, z1],
        [x1, y1, z2],
        [x1, y2, z1],
        [x1, y2, z2],
        [x2, y1, z1],
        [x2, y1, z2],
        [x2, y2, z1],
        [x2, y2, z2]
    ]

    # List of sides' polygons
    edges = [
        [vertices[0], vertices[1], vertices[3], vertices[2]], # left side
        [vertices[4], vertices[5], vertices[7], vertices[6]], # right side
        [vertices[0], vertices[1], vertices[5], vertices[4]], # bottom side
        [vertices[2], vertices[3], vertices[7], vertices[6]], # top side
        [vertices[0], vertices[2], vertices[6], vertices[4]], # front side
        [vertices[1], vertices[3], vertices[7], vertices[5]]  # back side
    ]

    # Plot each side
    for edge in edges:
        x = [vertex[0] for vertex in edge]
        y = [vertex[1] for vertex in edge]
        z = [vertex[2] for vertex in edge]
        ax.add_collection3d(Poly3DCollection([list(zip(x, y, z))], facecolors='r', linewidths=1, edgecolors='r', alpha=.25))


# state: [B, n_his+n_roll, state_dim]
# action_seq: [B, n_his+n_roll-1, action_dim]
def plot_obj_pile(state_gt, state_pred, pusher_poses, action_seq, \
    config, start_idx, save_path, labels=None, filename=None, **kwargs):
    B, steps, state_dim = state_gt.shape
    lo = 0
    hi = 400# 800
    if labels is None:
        labels = [0 for _ in range(state_dim//2)]
    for b in range(B):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        frames = []
        for step in range(steps):
            if step == 0:
                substeps = [1]
            elif step == steps-1:
                substeps = [0]
            else:
                substeps = [0,1]
            
            for substep in substeps:
                ax1.set_xlim(lo, hi)
                ax1.set_ylim(lo, hi)
                ax2.set_xlim(lo, hi)
                ax2.set_ylim(lo, hi)
                ax1.set_title(f'gt_{step}')
                ax2.set_title(f'pred_{step}')
                ax1.set_aspect('equal')
                ax2.set_aspect('equal')
                # print(f"-----------------step:{step}-----------------")
                plot_circles(ax1, state_gt[b, step], config['data']['obj_size'], labels)
                plot_circles(ax2, state_pred[b, step], config['data']['obj_size'], labels)
                if substep == 1:
                    plot_pusher(ax1, pusher_poses[b, step], config['data']['pusher_size'], action_seq[b, step])
                    plot_pusher(ax2, pusher_poses[b, step], config['data']['pusher_size'], action_seq[b, step])
                elif substep == 0:
                    plot_pusher(ax1, pusher_poses[b, step-1]+action_seq[b, step-1], config['data']['pusher_size'], action_seq[b, step-1])
                    plot_pusher(ax2, pusher_poses[b, step-1]+action_seq[b, step-1], config['data']['pusher_size'], action_seq[b, step-1])
                
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                width, height = fig.canvas.get_width_height()
                image = data.reshape((height, width, 4))
                
                frames.append(deepcopy(image)) 
                plt.cla()
                ax1.clear()
                ax2.clear()
        fps = 1
        if filename is not None:
            imageio.mimsave(f'{save_path}/{filename}.gif', frames, 'GIF', fps=fps)
            print(f"{save_path}/{filename}.gif saved")
        else:
            imageio.mimsave(f'{save_path}/obj_pile_{start_idx+b}.gif', frames, 'GIF', fps=fps)
            print(f"{save_path}/obj_pile_{start_idx+b}.gif saved")
        fig.clear()
        plt.close()

def plot_circles(ax, states, radius, labels, colors=None, alpha=0.5):
    states = states.reshape(-1, 2) # N, 2
    circle_num = states.shape[0]
    if colors is not None:
        for i in range(circle_num):
            ax.add_patch(Circle((states[i][0], states[i][1]), radius, color=colors[labels[i]]))
    else:
        for i in range(circle_num):
            ax.add_patch(Circle((states[i][0], states[i][1]), radius, color=COLORS[labels[i]]))
    return ax 

def create_rotated_rectangle(ax, center, width, height, angle):
    # Create an unrotated rectangle with the origin at the initial position
    rect = patches.Rectangle((-width/2, -height/2), width, height, 
                            facecolor='red', edgecolor='red')

    # Create a transformation object: first translate and then rotate
    trans = transforms.Affine2D().rotate_deg(angle).translate(*center)

    # Apply transformation to axes
    rect.set_transform(trans + ax.transData)
    return rect

def plot_pusher(ax, pusher_pos, pusher_size, action):
    # left bottom corner coordinate 
    pusher_dir = action / np.linalg.norm(action)
    angle = np.arctan2(pusher_dir[1], pusher_dir[0]) / np.pi *180 + 90
    rect = create_rotated_rectangle(ax, pusher_pos, pusher_size[0], pusher_size[1], angle)
    # rect = patches.Rectangle((-pusher_size[0]/2, -pusher_size[1]/2), pusher_size[0], pusher_size[1], 
    #                         facecolor='red', edgecolor='red')
    rect.set_zorder(10)
    ax.add_patch(rect)
    
    return ax

def _dict_convert_np_to_list(input_dict):
    for k, v in input_dict.items():
        if isinstance(v, np.ndarray):
            input_dict[k] = v.tolist()
        elif isinstance(v, list):
            input_dict[k] = _list_convert_np_to_list(v)
        elif isinstance(v, dict):
            input_dict[k] = _dict_convert_np_to_list(v)
        elif isinstance(v, np.float32):
            input_dict[k] = float(v)
        else:
            pass
    return input_dict

def _list_convert_np_to_list(input_list):
    for idx in range(len(input_list)):
        elem = input_list[idx]
        if isinstance(elem, np.ndarray):
            input_list[idx] = _list_convert_np_to_list(elem.tolist())
        elif isinstance(elem, list):
            input_list[idx] = _list_convert_np_to_list(elem)
        elif isinstance(elem, dict):
            input_list[idx] = _dict_convert_np_to_list(elem)
        elif isinstance(elem, np.float32):
            input_list[idx] = float(elem)
        else:
            pass
    return input_list

def plot_cost(pred_cost, gt_cost, cost_diff, save_path, filename):
    # plot 3 costs in one figure
    plt.figure()
    plt.plot(pred_cost, label='pred_cost')
    plt.plot(gt_cost, label='gt_cost')
    plt.plot(cost_diff, label='cost_diff')
    plt.legend()
    plt.savefig(f"{save_path}/{filename}")
    print(f"{save_path}/{filename} saved")
    plt.close()
    
