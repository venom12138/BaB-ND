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
import numpy as np
from scipy.optimize import minimize
import pickle
from transforms3d.euler import euler2mat, mat2euler
from real_wrapper_mp_pushing_T import RealWrapper
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
world2base = world2base
workspace2world = workspace2world
workspace2base = world2base @ workspace2world


# env
serial_numbers = ["311322303615", "215122252880", "151422251501", "213622251153"]
visible_areas = None
param_dict = {
    "capture_fps": 15,
    "record_fps": 5,
    "record_time": 10,
    "reset": False,
}
cad_path_list = [
    f"{file_dir}/object/t.obj",
]
object_color_list = [np.array([0.80, 0.35, 0.15], dtype=np.float32)]
real_wrapper = RealWrapper(param_dict, serial_numbers, visible_areas, cad_path_list, object_color_list, target_state=None)
real_wrapper.start()
manual_selected_points_from_get_pose_in_base = []
transposed_selected_points_by_transformation_mat = []
gt_points_from_T = []

while True:
    input_key = input("Press x to break...")
    if input_key == 'x':
        break
    import pdb; pdb.set_trace()
    state = real_wrapper.get_state(
                keypoint_offsets_2d_list=np.array([[[  0.,  -5.],
                                                    [ 30.,  -5.],
                                                    [ 60.,  -5.],
                                                    [ 30., -70.]]]),
                visualize=True,
            ).reshape(-1,2)
    manual_selected_points_from_get_pose_in_base.append(np.array(real_wrapper.get_robot_pose()[:3]))
    transposed_selected_points_by_transformation_mat.append(np.array(state[-1]))
    min_idx = np.argmin(np.linalg.norm(state[-1][np.newaxis,:] - state[:-1], axis=-1))
    gt_points_from_T.append(np.array(state[min_idx]))
print(f"manual_selected_points_from_get_pose_in_base:{manual_selected_points_from_get_pose_in_base}")
print(f"transposed_selected_points_by_transformation_mat:{transposed_selected_points_by_transformation_mat}")
print(f"gt_points_from_T:{gt_points_from_T}")
manual_selected_points_from_get_pose_in_base = np.array(manual_selected_points_from_get_pose_in_base) / 1000
transposed_selected_points_by_transformation_mat = np.array(transposed_selected_points_by_transformation_mat) / 1000
gt_points_from_T = np.array(gt_points_from_T) / 1000
# manual_selected_points_from_get_pose_in_base = np.array([[423.421844, -24.218121, 132.932556],
#                                                         [497.398285, -24.218161, 132.932663],
#                                                         [497.398285, 65.194397, 132.932709],
#                                                         ]) / 1000
# transposed_selected_points_by_transformation_mat = np.array([[148.44984528, 235.43454756],
#                                                             [146.68219208, 161.48559784],
#                                                             [236.03071293, 159.38469561],
#                                                             ]) / 1000

# gt_points_from_T = np.array([[136.94335937, 234.5501709],
#                             [134.03289795, 158.22113037],
#                             [219.36584473, 159.82543945],
#                             ]) / 1000
num_p = manual_selected_points_from_get_pose_in_base.shape[0]

# manual_selected_points_from_get_pose_in_base_extended = np.concatenate([manual_selected_points_from_get_pose_in_base, \
#                                                             np.ones((num_p,1))], axis=1)

# gt_hat = manual_selected_points_from_get_pose_in_base_extended @ np.linalg.inv(workspace2base).T
# print(f"gt_hat:{gt_hat}")


def error_function(params, source_points, target_points):
    eulers = params[:3]
    translation = params[3:]
    transformation_mat = np.zeros((4,4))
    transformation_mat[:3,:3] = euler2mat(*eulers)
    transformation_mat[:3, 3] = translation
    transformation_mat[3,3] = 1
    num_p = source_points.shape[0]

    source_points = np.concatenate([source_points, np.ones((num_p,1))], axis=1)
    gt_hat = source_points @ np.linalg.inv(transformation_mat @ workspace2world).T
    
    return np.sum((gt_hat[:,:2] - target_points) ** 2)

def finetune_transformation(transformation_mat, source_points, target_points):
    initial_params = np.zeros(6)
    eulers = mat2euler(transformation_mat[:3,:3])
    trans = transformation_mat[:3,3]
    initial_params[:3] = eulers
    initial_params[3:] = trans
    result = minimize(error_function, initial_params, args=(source_points, target_points), method='BFGS')
    print(f"result:{result}")
    optimized_angles = result.x[:3]
    optimized_translation = result.x[3:]
    transformation_mat = np.zeros((4,4))
    transformation_mat[:3,:3] = euler2mat(*optimized_angles)
    transformation_mat[:3, 3] = optimized_translation
    transformation_mat[3,3] = 1
    
    return transformation_mat

finetuned_mat = finetune_transformation(world2base, manual_selected_points_from_get_pose_in_base, gt_points_from_T)
print(f"finetune world2base: {finetuned_mat}")
print(f"raw world2base: {world2base}")