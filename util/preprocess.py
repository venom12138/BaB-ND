import os
import torch
import numpy as np
import glob
from PIL import Image
import cv2
import pickle as pkl
import json
import open3d as o3d
from dgl.geometry import farthest_point_sampler

from util.utils import label_colormap, opengl2cam


def extract_kp(data_dir, episode_idx, start_frame, end_frame):
    obj_ptcl_start = np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/{start_frame}_particles.npy"))
    obj_ptcl_end = np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/{end_frame}_particles.npy"))
    obj_ptcl_start = obj_ptcl_start[:, :3]
    obj_ptcl_end = obj_ptcl_end[:, :3]
    obj_kp = np.stack([obj_ptcl_start, obj_ptcl_end], axis=0)

    y = 0.5
    eef_ptcl_start = np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/{start_frame}_endeffector.npy"))
    eef_ptcl_end = np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/{end_frame}_endeffector.npy"))
    x_start = eef_ptcl_start[0]
    z_start = eef_ptcl_start[1]
    x_end = eef_ptcl_end[0]
    z_end = eef_ptcl_end[1]
    pos_start = np.array([[x_start, y, z_start]])  # (1, 3)
    pos_end = np.array([[x_end, y, z_end]])
    eef_kp = np.stack([pos_start, pos_end], axis=0)  # (2, 1, 3)
    eef_kp[:, :, 2] *= -1
    return obj_kp, eef_kp


def extract_kp_single_frame(data_dir, episode_idx, frame_idx):
    obj_ptcl = np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/{frame_idx}_particles.npy"))
    obj_ptcl = obj_ptcl[:, :3]
    obj_kp = [obj_ptcl]

    y = obj_kp[0][:, 1].mean()
    eef_ptcl = np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/{frame_idx}_endeffector.npy"))
    x = eef_ptcl[0]
    z = eef_ptcl[1]
    eef_kp = np.array([[x, y, -z]])  # (1, 3)
    return obj_kp, eef_kp


# new dataset
def extract_kp_single_frame_by(data_dir, episode_idx, frame_idx):
    # obtain object keypoints
    obj_ptcls = np.load(os.path.join(data_dir, f"episode_{episode_idx}/particles_pos.npy"))
    obj_ptcl = obj_ptcls[frame_idx]
    obj_kp = np.array([obj_ptcl])
    
    # obtain tool keypoints
    tool_ptcl = np.load(os.path.join(data_dir, f"episode_{episode_idx}/eef_states.npy"))
    tool_kp = tool_ptcl[frame_idx]
    tool_kp = np.array(tool_kp)
    return obj_kp, tool_kp

def extract_kp_single_frame_reorientation(data_dir, episode_idx, frame_idx):
    # obtain object keypoints
    obj_ptcls = np.load(os.path.join(data_dir, f"episode_{episode_idx}/obj_state.npy"))
    obj_ptcl = obj_ptcls[frame_idx]
    obj_kp = np.array([obj_ptcl])
    
    # obtain tool keypoints
    tool_ptcl = np.load(os.path.join(data_dir, f"episode_{episode_idx}/eef_states.npy"))
    tool_kp = tool_ptcl[frame_idx]
    tool_kp = np.array(tool_kp)
    return obj_kp, tool_kp


def extract_pushes(data_dir, save_dir, dist_thresh, n_his, n_future):
    # use overlapping samples
    # provide canonical frame info
    # compatible to other data layouts (make a general episode list)

    frame_idx_dir = os.path.join(save_dir, 'frame_pairs')
    os.makedirs(frame_idx_dir, exist_ok=True)

    num_episodes = len(list(glob.glob(os.path.join(data_dir, f"episode_*"))))

    print(f"Preprocessing starts. num_episodes: {num_episodes}")

    phys_params = []

    for episode_idx in range(num_episodes):
        num_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/*_color.jpg"))))
        print(f"Processing episode {episode_idx}, num_frames: {num_frames}")

        actions = np.load(os.path.join(data_dir, f"episode_{episode_idx}/actions.npy"))
        steps = np.load(os.path.join(data_dir, f"episode_{episode_idx}/steps.npy"))
        eef_particles = np.load(os.path.join(data_dir, f"episode_{episode_idx}/eef_states.npy"))
        steps_a = steps
        # steps_a = np.concatenate([[2], steps], axis=0)

        if len(actions) != len(steps):
            try:
                assert num_frames == steps[-1]
                assert actions[len(steps):].sum() == 0
                actions = actions[:len(steps)]
            except:
                import pdb; pdb.set_trace()

        frame_idxs = []

        # get start-end pairs
        cnt = 0
        for fj in range(num_frames):

            # get physical parameter
            if fj == 0:
                physics_path = os.path.join(data_dir, f"episode_{episode_idx}/property_params.json")
                assert os.path.join(data_dir, f"episode_{episode_idx}/property_params.json") == physics_path
                with open(physics_path) as f:
                    properties = json.load(f)
                # physics_param = np.array([
                #     properties['object_type'],
                #     properties['particle_radius'],
                #     properties['num_particles'],
                #     properties['mass'],
                #     properties['dynamic_friction'],
                # ]).astype(np.float32)
                physics_param = np.array([
                    properties['particle_radius'],
                    properties['num_particles'],
                    properties['length'],
                    properties['thickness'],
                    properties['dynamic_friction'],
                    properties['cluster_spacing'],
                    properties['global_stiffness'],
                ]).astype(np.float32)
                phys_params.append(physics_param)

            curr_step = None # 0, 1, depends on the steps_a's length
            for si in range(len(steps_a) - 1):
                # if step is [x1, x2]: [2, x1-2] is first push, (x1-1 is the final state of first push), 
                # [x1, x2-2] is second push
                if fj >= steps_a[si] and fj <= steps_a[si+1] - 2:
                    curr_step = si
                    break
            else:
                continue  # this frame is not valid
            assert curr_step is not None

            curr_frame = fj
            # this action's start and end frame
            start_frame = steps_a[curr_step]
            end_frame = steps_a[curr_step + 1] - 2  # inclusive

            eef_particles_curr = eef_particles[curr_frame]
            # np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/{curr_frame}_endeffector.npy"))

            frame_traj = [curr_frame]

            # search backward
            # to find a trajectory whose the interval is dist_thresh
            fi = fj
            while fi >= start_frame:
                eef_particles_fi = eef_particles[fi]
                # np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/{fi}_endeffector.npy"))
                x_curr = eef_particles_curr[0]
                z_curr = eef_particles_curr[1]
                x_fi = eef_particles_fi[0]
                z_fi = eef_particles_fi[1]
                dist_curr = np.sqrt((x_curr - x_fi) ** 2 + (z_curr - z_fi) ** 2)
                if dist_curr >= dist_thresh:
                    frame_traj.append(fi)
                    eef_particles_curr = eef_particles_fi
                fi -= 1
                if len(frame_traj) == n_his:
                    break
            else:
                # pad to n_his
                frame_traj = frame_traj + [frame_traj[-1]] * (n_his - len(frame_traj))

            frame_traj = frame_traj[::-1]
            fi = fj
            eef_particles_curr = eef_particles[curr_frame]
            # np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/{curr_frame}_endeffector.npy"))
            # search forward to find a future trajectory whose the interval is dist_thresh
            while fi <= end_frame:
                eef_particles_fi = eef_particles[fi]
                # np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/{fi}_endeffector.npy"))
                x_curr = eef_particles_curr[0]
                z_curr = eef_particles_curr[1]
                x_fi = eef_particles_fi[0]
                z_fi = eef_particles_fi[1]
                dist_curr = np.sqrt((x_curr - x_fi) ** 2 + (z_curr - z_fi) ** 2)

                if dist_curr >= dist_thresh or (fi == end_frame and dist_curr >= 0.75 * dist_thresh):
                    frame_traj.append(fi)
                    eef_particles_curr = eef_particles_fi

                    img_vis = False
                    if img_vis:
                        vis_push(data_dir, episode_idx, curr_frame, fi, start_frame, end_frame)
                        return  # for debug

                fi += 1
                if len(frame_traj) == n_his + n_future:
                    cnt += 1
                    break
            else:
                # (1) We can just continue but this will result in less pushes and no stopping condition
                # continue

                # (2) When assuming quasi-static, we can pad to n_his + n_future
                frame_traj = frame_traj + [frame_traj[-1]] * (n_his + n_future - len(frame_traj))
                cnt += 1

                # (3) We can go to next push, but this will ignore the gap between pushes (not implemented)
                # import ipdb; ipdb.set_trace()

            frame_idxs.append(frame_traj)

            # push_centered
            if fj == end_frame:
                frame_idxs = np.array(frame_idxs)
                np.savetxt(os.path.join(frame_idx_dir, f'{episode_idx}_{curr_step}.txt'), frame_idxs, fmt='%d')
                print(f'episode {episode_idx}, push {curr_step} has {cnt} pushes')
                frame_idxs = []

        # push_centered
        # frame_idxs = np.array(frame_idxs)
        # np.savetxt(os.path.join(frame_idx_dir, f'{episode_idx}.txt'), frame_idxs, fmt='%d')
        # print(f'episode {episode_idx} has {cnt} pushes')

    phys_params = np.stack(phys_params, axis=0)
    phys_params_max = phys_params.max(0)
    phys_params_min = phys_params.min(0)
    phys_params_range = np.stack([phys_params_min, phys_params_max], axis=0)
    print("phys_params_range:", phys_params_range)
    np.savetxt(os.path.join(os.path.join(save_dir, 'phys_range.txt')), phys_params_range)


def vis_push(data_dir, episode_idx, curr_frame, fi, start_frame, end_frame):
    cam_idx = 0
    obj_kp, eef_kp = extract_kp(data_dir, episode_idx, curr_frame, fi)
    intr = np.load(os.path.join(data_dir, f"camera_intrinsic_params.npy"))[cam_idx]
    extr = np.load(os.path.join(data_dir, f"camera_extrinsic_matrix.npy"))[cam_idx]
    img = cv2.imread(os.path.join(data_dir, f"episode_{episode_idx}/camera_{cam_idx}/{curr_frame}_color.jpg"))
    # transform particle to camera coordinate
    particle_cam = opengl2cam(obj_kp[0], extr)
    assert particle_cam[:, 2].min() > 0  # z > 0
    fx, fy, cx, cy = intr
    particle_projs = np.zeros((particle_cam.shape[0], 2))
    particle_projs[:, 0] = particle_cam[:, 0] * fx / particle_cam[:, 2] + cx
    particle_projs[:, 1] = particle_cam[:, 1] * fy / particle_cam[:, 2] + cy
    for pi in range(particle_projs.shape[0]):
        cv2.circle(img, (int(particle_projs[pi, 0]), int(particle_projs[pi, 1])), 3, (0, 0, 255), -1)
    # project eef particle
    eef_cam = opengl2cam(eef_kp[0], extr)
    eef_proj = np.zeros((1, 2))
    eef_proj[0, 0] = eef_cam[0, 0] * fx / eef_cam[0, 2] + cx
    eef_proj[0, 1] = eef_cam[0, 1] * fy / eef_cam[0, 2] + cy
    cv2.circle(img, (int(eef_proj[0, 0]), int(eef_proj[0, 1])), 3, (0, 255, 0), -1)
    # print(eef_kp[0].mean(0), eef_proj.mean(0))
    cv2.imwrite(f'test_{episode_idx}_{cam_idx}_{curr_frame}.jpg', img)


if __name__ == "__main__":
    data_dir_list = [
        "/home/venom/projects/RobotCrown/kf_dynamics/data/raw_rope",
    ]
    save_dir_list = [
        "/home/venom/projects/RobotCrown/kf_dynamics/data/processed/rope",
    ]
    dist_thresh = 0.20
    n_his = 4
    n_future = 3
    for data_dir, save_dir in zip(data_dir_list, save_dir_list):
        if os.path.isdir(data_dir):
            os.makedirs(save_dir, exist_ok=True)
            extract_pushes(data_dir, save_dir, dist_thresh=dist_thresh, n_his=n_his, n_future=n_future)
        # save metadata
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'metadata.txt'), 'w') as f:
            f.write(f'{dist_thresh},{n_future},{n_his}')
