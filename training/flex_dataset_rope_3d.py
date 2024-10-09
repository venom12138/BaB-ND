import os
import sys
sys.path.append(os.getcwd())

import abc
import glob
import json

import random
import re
import pickle
import time

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from util.utils import depth2fgpcd, np2o3d, fps_rad_idx
from util.preprocess import extract_pushes, extract_kp_single_frame_by
from dgl.geometry import farthest_point_sampler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_pairs(pairs_path, episode_range):
    pair_lists = []
    for episode_idx in episode_range:
        frame_pairs = np.loadtxt(os.path.join(pairs_path, f'{episode_idx}.txt'))
        if len(frame_pairs.shape) == 1: continue
        episodes = np.ones((frame_pairs.shape[0], 1)) * episode_idx
        pairs = np.concatenate([episodes, frame_pairs], axis=1)
        pair_lists.extend(pairs)
    pair_lists = np.array(pair_lists).astype(int)
    return pair_lists

def load_dataset(config, phase='train'):
    data_dir = os.path.join(config["data_root"], config["task_name"])

    # preprocess
    # save_dir = os.path.join(prep_save_dir, dataset["data_dir"].split('/')[-1])
    # check_preprocess(data_dir, save_dir, dist_thresh, n_future, n_his)

    # load kypts paths
    num_episodes = len(list(glob.glob(os.path.join(data_dir, f"episode*"))))
    print(f"Found num_episodes: {num_episodes}")

    # load data pairs
    num_train = int(num_episodes * config["train"]["train_valid_ratio"])
        
    if phase == "train":
        episode_range_phase = range(num_episodes-num_train, num_episodes) # range(0, num_train)
        pairs_path = os.path.join(data_dir, 'frame_pairs')
    elif phase == "valid":
        episode_range_phase = range(0, num_episodes-num_train)
        pairs_path = os.path.join(data_dir, 'frame_pairs')
    elif phase == 'rollout':
        episode_range_phase = range(0, num_episodes-num_train)
        pairs_path = os.path.join(data_dir, 'rollout_frame_idx')
    else:
        raise AssertionError("Unknown phase %s" % phase)
    
    pair_lists = load_pairs(pairs_path, episode_range_phase)
    print(f'{phase} dataset has {len(list(episode_range_phase))} episodes, {len(pair_lists)} frame pairs')

    return pair_lists

def pad(x, max_dim, dim=0):
    if dim == 0:
        x_dim = x.shape[0]
        x_pad = np.zeros((max_dim, x.shape[1]), dtype=np.float32)
        x_pad[:x_dim] = x
    elif dim == 1:
        x_dim = x.shape[1]
        x_pad = np.zeros((x.shape[0], max_dim, x.shape[2]), dtype=np.float32)
        x_pad[:, :x_dim] = x
    return x_pad

def pad_torch(x, max_dim, dim=0):
    if dim == 0:
        x_dim = x.shape[0]
        x_pad = torch.zeros((max_dim, x.shape[1]), dtype=x.dtype, device=x.device)
        x_pad[:x_dim] = x
    elif dim == 1:
        x_dim = x.shape[1]
        x_pad = torch.zeros((x.shape[0], max_dim, x.shape[2]), dtype=x.dtype, device=x.device)
        x_pad[:, :x_dim] = x
    return x_pad

class FlexRope3DDynamicsDataset(Dataset):
    def __init__(self, config, phase):
        self.config = config
        self.pair_lists = load_dataset(config, phase)
        self.pair_lists = np.array(self.pair_lists)
        
        self.state_dim = self.config["data"]["state_dim"]
        self.action_dim = self.config["data"]["action_dim"]
        n_his = self.config["train"]["n_history"]
        n_roll = self.config["train"]["n_rollout"]
        self.phase = phase
        if phase == "valid":
            n_roll = (
                self.config["train"]["n_rollout_valid"] if "n_rollout_valid" in self.config["train"] else n_roll * 2
            )

        n_sample = n_his + n_roll
        self.n_roll = n_roll
        self.n_sample = n_sample
        self.n_his = n_his
        # self.episode_length = config['data']['episode_length']

    def shuffle(self):
        return 

    def __len__(self):
        return len(self.pair_lists)

    def reorder_particles(self, particle_pos, fps_idx):
        # reorder the particles based on the distance to the end effector
        # particle_pos = particle_pos[:,[0,2]]
        # scaler = StandardScaler()
        # particle_pos_scaled = scaler.fit_transform(particle_pos)
        particle_pos_scaled = particle_pos - np.mean(particle_pos, axis=0)
        pca = PCA(n_components=1)  # We want to reduce the data to 2 principal components
        particle_pos_pca = pca.fit_transform(particle_pos_scaled).reshape(-1)
        sorted_idx = np.argsort(particle_pos_pca)
        fps_idx = fps_idx[sorted_idx]
        
        return fps_idx
    
    def __getitem__(self, i):
        # time1 = time.time()
        episode_idx = self.pair_lists[i][0].astype(int)
        pair = self.pair_lists[i][1:].astype(int) # (n_his+n_roll, 1)
        # print(f"pair:{len(pair)}")
        n_his = self.n_his
        n_future = self.n_roll

        max_tool = self.config['data']['max_tool']
        max_nobj = self.config['data']['max_nobj']
        max_ntool = self.config['data']['max_ntool']
        state_noise = self.config['data']['state_noise'][self.phase]

        # assert max_tool == 1, 'only support single tool'

        # get history keypoints
        obj_kps = [] # (n_his+n_roll, len(start, end), instance_num, all_points, 3)
        tool_kps = [] # (n_his+n_roll, len(start, end), instance_num, all_points, 3)
        data_dir = os.path.join(self.config["data_root"], self.config["task_name"])
        
        # extract key points from the first frame:
        first_frame_obj_kp, _ = extract_kp_single_frame_by(data_dir, episode_idx, 2) # (instance_num, all_points, 3)
        
        fps_idx_list = []
        ## sampling using raw particles
        for j in range(len(first_frame_obj_kp)): # in fact len(obj_kp_start) == 1
            # farthest point sampling
            particle_tensor = torch.from_numpy(first_frame_obj_kp[j]).float()[None, ...]
            obj_kp_num = min(max_nobj, particle_tensor.shape[1])
            fps_idx_tensor = farthest_point_sampler(particle_tensor, obj_kp_num, 
                                start_idx=np.random.randint(0, first_frame_obj_kp[j].shape[0]))[0]
            fps_idx_1 = fps_idx_tensor.numpy().astype(np.int32)       
            fps_idx_1 = self.reorder_particles(first_frame_obj_kp[j][fps_idx_1], fps_idx_1)
            
            fps_idx_list.append(fps_idx_1)
            
        for i in range(len(pair)):
            frame_idx = pair[i]
            obj_kp, tool_kp = extract_kp_single_frame_by(data_dir, episode_idx, frame_idx)
            obj_kps.append(obj_kp)
            tool_kps.append(tool_kp)
        
        # correct the fps_idx_list
        # the first point is the closest to the gripper
        tool_kp_start_pos = np.mean(tool_kps[0], axis=0)[[0,2]]
        obj_kp_start_pos = obj_kps[0][0][fps_idx_list[0]][:, [0,2]]
        tool_obj_dis = np.linalg.norm(tool_kp_start_pos - obj_kp_start_pos, axis=1)
        Is_reverse_fps_idx = (np.argmin(tool_obj_dis) > len(fps_idx_list[0]) // 2)
        if Is_reverse_fps_idx:
            fps_idx_list[0] = fps_idx_list[0][::-1]
        
        obj_kp_start = obj_kps[n_his-1] # start frame keypoints of the object
        instance_num = len(obj_kp_start)
        assert instance_num == 1, 'only support single object'
        
        tools = np.zeros((n_his + n_future, max_nobj + max_ntool * max_tool, 3), dtype=np.float32)
        states = np.zeros((n_his + n_future, max_nobj + max_ntool * max_tool, 3), dtype=np.float32)
        states_delta = np.zeros((n_his + n_future, max_nobj + max_ntool * max_tool, 3), dtype=np.float32)
        for fi in range(n_his+n_future):
            # downsample
            obj_kp_fi = [obj_kps[fi][j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
            obj_kp_fi = np.concatenate(obj_kp_fi, axis=0) # (2, N, 3)
            obj_kp_fi = pad(obj_kp_fi, max_nobj)
            # transform the tool_kps to world frame
            tool_kp_s = np.mean(tool_kps[fi], axis=0) # [[0,2,1]]*np.array([1,-1,1])
            # accidentally, I load pairlist one more time, so it is n_his+n_future+1
            tool_kp_e = np.mean(tool_kps[fi+1], axis=0) # [[0,2,1]]*np.array([1,-1,1]) 
            # print(f"obj_kp_s: {obj_kp_s.shape}, obj_kp_e: {obj_kp_e.shape}, tool_kp_s: {tool_kp_s.shape}, tool_kp_e: {tool_kp_e.shape}")
            # states:
            states[fi, :max_nobj] = obj_kp_fi
            states[fi, max_nobj:] = tool_kp_s[:3]
            states_delta[fi, max_nobj : max_nobj + max_ntool] = tool_kp_e[:3] - tool_kp_s[:3]
            # tools:
            tools[fi, max_nobj : max_nobj + max_ntool] = tool_kp_s[:3]
        # sanity check
        assert sum([len(fps_idx_list[j]) for j in range(len(fps_idx_list))]) == obj_kp_num
        
        # rotation randomness (already translation-invariant)
        if self.phase == 'train':
            # add randomness
            # state randomness
            states[:n_his] += np.random.uniform(-state_noise, state_noise, size=states[:n_his].shape)  # TODO tune noise level
            random_rot = np.random.uniform(-np.pi, np.pi)
            # rot_mat = np.array([[np.cos(random_rot), -np.sin(random_rot), 0],
            #                     [np.sin(random_rot), np.cos(random_rot), 0],
            #                     [0, 0, 1]], dtype=states.dtype)  # 2D rotation matrix in xy plane
            rot_mat = np.array([[np.cos(random_rot), 0, np.sin(random_rot)],
                                [0, 1, 0],
                                [-np.sin(random_rot), 0, np.cos(random_rot)]], dtype=states.dtype)  # 2D rotation matrix in xy plane, because the position is (x, z, -y)
            states = states @ rot_mat[None]
            states_delta = states_delta @ rot_mat[None]
            tools = tools @ rot_mat[None]

        # numpy to torch
        states = torch.from_numpy(states).float()
        states_delta = torch.from_numpy(states_delta).float()
        tools = torch.from_numpy(tools).float() # tool position
        assert self.action_dim == self.state_dim // max_nobj, f"action_dim: {self.action_dim}, state_dim: {self.state_dim}, max_nobj: {max_nobj}"
        # turn the input to my form
        if self.state_dim // max_nobj == 3:
            # (n_his+n_roll, n_p, 3)
            obs = states[:, :max_nobj, [0,2,1]] * np.array([1,-1,1])
            # (n_his+n_roll, n_s, 3), the last dim has no meaning
            pusher_pos = tools[:, max_nobj:, [0,2,1]] * np.array([1,-1,1])
            # (n_his+n_roll, n_s, 3), the last dim has no meaning
            pusher_dir = states_delta[:, max_nobj:, [0,2,1]] * np.array([1,-1,1])          
        else:
            raise NotImplementedError
        
        # COLOR_LIST = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink']
        # for b in range(obs.shape[0]):
        #     for i, particle in enumerate(obs[b]):
        #         plt.plot(particle[0], particle[1], 'o', color=COLOR_LIST[i], label=f"{fps_idx_list[0][i]}_{i}")
        #     plt.xlim(-4, 4)
        #     plt.ylim(-4, 4)
        #     plt.legend()
        #     plt.savefig(f'{b}_particle_pos.png')
        #     plt.cla() 
        # import pdb; pdb.set_trace()
        
        # (n_his+n_roll, n_s, 3)
        obs = obs - pusher_pos
        action = pusher_dir
        
        # save graph
        # x,y,z: x is forward, y is up, z is right
        return {
            "observations": obs.flatten(1,2), # (n_his+n_roll, state_dim)
            "actions": action.flatten(1,2), # (n_his+n_roll, action_dim)
            "weights": torch.tensor([1.0]*self.n_roll, dtype=torch.float32), # (n_his+n_roll)
            "pusher_pos": pusher_pos.flatten(1,2), # (n_his+n_roll, 3)
            "episode_idx": episode_idx,
            "pair": pair,
            "others": [],
        }

# graph = {
#     # input information
#     "state": state_history,  # (n_his, n_p+n_s, state_dim)
#     "action": states_delta,  # (n_p+n_s, state_dim)

#     # future information
#     "tool_future": tool_future,  # (n_future-1, n_p+n_s, state_dim) only n_s is non-zero
#     "action_future": states_delta_future,  # (n_future-1, n_p+n_s, state_dim) only n_s is non-zero

#     # ground truth information
#     "state_future": obj_kp_future,  # (n_future, n_p, state_dim) used as ground truth
#     "pushing_direction": pushing_direction, 
# }

if __name__ == "__main__":
    import yaml
    config_path = "/home/venom/projects/RobotCrown/neural-dynamics-crown/configs/rope_3d.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    dataset = FlexRope3DDynamicsDataset(config, "train")
    # print(len(dataset))
    print(dataset[0])
    for dp in dataset:
        print(dp)