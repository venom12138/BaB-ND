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
from preprocess import extract_pushes, extract_kp_single_frame_by
from dgl.geometry import farthest_point_sampler

def load_pairs(pairs_path, episode_range):
    pair_lists = []
    for episode_idx in episode_range:
        prev_pair_len = len(pair_lists)
        n_pushes = len(list(glob.glob(os.path.join(pairs_path, f'{episode_idx}_*.txt'))))
        for push_idx in range(n_pushes):
            frame_pairs = np.loadtxt(os.path.join(pairs_path, f'{episode_idx}_{push_idx}.txt'))
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
    num_episodes = len(list(glob.glob(os.path.join(data_dir, f"episode_*"))))
    print(f"Found num_episodes: {num_episodes}")

    # load data pairs
    num_train = int(num_episodes * config["train"]["train_valid_ratio"])
        
    if phase == "train":
        episode_range_phase = range(0, num_train)
    elif phase == "valid":
        episode_range_phase = range(num_train, num_episodes)
    else:
        raise AssertionError("Unknown phase %s" % phase)
    
    pairs_path = os.path.join(data_dir, 'frame_pairs')
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

class FlexDynamicsDataset(Dataset):
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

    def __getitem__(self, i):
        # time1 = time.time()
        episode_idx = self.pair_lists[i][0].astype(int)
        pair = self.pair_lists[i][1:].astype(int)
        
        n_his = self.n_his
        n_future = self.n_roll

        max_tool = self.config['data']['max_tool']
        max_nobj = self.config['data']['max_nobj']
        max_ntool = self.config['data']['max_ntool']
        state_noise = self.config['data']['state_noise'][self.phase]

        assert max_tool == 1, 'only support single tool'

        # get history keypoints
        obj_kps = [] # (n_his+n_roll, num_objs, all_points, 3)
        tool_kps = []
        data_dir = os.path.join(self.config["data_root"], self.config["task_name"])
        for i in range(len(pair)):
            frame_idx = pair[i]
            obj_kp, tool_kp = extract_kp_single_frame_by(data_dir, episode_idx, frame_idx)
            obj_kps.append(obj_kp)
            tool_kps.append(tool_kp)

        obj_kp_start = obj_kps[n_his-1]
        instance_num = len(obj_kp_start)
        assert instance_num == 1, 'only support single object'

        fps_idx_list = []
        ## sampling using raw particles
        for j in range(len(obj_kp_start)): # in fact len(obj_kp_start) == 1
            # farthest point sampling
            particle_tensor = torch.from_numpy(obj_kp_start[j]).float()[None, ...]
            fps_idx_tensor = farthest_point_sampler(particle_tensor, min(max_nobj, particle_tensor.shape[1]), 
                                start_idx=np.random.randint(0, obj_kp_start[j].shape[0]))[0]
            fps_idx_1 = fps_idx_tensor.numpy().astype(np.int32)

            # downsample to uniform radius
            # downsample_particle = particle_tensor[0, fps_idx_1, :].numpy()
            # fps_radius = np.random.uniform(fps_radius_range[0], fps_radius_range[1])
            # _, fps_idx_2 = fps_rad_idx(downsample_particle, fps_radius)
            # fps_idx_2 = fps_idx_2.astype(int)
            # fps_idx = fps_idx_1[fps_idx_2]
            fps_idx_list.append(fps_idx_1)

        # downsample to get current obj kp
        obj_kp_start = [obj_kp_start[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
        obj_kp_start = np.concatenate(obj_kp_start, axis=0) # (N, 3)
        obj_kp_num = obj_kp_start.shape[0]

        # get current state delta
        tool_kp = np.stack(tool_kps[n_his-1:n_his+1], axis=0)  # (2, 14) # [x, y, z, pre_x, pre_y, pre_z, quat, pre_quat]
        tool_kp_num = max_ntool # tool_kp.shape[1] // 14
        # action: only the tool has non-zero movement
        states_delta = np.zeros((max_nobj + max_ntool * max_tool, obj_kp_start.shape[-1]), dtype=np.float32)
        states_delta[max_nobj : max_nobj + tool_kp_num] = tool_kp[1, :3] - tool_kp[0, :3]

        # new: get pushing direction
        pushing_direction = states_delta[max_nobj] # (3, )

        # load future states
        obj_kp_future = np.zeros((n_future, max_nobj, obj_kp_start.shape[-1]), dtype=np.float32)
        
        for fi in range(n_future):
            obj_kp_fu = obj_kps[n_his+fi]
            obj_kp_fu = [obj_kp_fu[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
            obj_kp_fu = np.concatenate(obj_kp_fu, axis=0) # (n_p, 3)
            obj_kp_fu = pad(obj_kp_fu, max_nobj)
            obj_kp_future[fi] = obj_kp_fu

        # load future tool keypoints
        tool_future = np.zeros((n_future - 1, max_nobj + max_ntool * max_tool, obj_kp_start.shape[-1]), dtype=np.float32)
        states_delta_future = np.zeros((n_future - 1, max_nobj + max_ntool * max_tool, obj_kp_start.shape[-1]), dtype=np.float32)
        for fi in range(n_future - 1):
            tool_kp_future = tool_kps[n_his+fi:n_his+fi+2]
            tool_kp_future = np.stack(tool_kp_future, axis=0)  # (2, 14)
            tool_future[fi, max_nobj : max_nobj + tool_kp_num] = tool_kp_future[0, :3]
            states_delta_future[fi, max_nobj : max_nobj + tool_kp_num] = tool_kp_future[1, :3] - tool_kp_future[0, :3]
        
        # load history states
        # this serves as the state, which every particle has its own position
        # and after adding action, we can obtain the future state
        state_history = np.zeros((n_his, max_nobj + max_ntool * max_tool, obj_kp_start.shape[-1]), dtype=np.float32)
        states_delta_history = np.zeros((n_his, max_nobj + max_ntool * max_tool, obj_kp_start.shape[-1]), dtype=np.float32)
        for fi in range(n_his):
            obj_kp_his = obj_kps[fi]
            obj_kp_his = [obj_kp_his[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
            obj_kp_his = np.concatenate(obj_kp_his, axis=0)
            obj_kp_his = pad(obj_kp_his, max_nobj)
            state_history[fi, :max_nobj] = obj_kp_his

            tool_kp_his = tool_kps[fi:fi+2]
            tool_kp_his = np.stack(tool_kp_his, axis=0)
            # tool_kp_his = pad(tool_kp_his, max_ntool * max_tool)
            state_history[fi, max_nobj:] = tool_kp_his[0, :3]
            states_delta_history[fi, max_nobj : max_nobj + tool_kp_num] = tool_kp_his[1, :3] - tool_kp_his[0, :3]

        # sanity check
        assert sum([len(fps_idx_list[j]) for j in range(len(fps_idx_list))]) == obj_kp_num
        
        # add randomness
        # state randomness
        state_history += np.random.uniform(-state_noise, state_noise, size=state_history.shape)  # TODO tune noise level
        
        # rotation randomness (already translation-invariant)
        if self.phase == 'train':
            random_rot = np.random.uniform(-np.pi, np.pi)
            rot_mat = np.array([[np.cos(random_rot), -np.sin(random_rot), 0],
                                [np.sin(random_rot), np.cos(random_rot), 0],
                                [0, 0, 1]], dtype=state_history.dtype)  # 2D rotation matrix in xy plane
            state_history = state_history @ rot_mat[None]
            states_delta_history = states_delta_history @ rot_mat[None]
            states_delta = states_delta @ rot_mat
            tool_future = tool_future @ rot_mat[None]
            states_delta_future = states_delta_future @ rot_mat[None]
            obj_kp_future = obj_kp_future @ rot_mat[None]

        # numpy to torch
        state_history = torch.from_numpy(state_history).float()
        states_delta_history = torch.from_numpy(states_delta_history).float()
        states_delta = torch.from_numpy(states_delta).float()
        tool_future = torch.from_numpy(tool_future).float() # tool position
        states_delta_future = torch.from_numpy(states_delta_future).float() # tool action length
        obj_kp_future = torch.from_numpy(obj_kp_future).float()
        pushing_direction = torch.from_numpy(pushing_direction).float()
        assert self.action_dim == self.state_dim // max_nobj * 2
        # turn the input to my form
        if self.state_dim // max_nobj == 3:
            # (n_his+n_roll, n_p, 3)
            obs = torch.cat([state_history[:, :max_nobj], obj_kp_future[:, :max_nobj]], dim=0)
            # (n_his+n_roll, n_s, 3), the last dim has no meaning
            pusher_pos = torch.cat([state_history[:, max_nobj:], tool_future[:, max_nobj:], tool_future[-1:, max_nobj:]], dim=0)
            # (n_his+n_roll, n_s, 3), the last dim has no meaning
            pusher_dir = torch.cat([states_delta_history[:, max_nobj:], states_delta_future[:, max_nobj:], states_delta_future[-1:, max_nobj:]], dim=0)
            # (n_his+n_roll, n_s, 6)
            action = torch.cat([pusher_pos, pusher_dir], dim=-1)
        elif self.state_dim // max_nobj == 2:
            # (n_his+n_roll, n_p, 2)
            obs = torch.cat([state_history[:, :max_nobj,], obj_kp_future[:, :max_nobj,]], dim=0)[:,:,[0,2]]
            # (n_his+n_roll, n_s, 2), the last dim has no meaning
            pusher_pos = torch.cat([state_history[:, max_nobj:], tool_future[:, max_nobj:], tool_future[-1:, max_nobj:]], dim=0)[:,:,[0,2]]
            # (n_his+n_roll, n_s, 2), the last dim has no meaning
            pusher_dir = torch.cat([states_delta_history[:, max_nobj:], states_delta_future[:, max_nobj:], \
                states_delta_future[-1:, max_nobj:]], dim=0)[:,:,[0,2]]
            # (n_his+n_roll, n_s, 4)
            action = torch.cat([pusher_pos, pusher_dir], dim=-1)
        else:
            raise NotImplementedError
        
        # save graph
        # x,y,z: x is forward, y is up, z is right
        return {
            "observations": obs.flatten(1,2), # (n_his+n_roll, state_dim)
            "actions": action.flatten(1,2) , # (n_his+n_roll, action_dim)
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
    config_path = ""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    dataset = FlexDynamicsDataset(config, "train")
    print(len(dataset))
    print(dataset[0])