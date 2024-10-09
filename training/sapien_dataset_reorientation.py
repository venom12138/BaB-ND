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
from util.preprocess import extract_pushes, extract_kp_single_frame_reorientation
from dgl.geometry import farthest_point_sampler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import transforms3d
def load_pairs(pairs_path, episode_range):
    pair_lists = []
    for episode_idx in episode_range:
        # import pdb; pdb.set_trace()
        frame_pairs = np.loadtxt(os.path.join(pairs_path, f'{episode_idx}.txt'))
        if len(frame_pairs.shape) == 1: 
            frame_pairs = np.array([frame_pairs])
            # continue
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

class ReorientationDynamicsDataset(Dataset):
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
        self.weights = np.ones((len(self.pair_lists), n_roll))
        # self.episode_length = config['data']['episode_length']

    def update_weights(self, indices, new_weight):
        if len(indices.shape) == 1:
            self.weights[indices] *= new_weight.reshape(len(new_weight), 1)
            self.weights[indices] = np.clip(self.weights[indices], 1, self.config["data"]["weight_ub"])
        elif len(indices.shape) == 2:
            self.weights[indices[:, 0], indices[:, 1]] *= new_weight
            self.weights[indices[:, 0], indices[:, 1]] = np.clip(
                self.weights[indices[:, 0], indices[:, 1], 0], 1, self.config["data"]["weight_ub"]
            )
        else:
            raise AssertionError("Unknown indices shape")
        print("weights percentile", [round(np.percentile(self.weights, 5 * i), 5) for i in range(21)])
        self.shuffle()
    
    # only called when train.enhance_method="hn"
    def add_hard_negatives(self, indices):
        """
        Add hard negative cases to the dataset.

        :param indices: Indices of the hard negative samples in the dataset.
        """
        # Extract and append the hard negative samples
        self.pair_lists = np.concatenate((self.pair_lists, self.pair_lists[indices]))
        self.weights = np.concatenate((self.weights, self.weights[indices]))
        self.shuffle()
        
    def shuffle(self):
        idx = np.random.permutation(range(len(self.pair_lists)))
        self.pair_lists = self.pair_lists[idx]
        self.weights = self.weights[idx]

    def __len__(self):
        return len(self.pair_lists)

    def __getitem__(self, i):
        # time1 = time.time()
        # episode_idx = self.pair_lists[i][0].astype(int)
        # pair = self.pair_lists[i][1:].astype(int) # (n_his+n_roll, 1)
        episode_idx = self.pair_lists[i][0].astype(int)
        pair = self.pair_lists[i][1:].astype(int) # (n_his+n_roll, 1)
        # print(f"pair:{len(pair)}")
        n_his = self.n_his
        n_future = self.n_roll
        property_dict = json.load(open(os.path.join(self.config["data_root"], self.config["task_name"], f"episode_{episode_idx}", "property_params.json")))
        state_noise = 0.0005
        
        # get history keypoints
        obj_kps = [] # (n_his+n_roll, len(start, end), instance_num, all_points, 3)
        tool_kps = [] # (n_his+n_roll, len(start, end), instance_num, all_points, 3)
        data_dir = os.path.join(self.config["data_root"], self.config["task_name"])
        
        # extract key points from the first frame:
        for i in range(len(pair)):
            frame_idx = pair[i]
            obj_kp, tool_kp = extract_kp_single_frame_reorientation(data_dir, episode_idx, frame_idx)
            obj_kps.append(obj_kp)
            tool_kps.append(tool_kp)

        obj_kp_start = obj_kps[n_his-1] # start frame keypoints of the object
        instance_num = len(obj_kp_start)
        assert instance_num == 1, 'only support single object'
        
        tools = np.zeros((n_his + n_future, 2, 2), dtype=np.float32)
        states = np.zeros((n_his + n_future, 4, 2), dtype=np.float32)

        states_delta = np.zeros((n_his + n_future, 2, 2), dtype=np.float32)
        for fi in range(n_his+n_future):
            # downsample
            obj_center_pos = obj_kps[fi][0, :3]
            obj_ori = obj_kps[fi][0, 3:]
            obj_ori_mat = transforms3d.euler.quat2mat(obj_ori)
            
            obj_points = np.zeros((4,2)) # (x, z)
            obj_points[0] = obj_center_pos[[0,2]] + np.dot(obj_ori_mat, property_dict['object_size']*np.array([1,1,1]))[[0,2]]
            obj_points[1] = obj_center_pos[[0,2]] + np.dot(obj_ori_mat, property_dict['object_size']*np.array([-1,1,1]))[[0,2]]
            obj_points[2] = obj_center_pos[[0,2]] + np.dot(obj_ori_mat, property_dict['object_size']*np.array([-1,1,-1]))[[0,2]]
            obj_points[3] = obj_center_pos[[0,2]] + np.dot(obj_ori_mat, property_dict['object_size']*np.array([1,1,-1]))[[0,2]]
            
            obj_kp_fi = obj_points # (4, 2)
            
            # transform the tool_kps to world frame
            eef_pos_s = tool_kps[fi][8, :3]
            eef_ori_mat_s = transforms3d.euler.quat2mat(tool_kps[fi][8, 3:])
            tool_kp_s = np.zeros((2,2))
            tool_kp_s[0] = (eef_pos_s + np.dot(eef_ori_mat_s, [0,0,0.29]))[[0,2]]
            tool_kp_s[1] = (eef_pos_s + np.dot(eef_ori_mat_s, [0,0,0.25]))[[0,2]]
            
            # accidentally, I load pairlist one more time, so it is n_his+n_future+1
            eef_pos_e = tool_kps[fi+1][8, :3]
            eef_ori_mat_e = transforms3d.euler.quat2mat(tool_kps[fi+1][8, 3:])
            tool_kp_e = np.zeros((2,2))
            tool_kp_e[0] = (eef_pos_e + np.dot(eef_ori_mat_e, [0,0,0.29]))[[0,2]]
            tool_kp_e[1] = (eef_pos_e + np.dot(eef_ori_mat_e, [0,0,0.25]))[[0,2]]
            
            # print(f"obj_kp_s: {obj_kp_s.shape}, obj_kp_e: {obj_kp_e.shape}, tool_kp_s: {tool_kp_s.shape}, tool_kp_e: {tool_kp_e.shape}")
            # states:
            states[fi,] = obj_kp_fi
            # tools:
            tools[fi,] = tool_kp_s
            states_delta[fi, ] = tool_kp_e - tool_kp_s
        
        # rotation randomness (already translation-invariant)
        if self.phase == 'train':
            # add randomness
            # state randomness
            states[:n_his] += np.random.uniform(-state_noise, state_noise, size=states[:n_his].shape)  # TODO tune noise level
            

        # numpy to torch
        states = torch.from_numpy(states).float()
        states_delta = torch.from_numpy(states_delta).float()
        tools = torch.from_numpy(tools).float() # tool position
        
        obs = states # (n_his + n_future, 4, 2)
        pusher_pos = tools # (n_his + n_future, 2, 2)
        action = states_delta # (n_his + n_future, 2, 2)
        
        # change it to wall frame
        obs = obs - torch.tensor([[[0.25, 0]]])
        pusher_pos = pusher_pos - torch.tensor([[[0.25, 0.]]])
        
        # normalize data
        obs = obs / self.config['data']['scale']
        pusher_pos = pusher_pos / self.config['data']['scale']
        action = action / self.config['data']['scale']
        
        # save graph
        # x,y,z: x is forward, y is up, z is right
        return {
            "observations": obs.flatten(1,2), # (n_his+n_roll, state_dim)
            "actions": action.flatten(1,2), # (n_his+n_roll, action_dim)
            "weights": torch.tensor(self.weights[i], dtype=torch.float32), # (n_his+n_roll)
            "pusher_pos": pusher_pos.flatten(1,2), # (n_his+n_roll, 4)
            "episode_idx": episode_idx,
            "pair": pair,
            "others": [],
        }

if __name__ == "__main__":
    import yaml
    config_path = "/home/venom/projects/RobotCrown/neural-dynamics-crown/configs/reorientation.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    dataset = ReorientationDynamicsDataset(config, "train")
    # print(len(dataset))
    print(dataset[0])
    for dp in dataset:
        print(dp)