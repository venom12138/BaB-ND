import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())
cwd = os.getcwd()
from model.mlp import MLP


class wrapped_model(nn.Module):
    def __init__(self, model_path, model_config):
        super(wrapped_model, self).__init__()
        model: MLP = torch.load(model_path)
        enable_latent = model_config["enable_latent"]
        state_dim = model_config["state_dim"]
        action_dim = model_config["action_dim"]
        n_history = model_config["n_history"]
        horizon = model_config["horizon"]
        cost_norm = int(model_config["cost_norm"])
        assert cost_norm in [1, 2]
        only_final_cost = model_config["only_final_cost"]
        device = model_config["device"]
        self.penalty_type = model_config["penalty_type"]
        known_dim = (state_dim + action_dim) * n_history - action_dim
        self.enable_latent = enable_latent
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_history = n_history
        self.horizon = horizon
        self.cost_norm = cost_norm
        self.only_final_cost = only_final_cost
        self.known_dim = known_dim
        self.device = device
        torch.manual_seed(0)
        self.obs_pos_list = None
        self.obs_size_list = None
        self.obs_type = None
        if model_config["obs_pos_list"] is not None:
            self.obs_type = model_config.get("obs_type", "circle")
            assert self.obs_type in ["circle", "square"]
            self.obs_pos_list = [torch.tensor(obs_pos, device=device) for obs_pos in model_config["obs_pos_list"]]
            self.obs_enlarge = model_config.get("obs_enlarge", 0)
            if self.obs_type == "circle":
                # store the square of the radius of the obstacles
                self.obs_size_list = [((1+self.obs_enlarge)*obs_size)**2 for obs_size in model_config["obs_size_list"]]
            else:
                # half side length of the square obstacles
                self.obs_size_list = [(1+self.obs_enlarge)*obs_size for obs_size in model_config["obs_size_list"]]
            assert len(self.obs_pos_list) == len(self.obs_size_list)
            self.num_obs = len(self.obs_pos_list)

        if self.penalty_type != 0:
            assert self.obs_pos_list is not None
        target_state = torch.rand((1, state_dim), device=device)
        known_input = torch.rand((1, known_dim), device=device)
        refer_pos = torch.rand((1, action_dim), device=device)
        self.register_buffer("known_input", known_input)
        self.register_buffer("target_state", target_state)
        self.register_buffer("refer_pos", refer_pos)
        self.buffer_list = ["known_input", "target_state", "refer_pos"]

        # init model
        self.layers = []
        self.pre_bounds = []
        for i in range(0, len(model.layers) - 1, 2):
            weight = model.layers[i].weight.data
            bias = model.layers[i].bias.data
            linear = nn.Linear(weight.shape[1], weight.shape[0], bias=True)
            linear.weight.data = weight
            linear.bias.data = bias
            self.layers.append(linear)
            self.layers.append(nn.ReLU())
            lb_pre = torch.tensor(model.lb_activation[i // 2], dtype=torch.float32, device=device).unsqueeze(0)
            ub_pre = torch.tensor(model.ub_activation[i // 2], dtype=torch.float32, device=device).unsqueeze(0)
            self.pre_bounds.append((lb_pre, ub_pre))

        i = len(model.layers) - 1
        weight = model.layers[i].weight.data
        bias = model.layers[i].bias.data
        linear_last = nn.Linear(weight.shape[1], weight.shape[0], bias=True)
        linear_last.weight.data = weight
        linear_last.bias.data = bias
        self.layers.append(linear_last)
        # self.model = nn.Sequential(*self.layers)
        self.model = model.model
        self.step_weight_list = (np.arange(1, horizon + 1) / horizon).tolist()

    def forward(self, act_seq):
        # B, act_seq_dim = act_seq.shape
        device = self.known_input.device
        act_seq = act_seq.to(device)
        # act_seq_dim = self.action_dim * self.horizon
        # assert act_seq.shape[1] == act_seq_dim
        curr_state = self.known_input.expand((act_seq.shape[0], -1))
        pusher_pos_abs = self.refer_pos.expand((act_seq.shape[0], -1))
        num_kp = self.state_dim // 2
        
        cost = 0
        penalty = 0
        penalty_type = self.penalty_type
        penalty_factor = 100000
        if penalty_type != 0:
            obs_pos_list = [obs_pos.to(device) for obs_pos in self.obs_pos_list]
            obs_type = self.obs_type
        for j in range(self.horizon):
            curr_action = act_seq[:, j * self.action_dim : (j + 1) * self.action_dim]
            # input: [B, (state_dim + action_dim) * n_history - action_dim + action_dim]
            inputs = torch.cat((curr_state, curr_action), dim=1)
            outputs = self.model(inputs)
            curr_state = curr_state + outputs
            if self.enable_latent:
                abs_state = curr_state
            else:
                abs_state = curr_state + torch.cat([pusher_pos_abs] * num_kp, dim=1)
            # calculate the cost
            diff = self.target_state - abs_state
            if self.cost_norm == 2:
                step_cost = (diff * diff).sum(dim=1, keepdim=True)
            else:
                step_cost = (abs(diff)).sum(dim=1, keepdim=True)
            step_cost = step_cost * self.step_weight_list[j]
            cost += step_cost
            # update the pusher position, shift to next pusher frame
            pusher_pos_abs = pusher_pos_abs + curr_action
            if penalty_type == 1 or penalty_type == 3:
                step_penalty = 0
                if obs_type == "circle":
                    for i in range(self.num_obs):
                        diff_to_obs = pusher_pos_abs - obs_pos_list[i]
                        dist_to_obs = (diff_to_obs * diff_to_obs).sum(dim=-1, keepdim=True)
                        step_penalty += F.relu(-dist_to_obs+self.obs_size_list[i])
                    penalty += step_penalty
                else:
                    for i in range(self.num_obs):
                        step_penalty += F.relu(-abs(pusher_pos_abs - obs_pos_list[i])+self.obs_size_list[i])
                    penalty += step_penalty.min(dim=-1, keepdim=True).values
            if penalty_type == 2 or penalty_type == 3:
                step_penalty = 0
                if obs_type == "circle":
                    for i in range(self.num_obs):
                        diff_to_obs = abs_state.view(act_seq.shape[0], -1, 2) - obs_pos_list[i]
                        mid_point = (diff_to_obs[:, 1:2, :] + diff_to_obs[:, 3:4, :]) / 2
                        diff_to_obs = torch.cat([diff_to_obs, mid_point], dim=1)
                        dist_to_obs = (diff_to_obs * diff_to_obs).sum(dim=-1, keepdim=False)
                        step_penalty += F.relu(-dist_to_obs+self.obs_size_list[i])
                    penalty += step_penalty.sum(dim=-1, keepdim=True)
                else:
                    for i in range(self.num_obs):
                        step_penalty += F.relu(-abs(abs_state.view(act_seq.shape[0], -1, 2) - obs_pos_list[i])+self.obs_size_list[i])
                    penalty += step_penalty.min(dim=-1, keepdim=False).values.sum(dim=-1, keepdim=True)
            if not self.enable_latent:
                curr_state -= torch.cat([curr_action] * num_kp, dim=1)
        penalty = penalty_factor * penalty
        if self.only_final_cost:
            if self.cost_norm == 2:
                cost = (diff * diff).sum(dim=1, keepdim=True)
            else:
                cost = (abs(diff)).sum(dim=1, keepdim=True)
        cost += penalty

        del curr_state, pusher_pos_abs, diff, abs_state, inputs, outputs
        return cost

    def get_info(self):
        info_dict = {}
        for key in self.buffer_list:
            info_dict[key] = getattr(self, key)
        return info_dict

    def set_const(self, key, value):
        if hasattr(self, key):
            assert getattr(self, key).shape == value.shape
            setattr(self, key, value.to(self.device))

    def export_pth(self, pth_file):
        torch.save(self, pth_file)

    def export_onnx(self, onnx_file):
        dummy_input = torch.randn(1, self.action_dim * self.horizon, device="cpu")
        self.to("cpu")
        torch.onnx.export(self, dummy_input, onnx_file, verbose=True)
