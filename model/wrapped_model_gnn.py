import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print(sys.path)
from model.model import PyG_GNN


class wrapped_model_gnn(nn.Module):
    def __init__(self, model_path, model_config):
        super(wrapped_model_gnn, self).__init__()
        state_dim, action_dim, horizon, n_history = (
            model_config["state_dim"],
            model_config["action_dim"],
            model_config["horizon"],
            model_config["n_history"],
        )
        n_particle, n_relation, num_classes = (
            model_config["n_particle"],
            model_config["n_relation"],
            model_config["num_classes"],
        )
        device = model_config["device"]
        self.obj_size = model_config["obj_size"]
        self.close_dis = (self.obj_size+model_config["forbidden_radius"])**2
        self.far_dis = (model_config["far_factor"]*self.obj_size)**2
        self.forbidden = model_config["forbidden"]
        self.cls_idx = model_config["cls_idx"]
        self.fix_others = model_config["fix_others"]
        self.cost_weight = model_config["cost_weight"]
        assert n_history == 1
        cost_norm, only_final_cost = model_config["cost_norm"], model_config["only_final_cost"]
        assert cost_norm in [1, 2]
        self.cost_norm = cost_norm
        self.only_final_cost = only_final_cost
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_norm = torch.tensor(model_config["action_norm"], device=device)
        self.horizon = horizon
        self.num_classes = num_classes
        self.device = device
        model: PyG_GNN = torch.load(model_path, map_location=device)
        self.model = model.model.to(device)
        torch.manual_seed(0)
        known_input = torch.rand((1, n_particle, 2), device=device)
        target_state = torch.rand((1, n_particle, 2), device=device)
        initial_state = torch.rand((1, n_particle, 2), device=device)
        Rr = torch.rand((1, n_relation, n_particle), device=device)
        Rs = torch.rand((1, n_relation, n_particle), device=device)
        self.register_buffer("known_input", known_input)
        self.register_buffer("target_state", target_state)
        self.register_buffer("initial_state", initial_state)
        self.register_buffer("Rr", Rr)
        self.register_buffer("Rs", Rs)
        self.buffer_list = ["known_input", "target_state", "initial_state", "Rr", "Rs"]
        self.step_weight_list = (np.arange(1, horizon + 1) / horizon).tolist()

    # [B, horizon, action_dim], action_dim = 4
    # assume n_history = 1
    def forward(self, act_seq):
        act_seq = act_seq.view(-1, self.horizon, self.action_dim) * self.action_norm.to(act_seq.device)
        curr_state = self.known_input.clone().detach() # [B, N, 2]
        Rr = self.Rr.clone().detach().expand((act_seq.shape[0], -1, -1)) # [B, N, n_relation]
        Rs = self.Rs.clone().detach().expand((act_seq.shape[0], -1, -1)) # [B, N, n_relation]
        cost = 0
        penalty, penalty_factor = 0, 1e5 
        for j in range(self.horizon):
            curr_action = act_seq[:, j:j+1, :] # [B, 1, action_dim=4]
            state_ori = curr_state - curr_action[:, :, :2] # [B, N, 2]
            if self.forbidden:
                sqr_dist = (state_ori * state_ori).sum(dim=-1, keepdim=False) # [B, N]
                min_sqr_dist = torch.min(sqr_dist, dim=-1, keepdim=True).values # [B, 1]
                close_penalty = F.relu(self.close_dis - min_sqr_dist) # [B, 1]
                far_penalty = F.relu(min_sqr_dist - self.far_dis) # [B, 1]
                # far_penalty = 0 
                penalty += close_penalty + far_penalty
            # compute attributes for each node
            attr = torch.cat([state_ori.clone().detach(), state_ori - curr_action[:, :, 2:]], -1) # [B, N, 2+2]
            state = state_ori.clone().detach()
            output = self.model(attr, state, Rr, Rs) # [B, N, 2]
            # always predict the residual
            curr_state = output + curr_state
            # calculate cost
            cost += self.step_cost(curr_state) * self.step_weight_list[j]
        if self.only_final_cost:
            cost = self.step_cost(curr_state)
        penalty *= penalty_factor
        return cost + penalty

    def step_cost(self, state):
        cost = 0
        initial_state = self.initial_state.clone().detach().expand(state.shape[0], -1, -1)
        for i, idx_in_cls in enumerate(self.cls_idx):
            if self.fix_others and i != 0:
                diff = (state - initial_state)[:, idx_in_cls]
            else:
                diff = (state - self.target_state)[:, idx_in_cls]
            if self.cost_norm == 2:
                cls_cost = (diff ** 2).sum(dim=-1, keepdim=False).sum(dim=-1, keepdim=True)
            else:
                cls_cost = (diff.abs()).sum(dim=-1, keepdim=False).sum(dim=-1, keepdim=True)
            if self.fix_others and i != 0:
                cls_cost *= self.cost_weight
            elif (not self.fix_others) and (i == 0):
                cls_cost *= self.cost_weight
            cost += cls_cost
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
        dummy_input = torch.randn(1, self.horizon, self.action_dim, device=self.Rs.device)
        torch.onnx.export(self, dummy_input, onnx_file, verbose=True)