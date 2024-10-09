import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())
cwd = os.getcwd()
from model.model import Rope_MLP


class wrapped_model_rope(nn.Module):
    def __init__(self, model_path, model_config):
        super(wrapped_model_rope, self).__init__()
        model: Rope_MLP = torch.load(model_path)
        state_dim = model_config["state_dim"]
        action_dim = model_config["action_dim"]
        n_history = model_config["n_history"]
        horizon = model_config["horizon"]
        only_final_cost = model_config["only_final_cost"]
        device = model_config["device"]
        known_dim = (state_dim + action_dim) * n_history - action_dim
        self.state_dim = state_dim
        self.dim_of_work_space = 3
        self.num_kp = state_dim // self.dim_of_work_space
        self.action_dim = action_dim
        self.action_norm = torch.tensor(model_config["action_norm"], device=device)
        self.n_history = n_history
        assert n_history == 1 and action_dim == self.dim_of_work_space
        self.horizon = horizon
        self.only_final_cost = only_final_cost
        self.known_dim = known_dim
        self.device = device
        target_state, forbidden_area, rope_fixed_end, rope_length = \
        model_config["target_state"], model_config["forbidden_area"], model_config["rope_fixed_end"], model_config["rope_length"]
        self.register_buffer("target_state", torch.tensor(target_state, dtype=torch.float32, device=device))
        torch.manual_seed(0)
        curr_state = torch.rand((1, self.num_kp, self.dim_of_work_space), device=device)
        refer_pos = torch.rand((1, 1, self.dim_of_work_space), device=device)
        self.register_buffer("curr_state", curr_state)
        self.register_buffer("refer_pos", refer_pos)
        self.buffer_list = ["target_state", "curr_state", "refer_pos"]
        self.fixed_end = False
        if rope_fixed_end is not None:
            self.fixed_end = True
            self.register_buffer("rope_fixed_end", torch.tensor(rope_fixed_end, dtype=torch.float32, device=device))
            # self.buffer_list.append("rope_fixed_end")
        self.rope_length_norm = 2
        self.rope_length = rope_length
        if self.rope_length_norm == 2:
            self.rope_length = rope_length ** 2
        self.cost_norm = 1
        self.forbidden = False
        if forbidden_area is not None:
            self.forbidden = True
            forbidden_area = np.array(forbidden_area, dtype=np.float32)
            left_bottom, left_top, right_bottom, right_top = forbidden_area 
            left_center = (left_bottom + left_top) / 2
            left_half_edge = np.abs(left_top - left_bottom) / 2
            right_center = (right_bottom + right_top) / 2
            right_half_edge = np.abs(right_top - right_bottom) / 2
            self.register_buffer("left_center", torch.tensor(left_center, dtype=torch.float32, device=device))
            self.register_buffer("left_half_edge", torch.tensor(left_half_edge, dtype=torch.float32, device=device))
            self.register_buffer("right_center", torch.tensor(right_center, dtype=torch.float32, device=device))
            self.register_buffer("right_half_edge", torch.tensor(right_half_edge, dtype=torch.float32, device=device))
            # self.buffer_list.extend(["left_center", "left_half_edge", "right_center", "right_half_edge"])

        # # init model
        self.model = model.model
        self.step_weight_list = ((np.arange(1, horizon + 1) / horizon)).tolist()
        self.step_weights = torch.tensor(self.step_weight_list, device=device)

    def forward(self, act_seq):
        # return self.forward_new(act_seq)
        # B, act_seq_dim = act_seq.shape
        device = self.curr_state.device
        act_seq = act_seq.view(-1, self.horizon, self.action_dim) * self.action_norm.to(device)
        # act_seq_dim = self.action_dim * self.horizon
        # assert act_seq.shape[1] == act_seq_dim
        # curr_state: [B, num_kp, dim_of_work_space]
        curr_state = self.curr_state
        rope_length = self.rope_length
        # rope_length = 4
        batch = act_seq.shape[0]
        curr_state = curr_state.expand((batch, self.num_kp, self.dim_of_work_space))    # [B, num_kp, dim_of_work_space]
        # curr_state = curr_state.repeat((batch, 1, 1))
        pusher_pos_abs = self.refer_pos.expand((batch, 1, -1))         # [B, dim_of_work_space]
        # act_seq = act_seq.view(-1, self.horizon, self.action_dim)   # [B, horizon, action_dim]
        cost = 0
        fixed_end_penalty = 0
        penalty = 0
        forbidden_penalty = 0
        pusher_pos_penalty = 0
        penalty_factor = 1e5
        xyz_weight = torch.tensor([1, 1, 1.3], device=device)
        # forbidden_penalty_factor = penalty_factor
        # fixed_end_penalty_factor = penalty_factor
        # pusher_pos_penalty_factor = penalty_factor
        for j in range(self.horizon):
            # curr_action: [B, 1, action_dim (dim_of_work_space)]
            curr_action = act_seq[:, j:j+1, :]
            attr = torch.cat([curr_state, curr_state - curr_action], -1).view(batch, -1)
            outputs = self.model(attr).view(batch, self.num_kp, self.dim_of_work_space)
            curr_state = curr_state + outputs
            abs_state = curr_state + pusher_pos_abs
            # [B, num_kp, dim_of_work_space]
            diff = (self.target_state - abs_state)
            diff += diff[:, :, 0:1]*0.3
            # diff = diff * xyz_weight
            # diff = torch.cat([diff[:, :, 0:1], diff[:, :, 2:3]], -1)
            if self.cost_norm == 2:
                step_cost = (diff * diff).sum(dim=-1, keepdim=False)[:, :].sum(dim=-1, keepdim=True)
            else:
                step_cost = (torch.abs(diff)).sum(dim=-1, keepdim=False).sum(dim=-1, keepdim=True)
            step_cost = step_cost * self.step_weight_list[j]
            if self.fixed_end:
                rope_end_diff = abs_state - self.rope_fixed_end
                if self.rope_length_norm == 2:
                    rope_end_dis = (rope_end_diff * rope_end_diff).sum(dim=-1, keepdim=False)
                else:
                    rope_end_dis = torch.abs(rope_end_diff).sum(dim=-1, keepdim=False)
                rope_end_penalty = F.relu(rope_end_dis - rope_length).sum(dim=-1, keepdim=True)
                fixed_end_penalty += rope_end_penalty
            if self.forbidden:
                left_penalty, right_penalty, down_penalty = 0, 0, 0
                left_dis = torch.abs(abs_state - self.left_center)
                left_penalty = F.relu(self.left_half_edge - left_dis).min(dim=-1, keepdim=False).values
                right_dis = torch.abs(abs_state - self.right_center)
                right_penalty = F.relu(self.right_half_edge - right_dis).min(dim=-1, keepdim=False).values
                down_penalty = F.relu(0.5-abs_state[:, :, 2])
                forbidden_penalty += (left_penalty + right_penalty + down_penalty).sum(dim=-1, keepdim=True)
            
            cost += step_cost
            pusher_pos_abs = pusher_pos_abs + curr_action
            pusher_pos_penalty += F.relu(1.1 - pusher_pos_abs[:, :, 2]) + F.relu(-2.45 - pusher_pos_abs[:, :, 1])
            # height_reward += -abs(curr_action[:,:,-1])
            # if j == 7:
            #     height_reward = - abs(pusher_pos_abs[:, :, 2]-1.1)*0 + abs(pusher_pos_abs[:, :, 0]-0.56) 
                # height_reward =  - abs(abs_state[:, :, 2]-0.5).sum(dim=-1, keepdim=True)
            curr_state -= curr_action
        # penalty = forbidden_penalty_factor * forbidden_penalty + fixed_end_penalty_factor * fixed_end_penalty + pusher_pos_penalty_factor * pusher_pos_penalty
        penalty = (forbidden_penalty + fixed_end_penalty + pusher_pos_penalty) * penalty_factor
        if self.only_final_cost:
            # cost = (abs(diff)).sum(dim=1, keepdim=True)
            cost = ((diff)*(diff)).sum(dim=1, keepdim=False)[:, :].sum(dim=-1, keepdim=True)
        # cost *= cost_factor
        cost += penalty 
        # cost += height_reward*1

        return cost

    # def forward_new(self, act_seq):
    #     # B, act_seq_dim = act_seq.shape
    #     device = self.curr_state.device
    #     act_seq = act_seq.view(-1, self.horizon, self.action_dim) * self.action_norm.to(device)
    #     # act_seq_dim = self.action_dim * self.horizon
    #     # assert act_seq.shape[1] == act_seq_dim
    #     # curr_state: [B, num_kp, dim_of_work_space]
    #     curr_state = self.curr_state
    #     rope_length = self.rope_length
    #     xyz_weight = torch.tensor([1, 1, 1.3], device=device)
    #     # rope_length = 4
    #     batch = act_seq.shape[0]
    #     curr_state = curr_state.expand((batch, self.num_kp, self.dim_of_work_space))    # [B, num_kp, dim_of_work_space]
    #     # curr_state = curr_state.repeat((batch, 1, 1))
    #     pusher_pos_abs = self.refer_pos.expand((batch, 1, -1))         # [B, dim_of_work_space]
    #     act_seq = act_seq.view(-1, self.horizon, self.action_dim)   # [B, horizon, action_dim]
    #     cost = 0
    #     fixed_end_penalty = 0
    #     penalty = 0
    #     forbidden_penalty = 0
    #     pusher_pos_penalty = 0
    #     penalty_factor = 1e5
    #     forbidden_penalty_factor = penalty_factor
    #     fixed_end_penalty_factor = penalty_factor
    #     pusher_pos_penalty_factor = penalty_factor
    #     step_weights = self.step_weights.to(device)
    #     # abs_state_seq = torch.zeros((batch, self.horizon, self.num_kp, self.dim_of_work_space), device=device)
    #     # pusher_pos_abs_seq = torch.zeros((batch, self.horizon, self.action_dim), device=device)
    #     abs_state_seq = []
    #     pusher_pos_abs_seq = []
    #     for j in range(self.horizon):
    #         # curr_action: [B, 1, action_dim (dim_of_work_space)]
    #         curr_action = act_seq[:, j:j+1, :]
    #         attr = torch.cat([curr_state, curr_state - curr_action], -1).view(batch, -1)
    #         outputs = self.model(attr).view(batch, self.num_kp, self.dim_of_work_space)
    #         curr_state = curr_state + outputs
    #         abs_state = curr_state + pusher_pos_abs
    #         # abs_state_seq[:, j, :, :] += abs_state
    #         abs_state_seq.append(abs_state)

    #         pusher_pos_abs = pusher_pos_abs + curr_action
    #         # pusher_pos_abs_seq[:, j:j+1, :] += pusher_pos_abs
    #         pusher_pos_abs_seq.append(pusher_pos_abs)
            
    #         curr_state -= curr_action

    #     abs_state_seq = torch.stack(abs_state_seq, dim=1)
    #     pusher_pos_abs_seq = torch.stack(pusher_pos_abs_seq, dim=1)
    #     # abs_state_seq = torch.cat(abs_state_seq, dim=1).view(batch, self.horizon, self.num_kp, self.dim_of_work_space)
    #     # pusher_pos_abs_seq = torch.cat(pusher_pos_abs_seq, dim=1).view(batch, self.horizon, self.action_dim)

    #     # [B, horizon, num_kp, dim_of_work_space]
    #     diff = abs_state_seq - self.target_state.view(1, 1, self.num_kp, self.dim_of_work_space)
    #     # diff = diff * xyz_weight

    #     if self.cost_norm == 2:
    #         cost = (diff * diff).sum(dim=-1, keepdim=False)[:, :].sum(dim=-1, keepdim=False)
    #     else:
    #         cost = (torch.abs(diff)).sum(dim=-1, keepdim=False).sum(dim=-1, keepdim=False)
    #     cost = (cost * step_weights).sum(dim=1, keepdim=True)
    #     if self.fixed_end:
    #         rope_end_diff = abs_state_seq - self.rope_fixed_end
    #         if self.rope_length_norm == 2:
    #             rope_end_dis = (rope_end_diff * rope_end_diff).sum(dim=-1, keepdim=False)
    #         else:
    #             rope_end_dis = torch.abs(rope_end_diff).sum(dim=-1, keepdim=False)
    #         fixed_end_penalty = F.relu(rope_end_dis - rope_length).sum(dim=-1, keepdim=False).sum(dim=-1, keepdim=True)
    #     if self.forbidden:
    #         left_penalty, right_penalty, down_penalty = 0, 0, 0
    #         left_dis = torch.abs(abs_state_seq - self.left_center)
    #         # [B, horizon, num_kp]
    #         left_penalty = F.relu(self.left_half_edge - left_dis).min(dim=-1, keepdim=False).values
    #         right_dis = torch.abs(abs_state_seq - self.right_center)
    #         right_penalty = F.relu(self.right_half_edge - right_dis).min(dim=-1, keepdim=False).values
    #         down_penalty = F.relu(0.5-abs_state_seq[:, :, :, 2])
    #         forbidden_penalty = (left_penalty + right_penalty + down_penalty).sum(dim=-1, keepdim=False).sum(dim=-1, keepdim=True)
        
    #     pusher_pos_penalty = F.relu(1.1 - pusher_pos_abs_seq[:, :, :, 2]) + F.relu(-2.45 - pusher_pos_abs_seq[:, :, :, 1])
    #     pusher_pos_penalty = pusher_pos_penalty.sum(dim=-1, keepdim=False).sum(dim=-1, keepdim=True)
    #     penalty = forbidden_penalty_factor * forbidden_penalty + fixed_end_penalty_factor * fixed_end_penalty + pusher_pos_penalty_factor * pusher_pos_penalty
    #     if self.only_final_cost:
    #         # cost = (abs(diff)).sum(dim=1, keepdim=True)
    #         cost = ((diff)*(diff)).sum(dim=1, keepdim=False)[:, :].sum(dim=-1, keepdim=True)
    #     # cost *= cost_factor
    #     cost += penalty 
    #     # cost += height_reward*1

    #     return cost


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
