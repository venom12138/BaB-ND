import os, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def pos2posemb(pos, num_pos_feats=32, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats//2, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats//2)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    
    posemb = torch.cat((pos_x, pos_y), dim=-1)
    return posemb

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        # assert config["task_name"] in ["box_pushing", "pushing_T", "merging_L", "inserting"]
        self.config = config
        self.state_dim = config["data"]["state_dim"]
        if "latent" in config["task_name"]:
            self.state_dim = config["latent"]["latent_dim"]
        self.action_dim = config["data"]["action_dim"]
        assert self.action_dim == 2
        self.n_history = config["train"]["n_history"]  # how many steps in history we are using as input

        self.layers = []
        # self.norm = nn.ModuleList()
        input_dim = (self.state_dim + self.action_dim) * self.n_history
        for i in config["train"]["architecture"]:
            self.layers.append(nn.Linear(input_dim, i))
            self.layers.append(nn.ReLU())
            # self.norm.append(nn.BatchNorm1d(i))
            input_dim = i
        self.layers.append(nn.Linear(input_dim, self.state_dim))
        self.model = nn.Sequential(*self.layers)

        N = 3  # corresponds to three probabilities: ReLU, ID or Zero
        masks = []
        for i in config["train"]["architecture"]:
            masks.append(nn.Parameter(torch.log(torch.ones(i, N) / N)))
        self.mask_prob = nn.ParameterList(masks)

        # create the initial bounds for x
        self.lb_input = []
        self.ub_input = []
        for i in config["train"]["architecture"]:
            self.lb_input.append(np.ones(i) * np.inf)
            self.ub_input.append(np.ones(i) * (-np.inf))
        self.lb_input.insert(0, np.ones((self.state_dim + self.action_dim) * self.n_history) * np.inf)
        del self.lb_input[-1]
        self.ub_input.insert(0, np.ones((self.state_dim + self.action_dim) * self.n_history) * (-np.inf))
        del self.ub_input[-1]

        # create the initial bounds for w*x + b
        self.lb_activation = []
        self.ub_activation = []
        for i in config["train"]["architecture"]:
            self.lb_activation.append(np.ones(i) * np.inf)
            self.ub_activation.append(np.ones(i) * (-np.inf))

    def forward(
        self, input, verbose=False  # dict: w/ keys ['observation', 'action']
    ):  # torch.FloatTensor B x state_dim
        """
        input['observation'].shape is [B, n_history, obs_dim]
        input['action'].shape is [B, n_history, action_dim]
        mask: shape is [#Relu layers, 1, #output units at that Relu layer]
        1 means not affected, 0 means negative/zero, 2 means positive/ID
        """

        # [B, n_history, obs_dim]
        # state is relative to the current pusher position
        state = input["observation"]
        # [B, n_history, action_dim]
        action = input["action"]

        B, n_history, state_dim = state.shape
        # import pdb; pdb.set_trace()
        # flatten the observation and action inputs
        # then concatenate them
        # thus of shape (B, n_history * obs_dim + n_history * action_dim)
        input = torch.cat([state.view(B, -1), action.view(B, -1)], 1).float()

        activations = []
        inputs = []
        
        # go through all the layers
        for i in range(0, len(self.layers) - 1, 2):
            if verbose:
                inputs.append(input)
            # pass it through the affine function
            a = self.model[i](input)
            # print(list(self.model[i].parameters()))
            if verbose:
                activations.append(a)
            # a = self.norm[i//2](a)
            a = F.relu(a)
            input = a

        # apply the final affine layer
        output = self.model[len(self.layers) - 1](input)

        if verbose:
            activations = [[d.min(dim=0)[0].tolist(), d.max(dim=0)[0].tolist()] for d in activations]
            inputs = [[d.min(dim=0)[0].tolist(), d.max(dim=0)[0].tolist()] for d in inputs]

        # output: B x state_dim
        output = output + state[:, -1]
        if "latent" not in self.config["task_name"]:
            # adjust the output to be relative to the next pusher position
            output = output - action[:, -1, :].repeat(1, self.state_dim // self.action_dim)
        return output, inputs, activations

    # transform one object's keypoints with rotation and translation
    def rt_transform(self, curr_state, output):
        # curr_state: B x state_dim per object
        state_dim = curr_state.shape[1]
        keypoints = curr_state.reshape(-1, state_dim // 2, 2)
        P_c = torch.mean(keypoints, dim=1, keepdim=True)
        delta_theta = output[:, 0]
        delta_x = output[:, 1]
        delta_y = output[:, 2]
        cos_theta = torch.cos(delta_theta)
        sin_theta = torch.sin(delta_theta)
        R_t = torch.stack(
            [torch.stack([cos_theta, -sin_theta], dim=1), torch.stack([sin_theta, cos_theta], dim=1)], dim=1
        )
        rel_kps = keypoints - P_c
        rel_kps = rel_kps.permute(0, 2, 1)
        rel_kps = torch.bmm(R_t, rel_kps)
        rel_kps = rel_kps.permute(0, 2, 1)
        T_t = torch.stack([delta_x, delta_y], dim=1).unsqueeze(1)
        output = rel_kps + P_c + T_t
        output = output.reshape(-1, state_dim)
        return output

    def rollout_model(
        self,
        input_dict, # {"state_init": state_init,  "action_seq": action_seq,}
        grad=False,
        verbose=False,
    ):
        """
        Rolls out the dynamics model for the given number of steps
        """
        state_init = input_dict['state_init'] # [B, n_his, state_dim]
        action_seq = input_dict['action_seq'] # [B, n_his+n_roll-1, action_dim]
        assert len(state_init.shape) == 3
        assert len(action_seq.shape) == 3

        B, n_history, obs_dim = state_init.shape
        _, n_tmp, action_dim = action_seq.shape

        assert n_history == 1, "TODO: check the normalization is reasonable for n_history > 1"

        # if state_init and action_seq have same size in dim=1
        # then we are just doing 1 step prediction
        n_rollout = n_tmp - n_history + 1
        assert n_rollout > 0, "n_rollout = %d must be greater than 0" % (n_rollout)

        if grad:
            state_cur = state_init.requires_grad_(True)
        else:
            state_cur = state_init.clone().detach()
        state_pred_list = []
        activation_list = []
        input_list = []

        for i in range(n_rollout):
            # [B, n_history, action_dim]
            actions_cur = action_seq[:, i : i + n_history].clone().detach()
            # state_cur is [B, n_history, obs_dim]
            # # save previous absolute position
            # prev_absolute_position = state_cur[:, 0, :2].clone().detach()

            model_input = {"observation": state_cur, "action": actions_cur}

            # [B, obs_dim]
            obs_pred, inputs, activations = self.forward(model_input, verbose)

            activation_list.append(activations)
            input_list.append(inputs)

            # [B, n_history-1, action_dim] + [B, 1, action_dim] --> [B, n_history, action_dim]
            state_cur = torch.cat([state_cur[:, 1:].float(), obs_pred.unsqueeze(1)], 1)
            state_pred_list.append(obs_pred)

        # [B, n_rollout, obs_dim]
        state_pred_tensor = torch.stack(state_pred_list, axis=1)

        result_dict = {"state_pred": state_pred_tensor, "input": input_list, "activation": activation_list}

        return result_dict

    def ibp_forward(self, lb, ub):
        """
        Perform Interval Bound Propagation for a given model and input bounds.
        :param model: The neural network model.
        :param lb: Lower bound of the input.
        :param ub: Upper bound of the input.
        :return: Tuple of (lower bound, upper bound) of the output.
        """
        if lb.ndim == 1:
            lb = lb.unsqueeze(0)
            ub = ub.unsqueeze(0)
        bounds_per_layer = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                weight = layer.weight
                bias = layer.bias
                mid = (lb + ub) / 2.0
                diff = (ub - lb) / 2.0
                weight_abs = weight.abs()
                center = torch.addmm(bias, mid, weight.t())
                deviation = diff.matmul(weight_abs.t())
                ub = center + deviation
                lb = center - deviation
                bounds_per_layer.append((lb.clone(), ub.clone()))
            elif isinstance(layer, nn.ReLU):
                lb = F.relu(lb)
                ub = F.relu(ub)
            else:
                raise NotImplementedError("IBP is not implemented for layer: %s" % layer)
        bounds_per_layer.append((lb.clone(), ub.clone()))
        return bounds_per_layer

    def update_bounds(self, input, activation):
        # input: [rollout step 1: [layer 1 inputs: [min: [#neurons at layer 1], max: [#neurons at layer 1]], layer 2 inputs, ...], rollout step 2...]
        # activation: [data 1: [rollout step 1: [layer 1 activations: [batch_size, #neurons at layer 1], layer 2 activations, ...], rollout step 2...], data 2: ...]
        for rollout in input:
            for i, layer in enumerate(rollout):
                assert len(layer) == 2
                self.lb_input[i] = np.minimum(self.lb_input[i], layer[0])
                self.ub_input[i] = np.maximum(self.ub_input[i], layer[1])
        for rollout in activation:
            for i, layer in enumerate(rollout):
                assert len(layer) == 2
                self.lb_activation[i] = np.minimum(self.lb_activation[i], layer[0])
                self.ub_activation[i] = np.maximum(self.ub_activation[i], layer[1])

    def get_activation_stat(self):
        return self.config["train"]["architecture"]

