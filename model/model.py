import os, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules import PyG_GNNBlock, MLP_Block, PointNet_Block, Reori_MLP_Block


class Rope_Base(nn.Module):
    def __init__(self, config):
        super(Rope_Base, self).__init__()
        self.config = config

        self.state_dim = config['data']['state_dim']
        self.action_dim = config['data']['action_dim']

        self.n_history = config['train']['n_history']  # how many steps in history we are using as input
        self.model = None

    def preprocess(self, model_input):
        """
        input_dict: dictionary with keys 'observation' and 'action'
        """
        # reshape    
        # B, n_his, N, dim_of_work_space
        assert model_input['state'].shape[1] == self.n_history
        assert model_input['action'].shape[1] == self.n_history
        # 3d or 2d
        dim_of_work_space = self.config['data']['state_dim'] // self.config['data']['max_nobj']
        state_ori = model_input['state'].reshape(-1, self.n_history, self.state_dim//dim_of_work_space, dim_of_work_space) # [B, n_his, N, 2]
        action = model_input['action'] # [B, n_his, action_dim], action_dim = 4
        
        # use sdf
        state_ori = state_ori - action[:, :, None, :dim_of_work_space]
        state = state_ori.clone().detach()
        
        B, n_his, N, _ = state.shape
        device = state.device
        assert action.shape[-1] == dim_of_work_space*2
        
        # compute attributes for each node
        attr = torch.cat([state, state_ori - action[:, :, None, dim_of_work_space:]], -1) # [B, n_his, N, dim_of_work_space*2]
        
        # reshape
        state = state_ori.transpose(1, 2).reshape(B, N, -1) # [B, N, n_his*dim_of_work_space]
        attr = attr.transpose(1, 2).reshape(B, N, -1) # [B, N, n_his*dim_of_work_space*2]
        # print(f"state: {state.shape}")
        return attr, state
    
    def forward(self,
                input,  # dict: w/ keys ['observation', 'action']
                verbose=False
                ):
        # input['state']: [B, n_his, state_dim]
        # input['action']: [B, n_his, action_dim]
        attr, _ = self.preprocess(input)
        
        # output: B, N, dim_of_work_space
        # attr: [B, N, n_his*dim_of_work_space*2]
        # state: [B, N, n_his*dim_of_work_space]
        output = self.model(attr, )
        N = output.shape[1]
        # import pdb; pdb.set_trace()
        output = output.flatten(1, 2) # B, state_dim
        # always predict the residual
        # print(f"input['action'][:, -1].repeat((1,N,1)).flatten(1,2):{input['action'][:, -1].unsqueeze(1).repeat((1,N,1)).flatten(1,2).shape}")
        if self.config["short_push"]:
            output = output + input['state'][:, -1] - input['action'][:, -1] # B, state_dim
        else:
            output = output + input['state'][:, -1]
        
        # - input['action'][:, -1].unsqueeze(1).repeat((1,N,1)).flatten(1,2)
        return output

    def rollout_model(self,
                    input_dict,  # {"state_init": state_init, "pusher_pos": pusher_pos, "action_seq": action_seq,}
                    grad=False,
                    verbose=False,
                    ):
        state_init = input_dict['state_init'] # [B, n_his, state_dim]
        action_seq = input_dict['action_seq'] # [B, n_his+n_roll-1, action_dim]
        
        # sanity check
        B, n_history, obs_dim = state_init.shape
        _, n_tmp, action_dim = action_seq.shape
        # assert n_history == 1, "TODO: check the normalization is reasonable for n_history > 1"
        # if state_init and action_seq have same size in dim=1
        # then we are just doing 1 step prediction
        n_rollout = n_tmp - n_history + 1
        assert n_rollout > 0, "n_rollout = %d must be greater than 0" % (n_rollout)
        
        if grad:
            state_cur = state_init.requires_grad_(True)
        else:
            state_cur = state_init # .clone().detach()
        
        state_pred_list = []
        activation_list = []
        input_list = []
        
        for i in range(n_rollout):
            # [B, n_history, action_dim]
            actions_cur = action_seq[:, i:i + n_history] # .clone().detach()
            # state_cur is [B, n_history, state_dim]
            # action_cur is [B, n_history, action_dim]
            model_input = {'state': state_cur, 'action': actions_cur}

            # B, state_dim
            obs_pred = self.forward(model_input, verbose)
            # print(f"obs_pred: {obs_pred.shape} state_cur: {state_cur.shape} actions_cur: {actions_cur.shape}")
            # [B, n_history-1, state_dim] + [B, 1, state_dim] --> [B, n_history, state_dim]
            state_cur = torch.cat([state_cur[:, 1:].float(), obs_pred.unsqueeze(1)], 1)
            state_pred_list.append(obs_pred)

        # [B, n_rollout, state_dim] -> [B, n_rollout, state_dim]
        state_pred_tensor = torch.stack(state_pred_list, dim=1)
        # return absolute positions
        result_dict = {'state_pred': state_pred_tensor,
                        'input': input_list,
                        'activation': activation_list
                        }

        return result_dict

class Rope_MLP(Rope_Base):
    def __init__(self, config, ):
        super().__init__(config)
        self.model = MLP_Block(config)
    
    def preprocess(self, model_input):
        """
        input_dict: dictionary with keys 'observation' and 'action'
        """
        # reshape    
        # B, n_his, N, dim_of_work_space
        assert model_input['state'].shape[1] == self.n_history
        assert model_input['action'].shape[1] == self.n_history
        # 3d or 2d
        dim_of_work_space = self.config['data']['state_dim'] // self.config['data']['max_nobj']
        state_ori = model_input['state'].reshape(-1, self.n_history, self.state_dim//dim_of_work_space, dim_of_work_space) # [B, n_his, N, 3]
        action = model_input['action'] # [B, n_his, action_dim], action_dim = 3
        
        # the first point is the closest point to the pusher
        if self.config['train'].get('use_fixed_point_frame', False) and self.config["short_push"]:
            # [B, n_his, N, 3] - [B, n_his, 1, 3]
            state_ori = state_ori - state_ori[:, :, None, -1]
        # use sdf
        if self.config["short_push"]:
            state_ori = state_ori
        else:
            state_ori = state_ori - action[:, :, None, :dim_of_work_space]
        state = state_ori.clone().detach()
        
        B, n_his, N, _ = state.shape
        device = state.device
        
        # compute attributes for each node
        if self.config["short_push"]:
            assert action.shape[-1] == dim_of_work_space
            attr = torch.cat([state, state_ori - action[:, :, None]], -1) # [B, n_his, N, dim_of_work_space*2]
        else:
            assert action.shape[-1] == dim_of_work_space*2
            attr = torch.cat([state, state_ori - action[:, :, None, dim_of_work_space:]], -1) # [B, n_his, N, dim_of_work_space*2]
        
        # reshape
        state = state_ori.transpose(1, 2).reshape(B, N, -1).flatten(1,2) # [B, N, n_his*dim_of_work_space]
        attr = attr.transpose(1, 2).reshape(B, N, -1).flatten(1,2) # [B, N, n_his*dim_of_work_space*2]
        # print(f"state: {state.shape}")
        return attr, state
    
    def forward(self,
                input,  # dict: w/ keys ['observation', 'action']
                verbose=False
                ):
        # input['state']: [B, n_his, state_dim]
        # input['action']: [B, n_his, action_dim]
        attr, _ = self.preprocess(input)
        
        # output: B, N*dim_of_work_space
        # attr: [B, N*n_his*dim_of_work_space*2]
        # state: [B, N*n_his*dim_of_work_space]
        output = self.model(attr, )
        # always predict the residual
        # print(f"input['action'][:, -1].repeat((1,N,1)).flatten(1,2):{input['action'][:, -1].unsqueeze(1).repeat((1,N,1)).flatten(1,2).shape}")
        # the first point is the closest point to the pusher
        if self.config['train'].get('use_fixed_point_frame', False) and self.config["short_push"]:
            output = output + input['state'][:, -1] - input['action'][:, -1].repeat(1, self.state_dim//self.action_dim)
        elif self.config["short_push"]:
            # import pdb; pdb.set_trace()
            output = output + input['state'][:, -1] - input['action'][:, -1].repeat(1, self.state_dim//self.action_dim)
        else:
            output = output + input['state'][:, -1] # B, state_dim
        # - input['action'][:, -1].unsqueeze(1).repeat((1,N,1)).flatten(1,2)
        return output

class Reori_MLP(nn.Module):
    def __init__(self, config, ):
        super().__init__()
        self.config = config

        self.state_dim = config['data']['state_dim']
        self.action_dim = config['data']['action_dim']

        self.n_history = config['train']['n_history']  # how many steps in history we are using as input
        self.model = Reori_MLP_Block(config)
    
    def preprocess(self, model_input):
        """
        input_dict: dictionary with keys 'observation' and 'action'
        """
        # reshape    
        # B, n_his, N, dim_of_work_space
        assert model_input['state'].shape[1] == self.n_history
        assert model_input['action'].shape[1] == self.n_history
        
        # 3d or 2d
        dim_of_work_space = 2
        state_ori = model_input['state'].reshape(-1, self.n_history, self.state_dim//dim_of_work_space, dim_of_work_space) # [B, n_his, N, 2]
        action = model_input['action'].reshape(-1, self.n_history, self.action_dim//dim_of_work_space, dim_of_work_space) # [B, n_his, 2, 2], action_dim = 4
        pusher_pos = model_input['pusher_pos'].reshape(-1, self.n_history, self.action_dim//dim_of_work_space, dim_of_work_space) # [B, n_his, 2, 2]
        state = state_ori.clone().detach()
        
        B, n_his, N, _ = state.shape
        device = state.device
        
        # import pdb; pdb.set_trace()
        
        # compute attributes for each node
        if self.config["short_push"]:
            assert action.shape[-1] == dim_of_work_space
            attr = torch.cat([
                    torch.clip(0.25 - state[:,:,:,0:1], min=-0.05, max=0.05),\
                    state_ori - pusher_pos[:, :, 0:1], state_ori - pusher_pos[:, :, 1:2], \
                    state_ori - pusher_pos[:, :, 0:1] - action[:, :, 0:1], \
                    state_ori - pusher_pos[:, :, 1:2] - action[:, :, 1:2]], -1) # [B, n_his, N, dim_of_work_space*4+1]
            # attr = torch.cat([
            #         state_ori, pusher_pos, action], -2) # [B, n_his, N+2+2, dim_of_work_space]
        else:
            raise NotImplementedError
        
        # reshape
        state = state_ori.transpose(1, 2).reshape(B, N, -1).flatten(1,2) # [B, N*n_his*dim_of_work_space] # N = 4
        attr = attr.transpose(1, 2).reshape(B, -1) # [B, N*n_his*(dim_of_work_space*4+1)]
        # print(f"state: {state.shape}")
        return attr, state
    
    def forward(self,
                input,  # dict: w/ keys ['observation', 'action']
                verbose=False
                ):
        # input['state']: [B, n_his, state_dim]
        # input['action']: [B, n_his, action_dim]
        attr, _ = self.preprocess(input)
        
        # output: B, N*dim_of_work_space
        # attr: [B, N*n_his*dim_of_work_space*5]
        # state: [B, N*n_his*dim_of_work_space]
        output = self.model(attr, )
        # always predict the residual
        # print(f"input['action'][:, -1].repeat((1,N,1)).flatten(1,2):{input['action'][:, -1].unsqueeze(1).repeat((1,N,1)).flatten(1,2).shape}")
        if self.config["short_push"]:
            # import pdb; pdb.set_trace()
            output = output + input['state'][:, -1]
        else:
            raise NotImplementedError
        # - input['action'][:, -1].unsqueeze(1).repeat((1,N,1)).flatten(1,2)
        return output
    
    def rollout_model(self,
                    input_dict,  # {"state_init": state_init, "pusher_pos": pusher_pos, "action_seq": action_seq,}
                    grad=False,
                    verbose=False,
                    ):
        state_init = input_dict['state_init'] # [B, n_his, state_dim]
        action_seq = input_dict['action_seq'] # [B, n_his+n_roll-1, action_dim]
        pusher_pos = input_dict['pusher_pos'] # [B, n_his, 4]
        # sanity check
        B, n_history, obs_dim = state_init.shape
        _, n_tmp, action_dim = action_seq.shape
        # assert n_history == 1, "TODO: check the normalization is reasonable for n_history > 1"
        # if state_init and action_seq have same size in dim=1
        # then we are just doing 1 step prediction
        n_rollout = n_tmp - n_history + 1
        assert n_rollout > 0, "n_rollout = %d must be greater than 0" % (n_rollout)
        
        if grad:
            state_cur = state_init.requires_grad_(True)
        else:
            state_cur = state_init # .clone().detach()
        
        state_pred_list = []
        activation_list = []
        input_list = []
        
        for i in range(n_rollout):
            # [B, n_history, action_dim]
            actions_cur = action_seq[:, i:i + n_history] # .clone().detach()
            pusher_pos_cur = pusher_pos[:, i:i + n_history] # .clone().detach()
            # state_cur is [B, n_history, state_dim]
            # action_cur is [B, n_history, action_dim]
            model_input = {'state': state_cur, 'action': actions_cur, 'pusher_pos':pusher_pos_cur}

            # B, state_dim
            obs_pred = self.forward(model_input, verbose)
            # print(f"obs_pred: {obs_pred.shape} state_cur: {state_cur.shape} actions_cur: {actions_cur.shape}")
            # [B, n_history-1, state_dim] + [B, 1, state_dim] --> [B, n_history, state_dim]
            state_cur = torch.cat([state_cur[:, 1:].float(), obs_pred.unsqueeze(1)], 1)
            state_pred_list.append(obs_pred)

        # [B, n_rollout, state_dim] -> [B, n_rollout, state_dim]
        state_pred_tensor = torch.stack(state_pred_list, dim=1)
        # return absolute positions
        result_dict = {'state_pred': state_pred_tensor,
                        'input': input_list,
                        'activation': activation_list
                        }

        return result_dict

class PyG_GNN(nn.Module):
    def __init__(self, config):
        super(PyG_GNN, self).__init__()
        self.config = config

        self.state_dim = config['data']['state_dim']
        self.action_dim = config['data']['action_dim']

        self.n_history = config['train']['n_history']  # how many steps in history we are using as input
        self.model = PyG_GNNBlock(config)

    def get_Rr_Rs(self, N, device='cuda'):
        adj_matrix = torch.ones((N, N), dtype=torch.float32, device=device)
        rels = adj_matrix.nonzero()
        Rr = torch.zeros((N*N, N), dtype=torch.float32, device=device)
        Rs = torch.zeros((N*N, N), dtype=torch.float32, device=device)
        Rr[torch.arange(N*N), rels[:, 0]] = 1
        Rs[torch.arange(N*N), rels[:, 1]] = 1
        return Rr.unsqueeze(0), Rs.unsqueeze(0)
    
    def preprocess(self, model_input):
        state_ori = model_input['state'].reshape(-1, self.n_history, self.state_dim//2, 2) # [B, n_his, N, 2]
        B, n_his, N, _ = state_ori.shape
        action = model_input['action'] # [B, n_his, action_dim], action_dim = 4
        Rr, Rs = self.get_Rr_Rs(N)
        Rr = Rr.to(action.device).expand(B, -1, -1)
        Rs = Rs.to(action.device).expand(B, -1, -1)
        if self.config['task_name'] == 'pushing_rope':
            # 3d or 2d
            dim_of_work_space = self.config['data']['state_dim'] // self.config['data']['max_nobj']
        else:
            dim_of_work_space = 2
            
        if self.config["short_push"]:
            state_ori = state_ori
        else:
            state_ori = state_ori - action[:, :, None, :2]
        state = state_ori.clone().detach()
        if self.config["short_push"]:
            attr = torch.cat([state, state_ori - action[:, :, None, ]], -1)
        else:
            attr = torch.cat([state, state_ori - action[:, :, None, dim_of_work_space:]], -1) # [B, n_his, N, 2+2]
        state = state_ori.transpose(1, 2).reshape(B, N, -1) # [B, N, n_his*2]
        attr = attr.transpose(1, 2).reshape(B, N, -1) # [B, N, n_his*(2+2)]
        return attr, state, Rr, Rs
    
    def forward(self,
                input,  # dict: w/ keys ['observation', 'action']
                verbose=False
                ):
        # input['state']: [B, n_his, state_dim]
        # input['action']: [B, n_his, action_dim]
        attr, state, Rr, Rs = self.preprocess(input)
        
        # output: B, N, 2
        # attr: [B, N, n_his*4]
        # state: [B, N, n_his*2]
        # Rr: [B, n_rels, N]
        # Rs: [B, n_rels, N]
        output = self.model(attr, state, Rr, Rs)
        N = output.shape[1]
        output = output.flatten(1, 2) # B, state_dim
        # always predict the residual
        # print(f"input['action'][:, -1].repeat((1,N,1)).flatten(1,2):{input['action'][:, -1].unsqueeze(1).repeat((1,N,1)).flatten(1,2).shape}")
        output = output + input['state'][:, -1] # B, state_dim
        #  - input['action'][:, -1].unsqueeze(1).repeat((1,N,1)).flatten(1,2)
        return output

    def rollout_model(self,
                    input_dict,  # {"state_init": state_init, "pusher_pos": pusher_pos, "action_seq": action_seq,}
                    grad=False,
                    verbose=False,
                    ):
        state_init = input_dict['state_init'] # [B, n_his, state_dim]
        action_seq = input_dict['action_seq'] # [B, n_his+n_roll-1, action_dim]
        
        # sanity check
        B, n_history, obs_dim = state_init.shape
        _, n_tmp, action_dim = action_seq.shape
        # assert n_history == 1, "TODO: check the normalization is reasonable for n_history > 1"
        # if state_init and action_seq have same size in dim=1
        # then we are just doing 1 step prediction
        n_rollout = n_tmp - n_history + 1
        assert n_rollout > 0, "n_rollout = %d must be greater than 0" % (n_rollout)
        
        if grad:
            state_cur = state_init.requires_grad_(True)
        else:
            state_cur = state_init # .clone().detach()
        
        state_pred_list = []
        activation_list = []
        input_list = []
        
        for i in range(n_rollout):
            # [B, n_history, action_dim]
            actions_cur = action_seq[:, i:i + n_history] # .clone().detach()
            # state_cur is [B, n_history, state_dim]
            # action_cur is [B, n_history, action_dim]
            model_input = {'state': state_cur, 'action': actions_cur}

            # B, state_dim
            obs_pred = self.forward(model_input, verbose)
            # print(f"obs_pred: {obs_pred.shape} state_cur: {state_cur.shape} actions_cur: {actions_cur.shape}")
            # [B, n_history-1, state_dim] + [B, 1, state_dim] --> [B, n_history, state_dim]
            state_cur = torch.cat([state_cur[:, 1:].float(), obs_pred.unsqueeze(1)], 1)
            state_pred_list.append(obs_pred)

        # [B, n_rollout, state_dim] -> [B, n_rollout, state_dim]
        state_pred_tensor = torch.stack(state_pred_list, dim=1)
        # return absolute positions
        result_dict = {'state_pred': state_pred_tensor,
                        'input': input_list,
                        'activation': activation_list
                        }

        return result_dict


    def update_bounds(self, input, activation):
        # input: [rollout step 1: [layer 1 inputs: [min: [#neurons at layer 1], max: [#neurons at layer 1]], layer 2 inputs, ...], rollout step 2...]
        # activation: [data 1: [rollout step 1: [layer 1 activations: [batch_size, #neurons at layer 1], layer 2 activations, ...], rollout step 2...], data 2: ...]
        for rollout in input:
            for i, layer in enumerate(rollout):
                assert (len(layer) == 2)
                self.lb_input[i] = np.minimum(self.lb_input[i], layer[0])
                self.ub_input[i] = np.maximum(self.ub_input[i], layer[1])
        for rollout in activation:
            for i, layer in enumerate(rollout):
                assert (len(layer) == 2)
                self.lb_activation[i] = np.minimum(self.lb_activation[i], layer[0])
                self.ub_activation[i] = np.maximum(self.ub_activation[i], layer[1])

    def get_activation_stat(self):
        return [layer.out_features for layer in self.layers if isinstance(layer, nn.Linear)]