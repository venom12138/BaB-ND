import torch
import torch.nn as nn
import torch
import torch.nn as nn

from model.layers import GNN_layer, MLP_layer
from model.pointnet import PointNetfeat

class MLP_Block(nn.Module):
    def __init__(self, config,):
        super(MLP_Block, self).__init__()
        self.config = config        
        self.blocks = nn.ModuleList()
        dim_of_work_space = self.config['data']['state_dim'] // self.config['data']['max_nobj']
        # N * n_his * dim_of_work_space * 2
        attr_dim = self.config['data']['max_nobj'] * config['train']['n_history'] * dim_of_work_space * 2 
        output_dim = self.config['data']['max_nobj'] * dim_of_work_space
        self.blocks.append(MLP_layer(attr_dim, config['train']['model_param']['nf_particle']))
        for _ in range(config['train']['model_param']['layers']):
            self.blocks.append(MLP_layer(config['train']['model_param']['nf_particle'], config['train']['model_param']['nf_particle']))
        self.predictor = nn.Linear(config['train']['model_param']['nf_particle'], output_dim)
        
    def forward(self, attr):
        for blk in self.blocks:
            attr = blk(attr)
        return self.predictor(attr) # B, N*n_his*dim_of_work_space

class Reori_MLP_Block(nn.Module):
    def __init__(self, config,):
        super(Reori_MLP_Block, self).__init__()
        self.config = config        
        self.blocks = nn.ModuleList()
        dim_of_work_space = 2
        # [B, N*n_his*dim_of_work_space*5]
        attr_dim = 4 * config['train']['n_history'] * (dim_of_work_space * 4 + 1)
        # attr_dim = 8 * config['train']['n_history'] * dim_of_work_space
        output_dim = 4 * dim_of_work_space
        self.blocks.append(MLP_layer(attr_dim, config['train']['model_param']['nf_particle']))
        for _ in range(config['train']['model_param']['layers']):
            self.blocks.append(MLP_layer(config['train']['model_param']['nf_particle'], config['train']['model_param']['nf_particle']))
        self.predictor = nn.Linear(config['train']['model_param']['nf_particle'], output_dim)
        
    def forward(self, attr):
        for blk in self.blocks:
            attr = blk(attr)
        return self.predictor(attr) # B, N*dim_of_work_space

class PointNet_Block(nn.Module):
    def __init__(self, config,):
        super(PointNet_Block, self).__init__()
        self.config = config        
        dim_of_work_space = self.config['data']['state_dim'] // self.config['data']['max_nobj']
        attr_dim = config['train']['n_history'] * dim_of_work_space * 2 # action+state
        self.block = PointNetfeat(input_dim=attr_dim)
        self.predictor = nn.Sequential(*[nn.Linear(192, dim_of_work_space),])
    def forward(self, attr):
        particle_effect, _, _ = self.block(attr)
        return self.predictor(particle_effect) # B, N, dim_of_work_space

class PyG_GNNBlock(nn.Module):
    def __init__(self, config,):
        super(PyG_GNNBlock, self).__init__()
        self.config = config        
        self.blocks = nn.ModuleList()
        # 
        if self.config['task_name'] == 'pushing_rope':
            # 3d or 2d
            dim_of_work_space = self.config['data']['state_dim'] // self.config['data']['max_nobj']
        else:
            dim_of_work_space = 2
        for _ in range(config['train']['model_param']['layers']):
            self.blocks.append(GNN_layer(config))
        self.predictor = nn.Linear(config['train']['model_param']['nf_particle'], dim_of_work_space)
        self.nf_particle = self.config['train']['model_param']['nf_particle']
    def forward(self, attr, state_norm, Rr, Rs):
        # attr: (bs, N, attr_dim)
        # state: (bs, N, state_dim)
        # calculate particle encoding
        bs = attr.size(0)
        N = attr.size(1)
        particle_effect = torch.zeros((bs, N, self.nf_particle)).to(attr.device) # (bs, N, nf_effect)
        for blk in self.blocks:
            particle_effect = blk(particle_effect, attr, state_norm, Rr, Rs)
        
        return self.predictor(particle_effect) # B, N, 2