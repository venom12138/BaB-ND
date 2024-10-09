import os, sys
cwd = os.getcwd()
sys.path.append(cwd)
import numpy as np
import torch
import torch.nn as nn


class synthetic_model(nn.Module):
    def __init__(self):
        super(synthetic_model, self).__init__()

    def forward(self, u):
        y = self.get_y(u)
        return torch.sum(y, dim=(1), keepdim=True)

    
    def get_y(self, u):
        y = 5*(u**2) + 2 * torch.cos(50*u)
        return y