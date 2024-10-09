import torch
from auto_LiRPA import BoundedTensor, PerturbationLpNorm
import arguments
from load_model import Customized

@torch.no_grad()
def customized_input_split_branching(net, dom_lb, x_L, x_U, lA, thresholds,
                          branching_method, split_depth=1, num_iter=0, **kwargs):
    """
    Produce input split according to branching methods.
    """
    x_L = x_L.flatten(1)
    x_U = x_U.flatten(1)
    num_domains, num_inputs = x_L.shape[:2]
    sample_distribution = kwargs.get('sample_distribution', None)
    if sample_distribution is not None:
        # count_ge, count_le: number of good samples ge/le the center on every input dimension in every domain
        # shape: [1, 1, num_domains, num_inputs]
        count_ge = sample_distribution['count_ge']
        count_le = sample_distribution['count_le']
        # # loss_ge_mean, loss_le_mean: mean of the loss of good samples ge/le the center on every input dimension in every domain
        # # larger loss, better sample
        # # shape: [1, 1, num_domains, num_inputs]
        # loss_ge_mean = sample_distribution['loss_ge_mean']
        # loss_le_mean = sample_distribution['loss_le_mean']
        # # loss_ge_max, loss_le_max: max of the loss of good samples ge/le the center on every input dimension in every domain
        # # shape: [1, 1, num_domains, num_inputs]
        # loss_ge_max = sample_distribution['loss_ge_max']
        # loss_le_max = sample_distribution['loss_le_max']

        count_diff = (count_ge - count_le).abs().view(num_domains, num_inputs)
        x_diff = (x_U - x_L).view(num_domains, num_inputs)
        split_idx =  torch.topk(count_diff*x_diff, split_depth, dim=-1).indices
        # print(count_diff*x_diff)
    else: 
        # split_idx = torch.randint(low = 0, high = 4, 
        #                     size=(x_L.shape[0], 1), device=x_L.device)
        split_idx = torch.ones(x_L.shape[0], 1, device=x_L.device) * (int(x_L.shape[-1]//2))
    print("split_idx: ", split_idx.flatten())
    # return torch.randint(low = int(x_L.shape[-1]//2)-1, high = int(x_L.shape[-1]//2)+1, 
    #                         size=(x_L.shape[0], 1), device=x_L.device)
    # return torch.randint(low = 0, high = 4, 
    #                         size=(x_L.shape[0], 1), device=x_L.device)
    # return torch.ones(x_L.shape[0], 1, device=x_L.device) * (int(x_L.shape[-1]//2))
    return split_idx

@torch.no_grad()
def customized_input_split_branching_1(net, dom_lb, x_L, x_U, lA, thresholds,
                          branching_method, split_depth=1, num_iter=0, **kwargs):
    """
    Produce input split according to branching methods.
    """
    x_L = x_L.flatten(1)
    x_U = x_U.flatten(1)
    num_domains, num_inputs = x_L.shape[:2]
    sample_distribution = kwargs.get('sample_distribution', None)
    if sample_distribution is not None:
        # count_ge, count_le: number of good samples ge/le the center on every input dimension in every domain
        # shape: [1, 1, num_domains, num_inputs]
        count_ge = sample_distribution['count_ge']
        count_le = sample_distribution['count_le']
        x_diff = (x_U - x_L).view(num_domains, num_inputs)
        split_idx =  torch.topk(x_diff, split_depth, dim=-1).indices
    else: 
        split_idx = torch.ones(x_L.shape[0], 1, device=x_L.device) * (int(x_L.shape[-1]//2))
    print("split_idx: ", split_idx.flatten())
    return split_idx

@torch.no_grad()
def customized_input_split_branching_2(net, dom_lb, x_L, x_U, lA, thresholds,
                          branching_method, split_depth=1, num_iter=0, **kwargs):
    """
    Produce input split according to branching methods.
    """
    x_L = x_L.flatten(1)
    x_U = x_U.flatten(1)
    num_domains, num_inputs = x_L.shape[:2]
    sample_distribution = kwargs.get('sample_distribution', None)
    if sample_distribution is not None:
        # count_ge, count_le: number of good samples ge/le the center on every input dimension in every domain
        # shape: [1, 1, num_domains, num_inputs]
        count_ge = sample_distribution['count_ge']
        count_le = sample_distribution['count_le']
        count_diff = (count_ge - count_le).abs().view(num_domains, num_inputs)
        split_idx =  torch.topk(count_diff, split_depth, dim=-1).indices
    else: 
        split_idx = torch.ones(x_L.shape[0], 1, device=x_L.device) * (int(x_L.shape[-1]//2))
    print("split_idx: ", split_idx.flatten())
    return split_idx