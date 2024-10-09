#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
## Copyright (C) 2021-2022, Huan Zhang <huan@huan-zhang.com>           ##
##                     Kaidi Xu, Zhouxing Shi, Shiqi Wang              ##
##                     Linyi Li, Jinqi (Kathryn) Chen                  ##
##                     Zhuolin Yang, Yihan Wang                        ##
##                                                                     ##
##      See CONTRIBUTORS for author contacts and affiliations.         ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
r"""
This file is a simplified version of `custom_op.py`. a_i x_i + b_i -> x_i
This file is an example of defining a custom operation and providing its
relaxations for bound computation. Here we consider a modified ReLU
function which is a mixture between a identity function and ReLU function:
             / x_i   if m_i = 1,
    f(x_i) = |
             \ ReLU(x_i)       if m_i = 0.
where m is the mask controlling the behavior of this function. 
We consider perturbations on x.

Note that if you also want to conduct branch and bound on your customized
op, you may also need to customize BaB code, so the complete verifier is
skipped here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA import register_custom_op
from auto_LiRPA.bound_ops import BoundRelu, Interval


class IDMaskedReluOp(torch.autograd.Function):
    """A relu function with some neurons replaced with identity operations."""
    @staticmethod
    def forward(ctx, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply a masked ReLU operation.
        mask = 1: Identity operation, mask = 0: ReLU
        """
        ctx.save_for_backward(input, mask)
        return torch.where(mask == 1, input, input.clamp(min=0))

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute gradient of loss with respect to input.
        """
        input, mask = ctx.saved_tensors
        grad_input = torch.where(mask == 1, grad_output, grad_output * (input >= 0))
        return grad_input, None  # Return None if no gradient for mask

    @staticmethod
    def symbolic(g, input, mask):
        # This will be parsed as a custom operation when doing the ONNX conversion.
        return g.op("customOp::IDMaskedRelu", input, mask)


class IDMaskedRelu(nn.Module):
    """Create a module to wrap the parameters for IDMaskedReluOp."""
    def __init__(self, size, mask=None):
        super().__init__()
        if isinstance(size, int):
            size = (size, )
        # mask is element-wise.
        assert mask.size() == size
        self.register_buffer('mask', mask.to(dtype=torch.get_default_dtype()))

    def forward(self, input):
        # mask = 1 => using identity operation input, mask = 0 => using ReLU
        return IDMaskedReluOp.apply(input, self.mask)


class BoundIDMaskedRelu(BoundRelu):
    """This class defines how we compute the bounds for our customized Relu function."""

    def forward(self, x, mask):
        """Regular forward propagation (e.g., for evaluating clean accuracy)."""
        # Save the shape, which will be used in other parts of the verifier.
        self.shape = x.shape[1:]
        if self.flattened_nodes is None:
            self.flattened_nodes = x[0].reshape(-1).shape[0]
        return IDMaskedReluOp.apply(x, mask)

    def interval_propagate(self, x, mask):
        """Interval bound propagation (IBP)."""
        # Each x, mask is a tuple, or a Interval object representing lower and upper bounds.
        # We assume Linf norm perturbation on input.
        assert Interval.get_perturbation(x)[0] == float("inf")
        x_L, x_U = x[0], x[1]  # The inputs (x)
        # We assume no perturbations on mask.
        mask = mask[0]
        # relu_lb = x_L.clamp(min=0), linear_lb = x_L
        # relu_ub = x_U.clamp(min=0), linear_ub = x_U
        # Select the final bounds according to the mask.
        final_lb = torch.where(mask == 1, x_L, x_L.clamp(min=0))
        final_ub = torch.where(mask == 1, x_U, x_U.clamp(min=0))
        return final_lb, final_ub

    def _backward_relaxation(self, last_lA, last_uA, x, start_node, unstable_idx):
        """Element-wise CROWN relaxation for our special ReLU activation function."""
        # Call parent class to relax ReLU neurons.
        upper_d, upper_b, lower_d, lower_b, lb_lower_d, ub_lower_d, _, _, alpha_lookup_idx = super()._backward_relaxation(
                last_lA, last_uA, x, start_node, unstable_idx)
        # Modify the relaxation coefficients for these linear neurons.
        # Compute the relaxed coefficients with mask.
        relu_mask = 1.0 - self._mask
        id_slope = self._mask

        # # Update coefficients with the mask.
        # # Use in-place operations for efficiency.
        # upper_d.mul_(relu_mask).add_(id_slope)
        # upper_b.mul_(relu_mask)
        
        # if lower_d is not None:
        #     lower_d.mul_(relu_mask).add_(id_slope)
        # else:
        #     # Update slopes separately for lb and ub, if necessary.
        #     if lb_lower_d is not None:
        #         lb_lower_d.mul_(relu_mask).add_(id_slope)
        #     if ub_lower_d is not None:
        #         ub_lower_d.mul_(relu_mask).add_(id_slope)

        upper_d = upper_d.mul(relu_mask).add(id_slope)
        upper_b = upper_b.mul(relu_mask)

        if lower_d is not None:
            lower_d = lower_d.mul(relu_mask).add(id_slope)
        else:
            # Update slopes separately for lb and ub, if necessary.
            if lb_lower_d is not None:
                lb_lower_d = lb_lower_d.mul(relu_mask).add(id_slope)
            if ub_lower_d is not None:
                ub_lower_d = ub_lower_d.mul(relu_mask).add(id_slope)

        assert lower_b is None  # For ReLU, there is no lower bias (=0)
        # For ID, there is no lower bias (=0) too.
        return upper_d, upper_b, lower_d, lower_b, lb_lower_d, ub_lower_d, None, None, alpha_lookup_idx

    def bound_backward(self, last_lA, last_uA, x, mask, **kwargs):
        """Backward LiRPA (CROWN) bound propagation."""
        # These are additional variabels that will be used in _backward_relaxation(), so we save them here.
        self._mask = mask.buffer  # These are registered as buffers; see class BoundBuffer.
        # The parent class will call _backward_relaxation() and obtain the relaxations,
        # and that's all we need; after obtaining linear relaxations for each neuron, other
        # parts of class BoundRelu can be reused.
        As, lbias, ubias = super().bound_backward(last_lA, last_uA, x, **kwargs)
        # Returned As = [(lA, uA)]; these A matrices are for input x.
        # Our customized ReLU has ine additional buffer as inputs; we need to set their
        # corresponding A matrices to None. The length of As must match the number of inputs
        # of this customize function.
        As += [(None, None)]
        return As, lbias, ubias


class id_masked_relu_model(nn.Module):
    """Model for testing."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 4, stride=2, padding=1)
        # Using our customized ReLU function to replace the original ReLU function.
        self.id_masked_relu1 = IDMaskedRelu(size=(16,16,16))
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.id_masked_relu2 = IDMaskedRelu(size=(32,8,8))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(32*8*8,100)
        self.id_masked_relu3 = IDMaskedRelu(100)
        self.linear2 = nn.Linear(100, 10)
        # Register the customized op in auto_LiRPA library.
        register_custom_op("customOp::IDMaskedRelu", BoundIDMaskedRelu)

    def forward(self, x):
        out = self.conv1(x)
        out = self.id_masked_relu1(out)
        # out = torch.nn.functional.relu(out)
        out = self.conv2(out)
        out = self.id_masked_relu2(out)
        # out = torch.nn.functional.relu(out)
        out = self.flatten(out)  # Flatten must be after activation for most efficient computation.
        out = self.linear1(out)
        out = self.id_masked_relu3(out)
        # out = torch.nn.functional.relu(out)
        out = self.linear2(out)
        return out

