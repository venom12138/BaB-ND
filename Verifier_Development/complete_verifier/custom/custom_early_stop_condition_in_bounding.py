import torch
from auto_LiRPA.operators import Bound, BoundRelu, BoundLinear, BoundAdd, BoundSub, BoundAbs
from auto_LiRPA import BoundedModule
from auto_LiRPA.utils import prod
import math
from load_model import Customized

class customized_early_stop_conditioner():
    # self defined
    def __init__(self, max_relu_nodes, max_prop_nodes) -> None:
        if max_relu_nodes is None:
            max_relu_nodes = math.inf
        if max_prop_nodes is None:
            max_prop_nodes = math.inf
        self.max_relu_nodes = max_relu_nodes
        self.max_prop_nodes = max_prop_nodes
        self.clear()

    # api should be the same
    # function name, ...
    # clear the conditioner
    def clear(self):
        self.curr_relu_nodes = 0
        self.curr_prop_nodes = 0
        self.relu_nodes = []
        self.intermediate_nodes = []
        self.prop_nodes = []

    # api should be the same
    # function name, ...
    # reset the conditioner by increasing max_prop_nodes
    # since the bound can be unsound and too tight,
    # it might loosen the bounds
    def loosen(self, **kwargs):
        self.max_prop_nodes += 1
        self.clear()

    # api should be the same
    # function name, ...
    # reset the conditioner by decreasing max_prop_nodes
    # to tighten the bounds
    def tighten(self, **kwargs):
        self.max_prop_nodes = max(self.max_prop_nodes-1, 1)
        self.clear()

    # api should be the same
    # function name, node, ...
    # return the list of nodes will early stop at 
    def update(self, node: Bound, **kwargs):
        max_relu_nodes = self.max_relu_nodes
        if isinstance(node, BoundRelu) and isinstance(node.inputs[0], BoundLinear):
            self.relu_nodes.append(node)
            self.curr_relu_nodes += 1
            if self.curr_relu_nodes == max_relu_nodes:
                self.intermediate_nodes.append(node.inputs[0])

        if isinstance(node, BoundLinear) and isinstance(node.inputs[0], BoundRelu):
            if self.curr_relu_nodes >= max_relu_nodes and node not in self.intermediate_nodes:
                self.prop_nodes.append(node)

        return self.intermediate_nodes+self.prop_nodes

    # api should be the same
    # function name, node, ...
    # update attribute of the node, at least, lower and upper should be not None
    # since we will use them to concretize
    def process(self, node: Bound, bounded_module: BoundedModule, **kwargs):
        max_prop_nodes = self.max_prop_nodes

        if node in self.intermediate_nodes or node in self.prop_nodes:
            # print(f"Early stopping at {node.name}")

            if node in self.prop_nodes:
                if self.curr_prop_nodes < max_prop_nodes and (node.lA is not None or node.uA is not None):
                    self.curr_prop_nodes += 1
                elif self.curr_prop_nodes == max_prop_nodes and (node.lA is not None or node.uA is not None):
                    node.lA = node.uA = None

            # in our case, lower and upper should be ready or in reference_bounds
            if node.lower is None or node.upper is None:
                if node.name in bounded_module.reference_bounds:
                    node.lower, node.upper = bounded_module.reference_bounds[node.name]
                else:
                    raise ValueError(f"Node {node.name} has no bounds")
                    bound_lower = kwargs.get('bound_lower', True)
                    bound_upper = kwargs.get('bound_upper', True)
                    average_A = kwargs.get('average_A', False)
                    need_A_only = kwargs.get('need_A_only', False)
                    update_mask = kwargs.get('update_mask', None)
                    # store all nodes and lA, uA for later use
                    A_dict = {}
                    for node in bounded_module.nodes():
                        if hasattr(node, 'lA') and hasattr(node, 'uA'):
                            A_dict[node.name] = (node.lA, node.uA)
                    dim_output = int(prod(node.output_shape[1:]))
                    C_l = torch.eye(dim_output, device=bounded_module.device).expand(
                                    bounded_module.batch_size, dim_output, dim_output)
                    ret=bounded_module.backward_general(
                                    node, C_l,
                                    bound_lower=bound_lower, bound_upper=bound_upper,
                                    average_A=average_A, need_A_only=need_A_only,
                                    unstable_idx=None, update_mask=update_mask,
                                    apply_output_constraints_to=None,
                                    early_stop_conditioner = customized_early_stop_conditioner(self.max_relu_nodes, self.max_prop_nodes))
                    if bound_lower:
                        node.lower = ret[0]
                    if bound_upper:
                        node.upper = ret[1]
                    # restore lA, uA
                    for node in bounded_module.nodes():
                        if node.name in A_dict:
                            node.lA, node.uA = A_dict[node.name]


class customized_early_stop_conditioner_for_pile(customized_early_stop_conditioner):
    # def update(self, node: Bound, **kwargs):
    #     max_relu_nodes = self.max_relu_nodes

    #     if isinstance(node, BoundRelu) and isinstance(node.inputs[0], BoundSub):
    #         self.intermediate_nodes.append(node.inputs[0])

    #     if isinstance(node, BoundAbs):
    #         self.intermediate_nodes.append(node.inputs[0])

    #     return self.intermediate_nodes

    # def process(self, node: Bound, bounded_module: BoundedModule, **kwargs):
    #     return
    #     return super().process(node, bounded_module, **kwargs)

    def update(self, node: Bound, **kwargs):
        max_relu_nodes = self.max_relu_nodes
        if isinstance(node, BoundRelu) and isinstance(node.inputs[0], BoundAdd):
            self.relu_nodes.append(node)
            self.curr_relu_nodes += 1
            if self.curr_relu_nodes == max_relu_nodes:
                self.intermediate_nodes.append(node.inputs[0])

        if isinstance(node, BoundRelu) and isinstance(node.inputs[0], BoundAdd):
            if self.curr_relu_nodes >= max_relu_nodes and node.inputs[0] not in self.intermediate_nodes:
                self.prop_nodes.append(node.inputs[0])

        return self.intermediate_nodes+self.prop_nodes

    def loosen(self, **kwargs):
        self.max_relu_nodes += 1
        self.clear()

    def tighten(self, **kwargs):
        self.max_relu_nodes = max(self.max_relu_nodes-1, 1)
        self.clear()