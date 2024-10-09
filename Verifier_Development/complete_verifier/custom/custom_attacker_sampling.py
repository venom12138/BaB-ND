#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2024 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Kaidi Xu <kx46@drexel.edu>                      ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import arguments
from load_model import Customized
import torch
import torch.nn.functional as F
import time
import math
import pickle
from utils import output_time
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.operators import BoundRelu, BoundLinear, BoundSqr, BoundAbs, BoundAdd, BoundMatMul, BoundConcat
from attack.attack_pgd import boundary_attack, OSI_init_C, test_conditions, default_pgd_loss, AdamClipping

def mppi_attack_with_general_specs(model, X: torch.Tensor, data_min: torch.Tensor, data_max: torch.Tensor, C_mat, rhs_mat,
                                  cond_mat, same_number_const, alpha,
                                  use_adam=True, normalize=lambda x: x,
                                  initialization='uniform', GAMA_loss=False,
                                  num_restarts=None, pgd_steps=None,
                                  only_replicate_restarts=False,
                                  return_early_stopped=False,
                                  param_dict={}, **kwargs):
    prev_grad_enabled = torch.is_grad_enabled()
    if prev_grad_enabled:
        torch.torch.set_grad_enabled(False)
    per_spec = param_dict.get('per_spec', False)
    prev_adv_example = param_dict.get('prev_adv_example', None)
    reject_bad = param_dict.get('reject_bad', None)
    lirpa_model = param_dict.get('lirpa_model', None)
    require_distribution = param_dict.get('require_distribution', False)
    report_time = param_dict.get('report_time', False) or arguments.Config["find_feasible_solution"]["report_time"]
    
    device = X.device
    verbose = arguments.Config["find_feasible_solution"]["verbose"]
    mppi_config = arguments.Config["find_feasible_solution"]["mppi_config"]
    attack_iters = mppi_config["n_steps"] if pgd_steps is None else pgd_steps
    num_restarts = mppi_config["n_samples"] if num_restarts is None else num_restarts
    reject_bad = mppi_config["reject_bad"] if reject_bad is None else reject_bad
    early_stop, eps = mppi_config["early_stop"], mppi_config["noise_factor"]
    noise_decay, weight_factor = mppi_config["noise_decay"], mppi_config["weight_factor"]
    info_dict = {}

    X, best_loss, best_delta, delta_lower_limit, delta_upper_limit, input_shape, extra_dim, num_or_spec, num_classes, num_restarts = \
        attack_init(X, data_min, data_max, C_mat, cond_mat, only_replicate_restarts, num_restarts, per_spec, device)

    gama_lambda = arguments.Config["attack"]["gama_lambda"]

    delta, X, extra_dim = init_delta(model, X, data_min, data_max, delta_lower_limit, delta_upper_limit,
                C_mat, attack_iters, alpha, input_shape, initialization, extra_dim)
    delta = warm_start_delta(delta, prev_adv_example, X, num_restarts)
    store_fv_node_list = get_store_fv_node_list(lirpa_model, model, report_time)

    early_stopped = False
    # # hack to collect samples in crown
    # sample_file = "crown_configs/func_test/output/sample.pkl"

    start_time = time.time()
    for iteration in range(attack_iters + 1):
        if not reject_bad and iteration == attack_iters:
            break
        inputs, output, origin_out = get_inputs_and_outputs(model, X, delta, normalize, input_shape, extra_dim, num_classes, GAMA_loss)
        # # write inputs to sample file
        # with open(sample_file, 'ab') as f:
        #     pickle.dump({"iter": iteration, "samples":inputs}, f)

        loss, loss_gama = calculate_loss(origin_out, output, gama_lambda, GAMA_loss,
                                        C_mat, rhs_mat, cond_mat, same_number_const, model)
        gama_lambda *= arguments.Config["attack"]["gama_decay"]
        # shape of loss: [input_shape[0], num_restarts, num_or_spec, 1]
        # or float when gama_lambda > 0
        assert loss.shape == (input_shape[0], num_restarts, num_or_spec, 1)

        # ANONYMOUS: reject the update if the loss is not increasing (gradient ascent)
        if reject_bad:
            if iteration > 0:
                bad_update_indices = loss < loss_before
                bad_update_indices = bad_update_indices.view(bad_update_indices.shape[1:3])
                if verbose:
                    print(f"reject {bad_update_indices.sum().item()} bad updates from {num_restarts * num_or_spec} in iteration {iteration}")
                delta.data[:, bad_update_indices] = delta_before[:, bad_update_indices]
                output.data[:, bad_update_indices] = output_before[:, bad_update_indices]
                if GAMA_loss:
                    origin_out.data[:, bad_update_indices] = origin_out_before[:, bad_update_indices]
                loss, loss_gama = calculate_loss(origin_out, output, gama_lambda, GAMA_loss,
                                        C_mat, rhs_mat, cond_mat, same_number_const, model)
            if iteration == attack_iters:
                break
            delta_before = delta.clone()
            loss_before = loss.clone()
            output_before = output.clone()
            if GAMA_loss:
                origin_out_before = origin_out.clone()
        if verbose:
            print(f"eps: {eps}, output: {output.min().item()}, {output.mean().item()}, {output.max().item()}")

        # scores = (loss - loss.min(dim=1, keepdim=True)[0]) / (loss.max(dim=1, keepdim=True)[0] - loss.min(dim=1, keepdim=True)[0]).clamp(min=1e-6)
        # normalize the loss to var = 1
        scores = (loss - loss.mean(dim=1, keepdim=True)) / (loss.std(dim=1, keepdim=True) + 1e-6)
        # scores = (loss - loss.min(dim=1, keepdim=True)[0])
        weights = F.softmax(scores / weight_factor, dim=1)
        # print(f"weights: {weights.min().item()}, {weights.mean().item()}, {weights.max().item()}")
        weights = weights.view([*weights.shape[:3], *[1]*len(input_shape[1:])])
        new_delta = torch.sum(weights * delta, dim=1, keepdim=True).clamp(delta_lower_limit, delta_upper_limit)

        loss, indices, best_loss, best_delta = update_best_loss_and_delta(
            loss, delta, best_loss, best_delta,
            same_number_const, cond_mat, C_mat,
            per_spec, num_or_spec, input_shape
        )
        if early_stop:
            if eval(arguments.Config["attack"]["early_stop_condition"])(inputs, output, C_mat, rhs_mat,
                    cond_mat, same_number_const, data_max, data_min, model, indices, num_or_spec).all():
                print("mppi early stop")
                early_stopped = True
                break

        new_deltas = new_delta.expand_as(delta)
        noise = torch.normal(mean=0, std=eps, size=new_deltas.size(), device=new_deltas.device) * (delta_upper_limit-delta_lower_limit)
        # noise = torch.normal(mean=0, std=eps, size=new_deltas.size(), device=new_deltas.device)
        eps = eps * noise_decay
        # noise = torch.normal(mean=0, std=eps, size=new_deltas.size(), device=new_deltas.device)
        new_deltas = (noise + new_deltas).clamp(delta_lower_limit, delta_upper_limit)
        if verbose:
            print(f"{(new_deltas==delta_lower_limit).sum()} / {math.prod(new_deltas.shape)} clipped to lower limit, \
                {(new_deltas==delta_upper_limit).sum()} / {math.prod(new_deltas.shape)} clipped to upper limit")
        delta.data = new_deltas.data

    if require_distribution:
        loss = loss.view(input_shape[0], num_restarts, num_or_spec, 1)
        info_dict['sample_distribution'] = get_distribution_dict(loss, delta, delta_lower_limit, delta_upper_limit, param_dict, report_time)

    if report_time:
        output_time("searching", time.time()-start_time)
    if lirpa_model is not None:
        inputs = normalize(X + delta).view(-1, *input_shape[1:])
        update_reference_bounds(lirpa_model, inputs, num_restarts, num_or_spec, store_fv_node_list, report_time)

    if post_test_checking(early_stopped, best_delta, data_max, data_min, model, X,
                        C_mat, rhs_mat, cond_mat, same_number_const):
        best_loss = torch.full(size=(1,), fill_value=float('-inf'), device=best_loss.device)

    if prev_grad_enabled:
        torch.torch.set_grad_enabled(True)
    if return_early_stopped:
        return best_delta, delta, best_loss, early_stopped, info_dict
    else:
        return best_delta, delta, best_loss, info_dict


def cem_attack_with_general_specs(model, X: torch.Tensor, data_min: torch.Tensor, data_max: torch.Tensor, C_mat, rhs_mat,
                                  cond_mat, same_number_const, alpha,
                                  use_adam=True, normalize=lambda x: x,
                                  initialization='uniform', GAMA_loss=False,
                                  num_restarts=None, pgd_steps=None,
                                  only_replicate_restarts=False,
                                  return_early_stopped=False,
                                  param_dict={}, **kwargs):
    prev_grad_enabled = torch.is_grad_enabled()
    if prev_grad_enabled:
        torch.torch.set_grad_enabled(False)
    per_spec = param_dict.get('per_spec', False)
    prev_adv_example = param_dict.get('prev_adv_example', None)
    reject_bad = param_dict.get('reject_bad', None)
    lirpa_model = param_dict.get('lirpa_model', None)
    require_distribution = param_dict.get('require_distribution', False)
    report_time = param_dict.get('report_time', False) or arguments.Config["find_feasible_solution"]["report_time"]

    device = X.device
    verbose = arguments.Config["find_feasible_solution"]["verbose"]
    cem_config = arguments.Config["find_feasible_solution"]["cem_config"]
    attack_iters = cem_config["n_steps"] if pgd_steps is None else pgd_steps
    num_restarts = cem_config["n_samples"] if num_restarts is None else num_restarts
    reject_bad = cem_config["reject_bad"] if reject_bad is None else reject_bad
    early_stop, jitter = cem_config["early_stop"], cem_config["jitter_factor"]
    elite_ratio, min_n_elites = cem_config["elite_ratio"], cem_config["min_n_elites"]
    info_dict = {}

    X, best_loss, best_delta, delta_lower_limit, delta_upper_limit, input_shape, extra_dim, num_or_spec, num_classes, num_restarts = \
        attack_init(X, data_min, data_max, C_mat, cond_mat, only_replicate_restarts, num_restarts, per_spec, device)

    gama_lambda = arguments.Config["attack"]["gama_lambda"]

    assert input_shape[0] == 1, "First dimension of input shape must be 1 for CEM attack"
    n_input = math.prod(input_shape[1:])
    jitters = jitter * torch.eye(n_input, device=device).unsqueeze(0).repeat(num_or_spec,1,1)
    extra_item_dict = kwargs.get('extra_item_dict', {})
    if extra_item_dict != {}:
        means = extra_item_dict['means']
        covs = extra_item_dict['covs']
    else:
        means = torch.zeros((num_or_spec, n_input), device=device)
        covs = torch.eye(n_input, device=device).unsqueeze(0).repeat(num_or_spec,1,1)
        sigma = ((delta_upper_limit - delta_lower_limit) / 2).view(num_or_spec, n_input, 1).repeat(1,1,n_input)
        covs *= sigma
        covs += jitters
    n_elites = max(min_n_elites, int(num_restarts * elite_ratio))

    def sample_delta(means_p, covs_p):
        delta_p = torch.zeros(num_restarts, num_or_spec, n_input, device=device)
        for i in range(num_or_spec):
            reset_torch_seed(arguments.Config['general']['seed'])
            delta_p[:,i,:] = torch.distributions.MultivariateNormal(means_p[i], covs_p[i]).sample((num_restarts,))
        return delta_p.clamp(delta_lower_limit.view(num_or_spec, n_input), delta_upper_limit.view(num_or_spec, n_input)).view(-1, num_restarts, num_or_spec, *input_shape[1:])

    delta = sample_delta(means, covs) # [input_shape[0], num_restarts, num_or_spec, *input_shape[1:]]
    delta = warm_start_delta(delta, prev_adv_example, X, num_restarts)
    store_fv_node_list = get_store_fv_node_list(lirpa_model, model)

    early_stopped = False
    # # hack to collect samples in crown
    # sample_file = "crown_configs/func_test/output/sample.pkl"

    start_time = time.time()
    for iteration in range(attack_iters + 1):
        if not reject_bad and iteration == attack_iters:
            break
        inputs, output, origin_out = get_inputs_and_outputs(model, X, delta, normalize, input_shape, extra_dim, num_classes, GAMA_loss)
        # # write inputs to sample file
        # with open(sample_file, 'ab') as f:
        #     pickle.dump({"iter": iteration, "samples":inputs}, f)

        loss, loss_gama = calculate_loss(origin_out, output, gama_lambda, GAMA_loss,
                                        C_mat, rhs_mat, cond_mat, same_number_const, model)
        gama_lambda *= arguments.Config["attack"]["gama_decay"]
        # shape of loss: [input_shape[0], num_restarts, num_or_spec, 1]
        # or float when gama_lambda > 0
        assert loss.shape == (input_shape[0], num_restarts, num_or_spec, 1)
        if verbose:
            print(f"output: {output.min().item()}, {output.mean().item()}, {output.max().item()}")
            print(covs.abs().max().item())
        loss: torch.Tensor
        elite_idxs = loss.topk(n_elites, dim=1, largest=True)[1]
        # (input_shape[0], n_elites, num_or_spec, 1)
        elite_delta = torch.gather(delta.view(input_shape[0], num_restarts, num_or_spec, n_input), dim=1, index=elite_idxs.repeat(1,1,1,n_input)).squeeze(0)
        # [n_elites, num_or_spec, n_input]
        # TODO: implement rejection of bad samples
        # if reject_bad:
        #     elite_loss = torch.gather(loss.view(input_shape[0], num_restarts, num_or_spec, 1), dim=1, index=elite_idxs).squeeze(0)
        #     # [n_elites, num_or_spec, 1]
        #     if iteration > 0:
        #         # replace the worst elite with the best one
        #         worst_elite_loss, worst_elite_idx = elite_loss.min(dim=0)
        #         replace_idx = torch.where(worst_elite_loss < best_loss)
        #         pass
        #     if iteration == attack_iters:
        #         break
        means = elite_delta.mean(dim=0)
        # (num_or_spec, n_input)
        centered_delta = (elite_delta - means).transpose(0, 1)
        # [num_or_spec, n_elites, n_input]
        covs = torch.einsum('ijk,ijl->ikl', centered_delta, centered_delta) / (n_elites-1)
        covs += jitters
        # [num_or_spec, n_input, n_input]

        loss, indices, best_loss, best_delta = update_best_loss_and_delta(
            loss, delta, best_loss, best_delta,
            same_number_const, cond_mat, C_mat,
            per_spec, num_or_spec, input_shape
        )

        if early_stop:
            if eval(arguments.Config["attack"]["early_stop_condition"])(inputs, output, C_mat, rhs_mat,
                    cond_mat, same_number_const, data_max, data_min, model, indices, num_or_spec).all():
                print("cem early stop")
                early_stopped = True
                break

        delta = sample_delta(means, covs)

    info_dict['means'] = means
    info_dict['covs'] = covs
    if require_distribution:
        loss = loss.view(input_shape[0], num_restarts, num_or_spec, 1)
        info_dict['sample_distribution'] = get_distribution_dict(loss, delta, delta_lower_limit, delta_upper_limit, param_dict, report_time)

    if report_time:
        output_time("searching", time.time()-start_time)
    if lirpa_model is not None:
        inputs = normalize(X + delta).view(-1, *input_shape[1:])
        update_reference_bounds(lirpa_model, inputs, num_restarts, num_or_spec, store_fv_node_list, report_time)

    if post_test_checking(early_stopped, best_delta, data_max, data_min, model, X,
                        C_mat, rhs_mat, cond_mat, same_number_const):
        best_loss = torch.full(size=(1,), fill_value=float('-inf'), device=best_loss.device)

    if prev_grad_enabled:
        torch.torch.set_grad_enabled(True)
    if return_early_stopped:
        return best_delta, delta, best_loss, early_stopped, info_dict
    else:
        return best_delta, delta, best_loss, info_dict


def pgd_attack_with_general_specs(model, X: torch.Tensor, data_min: torch.Tensor, data_max: torch.Tensor, C_mat, rhs_mat,
                                  cond_mat, same_number_const, alpha,
                                  use_adam=True, normalize=lambda x: x,
                                  initialization='uniform', GAMA_loss=False,
                                  num_restarts=None, pgd_steps=None,
                                  only_replicate_restarts=False,
                                  return_early_stopped=False, 
                                  param_dict={},
                                  **kwargs
                                  ):
    per_spec = param_dict.get('per_spec', False)
    prev_adv_example = param_dict.get('prev_adv_example', None)
    reject_bad = param_dict.get('reject_bad', None)
    lirpa_model = param_dict.get('lirpa_model', None)
    require_distribution = param_dict.get('require_distribution', False)
    report_time = param_dict.get('report_time', False) or arguments.Config["find_feasible_solution"]["report_time"]

    device = X.device
    verbose = arguments.Config["find_feasible_solution"]["verbose"]
    pgd_config = arguments.Config["find_feasible_solution"]["pgd_config"]
    attack_iters = pgd_config["n_steps"] if pgd_steps is None else pgd_steps
    num_restarts = pgd_config["n_restarts"] if num_restarts is None else num_restarts
    reject_bad = pgd_config["reject_bad"] if reject_bad is None else reject_bad
    if pgd_config["lr_decay_mode"] == "auto":
        lr_decay = pgd_config["lr_target"] ** (1.0 / attack_iters)
    else:
        lr_decay = pgd_config["lr_decay"]
    early_stop = pgd_config["early_stop"]
    restart_when_stuck = pgd_config["restart_when_stuck"]
    info_dict = {}

    if restart_when_stuck:
        total_replaced_deltas = 0

    X, best_loss, best_delta, delta_lower_limit, delta_upper_limit, input_shape, extra_dim, num_or_spec, num_classes, num_restarts = \
        attack_init(X, data_min, data_max, C_mat, cond_mat, only_replicate_restarts, num_restarts, per_spec, device)

    gama_lambda = arguments.Config["attack"]["gama_lambda"]

    alpha = alpha * (delta_upper_limit - delta_lower_limit)
    delta, X, extra_dim = init_delta(model, X, data_min, data_max, delta_lower_limit, delta_upper_limit,
                C_mat, attack_iters, alpha, input_shape, initialization, extra_dim)
    delta = warm_start_delta(delta, prev_adv_example, X, num_restarts)
    store_fv_node_list = get_store_fv_node_list(lirpa_model, model)
    delta = delta.requires_grad_()

    if use_adam:
        opt = AdamClipping(params=[delta], lr=alpha)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay, verbose=False)

    early_stopped = False
    # # hack to collect samples in crown
    # sample_file = "crown_configs/func_test/output/sample.pkl"

    forward_time = 0
    backward_time = 0
    start_time = time.time()
    for iteration in range(attack_iters + 1):
        if not reject_bad and iteration == attack_iters:
            break
        inputs, output, origin_out = get_inputs_and_outputs(model, X, delta, normalize, input_shape, extra_dim, num_classes, GAMA_loss)
        # # write inputs to sample file
        # with open(sample_file, 'ab') as f:
        #     pickle.dump({"iter": iteration, "samples":inputs}, f)

        loss, loss_gama = calculate_loss(origin_out, output, gama_lambda, GAMA_loss,
                                        C_mat, rhs_mat, cond_mat, same_number_const, model)
        gama_lambda *= arguments.Config["attack"]["gama_decay"]
        # shape of loss: [num_example, num_restarts, num_or_spec]
        # or float when gama_lambda > 0
        assert loss.shape == (input_shape[0], num_restarts, num_or_spec, 1)

        # ANONYMOUS: reject the update if the loss is not increasing (gradient ascent)
        if reject_bad:
            if iteration > 0:
                bad_update_indices = loss < loss_before
                bad_update_indices = bad_update_indices.view(bad_update_indices.shape[1:3])
                if verbose:
                    print(f"reject {bad_update_indices.sum().item()} bad updates from {num_restarts * num_or_spec} in iteration {iteration}")
                delta.data[:, bad_update_indices] = delta_before[:, bad_update_indices]
                output.data[:, bad_update_indices] = output_before[:, bad_update_indices]
                if iteration == attack_iters:
                    break
                if GAMA_loss:
                    origin_out.data[:, bad_update_indices] = origin_out_before[:, bad_update_indices]
                loss, loss_gama = calculate_loss(origin_out, output, gama_lambda, GAMA_loss,
                                        C_mat, rhs_mat, cond_mat, same_number_const, model)
            if iteration == attack_iters:
                break
            delta_before = delta.clone()
            loss_before = loss.clone()
            output_before = output.clone()
            if GAMA_loss:
                origin_out_before = origin_out.clone()
        if verbose:
            print(f"output: {output.min().item()}, {output.mean().item()}, {output.max().item()}")

        backward_start_time = time.time()
        loss_gama.sum().backward()
        backward_time += time.time() - backward_start_time
        with torch.no_grad():
            loss, indices, best_loss, best_delta = update_best_loss_and_delta(
                loss, delta, best_loss, best_delta,
                same_number_const, cond_mat, C_mat,
                per_spec, num_or_spec, input_shape
            )
        if early_stop:
            if eval(arguments.Config["attack"]["early_stop_condition"])(inputs, output, C_mat, rhs_mat,
                    cond_mat, same_number_const, data_max, data_min, model, indices, num_or_spec).all():
                print("pgd early stop")
                early_stopped = True
                break

        if restart_when_stuck:
            old_delta = delta.clone().detach()
        if use_adam:
            opt.step(clipping=True, lower_limit=delta_lower_limit,
                     upper_limit=delta_upper_limit, sign=1)
            opt.zero_grad(set_to_none=True)
            scheduler.step()
        else:
            d = delta + alpha * torch.sign(delta.grad)
            d = torch.max(torch.min(d, delta_upper_limit), delta_lower_limit)
            delta = d.detach().requires_grad_()

        if restart_when_stuck:
            unchanged = ((delta - old_delta).abs().sum(list(range(2, delta.ndim)), keepdim=True) == 0).to(delta.dtype)
            total_replaced_deltas += int(unchanged.sum().item())
            new_init = (torch.empty_like(X).uniform_() * (delta_upper_limit - delta_lower_limit) + delta_lower_limit)
            delta.data.copy_(delta * (1 - unchanged) + new_init * unchanged)

    if restart_when_stuck:
        total_num_deltas = X.size(0) * delta.size(1) * (iteration + 1)
        replaced_percentage = total_replaced_deltas / total_num_deltas * 100
        print(f'Attack batch size: {X.size(0)}, restarts: {delta.size(1)}, iterations: {iteration + 1} '
              f'replaced deltas {total_replaced_deltas} ({replaced_percentage}%)')

    if require_distribution:
        loss = loss.view(input_shape[0], num_restarts, num_or_spec, 1)
        info_dict['sample_distribution'] = get_distribution_dict(loss, delta, delta_lower_limit, delta_upper_limit, param_dict, report_time)

    if report_time:
        output_time("searching", time.time()-start_time)
    if lirpa_model is not None:
        inputs = normalize(X + delta).view(-1, *input_shape[1:])
        update_reference_bounds(lirpa_model, inputs, num_restarts, num_or_spec, store_fv_node_list, report_time)

    if post_test_checking(early_stopped, best_delta, data_max, data_min, model, X,
                        C_mat, rhs_mat, cond_mat, same_number_const):
        best_loss = torch.full(size=(1,), fill_value=float('-inf'), device=best_loss.device)

    if return_early_stopped:
        return best_delta, delta, best_loss, early_stopped, info_dict
    else:
        return best_delta, delta, best_loss, info_dict


def attack_init(X, data_min, data_max, C_mat, cond_mat,
            only_replicate_restarts, num_restarts, per_spec, device, 
):
    if only_replicate_restarts:
        input_shape = (X.shape[0], *X.shape[2:])
    else:
        input_shape = X.size()
    num_classes = C_mat.shape[-1]

    num_or_spec = len(cond_mat[0])
    num_restarts = max(1, num_restarts//num_or_spec) if only_replicate_restarts else num_restarts

    extra_dim = (num_restarts, num_or_spec) if only_replicate_restarts == False else (num_restarts,)

    if per_spec:
        best_loss = torch.empty(X.size(0),X.size(1), device=device).fill_(float("-inf"))
        best_delta = torch.zeros(input_shape[0],num_or_spec, *input_shape[1:], device=device)
    else:
        best_loss = torch.empty(X.size(0), device=device).fill_(float("-inf"))
        best_delta = torch.zeros(input_shape, device=device)

    data_min = data_min.unsqueeze(1)
    data_max = data_max.unsqueeze(1)
    # [1, 1, num_spec, *input_shape]

    X_ndim = X.ndim

    X = X.view(X.shape[0], *[1] * len(extra_dim), *X.shape[1:])
    delta_lower_limit = data_min - X
    delta_upper_limit = data_max - X

    X = X.expand(-1, *extra_dim, *(-1,) * (X_ndim - 1))
    extra_dim = (X.shape[1], X.shape[2])
    # [num_restarts, num_or_spec]
    return X, best_loss, best_delta, delta_lower_limit, delta_upper_limit, input_shape, extra_dim, num_or_spec, num_classes, num_restarts

def init_delta(model, X, data_min, data_max, delta_lower_limit, delta_upper_limit,
                C_mat, attack_iters, alpha, input_shape, initialization, extra_dim):
    if initialization == 'osi':
        # X_init = OSI_init(model, X, y, epsilon, alpha, num_classes, iter_steps=attack_iters, extra_dim=extra_dim, upper_limit=upper_limit, lower_limit=lower_limit)
        osi_start_time = time.time()
        X_init = OSI_init_C(model, X, alpha, C_mat.shape[-1], attack_iters, data_min, data_max)
        osi_time = time.time() - osi_start_time
        print(f'diversed PGD initialization time: {osi_time:.4f}')
    if initialization == 'boundary':
        boundary_adv_examples = boundary_attack(model, X[:,0,...].view(-1, *input_shape[1:]), data_min.view(*input_shape), data_max.view(*input_shape))
        if boundary_adv_examples is not None:
            X_init = boundary_adv_examples.view(X.shape[0], -1, *X.shape[2:])
            X = X[:,:X_init.shape[1],...]
            extra_dim = (X.shape[1], X.shape[2])
        else:
            initialization = 'uniform'

    if initialization == 'osi' or initialization == 'boundary':
        delta = (X_init - X).detach()
    elif initialization == 'uniform':
        delta = (torch.empty_like(X).uniform_() * (delta_upper_limit - delta_lower_limit) + delta_lower_limit)
    elif initialization == 'none':
        delta = torch.zeros_like(X)
    else:
        raise ValueError(f"Unknown initialization method {initialization}")

    return delta, X, extra_dim

def warm_start_delta(delta, prev_adv_example, X, num_restarts):
    # prev_adv_example: [num_example, input_dim]
    if prev_adv_example is not None:
        mask = ~torch.isinf(prev_adv_example)
        num_dims = prev_adv_example.ndim
        for i in range(num_dims - 1):
            mask = mask.any(dim=-1)
        adv_indices = torch.nonzero(mask).squeeze(-1)
        # there is one domain and many adv examples: happens when it is initial massive attack
        if delta.shape[2] == 1 and delta.shape[2] < adv_indices.shape[0]:
            num_examples = min(num_restarts, adv_indices.shape[0])
            replace_id = torch.randperm(num_restarts)[:num_examples]
            delta[:,replace_id, ...] = prev_adv_example[adv_indices[:num_examples]].view(X[:,replace_id, ...].shape) - X[:,replace_id, ...]
        # there are many domains and many adv examples, assume domains and examples are matched
        else:
            replace_id = torch.randperm(num_restarts)[0].item()
            delta[:,replace_id,adv_indices, ...] = prev_adv_example[adv_indices] - X[:,replace_id,adv_indices, ...]
    return delta

def get_store_fv_node_list(lirpa_model, model, report_time=False):
    start_time = time.time()
    store_fv_node_list = [] 
    if lirpa_model is not None:
        # hack for ANONYMOUS's project
        if not (hasattr(lirpa_model, "default_reference_bounds") and lirpa_model.default_reference_bounds):
            lirpa_model.default_reference_bounds = {}
        if model.__class__.__name__ == 'wrapped_model':
            for node in lirpa_model.nodes():
                if isinstance(node, BoundLinear):
                    store_fv_node_list.append(node)
                elif isinstance(node, BoundSqr) or isinstance(node, BoundRelu):
                    for i in range(len(node.inputs)):
                        if i in node.requires_input_bounds:
                            store_fv_node_list.append(node.inputs[i])
        elif model.__class__.__name__ == 'wrapped_model_gnn':
            for node in lirpa_model.nodes():
                if node.perturbed and len(node.requires_input_bounds) > 0:
                    # if isinstance(node, BoundLinear):
                    #     store_fv_node_list.append(node)
                    # if isinstance(node, BoundAdd) and isinstance(lirpa_model[node.output_name], BoundRelu):
                    #     store_fv_node_list.append(node)
                    if isinstance(node, BoundRelu) or isinstance(node, BoundSqr):
                        store_fv_node_list.append(node.inputs[0])
                    if isinstance(node, BoundMatMul):
                        for input_id in node.requires_input_bounds:
                            if isinstance(node.inputs[input_id], BoundConcat):
                                store_fv_node_list.append(node.inputs[input_id])
                    # for input_id in node.requires_input_bounds:
                    #     if node.inputs[input_id].perturbed:
                    #         store_fv_node_list.append(node.inputs[input_id])
                    # if (isinstance(node, BoundRelu) and not isinstance(node, BoundAbs)) or isinstance(node, BoundAdd) and isinstance(node.inputs[1], BoundRelu):
                    #     for input_id in node.requires_input_bounds:
                    #         if node.inputs[input_id].perturbed:
                    #             store_fv_node_list.append(node.inputs[input_id])
        elif model.__class__.__name__ == 'wrapped_model_rope':
            for node in lirpa_model.nodes():
                if isinstance(node, BoundLinear):
                    store_fv_node_list.append(node)
                if node.perturbed and len(node.requires_input_bounds) > 0:
                    if not ((isinstance(node, BoundAbs) or isinstance(node, BoundSqr))
                                and isinstance(node.inputs[0], BoundAdd)):
                        for input_id in node.requires_input_bounds:
                            if node.inputs[input_id].perturbed:
                                store_fv_node_list.append(node.inputs[input_id])
                # elif (isinstance(node, BoundAbs) or isinstance(node, BoundSqr)) and (not isinstance(node.inputs[0], BoundAdd)):
                #     store_fv_node_list.append(node.inputs[0])
        else:
            pass
    # remove repeated nodes
    store_fv_node_list = list(set(store_fv_node_list))
    if report_time:
        output_time("get_nodes", time.time()-start_time)
    return store_fv_node_list

def update_best_loss_and_delta(loss, delta, best_loss, best_delta, 
                                same_number_const, cond_mat, C_mat,
                                per_spec, num_or_spec, input_shape):
    # Save the best loss so far.
    if same_number_const:
        loss = loss.amin(-1)
        # loss has shape [num_example, num_restarts, num_or_spec].
        # margins = (runnerup - groundtruth).view(groundtruth.size(0), -1)
    else:
        group_C = torch.zeros(len(cond_mat[0]), C_mat.shape[1]).to(loss.device) # [num_or_spec, num_total_spec]
        x_index = []
        y_index = []
        index = 0
        for i, cond in enumerate(cond_mat[0]):
            for _ in range(cond):
                x_index.append(i)
                y_index.append(index)
                index += 1
        group_C[x_index, y_index] = 1.0

        # loss shape: [batch_size, num_restarts, num_total_spec]
        loss = group_C.matmul(loss.unsqueeze(-1)).squeeze(-1)
        # loss shape: [batch_size, num_restarts, num_or_spec]

    if not per_spec:
        loss = loss.view(loss.shape[0], -1)
    # all_loss and indices have shape (batch, ),
    # and this is the best loss over all restarts and number of classes.
    all_loss, indices = loss.max(1)
    # delta has shape (batch, restarts, num_class-1, c, h, w).
    # For each batch element, we want to select from the best over
    # (restarts, num_classes-1) dimension.
    # delta_targeted has shape (batch, c, h, w).
    if per_spec:
        delta_targeted = delta.gather(
            dim=1, index=indices.view(
                -1,1, num_or_spec, *(1,) * (len(input_shape) - 1)).expand(
                    -1,-1,-1,*input_shape[1:])
        ).squeeze(1)
    else:
        delta_targeted = delta.view(
            delta.size(0), -1, *input_shape[1:]
        ).gather(
            dim=1, index=indices.view(
                -1,1,*(1,) * (len(input_shape) - 1)).expand(
                    -1,-1,*input_shape[1:])
        ).squeeze(1)

    best_delta[all_loss >= best_loss] = delta_targeted[all_loss >= best_loss]
    best_loss = torch.max(best_loss, all_loss)
    return loss, indices, best_loss, best_delta

def get_inputs_and_outputs(model, X, delta, normalize, input_shape, extra_dim, num_classes, GAMA_loss):
    inputs = normalize(X + delta)
    output = model(inputs.view(-1, *input_shape[1:])).view(input_shape[0], *extra_dim, num_classes)
    if GAMA_loss:
        # Output on original model is needed if gama loss is used.
        origin_out = torch.softmax(model(normalize(X.reshape(-1, *input_shape[1:]))), 1)
        origin_out = origin_out.view(output.shape)
    else:
        origin_out = None
    return inputs, output, origin_out

def calculate_loss(origin_out, output, gama_lambda, GAMA_loss,
                    C_mat, rhs_mat, cond_mat, same_number_const, model):
    loss, loss_gama = eval(arguments.Config["attack"]["pgd_loss"])(
        origin_out, output, C_mat, rhs_mat,
        cond_mat, same_number_const,
        gama_lambda if GAMA_loss else 0.0,
        mode=arguments.Config['attack']['pgd_loss_mode'], model=model)
    return loss, loss_gama

def get_distribution_dict(loss, delta, delta_lower_limit, delta_upper_limit, param_dict, report_time=False):
    assert len(loss.shape) == 4
    start_time = time.time()
    num_restarts = loss.shape[1]
    distribution_dict = {}
    # # analyze the distribution of the loss
    good_loss, good_loss_indices = loss.sort(dim=1, descending=True)
    elite_ratio_of_dist = param_dict.get('elite_ratio', 0.01)
    min_n_elites_for_dist = param_dict.get('min_n_elites', 10)
    n_elite_for_dist = max(min_n_elites_for_dist, int(num_restarts * elite_ratio_of_dist))
    good_loss = good_loss[:, :n_elite_for_dist]
    good_loss_indices = good_loss_indices[:, :n_elite_for_dist]
    # # get corresponding delta
    batch, num_selected, num_domain = good_loss_indices.shape[:3]
    batch_indices = torch.arange(batch).view(-1, 1, 1).expand(batch, num_selected, num_domain)
    domain_indices = torch.arange(num_domain).view(1, 1, -1).expand(batch, num_selected, num_domain)
    good_delta = delta[batch_indices, good_loss_indices.squeeze(-1), domain_indices]
    delta_center = (delta_lower_limit + delta_upper_limit) / 2
    ge_mask = good_delta >= delta_center
    le_mask = good_delta <= delta_center
    count_ge = ge_mask.sum(dim=1, keepdim=True)
    count_le = le_mask.sum(dim=1, keepdim=True)
    distribution_dict['count_ge'] = count_ge
    distribution_dict['count_le'] = count_le
    if report_time:
        output_time("get_distribution", time.time()-start_time)
    return distribution_dict

def update_reference_bounds(lirpa_model, inputs, num_restarts, num_or_spec, store_fv_node_list, report_time=False):
    bound_time = time.time()
    lirpa_model.forward(inputs, store_fv_node_list=store_fv_node_list)
    for node in store_fv_node_list:
        fv = node.forward_value.view(num_restarts, num_or_spec, *node.forward_value.shape[1:])
        lb = fv.min(dim=0, keepdim=False)[0]
        ub = fv.max(dim=0, keepdim=False)[0]
        lirpa_model.default_reference_bounds[node.name] = (lb, ub)
    lirpa_model.forward(lirpa_model.global_input)
    if report_time:
        output_time("update_reference", time.time()-bound_time)

def post_test_checking(early_stopped, best_delta, data_max, data_min, model, X,
                        C_mat, rhs_mat, cond_mat, same_number_const):
    if not early_stopped and 'Customized' in arguments.Config["attack"]["early_stop_condition"]:
        test_input = X[:, 0, 0, :] + best_delta
        test_output = model(test_input)
        test_input = test_input.unsqueeze(0).unsqueeze(0)
        test_output = test_output.unsqueeze(0).unsqueeze(0)
        if not test_conditions(test_input, test_output, C_mat, rhs_mat, cond_mat,
                           same_number_const, data_max, data_min).all():
            return True
    return False

def reset_torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)