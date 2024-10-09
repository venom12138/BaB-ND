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
"""Branch and bound for input space split."""

import time
import torch
import math
import pickle
import os
import sys

import arguments
from auto_LiRPA import BoundedTensor
from beta_CROWN_solver import LiRPANet
from input_split.branching_domains import UnsortedInputDomainList, show_domain_stat
from auto_LiRPA.utils import (stop_criterion_batch_any, stop_criterion_all,
                              AutoBatchSize)
from utils import check_auto_enlarge_batch_size, output_time
from input_split.attack import (massive_pgd_attack, check_adv,
                                attack_in_input_bab_parallel,
                                update_rhs_with_attack)
from input_split.branching_heuristics import input_split_branching
from input_split.clip import clip_domains
from input_split.split import input_split_parallel, get_split_depth
from input_split.utils import transpose_c_back, initial_verify_criterion
from load_model import Customized

Visited, storage_depth, total_picked_out = 0, 0, 0


def repeat_alphas(_alphas: dict, _split_depth: int, _split_partitions: int) -> dict:
    """
    Repeats along the batch dimension of every key in _alphas
    @param _alphas:              Dictionary of _alphas for network
    @param _split_depth:         The split depth for partitioning
    @param _split_partitions:    The number of partitions for each domain, is typically 2
    @return:
    """
    repeated_alphas = {}
    repeat_depth = _split_partitions ** _split_depth
    for key0 in _alphas.keys():
        repeated_alphas[key0] = {}
        for key1 in _alphas[key0].keys():
            # alpha[key0][key1] has shape (dim_in, spec_dim, batches, unstable size)
            repeated_alphas[key0][key1] = _alphas[key0][key1].repeat(1, 1, repeat_depth, 1)
    return repeated_alphas

# compute lbias at this point
def deconstruct_lbias(_x_L, _x_U, _lA, _dm_lb):
    _lA = _lA.flatten(2) # (batch, spec_dim, in_dim)
    xhat_vect = ((_x_U + _x_L) / 2).flatten(1) # (batch, in_dim)
    xhat_vect = xhat_vect.unsqueeze(2) # (batch, in_dim, 1)
    eps_vect = ((_x_U - _x_L) / 2).flatten(1) # (batch, in_dim)
    eps_vect = eps_vect.unsqueeze(2) # (batch, in_dim, 1)
    dm_lb_vect = _dm_lb.unsqueeze(2) # (batch, spec_dim, 1)
    _lbias = dm_lb_vect - (_lA.bmm(xhat_vect) - _lA.abs().bmm(eps_vect))
    return _lbias.squeeze(2) # (batch, spec_dim)


def reordered_batch_verification_input_split(
        d, net, batch, num_iter, decision_thresh, shape=None,
        bounding_method="crown", branching_method="sb",
        stop_func=stop_criterion_batch_any, split_partitions=2):
    """
    Reordering of the batch_verification_input_split method
    @param d:                   Domain list
    @param net:                 Bounded neural network
    @param batch:               Number of effective batches to evaluate
    @param num_iter:            The current iteration number of the input BaB run
    @param decision_thresh:     The specification threshold to verify against
    @param shape:               The shape of the network's input
    @param bounding_method:     The method to use when bounding the subdomains of the network
    @param branching_method:    The branching heuristic to use when splitting on input dimensions
    @param stop_func:           Criterion to stop naive lower bound of network
    @param split_partitions:    The number of partitions to create for subdomains, currently is always 2 for input split
    @return:
    """

    input_split_args = arguments.Config["bab"]["branching"]["input_split"]
    split_hint = input_split_args['split_hint']
    enable_clip_domains = input_split_args['enable_clip_domains']

    total_start_time = time.time()
    global Visited
    global total_picked_out

    # STEP 1: pick out domains
    pickout_start_time = time.time()
    ret = d.pick_out_batch(batch, device=net.x.device)
    alphas, dm_lb, x_L, x_U, cs, thresholds, _ = ret
    pickout_time = time.time() - pickout_start_time

    if input_split_args["update_rhs_with_attack"]:
        thresholds = update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb,
                                            net.model_ori)[0]

    # STEP 2: Compute bounds for all domains
    bounding_start_time = time.time()
    ret = net.get_lower_bound_naive(
        dm_lb=dm_lb if input_split_args["compare_with_old_bounds"] else None, dm_l=x_L, dm_u=x_U, alphas=alphas,
        bounding_method=bounding_method, branching_method=branching_method,
        C=cs, stop_criterion_func=stop_func, thresholds=thresholds)
    dm_lb, alphas, lA, _, lb_crown = ret  # here alphas is a dict
    dm_lb = dm_lb.to(device=thresholds.device)  # ensures it is on the same device as it may be different
    lb_crown = lb_crown.to(device=thresholds.device)
    bounding_time = time.time() - bounding_start_time

    lbias = deconstruct_lbias(x_L, x_U, lA, lb_crown)

    pickout_batch = len(x_L)
    total_picked_out += pickout_batch
    print(f"Current pickout batch: {pickout_batch}, total pickout batch: {total_picked_out}")
    Visited += pickout_batch

    # STEP 2.5: Filter out verified subdomains
    filt_time_start = time.time()
    # Since we have only bounded the domains and not clipped them, we only need to check thresholds
    ret_filt = d.filter_verified_domains(pickout_batch, dm_lb, x_L, x_U,
                                         alphas, cs, thresholds, lA, lbias,
                                         check_thresholds=True, check_bounds=False)
    num_unverified_domains, dm_lb, x_L, x_U, alphas, cs, thresholds, lA, lbias = ret_filt
    filt_time = time.time() - filt_time_start

    split_time, decision_time, clip_time, adddomain_time = 0, 0, 0, 0
    # when num_unverified_domains > 0, there are still unverified subdomains after filtering from step 2.5
    if num_unverified_domains > 0:

        # STEP 3: Make decisions
        decision_start_time = time.time()
        split_idx = input_split_branching(
            net, dm_lb, x_L, x_U, lA, thresholds,
            branching_method, storage_depth, num_iter=num_iter
        )
        decision_time = time.time() - decision_start_time

        # STEP 4: create new split domains.
        split_start_time = time.time()
        split_depth = get_split_depth(x_L, split_partitions=split_partitions)
        new_x_L, new_x_U, cs, thresholds, split_depth, dm_lb, lA, lbias = input_split_parallel(
            x_L, x_U, shape, cs, thresholds, split_depth=split_depth, i_idx=split_idx,
            split_partitions=split_partitions, split_hint=split_hint, dm_lb=dm_lb, lA=lA, lbias=lbias)
        # this will double alpha for the new domains
        if isinstance(alphas, dict):
            alphas = repeat_alphas(alphas, split_depth, split_partitions)
        split_time = time.time() - split_start_time

        # STEP 5: shrink these new domains
        clip_time = 0.
        if enable_clip_domains:
            clip_start_time = time.time()
            ret = clip_domains(new_x_L, new_x_U, thresholds, lA, dm_lb=None, lbias=lbias, calculate_dm_lb=True)
            new_x_L, new_x_U = ret
            clip_time = time.time() - clip_start_time

        # STEP 6: Add new domains back to domain list.
        adddomain_start_time = time.time()
        # Clipping only updates the input bounds but not the thresholds
        d.add(dm_lb, new_x_L.detach(), new_x_U.detach(),
              alphas, cs, thresholds, split_idx=None, check_thresholds=False, check_bounds=True)
        adddomain_time = time.time() - adddomain_start_time


    rest_time_start = time.time()
    def _print_final_results():
        rest_time = time.time() - rest_time_start
        total_time = time.time() - total_start_time
        print(
            f"Total time: {total_time:.4f}  pickout: {pickout_time:.4f} split: {split_time:.4f}  "
            f"decision: {decision_time:.4f}  bounding: {bounding_time:.4f}  filtering: {filt_time:.4f}  "
            f"clipping: {clip_time:.4f}  add_domain: {adddomain_time:.4f}  rest: {rest_time:.4f}"
        )
        print("Length of domains:", len(d))
        print(f"{Visited} branch and bound domains visited")

    if len(d) == 0:
        print("No domains left, verification finished!")
        if dm_lb is not None and len(dm_lb) > 0:
            dm_lb_min = dm_lb.min().item()
            print(f"The lower bound of last batch is {dm_lb_min}")
        _print_final_results()
        return decision_thresh.max() + 1e-7
    else:
        if input_split_args["skip_getting_worst_domain"]:
            # It can be costly to call get_topk_indices when the domain list is long
            worst_idx = 0
        else:
            worst_idx = d.get_topk_indices().item()
        worst_val = d[worst_idx]
        global_lb = worst_val[0] - worst_val[-1]
        if not input_split_args["skip_getting_worst_domain"]:
            if 1 < global_lb.numel() <= 5:
                print(f"Current (lb-rhs): {global_lb}")
            else:
                print(f"Current (lb-rhs): {global_lb.max().item()}")

    _print_final_results()

    if input_split_args["show_progress"]:
        print(f"Progress: {d.get_progess():.10f}")
    sys.stdout.flush()

    return global_lb

def batch_verification_input_split(
        d:UnsortedInputDomainList, net: LiRPANet, batch, num_iter, decision_thresh, shape=None,
        bounding_method="crown", branching_method="sb",
        stop_func=stop_criterion_batch_any, split_partitions=2, best_ub=torch.tensor(float('inf'))):
    input_split_args = arguments.Config["bab"]["branching"]["input_split"]

    total_start_time = time.time()
    global Visited
    # if num_iter > 20:
    #     print("debug")

    temperature = arguments.Config['bab']['branching']['input_split']['softmax_temperature']
    find_feasible_args = arguments.Config["find_feasible_solution"]
    enable_find_feasible =find_feasible_args['enable']
    fast_pass_bounding = False
    use_adv = False
    report_time =False
    if enable_find_feasible:
        report_time = find_feasible_args['report_time']
        fast_pass_bounding = find_feasible_args['fast_pass_bounding']
        use_adv = find_feasible_args['sample_with_solution']
        if bounding_method == 'crown-pgd' and num_iter % find_feasible_args['bound_pgd_interval'] != 0:
            bounding_method = 'crown'
    # ret = d.pick_out_batch(batch, device=net.x.device)
    store_upper_bound = d.store_upper_bound
    store_adv_example = d.store_adv_example

    extra_item_list = d.extra_item_list
    def parse_ret(ret):
        alphas, dm_lb, x_L, x_U, cs, thresholds, split_idx = ret["alpha"], ret["lb"], ret["dm_l"], ret["dm_u"], ret["cs"], ret["threshold"], ret["split_idx"]
        dm_ub, adv, inlist_round = ret.get("ub", None), ret.get("adv_example", None), ret.get("inlist_round", None)
        extra_item_dict = ret.get("extra_item_dict", {})
        return alphas, dm_lb, x_L, x_U, cs, thresholds, split_idx, dm_ub, adv, inlist_round, extra_item_dict
    # STEP 1: pick out domains
    pickout_start_time = time.time()
    if store_upper_bound and input_split_args["ub_ratio"] > 0:
        num_ub= max(round(batch*input_split_args["ub_ratio"]),1)
        # ANONYMOUS: sample by adv example and ub-lb
        ret_u = d.pick_out_batch_softmax(num_ub, device=net.x.device, temperature=temperature, use_upper_bound=True, use_adv_example=use_adv)
        ret_l = d.pick_out_batch_softmax(batch-num_ub, device=net.x.device, temperature=temperature, use_upper_bound=False, use_adv_example=False)
        # # hack
        # if ub_ratio > 0.5:
        #     ret_l = d.pick_out_batch_softmax(batch-num_ub, device=net.x.device, temperature=temperature, use_upper_bound=False, use_adv_example=False)
        # else:
        #     ret_l = d.pick_out_batch_softmax(num_ub, device=net.x.device, temperature=temperature, use_upper_bound=False, use_adv_example=False, use_inlist_round=True)
        # ret_u = d.pick_out_batch_softmax(num_ub, device=net.x.device, temperature=temperature, use_upper_bound=True, use_adv_example=False, use_inlist_round=True)
        # ret_l = d.pick_out_batch_softmax(batch-num_ub, device=net.x.device, temperature=temperature, use_upper_bound=True, use_adv_example=False)
        alphas, dm_lb, x_L, x_U, cs, thresholds, split_idx, dm_ub, adv, inlist_round, extra_item_dict = parse_ret(ret_u)
        if ret_l is not None:
            alphas_l, dm_lb_l, x_L_l, x_U_l, cs_l, thresholds_l, split_idx_l, dm_ub_l, adv_l, inlist_round_l, extra_item_dict_l = parse_ret(ret_l)
            alphas = alphas + alphas_l
            dm_lb = torch.cat([dm_lb, dm_lb_l], dim=0)
            x_L = torch.cat([x_L, x_L_l], dim=0)
            x_U = torch.cat([x_U, x_U_l], dim=0)
            cs = torch.cat([cs, cs_l], dim=0)
            thresholds = torch.cat([thresholds, thresholds_l], dim=0)
            split_idx = torch.cat([split_idx, split_idx_l], dim=0)
            dm_ub = torch.cat([dm_ub, dm_ub_l], dim=0)
            if adv_l is not None:
                adv = torch.cat([adv, adv_l], dim=0)
            if inlist_round_l is not None:
                inlist_round = torch.cat([inlist_round, inlist_round_l], dim=0)
            if len(extra_item_list) > 0:
                for key in extra_item_list:
                    extra_item_dict[key] = torch.cat([extra_item_dict[key], extra_item_dict_l[key]], dim=0)
    else:
        ret = d.pick_out_batch_softmax(batch, device=net.x.device, temperature=temperature, use_upper_bound=False)
        alphas, dm_lb, x_L, x_U, cs, thresholds, split_idx, dm_ub, adv, inlist_round, extra_item_dict = parse_ret(ret)
    d.inc_inlist_round()
    pickout_time = time.time() - pickout_start_time
    if report_time:
        output_time("pickout", pickout_time)

    # ANONYMOUS: show stat for picked out domains and attack
    print(f"---- Stat for picked out domains ----")
    current_space_size = show_domain_stat(thresholds, x_L, x_U, dm_lb, dm_ub, d.global_x_L, d.global_x_U, inlist_round)
    total_space_size = d.show_stat()
    best_adv = None
    attack_time = 0
    sample_distribution = None
    require_distribution = arguments.Config["find_feasible_solution"]["enable"] and arguments.Config["find_feasible_solution"]["record_distribution_for_branching"]
    if input_split_args['update_rhs_with_attack']:
        assert (torch.all(thresholds == thresholds[0])), f'thresholds are not the same: {thresholds}'
        attack_start_time = time.time()
        ret_attack = update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb, net.model_ori, dm_ub, 
                                            prev_adv_example=adv, lirpa_model=net.net if fast_pass_bounding else None, extra_item_dict=extra_item_dict)
        thresholds, best_adv, attack_examples, upper_bounds, info_dict = ret_attack
        if require_distribution:
            sample_distribution = info_dict.get('sample_distribution', None)
            assert sample_distribution is not None, "sample_distribution is None but required"
        torch.cuda.empty_cache()
        # ANONYMOUS: delete bad domains where lb > best ub after attack
        d.update_threshold(min(best_adv[-1], best_ub))
        d.update_adv(attack_examples, upper_bounds)
        attack_time = time.time() - attack_start_time
    if enable_find_feasible and find_feasible_args["report_per_iter"]:
        if num_iter == 1 and os.path.exists(find_feasible_args['save_bab_path']):
            os.remove(find_feasible_args['save_bab_path'])
        with open(find_feasible_args['save_bab_path'], 'ab') as f:
            pickle.dump({"num_iter": num_iter, "x_L": x_L, "x_U": x_U, "lb": dm_lb, "ub": upper_bounds, "sol": attack_examples, "total_space_size": current_space_size+ total_space_size}, f)

    # STEP 2: create new split domains.
    split_start_time = time.time()
    split_depth = get_split_depth(x_L, split_partitions=split_partitions)
    if sample_distribution is not None:
        sample_distribution["dm_ub"] = upper_bounds
        # calulate split_idx based on sample distribution
        split_idx = eval(arguments.Config['bab']['branching']['input_split']['branching_heuristic'])(
            net, dm_lb, x_L, x_U, None, thresholds,
            branching_method, storage_depth, num_iter=num_iter,
            sample_distribution=sample_distribution
        )
    new_x_L, new_x_U, cs, thresholds, split_depth, _, _, _ = input_split_parallel(
        x_L, x_U, shape, cs, thresholds, split_depth=split_depth, i_idx=split_idx,
        split_partitions=split_partitions)

    new_dm_lb = torch.zeros([new_x_L.shape[0], *dm_lb.shape[1:]], device=new_x_L.device)
    if input_split_args["compare_with_old_bounds"]:
        assert split_depth == 1
        dm_lb = dm_lb.repeat(2, *[1]*(dm_lb.ndim - 1))
    else:
        dm_lb = None

    alphas = alphas * (split_partitions ** (split_depth - 1))
    split_time = time.time() - split_start_time

    if report_time:
        output_time("split", split_time)

    # TODO: ANONYMOUS: init means and covs for cem if needed
    if "means" in extra_item_list and "covs" in extra_item_list:
        means: torch.Tensor = extra_item_dict["means"]
        covs: torch.Tensor = extra_item_dict["covs"]
        # means: [n_domains, n_input], covs: [n_domains, n_input, n_input]
        n_domains, n_input = means.shape[:2]
        new_means = means.repeat(2**split_depth, *[1]*(means.ndim-1))
        new_covs = covs.repeat(2**split_depth, *[1]*(covs.ndim-1))
        adjust_means, adjust_covs = True, True
        if adjust_means:
            # mask = new_means < new_x_L.view(-1, n_input)
            # new_x_center = (new_x_L + new_x_U).view(-1, n_input) / 2
            # new_means[mask] = new_x_center[mask]
            # mask = new_means > new_x_U.view(-1, n_input)
            # new_means[mask] = new_x_center[mask]
            # ------
            # mask = new_means < new_x_L.view(-1, n_input)
            # new_diff = (new_x_U - new_x_L).view(-1, n_input)
            # new_means[mask] += new_diff[mask]
            # mask = new_means > new_x_U.view(-1, n_input)
            # new_means[mask] -= new_diff[mask]
            # ------
            new_x_center = (new_x_L+new_x_U)/2
            x_center = ((x_L+x_U)/2).repeat(2**split_depth, *[1]*(x_L.ndim-1))
            new_means += x_center - new_x_center
            # assert torch.all(new_means >= new_x_L.view(-1, n_input)) and torch.all(new_means <= new_x_U.view(-1, n_input))
        new_means = new_means.clamp_(new_x_L.view(-1, n_input), new_x_U.view(-1, n_input))

        if adjust_covs:
            mask = torch.zeros([n_domains, n_input], dtype=torch.bool, device=means.device)
            mask.scatter_(1, split_idx.to(torch.int64), True)
            diag_elements = torch.diagonal(covs, dim1=1, dim2=2)
            diag_elements[mask] *= 0.5
            new_covs[:, torch.arange(covs.size(1)), torch.arange(covs.size(1))] = diag_elements.repeat(2**split_depth, 1)
            mask = mask.unsqueeze(2) | mask.unsqueeze(1)
            new_covs[mask.repeat(2**split_depth, 1, 1)] *= 0.5
        extra_item_dict["means"] = new_means
        extra_item_dict["covs"] = new_covs

    # ANONYMOUS: update upper bounds and solutions for new domains
    new_dm_ub = None
    new_adv = None

    if store_upper_bound or store_adv_example:
        attack_examples = attack_examples.squeeze(0)
        # start = time.time()
        new_num_domains = new_x_L.shape[0]
        new_ub = torch.full((new_num_domains, 1), float('inf'),device=x_L.device)
        new_adv = torch.full((new_num_domains, *attack_examples.shape[1:]), float('inf'),device=x_L.device)
        for i, example in enumerate(attack_examples):
            # Check if the example is within each subdomain
            is_within_bounds = (example >= new_x_L) & (example <= new_x_U)
            # Reshape to collapse all dimensions except the first
            is_within_bounds = is_within_bounds.view(new_num_domains, -1)
            is_within_bounds = torch.all(is_within_bounds, dim=1)

            # Find subdomains where the example is within bounds
            current_upper_bound = upper_bounds[i].item()
            if torch.any(is_within_bounds):
                match_indices = torch.where(is_within_bounds)[0]
                indices_to_update = match_indices[new_ub[match_indices].squeeze(1) > current_upper_bound]
                new_ub[indices_to_update] = current_upper_bound
                new_adv[indices_to_update] = attack_examples[i]
        # print(torch.nonzero(~torch.isinf(new_ub).any(dim=1)).squeeze(-1).shape[0], attack_examples.shape[0])
        random_example = torch.empty_like(new_x_L).uniform_()*(new_x_U-new_x_L)+new_x_L
        # [new_num_domains, output_dim]
        random_output = net.model_ori(random_example).T.unsqueeze(0).repeat(net.c.shape[0], 1, 1)
        # assume net.c.shape[0] == 1, net.c.shape[1] == 1, one output constraint
        new_dm_ub = net.c.matmul(random_output).squeeze(0).view(new_ub.shape)
        new_dm_ub = torch.min(new_dm_ub, new_ub)
        update_mask = (new_ub > new_dm_ub).flatten()
        new_adv[update_mask] = random_example[update_mask]
    
    if fast_pass_bounding:
        repeat_bound_start_time = time.time()
        for node in net.net.default_reference_bounds:
            lb, ub = net.net.default_reference_bounds[node]
            net.net.default_reference_bounds[node] = (lb.repeat(2**split_depth, *[1]*(lb.ndim-1)), ub.repeat(2**split_depth, *[1]*(lb.ndim-1)))
        store_fv_node_list = list(net.net.default_reference_bounds.keys())
        net.net.forward(new_adv, store_fv_node_list=store_fv_node_list)
        for node in store_fv_node_list:
            lb, ub = net.net.default_reference_bounds[node]
            fv= net.net[node].forward_value
            net.net.default_reference_bounds[node] = (torch.min(lb, fv), torch.max(ub, fv))
        net.net.forward(net.net.global_input)
        if report_time:
            output_time("repeat_bound", time.time() - repeat_bound_start_time)

    # STEP 3: Compute bounds for all domains and make decisions.
    param_dict = get_param_dict(bounding_method, net)
    new_x_L, new_x_U, new_dm_lb, alphas, split_idx, bounding_time, decision_time, clip_time = get_bound_and_decision(
        net, dm_lb, new_x_L, new_x_U, alphas, cs, thresholds,
        bounding_method, branching_method, stop_func, num_iter,
        param_dict=param_dict)

    print(f"    lower bound: min {new_dm_lb.min().item()}, median {new_dm_lb.median().item()}, max {new_dm_lb.max().item()}")
    # if bound_upper and ret is not None:
    #     print(f"    worst bound of gap: {(ret[1]-ret[0]).max().item()} on iter {num_iter}")
    #     print(f"    upper bound: min {ret[1].min().item()}, mean {ret[1].mean().item()}, max {ret[1].max().item()}")
    print(f"    upper bound used: min {upper_bounds.min().item()}, median {upper_bounds.median().item()}, max {upper_bounds.max().item()}")
    print(f"    New domains: {new_x_L.shape[0]}, bad domains: {(new_dm_lb > upper_bounds.min().to('cpu')).sum().item()}")

    # STEP 4: Add new domains back to domain list.
    adddomain_start_time = time.time()
    # print("new_lb\n", *(f"{lb:<10.2f}" for lb in new_dm_lb.flatten().tolist()))
    if store_upper_bound:
        # print("new_ub\n", *(f"{ub:<10.2f}" for ub in new_dm_ub.flatten().tolist()))
        if param_dict["early_stop_conditioner"] is None:
            assert torch.all(new_dm_lb <= new_dm_ub.to(new_dm_lb.device) + 1e-6)
        else: 
            new_dm_lb = loose_bound(net, dm_lb, alphas, cs, thresholds, new_x_L, new_x_U, new_dm_lb, new_dm_ub,
                        bounding_method, branching_method, stop_func, num_iter, param_dict=param_dict)

    d.add(new_dm_lb, new_x_L.detach(), new_x_U.detach(),
          alphas, cs, thresholds, split_idx, 
          ub=new_dm_ub if store_upper_bound else None,
          adv_example=new_adv if store_adv_example else None,
          extra_item_dict=extra_item_dict)
    adddomain_time = time.time() - adddomain_start_time

    Visited += len(new_x_L)
    rest_time_start = time.time()
    def _print_final_results():
        rest_time = time.time() - rest_time_start
        total_time = time.time() - total_start_time
        print(
            f"Total time: {total_time:.4f}  pickout: {pickout_time:.4f}  split: {split_time:.4f}  "
            f"attack: {attack_time:.4f}  decision: {decision_time:.4f}  bounding: {bounding_time:.4f}  "
            f"clipping: {clip_time:.4f}  add_domain: {adddomain_time:.4f}  rest: {rest_time:.4f}"
        )
        print("Length of domains:", len(d))
        print(f"{Visited} branch and bound domains visited")

    _print_final_results()
    # ANONYMOUS: return best_adv
    if len(d) == 0:
        print("No domains left, verification finished!")
        if new_dm_lb is not None:
            new_dm_lb_min = new_dm_lb.min().item()
            print(f"The lower bound of last batch is {new_dm_lb_min}")
        return decision_thresh.max() + 1e-7, best_adv
    else:
        if input_split_args["skip_getting_worst_domain"]:
            # It can be costly to call get_topk_indices when the domain list is long
            worst_idx = 0
        else:
            worst_idx = d.get_topk_indices().item()
        worst_val = d[worst_idx]
        global_lb = worst_val[0] - worst_val[4]
        if not input_split_args["skip_getting_worst_domain"]:
            if 1 < global_lb.numel() <= 5:
                print(f"Current (lb-rhs): {global_lb}")
            else:
                print(f"Current (lb-rhs): {global_lb.max().item()}")

    if input_split_args["show_progress"]:
        print(f"Progress: {d.get_progess():.10f}")
    sys.stdout.flush()

    return global_lb, best_adv

def loose_bound(net, dm_lb, alphas, cs, thresholds, new_x_L, new_x_U, new_dm_lb, new_dm_ub,
                bounding_method, branching_method, stop_func, num_iter, param_dict={}):
    retry_patience = param_dict.get('retry_patience', 3)
    retry = 0
    # (new_dm_lb <= new_dm_ub.to(new_dm_lb.device)).sum() < new_dm_lb.shape[0]//2
    while (not torch.all(new_dm_lb <= new_dm_ub.to(new_dm_lb.device))) and retry < retry_patience:
        print("lb <= ub violated, retrying...")
        early_stop_conditioner = param_dict["early_stop_conditioner"]
        early_stop_conditioner.loosen()
        new_x_L, new_x_U, new_dm_lb, alphas, split_idx, _, _, _ = get_bound_and_decision(
            net, dm_lb, new_x_L, new_x_U, alphas, cs, thresholds,
            bounding_method, branching_method, stop_func, num_iter,
            param_dict=param_dict)
        retry += 1
    return new_dm_lb

def get_bound_and_decision(net: LiRPANet, dm_lb, x_L, x_U, alphas, cs, thresholds,
                           bounding_method, branching_method, stop_func,
                           num_iter, split_partitions=2, param_dict={}):
    report_time = param_dict.get('report_time', False)
    bounding_start_time = time.time()
    ret = net.get_lower_bound_naive(
        dm_lb=dm_lb, dm_l=x_L, dm_u=x_U, alphas=alphas,
        bounding_method=bounding_method, branching_method=branching_method,
        C=cs, stop_criterion_func=stop_func, thresholds=thresholds,
        num_iter=num_iter, split_partitions=split_partitions, param_dict=param_dict)
    new_dm_lb, alphas, lA, lbias, lb_crown = ret  # here alphas is a dict
    bounding_time = time.time() - bounding_start_time

    new_dm_lb = new_dm_lb.to(device=thresholds.device)  # ensures it is on the same device as it may be different
    lb_crown = lb_crown.to(device=thresholds.device)

    # shrink these new domains
    enable_clip_domains = arguments.Config["bab"]["branching"]["input_split"]["enable_clip_domains"]
    clip_time = 0.
    if enable_clip_domains:
        clip_start_time = time.time()
        ret = clip_domains(x_L, x_U, thresholds, lA, lb_crown)
        x_L, x_U = ret
        clip_time = time.time() - clip_start_time

    decision_start_time = time.time()
    require_distribution = param_dict.get('require_distribution', False)
    if require_distribution:
        # dummy split_idx, split_idx will be calulated after the distribution is obtained
        split_idx = torch.zeros(x_L.shape[0], storage_depth, device=x_L.device)
    else:
        split_idx = eval(arguments.Config['bab']['branching']['input_split']['branching_heuristic'])(
            net, new_dm_lb, x_L, x_U, lA, thresholds,
            branching_method, storage_depth, num_iter=num_iter
        )
    decision_time = time.time() - decision_start_time
    if report_time:
        output_time("bound", bounding_time)
        output_time("split", decision_time)
    return x_L, x_U, new_dm_lb, alphas, split_idx, bounding_time, decision_time, clip_time

def get_param_dict(bounding_method, net:LiRPANet=None):
    find_feasible_args = arguments.Config["find_feasible_solution"]
    if not find_feasible_args['enable']:
        return {}
    param_dict = {}
    if ('pgd' in bounding_method or 'mppi' in bounding_method):
        param_dict['pgd_restarts'] = find_feasible_args['bound_pgd_restarts']
        param_dict['pgd_steps'] = find_feasible_args['bound_pgd_steps']
        param_dict['lr_decay'] = find_feasible_args['bound_pgd_lr_decay']
        param_dict['auto_factor'] = find_feasible_args['bound_pgd_auto_factor']
        param_dict['mppi_noise_factor'] = find_feasible_args['bound_mppi_noise_factor']
        param_dict['mppi_weight_factor'] = find_feasible_args['bound_mppi_weight_factor']
    param_dict['final_ibp'] = find_feasible_args['final_ibp_bounding']
    param_dict['fast_pass'] = find_feasible_args['fast_pass_bounding']
    param_dict['require_distribution'] = find_feasible_args['record_distribution_for_branching']
    param_dict['etile_ratio'] = find_feasible_args['elite_ratio_for_branching']
    param_dict['min_n_elites'] = find_feasible_args['min_n_elites_for_branching']
    early_stop_conditioner = find_feasible_args['early_stop_bounding']["conditioner"]
    retry_patience = find_feasible_args['early_stop_bounding']["retry_patience"]
    param_dict['early_stop_conditioner'] = None
    param_dict['retry_patience'] = retry_patience
    param_dict['report_time'] = find_feasible_args['report_time']
    if early_stop_conditioner != 'None' and net is not None:
        if getattr(net, 'early_stop_conditioner', None) is not None:
            param_dict['early_stop_conditioner'] = net.early_stop_conditioner
        else:
            param_dict['early_stop_conditioner'] = eval(early_stop_conditioner)()
            net.early_stop_conditioner = param_dict['early_stop_conditioner']
    return param_dict

def input_bab_parallel(net:LiRPANet, init_domain:torch.Tensor, x:BoundedTensor, rhs:torch.Tensor=None,
                       timeout=None, max_iterations=None,
                       vnnlib=None, c_transposed=False, return_domains=False):
    """Run input split bab.

    c_transposed: bool, by default False, indicating whether net.c matrix has
        transposed between dim=0 and dim=1. As documented in abcrown.py bab(),
        if using input split, and if there are multiple specs with shared single input,
        we transposed the c matrix from [multi-spec, 1, ...] to [1, multi-spec, ...] so that
        net.build() process in input_bab_parallel could share intermediate layer
        bounds across all specs. If such transpose happens, c_transposed is set
        to True, so that after net.build() in this func, we can transpose c
        matrix back, repeat x_LB & x_UB, duplicate alphas, to prepare for input domain bab.
    """
    global storage_depth

    start = time.time()
    # All supported arguments.
    global Visited
    global total_picked_out

    bab_args = arguments.Config["bab"]
    branching_args = bab_args["branching"]
    input_split_args = branching_args["input_split"]
    find_feasible_args = arguments.Config["find_feasible_solution"]

    timeout = timeout or bab_args["timeout"]
    batch = arguments.Config["solver"]["batch_size"]
    bounding_method = arguments.Config["solver"]["bound_prop_method"]
    init_bounding_method = arguments.Config["solver"]["init_bound_prop_method"]
    max_iterations = max_iterations or bab_args["max_iterations"]
    sort_domain_iter = bab_args["sort_domain_interval"]
    branching_method = branching_args['method']
    adv_check = input_split_args['adv_check']
    split_partitions = input_split_args['split_partitions']
    catch_assertion = input_split_args['catch_assertion']
    use_clip_domains = input_split_args["enable_clip_domains"]
    use_reordered_bab = input_split_args["reorder_bab"]
    split_hint = input_split_args["split_hint"]
    enable_check_adv = arguments.Config["attack"]["input_split_check_adv"]["enabled"]
    enable_check_adv = (
        arguments.Config["attack"]["pgd_order"] != "skip" if enable_check_adv == "auto"
        else enable_check_adv == "true"
    )
    # For this reordering, we are adjusting the effective batch size since (2*batches) number of domains should
    # be bounded per iteration no matter the BaB order
    batch = 2*batch if use_reordered_bab else batch
    enable_find_feasible = find_feasible_args['enable']
    save_solution, show_solution, report_per_iter, results_file, report_time = False, False, False, None, False
    stuck_patience, pruning_stuck_threshold, pruning_stuck_patience = -1, 1, -1
    if enable_find_feasible:
        assert use_reordered_bab == False
        report_time = find_feasible_args['report_time']
        use_upper_bound = find_feasible_args["use_upper_bound"]
        warm_start = find_feasible_args['warm_start']
        early_stop_patience = find_feasible_args['early_stop_patience']
        save_solution, show_solution = find_feasible_args['save_solution'], find_feasible_args['show_solution']
        report_per_iter, results_file = find_feasible_args['report_per_iter'], find_feasible_args['save_solution_path']
        arguments.Config['bab']['branching']['input_split']['update_rhs_with_attack'] = True
        arguments.Config["attack"]["pgd_order"] = "skip"
        print("Enable find feasible solution, update_rhs_with_attack and pgd_order are set to True and skip respectively.")
        best_adv, best_iter = None, -1
        best_ub = torch.tensor(float('inf'),device=x.device)
        stuck_patience, pruning_stuck_threshold, pruning_stuck_patience = find_feasible_args['stuck_patience'], find_feasible_args['pruning_stuck_threshold'], find_feasible_args['pruning_stuck_patience']
    else:
        warm_start = False
        use_upper_bound = False
        early_stop_patience = None

    if save_solution:
        assert results_file is not None
        with open(results_file, 'w') as f:
            f.write("")

    if init_bounding_method == "same":
        init_bounding_method = bounding_method

    if c_transposed or net.c.shape[1] == 1:
        # When c_transposed applied, previous checks have ensured that there is only single spec,
        # but we compressed multiple data samples to the spec dimension by transposing C
        # so we need "all" to be satisfied to stop.
        # Another case is there is only single spec, so batch_any equals to all. Then, we set to all
        # so we can allow prune_after_crown optimization
        stop_func = stop_criterion_all
    else:
        # Possibly multiple specs in each data sample
        stop_func = stop_criterion_batch_any

    Visited = 0
    total_picked_out = 0

    dm_l = x.ptb.x_L
    dm_u = x.ptb.x_U

    if (dm_u - dm_l > 0).int().sum() == 1:
        branching_method = "naive"

    # ANONYMOUS: always init adv example from massive_pgd_attack (batch, specs, c, h, w)
    attack_image = None
    upper_bound = None
    best_ub = None
    extra_item_list = []
    if enable_find_feasible:
        find_feasible_args['pre_pgd_restarts'] = arguments.Config["attack"]["input_split_check_adv"]["pgd_restarts"]
        _, _, attack_images, attack_outputs, info_dict = massive_pgd_attack(x, net.model_ori, vnnlib, find_feasible_args['pre_pgd_restarts'], 
                                                                find_feasible_args['pre_pgd_steps'], True, 
                                                                net.net if find_feasible_args['fast_pass_bounding'] else None)
        # x: (batch, c, h, w).
        # attack_image: [batch, spec, c, h, w], attack_output: [batch, num_or_spec, output_dim]
        # net.c: [num_example, num_spec, num_output]
        # rhs: [num_example, num_spec]
        # num_spec = 1
        assert attack_images.shape[1] == attack_outputs.shape[1] == 1
        # attack_images = attack_images.squeeze(0)
        attack_outputs = attack_outputs.permute(0, 2, 1)
        #  [num_example, num_spec, num_output] * [num_example, num_output, num_spec]
        upper_bound = net.c.matmul(attack_outputs).squeeze(-2)
        # # upper_bound: [batch, num_spec, num_restarts]
        upper_bound, upper_idx = upper_bound.min(dim=0, keepdim=True)
        # upper_bound: [batch, num_spec]
        rhs = torch.min(rhs, upper_bound)
        attack_image = attack_images[upper_idx[0], 0].clone().detach()
        best_attack_image = attack_image[0].flatten().tolist()
        best_attack_output = attack_outputs[upper_idx[0]].flatten().tolist()
        del attack_images, attack_outputs
        torch.cuda.empty_cache()
        best_ub = upper_bound[0,0]
        best_iter = 0
        best_adv = (best_attack_image, best_attack_output, best_ub)
        info_keys = info_dict.keys()
        extra_item_list = [key for key in info_keys if key != 'sample_distribution']
    
    # ANONYMOUS: adjust the order of bounding and attack. in some cases, we need bounds from attack.
    try:
        param_dict = get_param_dict(bounding_method, net)
        build_start_time = time.time()
        global_lb, ret = net.build(
            init_domain, x, stop_criterion_func=stop_func(rhs),
            bounding_method=init_bounding_method, decision_thresh=rhs, return_A=False, 
            bound_upper=False, param_dict=param_dict)
        if report_time:
            output_time("bound", time.time() - build_start_time)
    except AssertionError:
        if catch_assertion:
            global_lb = torch.ones(net.c.shape[0], net.c.shape[1],
                                   device=net.device) * torch.inf
            ret = {"alphas": {}}
        else:
            raise

    if getattr(net.net[net.net.input_name[0]], "lA", None) is not None:
        lA = net.net[net.net.input_name[0]].lA.transpose(0, 1)
    else:
        lA = None
        if bounding_method == "sb":
            raise ValueError("sb heuristic cannot be used without lA.")
        if use_clip_domains:
            raise ValueError("clip domains cannot be used without lA.")

    if c_transposed:
        lA, global_lb, rhs, dm_l, dm_u = transpose_c_back(
            lA, global_lb, rhs, dm_l, dm_u, ret, net)

    result = "unknown"

    if enable_find_feasible:
        if save_solution:
            output_solution("New", 0, best_attack_image, best_attack_output, best_ub, global_lb, results_file)
        if show_solution:
            output_solution("New", 0, best_attack_image, best_attack_output, best_ub, global_lb)
    # shrink the initial dm_l and dm_u
    if use_clip_domains and not use_reordered_bab:
        dm_l, dm_u = clip_domains(dm_l, dm_u, rhs, lA, global_lb)

    # compute storage depth
    use_alpha = init_bounding_method.lower() == "alpha-crown" or bounding_method == "alpha-crown"
    min_batch_size = (
        arguments.Config["solver"]["min_batch_size_ratio"]
        * arguments.Config["solver"]["batch_size"]
    )
    max_depth = max(int(math.log(max(min_batch_size, 1)) // math.log(split_partitions)), 1)
    storage_depth = min(max_depth, dm_l.shape[-1])
    if param_dict.get('require_distribution', False):
        storage_depth = 1
    domains = UnsortedInputDomainList(
        storage_depth, use_alpha=use_alpha, 
        sort_index=input_split_args["sort_index"],
        sort_descending=input_split_args["sort_descending"],
        use_split_idx=not use_reordered_bab,
        store_upper_bound=use_upper_bound, 
        store_adv_example=warm_start, extra_item_list=extra_item_list)
    domains.global_x_L = dm_l
    domains.global_x_U = dm_u

    initial_verified, remaining_index = initial_verify_criterion(global_lb, rhs)
    if initial_verified and not enable_find_feasible:
        result = "safe"
    else:
        if enable_find_feasible:
            early_stop_conditioner = param_dict["early_stop_conditioner"]
            if initial_verified and early_stop_conditioner is not None:
                global_lb = loose_bound(net, None, ret["alphas"] if use_alpha else [], net.c, rhs, dm_l, dm_u, global_lb, rhs, 
                            bounding_method, init_bounding_method, stop_func, 0, param_dict=param_dict)
                initial_verified, remaining_index = initial_verify_criterion(global_lb, rhs)
        # compute initial split idx
        split_start_time = time.time()
        split_idx = eval(arguments.Config['bab']['branching']['input_split']['branching_heuristic'])(
            net, global_lb, dm_l, dm_u, lA, rhs, branching_method, storage_depth)
        if report_time:
            output_time("split", time.time() - split_start_time)
        if use_reordered_bab:
            lbias = deconstruct_lbias(dm_l, dm_u, lA, global_lb)
            split_depth = get_split_depth(dm_l, split_partitions=split_partitions)
            dm_l, dm_u, cs, thresholds, split_depth, global_lb, lA, lbias = input_split_parallel(
                dm_l, dm_u, x.shape, net.c, rhs, split_depth=split_depth, i_idx=split_idx,
                split_partitions=split_partitions, split_hint=split_hint, dm_lb=global_lb, lA=lA, lbias=lbias)
            # shrink the initial dm_l and dm_u
            if use_clip_domains:
                dm_l, dm_u = clip_domains(dm_l, dm_u, rhs, lA, global_lb, lbias, calculate_dm_lb=True)
            alphas = ret["alphas"]
            # this will double alpha for the new domains
            if isinstance(alphas, dict):
                alphas = repeat_alphas(alphas, split_depth, split_partitions)
            domains.add(global_lb, dm_l.detach(), dm_u.detach(), alphas, cs, thresholds, split_idx=None,
                        check_thresholds=False, check_bounds=use_clip_domains)
        else:
            domains.add(global_lb, dm_l.detach(), dm_u.detach(),
                        ret["alphas"], net.c, rhs, split_idx=split_idx, remaining_index=remaining_index,
                        check_bounds=False,
                        ub = upper_bound if domains.store_upper_bound else None,
                        adv_example=attack_image if domains.store_adv_example else None, extra_item_dict = info_dict)
        if arguments.Config["attack"]["pgd_order"] == "after":
            if attack_in_input_bab_parallel(net.model_ori, domains, x, vnnlib=vnnlib):
                print("pgd attack succeed in input_bab_parallel")
                result = "unsafe"
        if input_split_args["presplit_domains"]:
            assert not use_alpha
            load_presplit_domains(
                domains, net, bounding_method, branching_method, stop_func,
            )

    auto_batch_size = AutoBatchSize(
        batch, net.device,
        enable=arguments.Config["solver"]["auto_enlarge_batch_size"])

    num_iter = 1
    stuck_iter, pruning_stuck_iter = 0, 0
    curr_space_size = prev_space_size = domains.get_total_space_size()
    enhanced_bound_initialized = False
    batch_verification_fn = reordered_batch_verification_input_split if use_reordered_bab else batch_verification_input_split
    while (result == "unknown" and len(domains) > 0
           and (max_iterations == -1 or num_iter <= max_iterations)):
        print(f'Iteration {num_iter}')
        # ANONYMOUS: add early stop
        if enable_find_feasible and (early_stop_patience != None) and (num_iter > best_iter + early_stop_patience):
            print(f"No improvement for {early_stop_patience} iterations, early stop.")
            break
        # sort the domains every certain number of iterations
        if sort_domain_iter > 0 and num_iter % sort_domain_iter == 0:
            sort_start_time = time.time()
            domains.sort()
            sort_time = time.time() - sort_start_time
            print(f"Sorting domains used {sort_time:.4f}s")

        last_glb = global_lb.max()

        if enable_check_adv:
            if adv_check != -1 and Visited >= adv_check:
                adv_check_start_time = time.time()
                # check whether adv example found
                if check_adv(domains, net.model_ori, x, vnnlib=vnnlib):
                    return global_lb.max(), Visited, "unsafe"
                adv_check_time = time.time() - adv_check_start_time
                print(f"Adv attack time: {adv_check_time:.4f}s")

        batch_ = batch
        if branching_method == "brute-force" and num_iter <= input_split_args["bf_iters"]:
            batch_ = input_split_args["bf_batch_size"]
        print("Batch size:", batch_)
        auto_batch_size.record_actual_batch_size(min(batch_, len(domains)))
        try:
            global_lb, best_adv = batch_verification_fn(
                domains, net, batch_,
                num_iter=num_iter, decision_thresh=rhs, shape=x.shape,
                bounding_method=bounding_method, branching_method=branching_method,
                stop_func=stop_func, split_partitions=split_partitions, best_ub=best_ub)
        except AssertionError:
            if catch_assertion:
                global_lb = torch.ones(net.c.shape[0], net.c.shape[1],
                                       device=net.device) * torch.inf
            else:
                raise
        batch = check_auto_enlarge_batch_size(auto_batch_size)

        if enable_find_feasible:
            # ANONYMOUS: update ub, remove bad domains
            curr_attack_image, curr_attack_output, curr_ub = best_adv
            ub_updated = False
            if curr_ub < best_ub:
                ub_updated = True
                best_ub = curr_ub
                best_attack_image = curr_attack_image
                best_attack_output = curr_attack_output
                best_iter = num_iter
            else:
                stuck_iter += 1
                if stuck_patience >= 0 and stuck_iter > stuck_patience:
                    print(f"No improvement on best objective for {stuck_iter} iterations.")
                    if use_upper_bound:
                        print("adjust the heuristics to select domains if possible.")
                        ub_ratio = input_split_args["ub_ratio"]
                        if ub_ratio != 0.5:
                            print(f"Update the upper bound ratio from {ub_ratio} to {1 - ub_ratio}.")
                            input_split_args["ub_ratio"] = 1 - ub_ratio
                            stuck_iter = 0
                        softmax_temperature = input_split_args["softmax_temperature"]
                        if softmax_temperature != 1:
                            print(f"Update the softmax temperature from {softmax_temperature} to {1 / softmax_temperature}.")
                            input_split_args["softmax_temperature"] = 1 / softmax_temperature
                            stuck_iter = 0
            domains.update_threshold(best_ub)
            if pruning_stuck_patience >= 0:
                curr_space_size = domains.get_total_space_size()
                if curr_space_size == prev_space_size and curr_space_size > pruning_stuck_threshold:
                    pruning_stuck_iter += 1
                    if pruning_stuck_iter >= pruning_stuck_patience:
                        print(f"No improvement on pruning subdomains for {pruning_stuck_iter} iterations.")
                        if early_stop_conditioner is not None:
                            print("Try to tighten the bound estimation.")
                            early_stop_conditioner.tighten()
                            pruning_stuck_iter = 0
            # ANONYMOUS: show and save the current best adv example.
            if save_solution or show_solution:
                if report_per_iter:
                    output_solution("New", num_iter, curr_attack_image, curr_attack_output, curr_ub, global_lb, results_file)
                if ub_updated:
                    if save_solution:
                        output_solution("Best", num_iter, best_attack_image, best_attack_output, best_ub, global_lb, results_file)
                    if show_solution:
                        output_solution("Best", num_iter, best_attack_image, best_attack_output, best_ub, global_lb)
                
        # once the lower bound stop improving we change to solve alpha mode
        # ANONYMOUS: not tested
        if (arguments.Config["solver"]["bound_prop_method"]
            != input_split_args["enhanced_bound_prop_method"]
            and time.time() - start > input_split_args["enhanced_bound_patience"]
            and global_lb.max().cpu() <= last_glb.cpu()
            and bounding_method != "alpha-crown"
            and not enhanced_bound_initialized
        ):
            enhanced_bound_initialized = True
            global_lb, domains, branching_method, bounding_method = enhanced_bound_init(
                net, init_domain, x, stop_func, rhs, storage_depth, num_iter)

        if arguments.Config["attack"]["pgd_order"] != "skip":
            if time.time() - start > input_split_args["attack_patience"]:
                print("Perform PGD attack with massively random starts finally.")
                ret_adv = massive_pgd_attack(x, net.model_ori, vnnlib=vnnlib)[1]
                if ret_adv:
                    result = "unsafe"
                    break

        if time.time() - start > timeout:
            print("Time out!")
            break

        print(f"Cumulative time: {time.time() - start}\n")
        num_iter += 1

    if result == "unknown" and len(domains) == 0:
        result = "safe"

    # ANONYMOUS: show and save the final best adv example.
    if enable_find_feasible and (best_adv is not None):
        if save_solution:
            output_solution("Final", best_iter, best_attack_image, best_attack_output, best_ub, global_lb, results_file)
        if show_solution:
            output_solution("Final", best_iter, best_attack_image, best_attack_output, best_ub, global_lb)
    
    if return_domains:
        # Thresholds may have been updated by PGD attack so that different
        # domains may have different thresholds. Restore thresholds to the
        # default RHS for the sorting.
        domains.threshold._storage.data[:] = rhs
        domains.sort()
        if return_domains == -1:
            return_domains = len(domains)
        lower_bound, x_L, x_U = domains.pick_out_batch(
            return_domains, device="cpu")[1:4]
        return lower_bound, x_L, x_U
    else:
        del domains
        return global_lb.max(), Visited, result

def filter_nodes(nodes, index_list=None, operation_list=None, filename_list=None, linenumber_list=None, mode='any'):
    filtered_nodes = []

    def is_in_list(value, lst):
        return lst is None or value in lst

    for node in nodes:
        index, operation, filename, linenumber = parse_node(node)

        if mode == 'all':
            if all([is_in_list(index, index_list),
                    is_in_list(operation, operation_list),
                    is_in_list(filename, filename_list),
                    is_in_list(linenumber, linenumber_list)]):
                filtered_nodes.append(node)
        elif mode == 'any':
            if any([is_in_list(index, index_list),
                    is_in_list(operation, operation_list),
                    is_in_list(filename, filename_list),
                    is_in_list(linenumber, linenumber_list)]):
                filtered_nodes.append(node)

    return filtered_nodes

def parse_node(node):
    parts = node.split('_')
    index = parts[0].split('/')[1]  # Extract index
    operation = parts[1]  # Extract operation

    # Handle filename and linenumber
    linenumber = parts[-1] if parts[-1].isdigit() else None
    filename = '_'.join(parts[2:-1]) if linenumber else '_'.join(parts[2:])

    return index, operation, filename, linenumber

def parse_node_names(nodes):
    index_list = set()
    operation_list = set()
    filename_list = set()
    linenumber_list = set()

    for node in nodes:
        index, operation, filename, linenumber = parse_node(node)

        index_list.add(index)
        operation_list.add(operation)
        if filename:
            filename_list.add(filename)
        if linenumber:
            linenumber_list.add(linenumber)

    return {
        'index_list': list(index_list),
        'operation_list': list(operation_list),
        'filename_list': list(filename_list),
        'linenumber_list': list(linenumber_list)
    }

def parse_node(node):
    parts = node.split('_')
    index = parts[0].split('/')[1]  # Extract index
    operation = parts[1]  # Extract operation

    # Handle filename and linenumber
    linenumber = parts[-1] if parts[-1].isdigit() else None
    filename = '_'.join(parts[2:-1]) if linenumber else '_'.join(parts[2:])

    return index, operation, filename, linenumber

def output_solution(mode, iter, attack_image, attack_output, ub, global_lb, results_file=None):
    """Print or save the solution."""
    # mode can be "Final" or "Best" or "New"
    output_str = f"{mode} solution found in {iter}:\n"
    output_str += f"  Adv example: {attack_image}\n"
    output_str += f"  Adv output: {attack_output}\n"
    output_str += f"  best upper bound: {ub}\n"
    output_str += f"  Current (lb-rhs): {global_lb.min()}\n"
    if results_file is not None:
        with open(results_file, 'a') as f:
            f.write(output_str)
            f.write('\n')
    else:
        print(output_str)

def load_presplit_domains(domains, net, bounding_method, branching_method, stop_func):
    input_split_args = arguments.Config["bab"]["branching"]["input_split"]
    use_reordered_bab = input_split_args["reorder_bab"]
    batch_size = arguments.Config["solver"]["batch_size"]
    batch_size = batch_size*2 if use_reordered_bab else batch_size
    ret = domains.pick_out_batch(len(domains))
    alphas, dm_lb, x_L, x_U, cs, thresholds, split_idx = ret

    presplit_dm_l, presplit_dm_u = torch.load(
        input_split_args["presplit_domains"])
    presplit_dm_l = presplit_dm_l.to(dm_lb)
    presplit_dm_u = presplit_dm_u.to(dm_lb)
    num_presplit_domains = presplit_dm_l.shape[0]
    print(f"Loaded {num_presplit_domains} pre-split domains")

    dm_lb = dm_lb.expand(batch_size, -1)
    cs = cs.expand(batch_size, -1, -1)
    thresholds = thresholds.expand(batch_size, -1)
    num_batches = (num_presplit_domains + batch_size - 1) // batch_size

    for i in range(num_batches):
        print(f"Pre-split domains batch {i+1}/{num_batches}:")
        x_L = presplit_dm_l[i*batch_size:(i+1)*batch_size]
        x_U = presplit_dm_u[i*batch_size:(i+1)*batch_size]
        size = x_L.shape[0]
        x_L, x_U, new_dm_lb, alphas, split_idx, _, _, _ = get_bound_and_decision(
            net, dm_lb[:size], x_L, x_U, alphas, cs[:size], thresholds[:size],
            bounding_method, branching_method, stop_func, num_iter=1
        )
        num_domains_pre = len(domains)
        domains.add(new_dm_lb, x_L, x_U, alphas, cs[:size], thresholds[:size], None if use_reordered_bab else split_idx)
        print(f"{len(domains) - num_domains_pre} domains added, "
              f"{len(domains)} in total")
        print()

    print(f"{len(domains)} pre-split domains added out of {presplit_dm_l.shape[0]}")
    verified_ratio = 1 - len(domains) * 1. / presplit_dm_l.shape[0]
    print(f"Verified ratio: {verified_ratio}")


def enhanced_bound_init(net, init_domain, x, stop_func, rhs, storage_depth,
                        num_iter):
    input_split_args = arguments.Config["bab"]["branching"]["input_split"]
    branching_method = input_split_args["enhanced_branching_method"]
    bounding_method = input_split_args["enhanced_bound_prop_method"]
    split_partitions = input_split_args["split_partitions"]
    split_hint = input_split_args["split_hint"]
    use_reordered_bab = input_split_args["reorder_bab"]
    print(f"Using enhanced bound propagation method {bounding_method} "
            f"with {branching_method} branching.")

    global_lb, ret = net.build(
        init_domain, x, stop_criterion_func=stop_func(rhs),
        bounding_method=bounding_method)
    if hasattr(net.net[net.net.input_name[0]], "lA"):
        lA = net.net[net.net.input_name[0]].lA.transpose(0, 1)
    else:
        raise ValueError("sb heuristic cannot be used without lA.")
    dm_l = x.ptb.x_L
    dm_u = x.ptb.x_U

    # compute initial split idx for the enhanced method
    split_idx = eval(arguments.Config['bab']['branching']['input_split']['branching_heuristic'])(
        net, global_lb, dm_l, dm_u, lA, rhs, branching_method,
        storage_depth, num_iter=num_iter)

    use_alpha = input_split_args["enhanced_bound_prop_method"] == "alpha-crown"
    if use_reordered_bab:
        domains = UnsortedInputDomainList(storage_depth, use_alpha=use_alpha, use_split_idx=False)
        split_depth = get_split_depth(dm_l, split_partitions=split_partitions)
        dm_l, dm_u, cs, thresholds, split_depth, global_lb, lA, _ = input_split_parallel(
            dm_l, dm_u, x.shape, net.c, rhs, split_depth=split_depth, i_idx=split_idx,
            split_partitions=split_partitions, split_hint=split_hint, dm_lb=global_lb, lA=lA)
        alphas = ret["alphas"]
        # this will double alpha for the new domains
        if isinstance(alphas, dict):
            alphas = repeat_alphas(alphas, split_depth, split_partitions)
        domains.add(
            global_lb, dm_l.detach(), dm_u.detach(), alphas,
            cs, thresholds, split_idx=None)
        global_lb = global_lb.max()
    else:
        domains = UnsortedInputDomainList(storage_depth, use_alpha=use_alpha, use_split_idx=True)
        # This is the first batch of initial domain(s) after the branching method changed.
        domains.add(
            global_lb, dm_l.detach(), dm_u.detach(), ret["alphas"],
            net.c, rhs, split_idx=split_idx)
        global_lb = global_lb.max()

    return global_lb, domains, branching_method, bounding_method


def analyze_search_space(x, x_L, x_U):
    return
    num_domains, num_inputs = x_L.shape[:2]
    global_x_L = x.ptb.x_L
    global_x_U = x.ptb.x_U
    
    # [num_domain, num_input]
    normed_x_L = (x_L - global_x_L) / (global_x_U - global_x_L)
    normed_x_U = (x_U - global_x_L) / (global_x_U - global_x_L)
    domain_center = (normed_x_L + normed_x_U) / 2
    # sort domain center by distance to origin
    sorted_indices = torch.norm(domain_center, p=1, dim=1).argsort()
    domain_center = domain_center[sorted_indices]
    normed_x_L = normed_x_L[sorted_indices]
    normed_x_U = normed_x_U[sorted_indices]
    covered_space = normed_x_U - normed_x_L
    space_size = covered_space.prod(-1)

    domain_distribution = torch.stack([normed_x_L,normed_x_U], -1)
    for i in range(num_domains):
        # print(f"Domain {i}: {normed_x_L[i].flatten().tolist()} to {normed_x_U[i].flatten().tolist()}")
        print(f"Domain {i}")
        print(f"  Center: {domain_center[i].flatten().tolist()}")
        print(f"  Converage: {covered_space[i].flatten().tolist()}")
        print(f"  Space size: {space_size[i].item()}")

    total_space_size = space_size.sum().item()
    print(f"Current search space ratio: {total_space_size:.4f}")

    # heatmap of distances
    domain_center_for_dist = domain_center.transpose(0, 1).unsqueeze(-1)
    # [num_input, num_domain, num_domain]
    distances_per_input = torch.cdist(domain_center_for_dist, domain_center_for_dist, p=1)
    # [num_domain, num_domain]
    distances = distances_per_input.sum(dim=0)
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 20))
    sns.heatmap(distances.cpu().numpy(), annot=True, fmt=".2f")
    plt.title('Distance between domain centers')
    plt.xlabel('Domain index')
    plt.ylabel('Domain index')
    plt.tight_layout()
    plt.savefig('./domain_distance.png')
    print("Domain distance saved to ./domain_distance.png")
    
    # heatmap of cosine similarity
    domain_center_for_cos = 2 * domain_center - 1
    domain_center_for_cos = domain_center_for_cos / torch.norm(domain_center_for_cos, p=2, dim=1, keepdim=True)
    cosine_similarity = torch.mm(domain_center_for_cos, domain_center_for_cos.transpose(0, 1))
    plt.figure(figsize=(20, 20))
    sns.heatmap(cosine_similarity.cpu().numpy(), annot=True, fmt=".2f")
    plt.title('Cosine similarity between domain centers')
    plt.xlabel('Domain index')
    plt.ylabel('Domain index')
    plt.tight_layout()
    plt.savefig('./domain_cosine_similarity.png')
    print("Domain cosine similarity saved to ./domain_cosine_similarity.png")

    num_samples = 1000 + 1
    x_values = torch.linspace(0, 1, steps=num_samples, device=domain_distribution.device)
    domain_distribution = domain_distribution.unsqueeze(2)
    is_within_bounds = (x_values >= domain_distribution[..., 0]) & (x_values < domain_distribution[..., 1])
    domain_sum = is_within_bounds.sum(dim=0).float()/ num_domains
    
    x_values = x_values.cpu().numpy()
    domain_sum = domain_sum.cpu().numpy()
    import numpy as np
    previous_sum = np.zeros(num_samples)
    plt.figure(figsize=(5, num_inputs))
    for i in range(num_inputs):
        plt.fill_between(x_values, previous_sum, previous_sum + domain_sum[i], label=f'Input dimension {i}')
        previous_sum += 1
    plt.xlim(0, 1)
    plt.ylim(0, num_inputs)
    plt.title('Stacked domain focus across all input dimensions')
    plt.xlabel('Input range [0, 1]')
    plt.ylabel('Stacked sum of domain indicators')
    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.savefig('./domain_focus.png')
    plt.close()
    print("Domain focus saved to ./domain_focus.png")
    pass

