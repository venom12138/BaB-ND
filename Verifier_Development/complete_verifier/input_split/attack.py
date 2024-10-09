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
import torch
import arguments
from attack.attack_pgd import (
    pgd_attack_with_general_specs, default_adv_example_finalizer, test_conditions, check_and_save_cex,
    process_vnn_lib_attack, build_conditions)
from load_model import Customized


def attack_in_input_bab_parallel(model_ori, domains, x, vnnlib=None):
    ## pack the domain list
    lbs, ubs, Cs, rhs = [], [], [], []
    for idx in range(len(domains)):
        val = domains[idx]
        lbs.append(val[1][None, ...])
        ubs.append(val[2][None, ...])
        Cs.append(val[3][None, ...])
        rhs.append(val[4][None, ...])

    lbs = torch.cat(lbs, dim=0)
    # [num_or_spec, input_shape]
    ubs = torch.cat(ubs, dim=0)
    # [num_or_spec, input_shape]
    Cs = torch.cat(Cs, dim=0)
    # [num_or_spec, num_and_spec, output_dim]
    rhs = torch.cat(rhs, dim=0)
    # [num_or_spec, num_and_spec]

    cond_mat = [[Cs.shape[1]]*Cs.shape[0]]
    Cs = Cs.view(1, -1, Cs.shape[-1])
    # [num_example, num_spec, num_output]
    rhs = rhs.view(1, -1)
    # [num_example, num_spec]
    lbs = lbs.unsqueeze(0)
    ubs = ubs.unsqueeze(0)
    # [num_example, num_or_spec, input_shape]

    if arguments.Config["attack"]["input_split"]["pgd_alpha"] == "auto":
        alpha = (ubs - lbs).max() / 4
    else:
        alpha = float(arguments.Config["attack"]["input_split"]["pgd_alpha"])
    # pack the domains as a large spec matrix

    num_restarts = arguments.Config["attack"]["input_split"]["pgd_restarts"]
    num_steps = arguments.Config["attack"]["input_split"]["pgd_steps"]

    device = x.device
    lbs = lbs.to(device)
    ubs = ubs.to(device)
    rhs = rhs.to(device)
    Cs = Cs.to(device)

    attack_x = ((lbs + ubs)/2).squeeze(0)

    best_deltas = pgd_attack_with_general_specs(
        model_ori, attack_x, lbs, ubs, Cs, rhs, cond_mat,
        same_number_const=True, alpha=alpha,
        pgd_steps=num_steps, num_restarts=num_restarts)[0]
    attack_image, attack_output, _ = eval(
        arguments.Config["attack"]["adv_example_finalizer"]
    )(model_ori, attack_x, best_deltas, ubs, lbs, Cs, rhs, cond_mat)

    res, idx = test_conditions(attack_image.unsqueeze(1), attack_output.unsqueeze(1),
                           Cs, rhs, cond_mat, True, ubs, lbs, return_success_idx=True)
    if res.all():
        print("pgd attack succeed in input bab parallel, with idx:", idx)
        _, verified_success = check_and_save_cex(attack_image[:, idx], attack_output[:, idx], vnnlib,
                                                 arguments.Config["attack"]["cex_path"], "unsafe")
                
        return verified_success
        
    return False


def massive_pgd_attack(x, model_ori, vnnlib=None, num_restarts=None, num_steps=None, preload=False, lirpa_model=None):
    """pgd attack with very large number of random starts
    init_domain: [input_shape, 2]
    rhs_mat: [num_or(1), num_and]
    x: [batch(1), input_shape]
    C_mat: [num_and, out_dim]
    """
    if num_restarts is None:
        num_restarts = arguments.Config["attack"]["input_split_enhanced"]["pgd_restarts"]
    if num_steps is None:
        num_steps = arguments.Config["attack"]["input_split_enhanced"]["pgd_steps"]

    list_target_label_arrays, data_min, data_max = process_vnn_lib_attack(vnnlib, x)
    C_mat, rhs_mat, cond_mat, same_number_const = build_conditions(x, list_target_label_arrays)
    data_min = data_min[:, :len(list_target_label_arrays[0]), ...]
    data_max = data_max[:, :len(list_target_label_arrays[0]), ...]

    if arguments.Config["attack"]["pgd_alpha"] == "auto":
        alpha = (data_max - data_min).max() / arguments.Config["attack"]["pgd_auto_factor"]
    else:
        alpha = float(arguments.Config["attack"]["input_split_enhanced"]["pgd_alpha"])

    adv_examples, adv_outputs = None, None
    if preload and arguments.Config["find_feasible_solution"]["preload_path"]:
        preload_path = arguments.Config["find_feasible_solution"]["preload_path"]
        print(f'Loading preloaded adv examples from {preload_path}')
        try:
            adv_examples, adv_outputs = torch.load(preload_path)
            adv_examples = adv_examples.to(x.device).view(-1, *x.shape[1:])
            adv_outputs = adv_outputs.to(x.device)
            model_outputs = model_ori(adv_examples)
            # assume shape consistent: [num_restarts, output_dim]
            mask = torch.isclose(model_outputs, adv_outputs, rtol=5e-2).flatten()
            # import pdb; pdb.set_trace()
            adv_examples = adv_examples[mask][:num_restarts]
            adv_outputs = adv_outputs[mask][:num_restarts]
            if len(adv_examples) > 0:
                print(f'Preloaded adv examples are valid, loaded {len(adv_examples)} adv examples')
                print('The first adv example and output: ', adv_examples[0], adv_outputs[0])
            else:
                print('Preloaded adv examples are invalid')
                adv_examples, adv_outputs = None, None
        except Exception as e:
            print(str(e))
            print('Failed to load preloaded adv examples')
            adv_examples, adv_outputs = None, None 

    ret_attack = pgd_attack_with_general_specs(
        model_ori, x, data_min, data_max, C_mat, rhs_mat, cond_mat,
        same_number_const=True, alpha=alpha, num_restarts=num_restarts,
        pgd_steps=num_steps, 
        param_dict={"prev_adv_example": adv_examples, "lirpa_model": lirpa_model})
    best_deltas, deltas = ret_attack[0], ret_attack[1]
    info_dict = None
    if arguments.Config["find_feasible_solution"]["enable"] and arguments.Config["find_feasible_solution"]["search_func"] != "None":
        info_dict = ret_attack[-1]
    deltas = deltas.view(-1, *deltas.shape[3:])
    # [batch, *input_shape]
    # default_adv_example_finalizer
    # x and best_deltas has shape (batch, c, h, w).
    # data_min and data_max have shape (batch, spec, c, h, w).
    num_spec = data_min.shape[1]
    if num_spec == 1 and adv_examples is not None:
        # ANONYMOUS: add
        # deltas has shape (batch, num_restarts, spec, c, h, w).
        # adv_examples-x: (num_restarts, c, h, w).
        #  => deltas: (batch, num_restarts, c, h, w).
        ori_deltas = (adv_examples-x)
        deltas = torch.cat([deltas, ori_deltas], dim=0)
    else:
        deltas = best_deltas
    with torch.no_grad():
        attack_image, attack_output, attack_margin = eval(arguments.Config["attack"]["adv_example_finalizer"])(
            model_ori, x, deltas, data_max, data_min, C_mat, rhs_mat, cond_mat)
    num_restarts = attack_margin.shape[1]
    # attack_image: [batch, num_or_spec, *input_shape]
    # attack_output: [batch_size, num_or_spec, out_dim]
    # attack_margin: [num_example, num_restarts, num_spec]
    # if num_spec == 1 and adv_examples is not None:
    #     # ANONYMOUS: add
    #     # attack_image: [batch, spec, num_restarts, *input_shape]
    #     # attack_output: [batch, spec, num_restarts, output_dim]
    #     attack_image = attack_image.unsqueeze(2).repeat(1, 1, num_restarts, *([1]*len(attack_image.shape[2:])))
    #     attack_output = attack_output.unsqueeze(2).repeat(1, 1, num_restarts, 1)
    # else:
    #     # num_restarts = 1
    #     # attack_image: [batch, spec, num_restarts, *input_shape]
    #     # attack_output: [batch, num_or_spec, num_restarts, output_dim]
    #     # attack_margin: [num_example, num_restarts, num_spec]
    #     attack_image = attack_image.unsqueeze(2).repeat(1, 1, num_restarts, *([1]*len(attack_image.shape[2:])))
    #     attack_output = attack_output.unsqueeze(2).repeat(1, 1, num_restarts, 1)
    # attack_image = attack_image.unsqueeze(1).repeat(1, num_restarts, 1, *([1]*len(attack_image.shape[2:])))
    # attack_output = attack_output.unsqueeze(1).repeat(1, num_restarts, 1, 1)
    # input: [batch, num_restarts, num_or_spec, *input_shape]
    # output: [batch, num_restarts, num_or_spec, num_output]
    if test_conditions(attack_image.unsqueeze(1).repeat(1, num_restarts, 1, *([1]*len(attack_image.shape[2:]))), attack_output.unsqueeze(1).repeat(1, num_restarts, 1, 1),
                       C_mat, rhs_mat, cond_mat, True, data_max, data_min).all():
        print("pgd attack succeed in massive attack")
        _, verified_success = check_and_save_cex(attack_image[0:1, 0:1].squeeze(1), attack_output[0:1, 0:1].squeeze(1),
                                                 vnnlib, arguments.Config["attack"]["cex_path"], "unsafe")
        return attack_margin, verified_success, attack_image, attack_output, info_dict
    else:
        return attack_margin, False, attack_image, attack_output, info_dict


def check_adv(domains, model_ori, x, vnnlib=None):
    """check whether exiting domains have adv example or not.
    By using inputs' lower and upper bound as attack starting points."""
    if len(vnnlib) != 1:
        print('Multiple x in check_adv() is not supported so far!')
        return False

    device = x.device
    max_num_domains = arguments.Config['attack']['input_split_check_adv']['max_num_domains']
    worst_indices = domains.get_topk_indices(k=min(max_num_domains, len(domains)))
    best_idx = domains.get_topk_indices(largest=True).item()
    indices = list(worst_indices.numpy()) + [best_idx]

    dm_l, dm_u, c, threshold = [], [], [], []
    for idx in indices:
        val = domains[idx]
        dm_l.append(val[1][None, ...].detach().cpu())
        dm_u.append(val[2][None, ...].detach().cpu())
        c.append(val[3][None, ...].detach().cpu())
        threshold.append(val[4].detach().cpu())

    # we pick the worst domains (smallest lower bounds) since they are less likely to be verified.
    # we use their input range: dm_l and dm_u as attacking starting points.
    starting_points = torch.cat([torch.cat([dm_l[i], dm_u[i]]) for i in range(len(worst_indices))])
    # we also include the most recent added domain to have a try.
    starting_points = torch.cat([starting_points, dm_l[-1], dm_u[-1]])
    starting_points = starting_points.unsqueeze(0).to(device, non_blocking=True)
    # [1, num_starting_points, *input_shape], num_starting_points = 2 * (worst_indices + 1)

    C_mat = torch.cat([torch.cat([c[i], c[i]]) for i in range(len(worst_indices))])
    C_mat = torch.cat([C_mat, c[-1], c[-1]]).to(device, non_blocking=True)
    # [num_starting_points, num_and, output_dim]

    rhs_mat = [threshold[i] for i in range(len(worst_indices))]
    rhs_mat.append(threshold[-1])
    rhs_mat = torch.stack(rhs_mat).repeat_interleave(2, dim=0)
    # [num_starting_points, num_and]

    # we need to manually construct condition/property/rhs matrix with the PGD random starts as num_starting_points
    cond_mat = [[C_mat.shape[1]] * C_mat.shape[0]]
    # list: [num_and, num_and, num_and ,...] with the length of num_starting_points
    prop_mat = C_mat.view(1, -1, C_mat.shape[-1])
    # [1, num_starting_points * num_and, output_dim]
    rhs_mat = rhs_mat.view(1, -1).to(device, non_blocking=True)
    # [1, num_starting_points * num_and]

    data_min = x.ptb.x_L.unsqueeze(1)
    data_max = x.ptb.x_U.unsqueeze(1)
    # [1, 1, *input_shape], all attack_images share the same data_min/max

    pgd_steps = arguments.Config["attack"]["input_split_check_adv"]["pgd_steps"]

    if arguments.Config["attack"]["input_split_check_adv"]["pgd_alpha"] == "auto":
        alpha = (data_max - data_min).max() / 4
    else:
        alpha = float(arguments.Config["attack"]["input_split_check_adv"]["pgd_alpha"])

    best_deltas = pgd_attack_with_general_specs(
        model_ori, starting_points, data_min, data_max, prop_mat, rhs_mat,
        cond_mat, same_number_const=True, alpha=alpha,
        pgd_steps=pgd_steps, only_replicate_restarts=True)[0]

    attack_image = best_deltas + starting_points.squeeze(1)
    attack_image = torch.min(torch.max(attack_image, data_min), data_max)
    # [1, num_starting_points, *input_shape]

    attack_output = model_ori(attack_image.view(-1, *attack_image.shape[2:])).view(
        *attack_image.shape[:2], -1)
    # [1, num_starting_points, output_dim]

    # in test_conditions() the attack_image and attack_output requires the shape:
    # [num_example, num_restarts, num_or_spec, *input_shape]
    # We currently don't have num_restarts dim, so we unsqueeze(1) for them.
    res, idx = test_conditions(attack_image.unsqueeze(1), attack_output.unsqueeze(1), prop_mat.unsqueeze(1), rhs_mat,
                               cond_mat, True, data_max.unsqueeze(1), data_min.unsqueeze(1), return_success_idx=True)
    if res.all():
        print("pgd attack succeed in check_adv, with idx:", idx)
        _, verified_success = check_and_save_cex(attack_image[:, idx], attack_output[:, idx], vnnlib,
                                                 arguments.Config["attack"]["cex_path"], "unsafe")
        return verified_success

    return False


def update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb, model, dm_ub=None, prev_adv_example=None, lirpa_model=None, extra_item_dict={}):
    device = x_L.device
    max_num_domains = arguments.Config['attack']['input_split_check_adv']['max_num_domains']
    num_domains = x_L.shape[0]
    assert num_domains <= max_num_domains, f'Number of domains {num_domains} exceeds the limit {max_num_domains}'
    print(f'Running PGD attack on {num_domains} domains')
    rhs = thresholds

    adv_example = ((x_L + x_U) / 2).unsqueeze(0)
    cond_mat = [[cs.shape[1]] * cs.shape[0]]
    prop_mat = cs.view(1, -1, cs.shape[-1])
    prop_rhs = rhs.view(1, -1)
    data_max = x_U.unsqueeze(0)
    data_min = x_L.unsqueeze(0)
    alpha = (data_max - data_min).max() / arguments.Config["attack"]["pgd_auto_factor"]

    # ANONYMOUS: use restarts and steps in the same part
    pgd_steps = arguments.Config["attack"]["input_split_check_adv"]["pgd_steps"]
    pgd_restarts = arguments.Config["attack"]["input_split_check_adv"]["pgd_restarts"]
    # # more restarts if there are less domains
    # pgd_restarts = int(pgd_restarts*max_num_domains/num_domains)
    # ANONYMOUS: add per_spec to get adv example for each sundomains (spec)
    per_spec = True
    require_distribution = arguments.Config["find_feasible_solution"]["enable"] and arguments.Config["find_feasible_solution"]["record_distribution_for_branching"]
    param_dict = {"per_spec":per_spec, "prev_adv_example":prev_adv_example, "lirpa_model":lirpa_model,
                  "require_distribution": require_distribution}
    ret = pgd_attack_with_general_specs(
        model, adv_example, data_min, data_max, prop_mat, prop_rhs,
        cond_mat, same_number_const=True, alpha=alpha, num_restarts=pgd_restarts,pgd_steps=pgd_steps,
        only_replicate_restarts=True, param_dict=param_dict, extra_item_dict=extra_item_dict)
    best_deltas = ret[0]
    info_dict = ret[-1]
    # # print memory allocation
    # print(f"Mem allocated: {torch.cuda.memory_allocated() / 1024 / 1024}")
    # best_deltas: (batch, specs, c, h, w), _: (batch, restarts, specs, c, h, w)
    # TODO: compare with the result before pgd, use the better one.
    attack_image = best_deltas + adv_example.squeeze(1)
    attack_image = torch.clamp(attack_image, data_min, data_max)
    attack_output = model(attack_image.view(-1, *attack_image.shape[2:])).view(
        *attack_image.shape[:2], -1
    )
    attack_output = attack_output.view(cs.shape[0], -1, 1)
    upper_bound = cs.matmul(attack_output).squeeze(-1)
    # ANONYMOUS: assume only on output constraint, only one ub, id
    best_ub, best_id = torch.min(upper_bound, dim=0)
    best_image = attack_image[:, best_id]
    best_output = attack_output[best_id]
    if dm_ub is not None:
        best_ub_prev, best_id_prev = torch.min(dm_ub, dim=0)
        if best_ub_prev < best_ub:
            best_image = prev_adv_example[best_id_prev]
            best_output = model(best_image)

    best_image_list = best_image.flatten().tolist()
    best_output_list = best_output.flatten().tolist()

    # # ANONYMOUS: try to update rhs with best upper bound and all thresholds, preform better but result changes???
    print('Trying to update RHS with attack')
    print(f'  Current RHS: mean {rhs.mean().item()}')
    print(f'  New upper bound: mean {best_ub.mean().item()}')
    print(f'  Number of updated RHS: {(best_ub < rhs).sum()}/{rhs.numel()}')   
    rhs = torch.min(rhs, best_ub)
    assert (torch.all(thresholds == thresholds[0])), f'thresholds are not the same: {thresholds}'
    thresholds = torch.min(thresholds, best_ub)

    gap = rhs - dm_lb
    min_gap = gap.min()
    print('  Gap between lower/upper bounds: '
          f'mean {gap.mean().item()}, min {min_gap.item()}')
    # ANONYMOUS: increase the gap between lower and upper bounds
    # assert min_gap >= -1e-1, f'Gap between lower and upper bounds is negative: {min_gap}'

    # ANONYMOUS: return the best adv example
    best_adv = (best_image_list, best_output_list, best_ub.item())
    return thresholds, best_adv, attack_image, upper_bound, info_dict
