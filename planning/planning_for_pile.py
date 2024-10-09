import os
import sys

sys.path.append(os.getcwd())
cwd = os.getcwd()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import time
import copy
import torch
import torch.nn.functional as F
import yaml
from functools import partial

from model.wrapped_model_gnn import wrapped_model_gnn
from planning.sampling_planner import Sampling_Planner
from others.helper import *
from others.meter import CostMeter
from Verifier_Development.complete_verifier.abcrown import ABCROWN
from planning.utils import *
import tasks.obj_pile.objpile_sim as objpile_sim

# state: obj_num, 2
# state: n_sample, length, state_dim
def cost_function(state, cls_idx, cost_norm, target_state, initial_state=None, cost_weight=1.0, final_state=None, fix_others=False):
    # print(f"state: {state.shape}")
    assert target_state is not None
    n_sample, length, state_dim = state.shape
    state = state.reshape(n_sample, length, state_dim//2, 2)
    initial_state = initial_state.reshape(state_dim//2, 2) if initial_state is not None else None
    return_np = False
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32)
        return_np = True
    if initial_state is not None:
        initial_state = torch.tensor(initial_state, dtype=state.dtype, device=state.device)
    target_state = target_state.reshape(state_dim//2, 2)
    target_state = torch.tensor(target_state, dtype=state.dtype, device=state.device)
    final_state = final_state.reshape(state_dim//2, 2).to(state.dtype).to(state.device) if final_state is not None else None

    cost = 0
    for i, idx_in_cls in enumerate(cls_idx):
        # B, n_his, n_obj_one_cls, 2
        state_in_one_cls = state[:, :, idx_in_cls]
        if fix_others and i != 0:
            cost += cost_weight * torch.norm((state_in_one_cls - initial_state[idx_in_cls]).reshape(n_sample, length, -1), dim=-1, p=cost_norm) ** cost_norm
        else:
            cls_cost = torch.norm((state_in_one_cls - target_state[idx_in_cls]).reshape(n_sample, length, -1), dim=-1, p=cost_norm) ** cost_norm
            if (not fix_others) and i == 0:
                cls_cost *= cost_weight
            cost += cls_cost
            # from real_new
            # if final_state is not None:
            #     cost += final_state_weight * torch.norm((state_in_one_cls - final_state[idx_in_cls]).reshape(n_sample, length, -1), dim=-1, p=cost_norm) ** cost_norm
            # if i == 0:
            #     cost += cls_weight * torch.norm((state_in_one_cls - target_state[idx_in_cls]).reshape(n_sample, length, -1), dim=-1, p=cost_norm) ** cost_norm
            # cost += torch.norm((state_in_one_cls - target_state[idx_in_cls]).reshape(n_sample, length, -1), dim=-1, p=cost_norm) ** cost_norm

    if return_np:
        cost = cost.cpu().numpy()
    return cost

# [n_his=1, state_dim], [n_sample, n_look_ahead, action_dim] -> [n_sample, n_look_ahead, state_dim]
def rollout(state_cur: torch.Tensor, act_seqs: torch.Tensor, model, state_dim, action_norm):
    # [n_sample, n_his, state_dim]
    n_sample = act_seqs.shape[0]
    
    state_cur = state_cur.unsqueeze(0).repeat(n_sample, 1, 1)
    assert state_cur.shape[1] == 1 # n_his=1
    n_his = state_cur.shape[1]
    output = []
    n_look_ahead = act_seqs.shape[1]
    # state_cur: [n_sample, n_his, state_dim]
    # act_seqs: [n_sample, n_look_ahead, action_dim]
    # n_his=1 simplified the operation
    act_seqs = act_seqs * action_norm
    for i in range(n_look_ahead):
        input = {
            "state": state_cur[:, :, :state_dim],
            "action": act_seqs[:, i : i + 1, :],
        }
        # [n_sample, state_dim]
        state_nxt = model(input)
        # [n_sample, 1, state_dim]
        state_nxt = state_nxt.unsqueeze(1)
        
        state_cur = torch.cat([state_cur[:, 1:], state_nxt], 1)
        
        output.append(state_nxt)

    output = torch.cat(output, 1)
    return {"state_seqs": output}  # [n_sample, n_look_ahead, state_dim]

# [n_sample, n_look_ahead, state_dim], [n_sample, n_look_ahead, action_dim] -> [n_sample]
def reward(state_mppi, action_mppi, param_dict, forbidden=False):
    cost_func = param_dict["cost_func"]
    only_final_cost = param_dict["only_final_cost"]
    target_state = param_dict["target_state"]
    state_dim = param_dict["state_dim"]
    scale = param_dict["scale"]
    initial_state = param_dict.get("initial_state", None)
    fix_others = param_dict["fix_others"]
    n_sample, n_look_ahead = action_mppi.shape[:2]
    device = state_mppi.device
    penalty = 0
    
    action_mppi = action_mppi * param_dict["action_norm"]
    # [n_sample, n_look_ahead]
    if forbidden is True:
        # forbid the pusher start pos to be too close to the object
        # n_sample, n_look_ahead, obj_num, 2
        forbidden_radius = param_dict["forbidden_radius"]
        far_factor = param_dict["far_factor"]
        curr_state = param_dict["curr_state"].view((1, 1, state_dim//2, 2)).repeat(n_sample, 1, 1, 1)
        forbidden_centers = state_mppi.view((n_sample, n_look_ahead, state_dim//2, 2))
        forbidden_centers = torch.cat([curr_state, forbidden_centers], dim=1)[:, :-1] # [n_sample, n_look_ahead, obj_num, 2]
        obj_size = param_dict["obj_size"] / scale
        pusher_start_pos = action_mppi[:, :, :2].unsqueeze(2) # n_sample, n_look_ahead, 2
        dist = pusher_start_pos - forbidden_centers
        sqr_dist = (dist ** 2).sum(-1)
        min_sqr_dist = torch.min(sqr_dist, dim=-1)[0] # [n_sample, n_look_ahead]
        close_penalty = F.relu((obj_size + forbidden_radius)**2 - min_sqr_dist)
        far_penalty = F.relu(min_sqr_dist - (far_factor*obj_size)**2)
        # far_penalty = 0
        penalty_factor = 1e5
        penalty = penalty_factor * (close_penalty + far_penalty).sum(dim=-1)
        

    cost_seqs = cost_func(state_mppi, target_state=target_state, initial_state=initial_state, fix_others=fix_others) # [n_sample, n_look_ahead]
    step_weight = torch.arange(1, n_look_ahead + 1, device=device).float() / n_look_ahead
    return {
        "rewards": -(cost_seqs*step_weight).sum(dim=-1)-penalty if not only_final_cost else -cost_seqs[:, -1]-penalty,
        "cost_seqs": cost_seqs,
    }

def planning(
    config,
    original_config, 
    open_loop,
    method_type,
    model,
    init_pose,
    target_pose,
    init_pusher_pos,
    vis_file_name_prefix=None,
    verbose=True
):
    assert method_type in ["MPPI_GD", "MPPI", "GD", "CROWN", "MPPI_BF", "GD_BF", "MIP", "CEM", "DecentCEM"]
    start = time.time()
    task_name = config["task_name"]
    real_exp_config = config["real_exp_config"]
    enable_real = real_exp_config["enable"]
    enable_vis, save_result = config["enable_vis"], config["save_result"]
    use_prev_sol, fixed_horizon = config["use_prev_sol"], config["fixed_horizon"]
    only_final_cost, cost_norm = config["only_final_cost"], config["cost_norm"]
    assert cost_norm in [1, 2]
    n_his = config["n_his"]
    assert n_his == 1
    n_sim_step = config["n_sim_step"]
    if open_loop:
        max_horizon = config["horizon"]
        max_n_act_step = max_horizon
        n_sim_step = max_horizon
    else:
        max_horizon = min(config["horizon"], n_sim_step)
        max_n_act_step = min(config["n_act_step"], max_horizon)
    if enable_real:
        time_limit = real_exp_config["time_limit"]
        fixed_horizon = True
    else:
        time_limit = n_sim_step
    device, seed = config["device"], config["seed"]
    state_dim, action_dim = config["state_dim"], config["action_dim"]
    fix_others = config["fix_others"]
    cost_weight = config["cost_weight"]
    warm_start_from_sampling, warmstart_file = config["warm_start_from_sampling"], config["crown_name_dict"]["warmstart_file"]
    lo, hi = config["pusher_lo"], config["pusher_hi"]
    action_bound, scale = config["action_bound"], config["scale"]
    pusher_lo = config["pusher_lo"] / scale
    pusher_hi = config["pusher_hi"] / scale
    action_lb = torch.tensor([pusher_lo, pusher_lo, -action_bound, -action_bound], device=device)
    action_ub = torch.tensor([pusher_hi, pusher_hi, action_bound, action_bound], device=device)
    action_norm = action_ub - action_lb
    action_lb /= action_norm
    action_ub /= action_norm
    task_spec_dict = config["task_spec_dict"]
    task_spec_dict["save_img"] = save_result
    task_spec_dict["enable_vis"] = enable_vis
    
    if task_name == "obj_pile":
        env = objpile_sim.ObjectPileSim(task_spec_dict, init_poses=init_pose, pusher_pos=init_pusher_pos)
    else:
        raise NotImplementedError

    # define cost function
    cls_num = env.classes
    labels = env.label_list
    
    cls_idx = []
    for i in range(cls_num):
        cls_idx.append(np.where(np.array(labels) == i)[0])
    
    # allow the simulator to a resting position
    for i in range(2):
        env.update(init_pusher_pos)
        env.wait(1)
    env_state = env.get_current_state().flatten()
    initial_state = env_state[:state_dim]/scale if fix_others else None
    # target_state = np.array(target_pose) / scale
    target_state = []
    for obj in labels:
        target_state.append(target_pose[obj])
    target_state = np.array(target_state).flatten()/scale
    cost_func = partial(cost_function, cls_idx=cls_idx, cost_norm=cost_norm, target_state=target_state, initial_state=initial_state, cost_weight=cost_weight, fix_others=fix_others)

    # Initialize planner based on method_type
    enable_sampling = "MPPI" in method_type or "GD" in method_type or 'CEM' in method_type or (method_type == "CROWN" and warm_start_from_sampling)
    if enable_sampling:
        reward_config = {"scale": scale, "state_dim": state_dim, "cost_func": cost_func, "obj_size": original_config['data']["obj_size"], "fix_others": fix_others,
                            "forbidden_radius": original_config['planning']["forbidden_radius"], "far_factor": original_config['planning']["far_factor"],
                            "only_final_cost": only_final_cost, "target_state":target_state, "initial_state":initial_state, "action_norm": action_norm}
        reward_func = partial(reward, param_dict=reward_config, forbidden=original_config["planning"]["forbidden"])
        
        config["planner_type"] = method_type
        if method_type == "CROWN" and warm_start_from_sampling:
            config["planner_type"] = original_config["planning"].get("warm_start_method", "MPPI")
        config["model_rollout_fn"] = partial(rollout, model=model, state_dim=state_dim, action_norm=action_norm)
        config["evaluate_traj_fn"] = reward_func
        config["action_lower_lim"] = action_lb
        config["action_upper_lim"] = action_ub
        config["n_look_ahead"] = max_horizon
        planner_mppi = Sampling_Planner(config)
    if method_type == "CROWN":
        abcrown: ABCROWN = config["abcrown"]
        cost_model:wrapped_model_gnn = config["cost_model"]
        const_nodes_dict = config["const_nodes_dict"]
        crown_config_dict = config["crown_config_dict"]
        model_config = config["model_config"]
        crown_name_dict, crown_config_dict = config["crown_name_dict"], config["crown_config_dict"]
        crown_config_prefix = crown_name_dict["crown_config_prefix"]
        vnnlib_name = crown_name_dict["vnnlib_name"]
        vnnlib_path = f"{crown_config_prefix}{vnnlib_name}"
        crown_config_name = crown_name_dict["crown_config_name"]
        model_file_name = config["model_file_name"]
        crown_sol_file = crown_name_dict["crown_sol_file"]
        abcrown_verbose = config["abcrown_verbose"]
        if abcrown_verbose:
            bab_file = crown_name_dict["bab_file"]
        use_empirical_bounds = config["use_empirical_bounds"]
        # [1, N*N, N], [1, N*N, N]
        Rr, Rs = model.get_Rr_Rs(state_dim//2)
        # target_pose: n_cls, 2
        update_const_nodes(const_nodes_dict, abcrown, cost_model, 
                            {"target_state": torch.tensor(target_state, dtype=torch.float32).view(1, state_dim//2, 2), "Rr": Rr, "Rs": Rs})
        if fix_others:
            update_const_nodes(const_nodes_dict, abcrown, cost_model, 
                                {"initial_state": torch.tensor(initial_state.reshape(1, state_dim//2, 2), dtype=torch.float32)})
        cost_model.cls_idx = cls_idx
        abcrown.model.model_ori.cls_idx = cls_idx
    # Core simulation loop
    all_res = []
    cost_meter = CostMeter()
    exec_action_seq, pred_state_seq = [], []
    success = True
    gt_states = []
    # number of steps to actuate the pusher, increase after planning, decrease after acting
    n_to_act = 0
    i = 0
    start_time = time.time()
    frame_time = time.time()
    stop_time = 0.2
    exec_time = 0
    planning_time = 0

    while exec_time <= time_limit:
        if enable_real:
            while time.time() - frame_time < stop_time:
                time.sleep(0.01)
            frame_time = time.time()
        # get state
        if enable_real:
            raise NotImplementedError
        else:
            env_state = env.get_current_state().flatten()
        # update the cost meter
        if i > 0:
            gt_states[-1][-4:-2] = [x_pusher_start, y_pusher_start]
            gt_states[-1][-2:] = env_state[-2:]
            planned_step_cost = cost_func((state_planner[:state_dim])[np.newaxis,np.newaxis,:,])[-1, -1]
            actual_step_cost = cost_func((env_state[:state_dim]/scale)[np.newaxis,np.newaxis,:,])[-1, -1]
            cost_meter.update(actual_step_cost, planned_step_cost, i-1)
        gt_states.append(env_state.tolist())

        if exec_time >= time_limit:
            # final step
            actual_final_cost = actual_step_cost
            break
        # do planning only when there is no action to actuate the pusher
        if n_to_act == 0:
            reset_seed(seed)
            horizon = max_horizon if fixed_horizon else min(max_horizon, n_sim_step - i)
            state = env_state[:state_dim] / scale
            n_act_step = min(max_n_act_step, horizon)
            x_pusher, y_pusher = env.get_pusher_position()
            if enable_sampling:
                if use_prev_sol and (i != 0):
                    mppi_init_act_seq = torch.cat(
                        (
                            mppi_init_act_seq[prev_n_act_step:],
                            torch.rand(
                                (
                                    horizon - (len(mppi_init_act_seq) - prev_n_act_step),
                                    action_ub.shape[0],
                                ),
                                device=device,
                            )
                            * (action_ub - action_lb)
                            + action_lb,
                        ),
                        dim=0,
                    )
                else:
                    mppi_init_act_seq = (
                        torch.rand((horizon, action_lb.shape[0]), device=device) * (action_ub - action_lb) + action_lb
                    )
                reward_config["curr_state"] = env_state_to_input(state[:state_dim], device=device)
                planning_start = time.time()
                res = planner_mppi.trajectory_optimization(env_state_to_input(state[:state_dim], device=device), mppi_init_act_seq)
                planning_time += time.time() - planning_start
                mppi_init_act_seq = res["act_seq"]
                if open_loop:
                    planned_final_cost = -res['best_eval_output']['rewards'].item()
                    print(f"mppi cost: {planned_final_cost}") 
                # could save the result to warmstart crown
                if warm_start_from_sampling and method_type == "CROWN":
                    # [n_sample, horizon, action_dim] -> [n_sample, horizon*action_dim]
                    warmstart_act_seqs = torch.cat(
                        [
                            res["act_seq"].view(-1, action_dim * horizon),
                            res["act_seqs_lst"][-1].view(-1, action_dim * horizon),
                        ]
                    )
                    # [n_sample]
                    warmstart_cost_seqs = -torch.cat(
                        [
                            res["best_eval_output"]["rewards"],
                            res["eval_outputs"][-1]["rewards"],
                        ]
                    )
                    warmstart_cost_seqs, sort_idx = warmstart_cost_seqs.sort(descending=False)
                    warmstart_act_seqs = warmstart_act_seqs[sort_idx]
                    # [n_sample, 1]
                    warmstart_cost_seqs = warmstart_cost_seqs.unsqueeze(1)
                    torch.save([warmstart_act_seqs, warmstart_cost_seqs], warmstart_file)
                    del warmstart_act_seqs, warmstart_cost_seqs
                # res['act_seq']: [horizon, action_dim]
                res["act_seq"] = res["act_seq"] * action_norm
                action_sequence = res["act_seq"].detach().cpu().numpy()
                state_sequence = res["best_model_output"]["state_seqs"][0].detach().cpu().numpy()
                del res["model_outputs"], res["eval_outputs"], res["act_seqs_lst"]
            reset_seed(seed)
            if method_type == "CROWN":
                if (i != 0) and (horizon != prev_horizon):
                    model_config["horizon"] = horizon
                    (
                        abcrown,
                        cost_model,
                        const_nodes_dict
                    ) = create_abcrown(
                        crown_name_dict,
                        crown_config_dict,
                        model_file_name,
                        model_config,
                        use_empirical_bounds,
                    )
                    # target_pose: n_cls, 2
                    update_const_nodes(const_nodes_dict, abcrown, cost_model, {"target_state": torch.tensor(target_state, dtype=torch.float32).view(1, state_dim//2, 2), "Rr": Rr, "Rs": Rs})
                    if fix_others:
                        update_const_nodes(const_nodes_dict, abcrown, cost_model, {"initial_state": torch.tensor(initial_state.reshape(1, state_dim//2, 2), dtype=torch.float32)})
                with open(vnnlib_path, "w") as f:
                    print_prop(i, f, action_dim*horizon, action_lb, action_ub)
                # # update model
                known_input = env_state_to_input(state[:state_dim], device).view(1, state_dim//2, 2)
                update_const_nodes(const_nodes_dict, abcrown, cost_model, {"known_input": known_input})
                if use_prev_sol and (not warm_start_from_sampling) and (i != 0):
                    # action_sequence_copy[prev_n_act_step:]: [prev_horizon-prev_n_act_step, action_dim]
                    # repeat it to [B, prev_horizon-prev_n_act_step, action_dim] and concat with random action
                    num_warmstart = 10000
                    warmstart_act_seqs = (
                        torch.Tensor(action_sequence_copy[prev_n_act_step:])[None]
                        .to(device)
                        .repeat(num_warmstart, 1, 1)
                    )
                    warmstart_act_seqs = torch.cat(
                        (
                            warmstart_act_seqs,
                            torch.rand(
                                (
                                    num_warmstart,
                                    horizon - (action_sequence_copy.shape[0] - prev_n_act_step),
                                    action_ub.shape[0],
                                ),
                                device=device,
                            )
                            * (action_ub - action_lb)
                            + action_lb,
                        ),
                        dim=1,
                    )
                    # warmstart_act_seqs: [B, horizon, action_dim], warmstart_cost_seqs: [B, 1]
                    warmstart_cost_seqs = cost_model(warmstart_act_seqs).view(-1, 1)
                    warmstart_cost_seqs, sort_idx = warmstart_cost_seqs.sort(descending=False, dim=0)
                    warmstart_act_seqs = warmstart_act_seqs[sort_idx.squeeze()]
                    torch.save(
                        [
                            warmstart_act_seqs,
                            warmstart_cost_seqs,
                        ],
                        warmstart_file,
                    )
                    del warmstart_act_seqs, warmstart_cost_seqs
                # # export model for debugging
                # dummy_input = torch.randn(1, horizon, action_dim, device=device)
                # torch.onnx.export(abcrown.model.model_ori, dummy_input, "model_ori.onnx", verbose=True)
                # torch.onnx.export(abcrown.model.net, dummy_input, "model_lirpa.onnx", verbose=True)
                planning_start = time.time()
                abcrown.main()
                planning_time += time.time() - planning_start
                os.system(f"rm -rf {warmstart_file}")
                feasible_sol, best_output, _, intermediate_best_outputs = parse_result(crown_sol_file, abcrown_verbose)[:4]
                assert feasible_sol is not None
                if open_loop:
                    planned_final_cost = best_output[0]
                    print(f"crown cost: {planned_final_cost}") 
                action_sequence = feasible_sol.reshape(horizon, action_dim) * action_norm.cpu().numpy()
                # [B, n_history, obs_dim], [B, n_history + n_rollout - 1, action_dim], n_history = 1
                state_cur_tensor = torch.from_numpy(state[:state_dim][None][None]).float().to(device)
                action_cur_tensor = torch.from_numpy(action_sequence[None]).float().to(device)
                # state_pred: [B=1, n_rollout, obs_dim]
                state_sequence = model.rollout_model({"state_init": state_cur_tensor, "action_seq": action_cur_tensor})["state_pred"]
                state_sequence = state_sequence.squeeze(0).detach().cpu().numpy()
                res = {
                    "act_seq": action_sequence,
                    "state_seq": state_sequence,
                    "planned_cost": best_output,
                    "intermediate_best_outputs": intermediate_best_outputs,
                }
                if abcrown_verbose:
                    space_size_dict = analyze_search_progress(bab_file, vis_file_name_prefix)
                    res["space_size_dict"] = space_size_dict
            n_to_act = n_act_step
            prev_n_act_step = n_act_step
            prev_horizon = horizon
            
            # copy for later use
            action_sequence_copy = action_sequence.copy()
            state_sequence_copy = state_sequence.copy()
            
            res["state_seq"] = (state_sequence * scale).tolist()
            res["start_step"] = i
            all_res.append(res)

        # calculate the action
        action = action_sequence_copy[n_act_step - n_to_act]
        state_planner = state_sequence_copy[n_act_step - n_to_act]

        x_pusher_start, y_pusher_start = action[:2] * scale
        dx, dy = action[2:] * scale
        
        exec_action_seq.append((scale * action[2:]).tolist())
        pred_state_seq.append((scale * state_planner[:state_dim]).tolist())
        x_pusher = np.clip(x_pusher_start + dx, lo, hi)
        y_pusher = np.clip(y_pusher_start + dy, lo, hi)
        if verbose:
            print(f"{'-'*5}step {i}{'-'*5}  Planned action: {(x_pusher_start,y_pusher_start,x_pusher,y_pusher)}")
        # execute the action
        if enable_real:
            raise NotImplementedError
        else:
            # env.set_pusher_pose([x_pusher_start, y_pusher_start], [x_pusher-x_pusher_start, y_pusher-y_pusher_start])
            env.set_pusher_position(x_pusher_start, y_pusher_start)
            env.update((x_pusher_start + (x_pusher-x_pusher_start)/1e4, y_pusher_start + (y_pusher-y_pusher_start)/1e4), steps=1)
            env.update((x_pusher, y_pusher))
            if open_loop and enable_vis:
                time.sleep(0.1)

        if target_pose is not None:
            print(f"state_planner: {state_planner[:state_dim]} target_state:{target_state}")

        n_to_act -= 1
        i += 1
        if enable_real:
            exec_time = time.time() - start_time
        else:
            exec_time = i

    if success:
        cost_meter.print_metric()
    if not enable_real and save_result:
        assert vis_file_name_prefix is not None
        imageio.mimsave(vis_file_name_prefix + f"_ori_{method_type}_{round(actual_final_cost, 3):.3f}.gif", env.frames, fps=5)
        
    if enable_real:
        env.stop()
    # reformat the result,
    all_res = format_all_res(method_type, scale, all_res)

    exp_setting = {
        "open_loop": open_loop,
        "method_type": method_type,
        "init_pose": init_pose,
        "target_pose": target_pose,
        "init_pusher_pos": init_pusher_pos,
        "n_sim_step": n_sim_step,
        "horizon": horizon,
        "enable_vis": enable_vis,
        "save_result": save_result,
        "fixed_horizon": fixed_horizon,
    }
    if success:
        result_summary = {
            "success": success,
            "final_cost": actual_final_cost,
            "gt_states": gt_states,
            "exec_action_seq": exec_action_seq,
            "pred_state_seq": pred_state_seq,
            "runtime": planning_time,
            "labels": env.label_list,
        }
        result_summary.update(cost_meter.get_metric())
        if open_loop:
            planned_cost = cost_meter.planned_cost_seq[-1] if only_final_cost else planned_final_cost
            actual_cost = cost_meter.actual_cost_seq[-1] if only_final_cost else cost_meter.actual_acc_cost
            result_summary["planned_cost"] = planned_cost
            result_summary["actual_cost"] = actual_cost
            result_summary["cost_diff"] = actual_cost - planned_cost
    else:
        result_summary = {"success": success, "runtime": planning_time}
    del env
    return {
        "exp_setting": exp_setting,
        "result_summary": result_summary,
        "all_res": all_res,
    }
